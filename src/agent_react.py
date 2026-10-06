import json
import sys
import time
from typing import Any

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
_src_dir = Path(__file__).resolve().parent
_root_dir = _src_dir.parent
for _p in [str(_src_dir), str(_root_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import tools
from harness import Constraints, ExecutionHarness


SYSTEM_PROMPT = (
    "Bạn là trợ lý đặt vé máy bay thông minh. "
    "Nhiệm vụ: Sử dụng các công cụ được cung cấp để tìm và đặt vé thỏa mãn TẤT CẢ ràng buộc của người dùng. "
    "Quy trình: 1. search_flights -> 2. Chọn chuyến bay thỏa mãn -> 3. book_seat -> 4. pay. "
    "Nếu không có chuyến bay nào thỏa mãn hoặc bị từ chối, hãy dừng lại và giải thích lý do."
)


def _heuristic_react_step(history: list[dict], constraints: Constraints, bookings: dict) -> tuple[str, dict] | None:
    """Hàm mô phỏng quyết định ReAct logic khi chạy offline không có LLM API."""
    searched = any(h["tool"] == "search_flights" and h["result"].get("status") == "ok" for h in history)
    if not searched:
        return "search_flights", {
            "origin": constraints.origin,
            "destination": constraints.destination,
            "date": constraints.date,
        }

    # Đã search, tìm kết quả search gần nhất
    search_res = next(
        (h["result"] for h in reversed(history) if h["tool"] == "search_flights" and h["result"].get("status") == "ok"),
        None,
    )
    flights = search_res.get("flights", []) if search_res else []
    valid_flights = [f for f in flights if constraints.is_ok(f)]

    # Kiểm tra xem đã có booking nào chưa
    # Tìm danh sách các chuyến bay đã thử book nhưng thất bại
    failed_flights = [
        h["args"].get("flight")
        for h in history
        if h["tool"] == "book_seat" and h["result"].get("status") != "ok"
    ]
    candidate_flights = [f for f in valid_flights if f["flight"] not in failed_flights]

    # Kiểm tra xem đã có booking thành công nào chưa
    booked = any(h["tool"] == "book_seat" and h["result"].get("status") == "ok" for h in history)
    if not booked:
        if candidate_flights:
            # Chọn chuyến bay hợp lệ tiếp theo
            return "book_seat", {"flight": candidate_flights[0]["flight"]}
        else:
            # Không còn chuyến hợp lệ
            return None

    # Đã book, tìm mã booking
    book_res = next(
        (h["result"] for h in reversed(history) if h["tool"] == "book_seat" and h["result"].get("status") == "ok"),
        None,
    )
    if book_res:
        code = book_res.get("code")
        paid = any(h["tool"] == "pay" and h["result"].get("status") == "ok" for h in history)
        if not paid and code:
            return "pay", {"code": code}

    return None


def run_react_agent(
    constraints: Constraints,
    max_steps: int = 10,
    use_real_llm: bool = True,
    adversarial_mode: str | None = None,
) -> dict[str, Any]:
    """Thực thi ReAct agent với LLM thật chọn tool và Harness kiểm soát."""
    start_time = time.time()
    harness = ExecutionHarness(constraints=constraints, max_steps=max_steps)
    llm = tools.get_llm() if use_real_llm else None

    history: list[dict[str, Any]] = []

    for step_i in range(1, max_steps + 1):
        decision = None

        if adversarial_mode == "loop":
            decision = ("search_flights", {"origin": constraints.origin, "destination": constraints.destination, "date": constraints.date}, "Bẫy lặp vô hạn theo kịch bản.")
        elif adversarial_mode == "violate_constraint":
            expensive = next((f["flight"] for f in tools.CURRENT_FLIGHTS if not constraints.is_ok(f)), "VJ604")
            decision = ("book_seat", {"flight": expensive}, "Cố tình chọn chuyến vi phạm ràng buộc.")
        elif llm:
            history_text = "\n".join([
                f"- Bước {h['step']}: Tool '{h['tool']}' với args {json.dumps(h['args'], ensure_ascii=False)} -> Kết quả Mockup Tool: {json.dumps(h['result'], ensure_ascii=False)}"
                for h in history
            ]) if history else "Chưa có hành động nào trước đó."

            prompt = (
                f"Bạn là trợ lý đặt vé máy bay thông minh theo mô hình ReAct (Reasoning + Acting).\n"
                f"Nhiệm vụ: Sử dụng các công cụ để tìm và đặt 1 vé máy bay thỏa mãn TẤT CẢ ràng buộc.\n"
                f"Ràng buộc nghiệp vụ:\n"
                f"- Tuyến bay: {constraints.origin} -> {constraints.destination}\n"
                f"- Ngày bay: {constraints.date}\n"
                f"- Khởi hành trước: {constraints.depart_before}\n"
                f"- Giá vé tối đa: {constraints.max_price:,} VND\n\n"
                f"Các công cụ khả dụng:\n"
                f"1. search_flights(origin, destination, date): Tìm chuyến bay.\n"
                f"2. check_price(flight): Kiểm tra giá và số ghế trống của một chuyến.\n"
                f"3. book_seat(flight): Giữ chỗ một chuyến bay (nhận mã booking).\n"
                f"4. pay(code): Thanh toán mã booking đã giữ chỗ (hành động nhạy cảm không thể hoàn tác).\n"
                f"5. done(): Dừng lại khi đã hoàn thành hoặc không có chuyến bay nào thỏa mãn.\n\n"
                f"Lịch sử thực thi các bước trước đó:\n{history_text}\n\n"
                f"Hãy suy luận lý do và chọn 1 công cụ tiếp theo dưới dạng JSON duy nhất:\n"
                f"```json\n"
                f"{{\n"
                f'  "thought": "<suy luận chi tiết dựa trên quan sát trước đó>",\n'
                f'  "tool": "<tên công cụ>",\n'
                f'  "args": {{<các tham số>}}\n'
                f"}}\n"
                f"```"
            )

            try:
                response = llm.invoke(prompt)
                parsed = tools.parse_llm_json(response.content)
                if parsed and isinstance(parsed, dict):
                    tool_name = parsed.get("tool")
                    args = parsed.get("args", {})
                    thought = parsed.get("thought", "")
                    if tool_name == "done" or not tool_name:
                        break
                    decision = (tool_name, args, thought)
            except Exception as e:
                print(f"[Warning] Gọi LLM thất bại ({e}), chuyển sang heuristic...")
                decision = None

        if not decision:
            # Fallback heuristic nếu không có LLM hoặc LLM lỗi
            heu = _heuristic_react_step(harness.log, constraints, tools.BOOKINGS)
            if not heu:
                break
            decision = (heu[0], heu[1], "Heuristic decision")

        tool_name, tool_args, thought = decision
        tool_fn = tools.TOOLS_BY_NAME.get(tool_name)
        if not tool_fn:
            break

        # Thực thi Mockup tool thông qua chốt chặn an toàn của Harness
        res = harness.execute_tool(
            tool_fn=tool_fn,
            args=tool_args,
            current_flights=tools.CURRENT_FLIGHTS,
            bookings=tools.BOOKINGS,
        )

        history.append({
            "step": harness.step_count,
            "tool": tool_name,
            "args": tool_args,
            "result": res,
            "thought": thought,
        })

        if res.get("status") == "denied":
            if "LOOP" in str(res.get("reason", "")):
                break

        if harness.evaluate_finish(tools.BOOKINGS)["success"]:
            break

    latency = round((time.time() - start_time) * 1000, 2)
    final_eval = harness.evaluate_finish(tools.BOOKINGS)
    final_eval["latency_ms"] = latency
    final_eval["trace"] = harness.log
    return final_eval


if __name__ == "__main__":
    print("=== CHẠY KIỂM THỬ REACT AGENT ===")
    tools.reset_db()
    c = Constraints()
    print("Yêu cầu:", c.to_prompt())
    output = run_react_agent(c)
    print(f"Kết quả: {output['result']} | Số bước: {output['step_count']} | Vi phạm: {output['violations_count']} | Độ trễ: {output['latency_ms']}ms")
    print(json.dumps(output.get("bookings") or output.get("handoff"), indent=2, ensure_ascii=False))

    print("\n--- TRACE CHI TIẾT CÁC BƯỚC ---")
    for step in output.get("trace", []):
        print(f"Bước {step['step']}: Tool '{step['tool']}'")
        print(f"  -> Tham số: {step['args']}")
        print(f"  -> Kết quả: {step['result']}\n")
