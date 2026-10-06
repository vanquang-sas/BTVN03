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

from pydantic import BaseModel, Field

import tools
from harness import Constraints, ExecutionHarness


class HybridStep(BaseModel):
    tool: str = Field(description="Tên công cụ")
    args: dict[str, Any] = Field(description="Tham số truyền vào")
    description: str = Field(default="", description="Mục tiêu của bước")


class HybridPlan(BaseModel):
    steps: list[HybridStep] = Field(description="Danh sách các bước trong kế hoạch")


def make_plan(
    constraints: Constraints,
    flights: list[dict[str, Any]],
    excluded_flights: list[str],
    llm=None,
    error_context: str = "",
) -> HybridPlan:
    """Lập hoặc tái lập kế hoạch bằng LLM thật (loại trừ các chuyến đã thử thất bại)."""
    if llm is None:
        llm = tools.get_llm()

    if llm:
        candidates = [f for f in flights if f.get("flight") not in excluded_flights]
        err_msg = f"\nLƯU Ý BIẾN ĐỘNG / SỰ CỐ TỪ QUAN SÁT TRƯỚC ĐÓ:\n{error_context}\n" if error_context else ""
        prompt = (
            f"Bạn là chuyên gia lập kế hoạch theo mẫu Hybrid (ReAct + Plan).\n"
            f"Nhiệm vụ: Lập kế hoạch từng bước (book_seat, pay) để đặt vé máy bay thỏa mãn ràng buộc.\n\n"
            f"Ràng buộc nghiệp vụ:\n"
            f"- Tuyến: {constraints.origin} -> {constraints.destination}\n"
            f"- Ngày bay: {constraints.date}\n"
            f"- Khởi hành trước: {constraints.depart_before}\n"
            f"- Giá vé tối đa: {constraints.max_price:,} VND\n"
            f"{err_msg}\n"
            f"Danh sách chuyến bay khả dụng hiện tại (đã loại trừ các chuyến bị lỗi {excluded_flights}):\n"
            f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            f"Công cụ khả dụng:\n"
            f"- book_seat(flight): Giữ chỗ chuyến bay\n"
            f"- pay(code): Thanh toán (dùng placeholder '$booking_code')\n\n"
            f"Quy tắc:\n"
            f"1. Chỉ chọn chuyến bay thỏa mãn TẤT CẢ các ràng buộc và còn ghế (seats > 0).\n"
            f"2. Nếu không còn chuyến bay nào phù hợp, trả về steps: [].\n\n"
            f"Định dạng JSON duy nhất:\n"
            f"```json\n"
            f"{{\n"
            f'  "thought": "<suy luận lý do chọn chuyến bay và cách thích ứng với biến động>",\n'
            f'  "steps": [\n'
            f'    {{"tool": "book_seat", "args": {{"flight": "<mã_chuyến>"}}, "description": "Giữ chỗ chuyến bay"}},\n'
            f'    {{"tool": "pay", "args": {{"code": "$booking_code"}}, "description": "Thanh toán vé"}}\n'
            f"  ]\n"
            f"}}\n"
            f"```"
        )
        try:
            res = llm.invoke(prompt)
            parsed = tools.parse_llm_json(res.content)
            if parsed and isinstance(parsed, dict) and "steps" in parsed:
                steps_data = []
                for s in parsed["steps"]:
                    steps_data.append(HybridStep(tool=s["tool"], args=s.get("args", {}), description=s.get("description", "")))
                return HybridPlan(steps=steps_data)
        except Exception as e:
            print(f"[Warning] Gọi LLM Hybrid Planner thất bại ({e}), dùng heuristic...")

    # Heuristic fallback khi không có LLM
    candidates = [
        f for f in flights
        if constraints.is_ok(f) and f["flight"] not in excluded_flights and f.get("seats", 1) > 0
    ]
    if not candidates:
        return HybridPlan(steps=[])

    chosen = candidates[0]["flight"]
    return HybridPlan(steps=[
        HybridStep(tool="book_seat", args={"flight": chosen}, description=f"Giữ chỗ chuyến bay {chosen}"),
        HybridStep(tool="pay", args={"code": "$booking_code"}, description="Thanh toán vé đã giữ"),
    ])


def run_hybrid_agent(
    constraints: Constraints,
    max_steps: int = 10,
    max_replans: int = 2,
    use_real_llm: bool = True,
) -> dict[str, Any]:
    """Thực thi Hybrid Agent kết hợp Plan và ReAct."""
    start_time = time.time()
    harness = ExecutionHarness(constraints=constraints, max_steps=max_steps)
    llm = tools.get_llm() if use_real_llm else None
    replan_count = 0
    failed_flights: list[str] = []
    error_context = ""

    # 1. Tìm kiếm ban đầu qua Mockup Tool
    search_res = harness.execute_tool(
        tool_fn=tools.search_flights,
        args={"origin": constraints.origin, "destination": constraints.destination, "date": constraints.date},
        current_flights=tools.CURRENT_FLIGHTS,
        bookings=tools.BOOKINGS,
    )
    all_flights = search_res.get("flights", [])

    # 2. Vòng lặp Hybrid: Plan -> Execute -> Evaluate Observation -> Re-plan
    while harness.step_count < max_steps:
        # Lập hoặc tái lập kế hoạch bằng LLM thật
        plan = make_plan(constraints, all_flights, excluded_flights=failed_flights, llm=llm, error_context=error_context)
        if not plan.steps:
            # Không còn phương án nào thỏa mãn
            break

        plan_interrupted = False
        for step in plan.steps:
            if harness.step_count >= max_steps:
                break

            # Giải quyết placeholder
            code = ""
            for item in reversed(harness.log):
                if item["tool"] == "book_seat" and item["result"].get("status") == "ok":
                    code = item["result"].get("code", "")
                    break

            args = dict(step.args)
            for k, v in args.items():
                if isinstance(v, str) and "$booking_code" in v:
                    args[k] = v.replace("$booking_code", code)

            tool_fn = tools.TOOLS_BY_NAME.get(step.tool)
            if not tool_fn:
                plan_interrupted = True
                break

            # Thực thi Mockup tool với Harness bảo vệ
            res = harness.execute_tool(
                tool_fn=tool_fn,
                args=args,
                current_flights=tools.CURRENT_FLIGHTS,
                bookings=tools.BOOKINGS,
            )

            # Kiểm tra quan sát: Nếu gặp lỗi/thất bại -> Kích hoạt Re-plan
            if res.get("status") != "ok":
                flight_id = args.get("flight")
                if flight_id:
                    failed_flights.append(flight_id)
                error_context = f"Thao tác '{step.tool}' với {args} thất bại: {res}. Chuyến bay {flight_id} không khả dụng."
                plan_interrupted = True
                replan_count += 1
                break  # Kích hoạt Re-planning

            # Kiểm tra hoàn thành sớm
            if harness.evaluate_finish(tools.BOOKINGS)["success"]:
                break

        if harness.evaluate_finish(tools.BOOKINGS)["success"]:
            break

        # Nếu hoàn thành trọn plan mà chưa đạt mục tiêu hoặc bị ngắt, kiểm tra ngân sách replan
        if replan_count > max_replans or not plan_interrupted:
            break

    latency = round((time.time() - start_time) * 1000, 2)
    final_eval = harness.evaluate_finish(tools.BOOKINGS)
    final_eval["latency_ms"] = latency
    final_eval["replan_count"] = replan_count
    final_eval["trace"] = harness.log
    return final_eval


if __name__ == "__main__":
    print("=== CHẠY KIỂM THỬ HYBRID AGENT ===")
    tools.reset_db()
    c = Constraints()
    print("Yêu cầu:", c.to_prompt())
    output = run_hybrid_agent(c)
    print(f"Kết quả: {output['result']} | Số bước: {output['step_count']} | Số lần Replan: {output.get('replan_count', 0)} | Vi phạm: {output['violations_count']} | Độ trễ: {output['latency_ms']}ms")
    print(json.dumps(output.get("bookings") or output.get("handoff"), indent=2, ensure_ascii=False))

    print("\n--- TRACE CHI TIẾT CÁC BƯỚC ---")
    for step in output.get("trace", []):
        print(f"Bước {step['step']}: Tool '{step['tool']}'")
        print(f"  -> Tham số: {step['args']}")
        print(f"  -> Kết quả: {step['result']}\n")
