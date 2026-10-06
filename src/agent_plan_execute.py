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

from langchain_core.prompts import ChatPromptTemplate

import tools
from harness import Constraints, ExecutionHarness


# =====================================================================
# 1. CẤU TRÚC KẾ HOẠCH (STRUCTURED OUTPUT)
# =====================================================================
class PlanStep(BaseModel):
    tool: str = Field(description="Tên công cụ cần thực thi (search_flights, book_seat, pay)")
    args: dict[str, Any] = Field(description="Tham số truyền vào cho công cụ")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description="Danh sách các bước thực thi tuần tự")


def fill_placeholders(args: dict[str, Any], log: list[dict[str, Any]]) -> dict[str, Any]:
    """Thay thế placeholder như '$booking_code' bằng mã booking trả về từ bước book_seat trước đó."""
    resolved_code = ""
    for item in reversed(log):
        if item["tool"] == "book_seat" and item["result"].get("status") == "ok":
            resolved_code = item["result"].get("code", "")
            break

    new_args = {}
    for k, v in args.items():
        if isinstance(v, str) and "$booking_code" in v:
            new_args[k] = v.replace("$booking_code", resolved_code) if resolved_code else v
        else:
            new_args[k] = v
    return new_args


def generate_plan(constraints: Constraints, available_flights: list[dict[str, Any]], llm=None) -> Plan:
    """Tạo bản kế hoạch hoàn chỉnh bằng LLM thật (hoặc fallback nếu không có LLM)."""
    if llm is None:
        llm = tools.get_llm()

    if llm:
        planner_prompt = (
            f"Bạn là chuyên gia lập kế hoạch đặt vé máy bay theo mẫu Plan-then-Execute.\n"
            f"Nhiệm vụ: Lập toàn bộ kế hoạch từng bước tuần tự để hoàn tất đặt vé thỏa mãn ràng buộc.\n\n"
            f"Ràng buộc nghiệp vụ:\n"
            f"- Tuyến: {constraints.origin} -> {constraints.destination}\n"
            f"- Ngày bay: {constraints.date}\n"
            f"- Khởi hành trước: {constraints.depart_before}\n"
            f"- Giá vé tối đa: {constraints.max_price:,} VND\n\n"
            f"Danh sách chuyến bay nhận được từ hệ thống (Mockup Tool):\n"
            f"{json.dumps(available_flights, ensure_ascii=False, indent=2)}\n\n"
            f"Các công cụ khả dụng để đưa vào kế hoạch:\n"
            f"- book_seat(flight): Giữ chỗ một chuyến bay\n"
            f"- pay(code): Thanh toán mã đặt chỗ (dùng placeholder '$booking_code')\n\n"
            f"Quy tắc quan trọng:\n"
            f"1. Chỉ chọn chuyến bay thỏa mãn TẤT CẢ các ràng buộc và còn ghế (seats > 0).\n"
            f"2. Ở bước pay, đặt tham số code là '$booking_code'.\n"
            f"3. Nếu không có chuyến bay nào thỏa mãn, trả về danh sách bước rỗng [].\n\n"
            f"Trả về kết quả bằng JSON duy nhất theo định dạng:\n"
            f"```json\n"
            f"{{\n"
            f'  "thought": "<suy luận lý do chọn chuyến bay>",\n'
            f'  "steps": [\n'
            f'    {{"tool": "book_seat", "args": {{"flight": "<mã_chuyến>"}}}},\n'
            f'    {{"tool": "pay", "args": {{"code": "$booking_code"}}}}\n'
            f"  ]\n"
            f"}}\n"
            f"```"
        )
        try:
            res = llm.invoke(planner_prompt)
            parsed = tools.parse_llm_json(res.content)
            if parsed and isinstance(parsed, dict) and "steps" in parsed:
                steps_data = []
                for s in parsed["steps"]:
                    steps_data.append(PlanStep(tool=s["tool"], args=s.get("args", {})))
                return Plan(steps=steps_data)
        except Exception as e:
            print(f"[Warning] Gọi LLM Planner thất bại ({e}), dùng heuristic...")

    # Heuristic Planner khi chạy offline hoặc khi không có LLM
    valid_flights = [f for f in available_flights if constraints.is_ok(f) and f.get("seats", 1) > 0]
    if not valid_flights:
        return Plan(steps=[])

    target_flight = valid_flights[0]["flight"]
    return Plan(steps=[
        PlanStep(tool="book_seat", args={"flight": target_flight}),
        PlanStep(tool="pay", args={"code": "$booking_code"}),
    ])


def run_plan_execute_agent(
    constraints: Constraints,
    max_steps: int = 10,
    use_real_llm: bool = True,
    adversarial_plan: Plan | None = None,
) -> dict[str, Any]:
    """Thực thi Plan-then-Execute agent với Harness bảo vệ."""
    start_time = time.time()
    harness = ExecutionHarness(constraints=constraints, max_steps=max_steps)
    llm = tools.get_llm() if use_real_llm else None

    # Bước 1: Tra cứu chuyến bay hiện có qua Mockup Tool (search_flights)
    search_res = harness.execute_tool(
        tool_fn=tools.search_flights,
        args={"origin": constraints.origin, "destination": constraints.destination, "date": constraints.date},
        current_flights=tools.CURRENT_FLIGHTS,
        bookings=tools.BOOKINGS,
    )
    available_flights = search_res.get("flights", [])

    # Bước 2: Gọi LLM thật để lập kế hoạch tổng thể (Plan generation)
    if adversarial_plan:
        plan = adversarial_plan
    else:
        plan = generate_plan(constraints, available_flights, llm=llm)

    # Bước 3: Thực thi tuần tự bằng code (Executor - không gọi lại LLM)
    if not plan.steps:
        latency = round((time.time() - start_time) * 1000, 2)
        res = harness.evaluate_finish(
            tools.BOOKINGS,
            custom_question="Kế hoạch rỗng do không tìm thấy chuyến bay phù hợp ràng buộc. Bạn muốn điều chỉnh tiêu chí nào?",
        )
        res["latency_ms"] = latency
        res["trace"] = harness.log
        res["plan"] = [s.model_dump() for s in plan.steps]
        return res

    for step in plan.steps:
        if harness.step_count >= max_steps:
            break

        actual_args = fill_placeholders(step.args, harness.log)
        tool_fn = tools.TOOLS_BY_NAME.get(step.tool)
        if not tool_fn:
            break

        res = harness.execute_tool(
            tool_fn=tool_fn,
            args=actual_args,
            current_flights=tools.CURRENT_FLIGHTS,
            bookings=tools.BOOKINGS,
        )

        # Plan-then-Execute dừng ngay khi một bước thất bại (kế hoạch tĩnh)
        if res.get("status") != "ok":
            break

        if harness.evaluate_finish(tools.BOOKINGS)["success"]:
            break

    latency = round((time.time() - start_time) * 1000, 2)
    final_eval = harness.evaluate_finish(tools.BOOKINGS)
    final_eval["latency_ms"] = latency
    final_eval["trace"] = harness.log
    final_eval["plan"] = [s.model_dump() for s in plan.steps]
    return final_eval


if __name__ == "__main__":
    print("=== CHẠY KIỂM THỬ PLAN-THEN-EXECUTE AGENT ===")
    tools.reset_db()
    c = Constraints()
    print("Yêu cầu:", c.to_prompt())
    output = run_plan_execute_agent(c)
    print(f"Kết quả: {output['result']} | Số bước: {output['step_count']} | Vi phạm: {output['violations_count']} | Độ trễ: {output['latency_ms']}ms")
    print(json.dumps(output.get("bookings") or output.get("handoff"), indent=2, ensure_ascii=False))

    print("\n--- TRACE CHI TIẾT CÁC BƯỚC ---")
    for step in output.get("trace", []):
        print(f"Bước {step['step']}: Tool '{step['tool']}'")
        print(f"  -> Tham số: {step['args']}")
        print(f"  -> Kết quả: {step['result']}\n")
