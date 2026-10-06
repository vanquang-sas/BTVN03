import json
import tools
from harness import Constraints
from agent_react import run_react_agent
from agent_plan_execute import run_plan_execute_agent, Plan, PlanStep
from agent_hybrid import run_hybrid_agent


def print_test_header(case_title: str):
    print("\n" + "=" * 80)
    print(f"{case_title}")
    print("=" * 80)


def print_result_summary(res: dict):
    print(f"Kết quả đánh giá: {res['result']} (Success={res['success']})")
    print(f"Số bước: {res['step_count']} | Số vi phạm bị chặn: {res['violations_count']} | Độ trễ: {res['latency_ms']}ms")
    
    if not res["success"] and "handoff" in res:
        print("\n[HANDOFF BÀN GIAO CHO CON NGƯỜI]:")
        print(json.dumps(res["handoff"], indent=2, ensure_ascii=False))

    # ĐỔI THÀNH FORMAT NÀY:
    print("\n--- TRACE CHI TIẾT CÁC BƯỚC ---")
    for step in res.get("trace", []):
        print(f"Bước {step['step']}: Tool '{step['tool']}'")
        print(f"  -> Tham số: {step['args']}")
        print(f"  -> Kết quả: {step['result']}\n")


# =====================================================================
# Case 2 (ReAct): Không có chuyến bay phù hợp -> LLM thật tự nhận diện & dừng
# =====================================================================
def test_case_2():
    print_test_header("Case 2 (ReAct): Không có chuyến phù hợp (LLM thật tự Handoff)")
    # Database chỉ có chuyến đắt tiền hoặc chuyến buổi chiều
    tight_flights = [
        {"flight": "VJ604", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T08:10", "price": 2_480_000, "seats": 3},
        {"flight": "QH118", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T15:40", "price": 1_640_000, "seats": 4},
    ]
    tools.reset_db(tight_flights)
    c = Constraints(origin="SGN", destination="DAD", date="2026-10-07", depart_before="12:00", max_price=2_000_000)
    
    # use_real_llm=True: LLM tự search, thấy không có chuyến nào thỏa mãn -> tự kết thúc
    res = run_react_agent(c, use_real_llm=True)
    print_result_summary(res)


# =====================================================================
# Case 3 (ReAct): Trôi mục tiêu -> Fault Injection để test Harness
# =====================================================================
def test_case_3():
    print_test_header("Case 3 (ReAct): Trôi mục tiêu chọn chuyến vi phạm (Kiểm tra PermissionHarness)")
    tools.reset_db()
    c = Constraints(origin="SGN", destination="DAD", date="2026-10-07", depart_before="12:00", max_price=2_000_000)
    
    # Cố tình giả lập hành vi vi phạm (Model bị jailbreak hoặc hallucination)
    res = run_react_agent(c, use_real_llm=False, adversarial_mode="violate_constraint")
    print_result_summary(res)


# =====================================================================
# Case 4 (ReAct): Bẫy lặp vô hạn -> Fault Injection để test LoopDetector
# =====================================================================
def test_case_4():
    print_test_header("Case 4 (ReAct): Lặp vô hạn lời gọi công cụ (Kiểm tra LoopDetector)")
    tools.reset_db()
    c = Constraints()
    
    # Cố tình giả lập lỗi kẹt vòng lặp
    res = run_react_agent(c, use_real_llm=False, adversarial_mode="loop")
    print_result_summary(res)


# =====================================================================
# Case 6 (Plan-then-Execute): Kế hoạch rỗng -> LLM thật tự sinh steps=[]
# =====================================================================
def test_case_6():
    print_test_header("Case 6 (Plan-then-Execute): LLM thật sinh Kế hoạch rỗng khi không có chuyến")
    tight_flights = [
        {"flight": "VJ604", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T08:10", "price": 2_480_000, "seats": 3},
        {"flight": "QH118", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T15:40", "price": 1_640_000, "seats": 4},
    ]
    tools.reset_db(tight_flights)
    c = Constraints(origin="SGN", destination="DAD", date="2026-10-07", depart_before="12:00", max_price=2_000_000)
    
    # use_real_llm=True: LLM đọc danh sách chuyến bay, nhận thấy không có chuyến hợp lệ -> trả về Plan(steps=[])
    res = run_plan_execute_agent(c, use_real_llm=True)
    print_result_summary(res)
    print(f"Kế hoạch LLM sinh ra: {res.get('plan')}")


# =====================================================================
# Case 7 (Plan-then-Execute): Bước thực thi bị chặn -> Test Executor fail
# =====================================================================
def test_case_7():
    print_test_header("Case 7 (Plan-then-Execute): Kế hoạch tĩnh bị chặn quyền thực thi")
    tools.reset_db()
    c = Constraints(origin="SGN", destination="DAD", date="2026-10-07", depart_before="12:00", max_price=2_000_000)
    
    # Đưa vào một kế hoạch sai để chứng minh: Plan-then-Execute không thể tự sửa sai khi bị Harness chặn
    bad_plan = Plan(steps=[
        PlanStep(tool="book_seat", args={"flight": "QH118"}),
        PlanStep(tool="pay", args={"code": "$booking_code"})
    ])
    res = run_plan_execute_agent(c, adversarial_plan=bad_plan)
    print_result_summary(res)


# =====================================================================
# Case 8 (Hybrid): LLM thật tự động Re-plan khi gặp lỗi môi trường
# =====================================================================
# =====================================================================
# Case 8 (Hybrid): LLM thật tự động Re-plan khi gặp lỗi môi trường
# =====================================================================
# =====================================================================
# Case 3 / Case 8 (Hybrid): Thích ứng đổi chuyến khi phương án đầu bị lỗi
# (Adaptive Recovery -> Đạt kết quả DONE)
# =====================================================================
def test_case_8():
    print_test_header("Case 3 (Hybrid): Thích ứng đổi chuyến khi phương án đầu bị lỗi (Adaptive Recovery)")
    
    # 1. Khởi tạo danh sách chuyến bay: VN122 (08:10, 1.850.000đ) rẻ nhất và VN128 (06:30, 1.950.000đ)
    dynamic_flights = [
        {"flight": "VN122", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T08:10", "price": 1_850_000, "seats": 1},
        {"flight": "VN128", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T06:30", "price": 1_950_000, "seats": 3},
        {"flight": "QH118", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T15:40", "price": 1_640_000, "seats": 4},
    ]
    tools.reset_db(dynamic_flights)
    c = Constraints(origin="SGN", destination="DAD", date="2026-10-07", depart_before="12:00", max_price=2_000_000)

    # 2. Giả lập biến động môi trường: Chuyến VN122 bị hết chỗ đột ngột khi gọi book_seat
    original_tool = tools.TOOLS_BY_NAME["book_seat"]

    def dynamic_book_seat(flight: str) -> dict:
        if flight == "VN122":
            # Trả về lỗi hết chỗ để kích hoạt Re-plan
            return {"status": "error", "error": "Flight VN122 has no available seats"}
        # Chuyến VN128 thực thi book_seat bình thường qua tool gốc
        return original_tool.invoke({"flight": flight})

    # QUAN TRỌNG: Giữ nguyên thuộc tính tên tool là "book_seat" để bộ giải quyết placeholder bắt được mã booking
    dynamic_book_seat.name = "book_seat"
    dynamic_book_seat.__name__ = "book_seat"
    tools.TOOLS_BY_NAME["book_seat"] = dynamic_book_seat

    try:
        res = run_hybrid_agent(c, use_real_llm=True)
        print_result_summary(res)
        print(f"\n=> SỐ LẦN RE-PLANNING KÍCH HOẠT THÀNH CÔNG: {res.get('replan_count')}")
    finally:
        # Khôi phục lại tool gốc cho các test case sau
        tools.TOOLS_BY_NAME["book_seat"] = original_tool


if __name__ == "__main__":

    test_case_8()