from collections import deque
from dataclasses import dataclass
from typing import Any
import json


# =====================================================================
# 1. RÀNG BUỘC LÀ DỮ LIỆU (CONSTRAINTS AS DATA)
# =====================================================================
@dataclass
class Constraints:
    origin: str = "SGN"
    destination: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"    
    max_price: int = 2_000_000    

    def to_prompt(self) -> str:
        return (
            f"Đặt một vé máy bay {self.origin} -> {self.destination} vào ngày {self.date}, "
            f"khởi hành trước {self.depart_before}, giá tối đa {self.max_price:,} VND."
        )

    def is_ok(self, flight: dict[str, Any]) -> bool:
        """Kiểm tra chuyến bay có thỏa mãn TẤT CẢ ràng buộc hay không."""
        depart = str(flight.get("depart", ""))
        price = flight.get("price", float("inf"))
        
        # Kiểm tra ngày bay và giờ bay
        valid_date = depart.startswith(self.date)
        valid_time = len(depart) >= 16 and depart[11:16] < self.depart_before
        valid_price = price <= self.max_price
        return valid_date and valid_time and valid_price


# =====================================================================
# 2. PHÁT HIỆN LẶP 
# =====================================================================
class LoopDetector:
    def __init__(self, window: int = 6, repeat_k: int = 2, stall_n: int = 4):
        self.recent: deque = deque(maxlen=window)
        self.k = repeat_k
        self.n = stall_n
        self.last_progress = None
        self.stall = 0

    def check(self, tool_name: str, args: dict[str, Any], progress: Any = None) -> str | None:
        """Kiểm tra xem lời gọi có bị lặp lại hoặc đứng yên hay không."""
        # Biểu diễn fingerprint của (tool, args)
        fp = (tool_name, repr(sorted(args.items())))
        if self.recent.count(fp) + 1 >= self.k:
            return f"LOOP: Tool '{tool_name}' với tham số {args} đã được gọi {self.k} lần liên tục."
        self.recent.append(fp)

        # Kiểm tra stall (không tiến triển)
        if progress is not None:
            self.stall = self.stall + 1 if progress == self.last_progress else 0
            self.last_progress = progress
            if self.stall >= self.n:
                return f"STALL: Quá trình không có tiến triển sau {self.n} bước liên tiếp."
        return None


# =====================================================================
# 3. KIỂM QUYỀN TRƯỚC KHI THỰC THI
# =====================================================================
def check_permission(
    tool_name: str,
    args: dict[str, Any],
    constraints: Constraints,
    current_flights: list[dict[str, Any]],
    bookings: dict[str, dict[str, Any]],
) -> str | None:
    """Trả về None nếu hợp lệ, hoặc lý do chặn nếu vi phạm ràng buộc / quyền hạn."""
    # book_seat không được vi phạm ràng buộc
    if tool_name == "book_seat":
        flight_id = args.get("flight")
        flight = next((f for f in current_flights if f["flight"] == flight_id), None)
        if flight is None:
            return f"Chuyến bay '{flight_id}' không tồn tại trong hệ thống."
        if not constraints.is_ok(flight):
            return (
                f"Chuyến bay '{flight_id}' vi phạm ràng buộc người dùng: "
                f"Giá {flight.get('price', 0):,} VND (max: {constraints.max_price:,} VND), "
                f"Giờ khởi hành: {flight.get('depart', '')} (yêu cầu trước {constraints.depart_before})."
            )

    # Guardrail 2: pay phải có booking hợp lệ và chưa thanh toán
    elif tool_name == "pay":
        code = args.get("code")
        if code not in bookings:
            return f"Không thể thanh toán: Mã booking '{code}' không tồn tại hoặc chưa được giữ chỗ."
        if bookings[code].get("paid"):
            return f"Mã booking '{code}' đã được thanh toán trước đó."

    # Guardrail 3: cancel_booking là hành động nhạy cảm
    elif tool_name == "cancel_booking":
        code = args.get("code")
        if code not in bookings:
            return f"Không thể hủy: Mã booking '{code}' không tồn tại."

    return None


# =====================================================================
# 4. TIÊU CHÍ HOÀN THÀNH KIỂM BẰNG CODE (COMPLETION CHECK - Slide 36 Buổi 3)
#    Sensor computational: Không tin câu trả lời của Model, chỉ tin DB state.
# =====================================================================
def is_done(bookings: dict[str, dict[str, Any]], constraints: Constraints) -> bool:
    """Hoàn thành = Có ít nhất 1 vé đã thanh toán (paid=True) và thỏa mãn ràng buộc."""
    for booking in bookings.values():
        if booking.get("paid") and constraints.is_ok(booking):
            return True
    return False


# =====================================================================
# 5. BÀN GIAO CHO CON NGƯỜI (HANDOFF - Slide 48 Buổi 3)
#    Khi thất bại hoặc bế tắc: cung cấp 3 thông tin để người duyệt trong 30s.
# =====================================================================
def handoff(
    bookings: dict[str, dict[str, Any]],
    log: list[dict[str, Any]],
    question: str = "Không tìm thấy chuyến bay thỏa mãn. Bạn muốn nới lỏng giờ bay hay tăng mức giá tối đa?",
) -> dict[str, Any]:
    """Tạo payload bàn giao cho con người."""
    done_so_far = [
        f"{code}: status={b.get('booking_status', b.get('status'))}, paid={b.get('paid')}, flight={b.get('flight')}"
        for code, b in bookings.items()
    ] or ["Chưa giữ chỗ và chưa thanh toán vé nào."]

    tried = [
        f"{item['tool']}({item['args']}) -> status={item['result'].get('status')}"
        for item in log
    ]

    return {
        "done_so_far": done_so_far,
        "tried": tried,
        "question": question,
    }


# =====================================================================
# 6. LỚP ĐIỀU PHỐI HARNESS TỔNG HỢP (EXECUTION HARNESS)
# =====================================================================
class ExecutionHarness:
    """Lớp bọc điều phối thực thi và ghi nhận vi phạm an toàn cho Agent."""

    def __init__(self, constraints: Constraints, max_steps: int = 10):
        self.constraints = constraints
        self.max_steps = max_steps
        self.step_count = 0
        self.violations_count = 0
        self.loop_detector = LoopDetector(window=6, repeat_k=2, stall_n=4)
        self.log: list[dict[str, Any]] = []

    def execute_tool(
        self,
        tool_fn,
        args: dict[str, Any],
        current_flights: list[dict[str, Any]],
        bookings: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Thực thi tool qua các cổng kiểm tra an toàn của harness."""
        self.step_count += 1
        tool_name = tool_fn.name if hasattr(tool_fn, "name") else tool_fn.__name__

        # 1. Kiểm tra ngân sách bước (Budget / Max steps)
        if self.step_count > self.max_steps:
            self.violations_count += 1
            result = {"status": "denied", "reason": f"Hết ngân sách thực thi: Vượt quá {self.max_steps} bước."}
            self.log.append({"step": self.step_count, "tool": tool_name, "args": args, "result": result})
            return result

        # 2. Kiểm tra phát hiện lặp (Loop Detection)
        loop_err = self.loop_detector.check(tool_name, args, progress=len(bookings))
        if loop_err:
            self.violations_count += 1
            result = {"status": "denied", "reason": loop_err}
            self.log.append({"step": self.step_count, "tool": tool_name, "args": args, "result": result})
            return result

        # 3. Kiểm quyền (Permission Check)
        perm_err = check_permission(tool_name, args, self.constraints, current_flights, bookings)
        if perm_err:
            self.violations_count += 1
            result = {"status": "denied", "reason": perm_err}
            self.log.append({"step": self.step_count, "tool": tool_name, "args": args, "result": result})
            return result

        # 4. Thực thi công cụ thật
        try:
            if hasattr(tool_fn, "invoke"):
                res = tool_fn.invoke(args)
            else:
                res = tool_fn(**args)
            result = res if isinstance(res, dict) else {"status": "ok", "data": res}
        except Exception as e:
            result = {"status": "error", "error": str(e)}

        self.log.append({"step": self.step_count, "tool": tool_name, "args": args, "result": result})
        return result

    def evaluate_finish(self, bookings: dict[str, dict[str, Any]], custom_question: str | None = None) -> dict[str, Any]:
        """Đánh giá kết quả cuối cùng hoàn toàn bằng code logic."""
        if is_done(bookings, self.constraints):
            return {
                "result": "DONE",
                "success": True,
                "step_count": self.step_count,
                "violations_count": self.violations_count,
                "bookings": list(bookings.values()),
            }
        
        q = custom_question or "Không tìm thấy chuyến bay thỏa mãn. Bạn muốn nới lỏng giờ bay hay tăng mức giá tối đa?"
        return {
            "result": "FAILED",
            "success": False,
            "step_count": self.step_count,
            "violations_count": self.violations_count,
            "handoff": handoff(bookings, self.log, question=q),
        }
