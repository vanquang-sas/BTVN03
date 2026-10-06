import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

_src_env = Path(__file__).resolve().parent / ".env"
_root_env = Path(__file__).resolve().parent.parent / ".env"
if _src_env.exists():
    load_dotenv(_src_env)
elif _root_env.exists():
    load_dotenv(_root_env)
else:
    load_dotenv()

# =====================================================================
# 1. MOCK DATABASE
# =====================================================================
DEFAULT_FLIGHTS = [
    {"flight": "VN122", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T08:10", "price": 1_850_000, "seats": 5},
    {"flight": "VJ604", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T09:30", "price": 2_480_000, "seats": 3},
    {"flight": "QH118", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T15:40", "price": 1_640_000, "seats": 4},
    {"flight": "VN128", "origin": "SGN", "destination": "DAD", "depart": "2026-10-07T06:30", "price": 1_950_000, "seats": 0},  # Hết vé
]

CURRENT_FLIGHTS = [dict(f) for f in DEFAULT_FLIGHTS]
BOOKINGS: dict[str, dict[str, Any]] = {}


def reset_db(custom_flights: list[dict] | None = None):
    """Đặt lại trạng thái mock database cho mỗi phiên kiểm thử."""
    global CURRENT_FLIGHTS, BOOKINGS
    CURRENT_FLIGHTS = [dict(f) for f in (custom_flights if custom_flights is not None else DEFAULT_FLIGHTS)]
    BOOKINGS = {}


# =====================================================================
# 2. TOOL INPUT SCHEMAS
# =====================================================================
class SearchFlightsInput(BaseModel):
    origin: str = Field(description="Mã sân bay đi (VD: SGN)")
    destination: str = Field(description="Mã sân bay đến (VD: DAD)")
    date: str = Field(description="Ngày bay định dạng YYYY-MM-DD")


class CheckPriceInput(BaseModel):
    flight: str = Field(description="Mã chuyến bay (VD: VN122)")


class BookSeatInput(BaseModel):
    flight: str = Field(description="Mã chuyến bay cần giữ chỗ (VD: VN122)")


class PayInput(BaseModel):
    code: str = Field(description="Mã đặt chỗ cần thanh toán (VD: VN122-12A)")


class GetBookingInput(BaseModel):
    code: str = Field(description="Mã đặt chỗ cần kiểm tra trạng thái")


class CancelBookingInput(BaseModel):
    code: str = Field(description="Mã đặt chỗ cần hủy")


# =====================================================================
# 3. MOCKUP TOOLS
# =====================================================================
@tool("search_flights", args_schema=SearchFlightsInput)
def search_flights(origin: str, destination: str, date: str) -> dict:
    """Tìm kiếm các chuyến bay theo điểm đi, điểm đến và ngày bay (YYYY-MM-DD)."""
    matched = [
        f for f in CURRENT_FLIGHTS
        if f["origin"].upper() == origin.upper()
        and f["destination"].upper() == destination.upper()
        and f["depart"].startswith(date)
    ]
    return {"status": "ok", "flights": matched}


@tool("check_price", args_schema=CheckPriceInput)
def check_price(flight: str) -> dict:
    """Kiểm tra giá vé hiện tại và số ghế trống của một chuyến bay cụ thể."""
    f = next((item for item in CURRENT_FLIGHTS if item["flight"] == flight), None)
    if not f:
        return {"status": "not_found", "flight": flight}
    return {"status": "ok", "flight": flight, "price": f["price"], "seats": f["seats"]}


@tool("book_seat", args_schema=BookSeatInput)
def book_seat(flight: str) -> dict:
    """Giữ chỗ cho một chuyến bay. Trả về mã booking code."""
    f = next((item for item in CURRENT_FLIGHTS if item["flight"] == flight), None)
    if not f:
        return {"status": "not_found", "error": f"Flight {flight} not found"}
    if f["seats"] <= 0:
        return {"status": "error", "error": f"Flight {flight} has no available seats"}

    code = f"{flight}-12A"
    BOOKINGS[code] = {
        "code": code,
        "flight": flight,
        "origin": f["origin"],
        "destination": f["destination"],
        "depart": f["depart"],
        "price": f["price"],
        "paid": False,
        "booking_status": "HOLD",
    }
    return {"status": "ok", **BOOKINGS[code]}


@tool("pay", args_schema=PayInput)
def pay(code: str) -> dict:
    """Xác nhận thanh toán cho mã booking đã giữ chỗ."""
    if code not in BOOKINGS:
        return {"status": "not_found", "error": f"Booking code {code} not found"}
    BOOKINGS[code]["paid"] = True
    BOOKINGS[code]["booking_status"] = "CONFIRMED"
    return {"status": "ok", **BOOKINGS[code]}


@tool("get_booking", args_schema=GetBookingInput)
def get_booking(code: str) -> dict:
    """Đọc thông tin vé đã đặt từ hệ thống cơ sở dữ liệu để kiểm tra trạng thái thực tế."""
    if code in BOOKINGS:
        return {"status": "ok", **BOOKINGS[code]}
    return {"status": "not_found", "code": code}


@tool("cancel_booking", args_schema=CancelBookingInput)
def cancel_booking(code: str) -> dict:
    """Hủy đặt chỗ hoặc vé đã thanh toán. Hành động nhạy cảm cần kiểm quyền."""
    if code not in BOOKINGS:
        return {"status": "not_found", "code": code}
    BOOKINGS[code]["booking_status"] = "CANCELLED"
    return {"status": "ok", **BOOKINGS[code]}


ALL_TOOLS = [search_flights, check_price, book_seat, pay, get_booking, cancel_booking]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


# =====================================================================
# 4. LLM CONFIGURATION & HELPER
# =====================================================================
from langchain_openai import ChatOpenAI


def parse_llm_json(text: str) -> dict | list | None:
    """Trích xuất và parse JSON từ chuỗi sinh bởi LLM (hỗ trợ cả khối markdown ```json)."""
    if not text:
        return None
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        start_curly = cleaned.find("{")
        end_curly = cleaned.rfind("}")
        if start_curly != -1 and end_curly != -1 and end_curly > start_curly:
            try:
                return json.loads(cleaned[start_curly:end_curly + 1])
            except Exception:
                pass
        start_sq = cleaned.find("[")
        end_sq = cleaned.rfind("]")
        if start_sq != -1 and end_sq != -1 and end_sq > start_sq:
            try:
                return json.loads(cleaned[start_sq:end_sq + 1])
            except Exception:
                pass
        return None


def get_llm(temperature: float = 0.0) -> ChatOpenAI | None:
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    model_name = os.getenv("MODEL_NAME", "gemma-4-26b")
    if not api_key:
        return None
    if base_url and base_url.endswith("/chat/completions"):
        base_url = base_url[:-len("/chat/completions")]
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )