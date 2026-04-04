from decimal import Decimal
from typing import Any
from typing_extensions import Literal

ChatEventType = Literal["info", "token", "products", "final", "done", "error"]


def json_default(value: Any):
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return str(value)


def build_sse_payload(event_type: ChatEventType, data: dict[str, Any]) -> dict[str, Any]:
    return {"type": event_type, **data}
