import json
from final_ai.contracts.sse import build_sse_payload, json_default


def sse_event(event_type: str, data: dict) -> str:
    payload = build_sse_payload(event_type, data)
    return f"data: {json.dumps(payload, ensure_ascii=False, default=json_default)}\n\n"
