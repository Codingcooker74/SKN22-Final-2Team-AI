from dataclasses import dataclass
from typing import Any

from final_ai.contracts.chat import ChatRequest


@dataclass(frozen=True)
class ChatExecutionRequest:
    initial_state: dict[str, Any]
    config: dict[str, Any]


def build_chat_execution_request(req: ChatRequest) -> ChatExecutionRequest:
    metadata = {
        "request_id": getattr(req, "request_id", None),
        "session_id": req.thread_id,
        "user_id": req.user_id,
        "target_pet_id": req.target_pet_id,
    }
    return ChatExecutionRequest(
        initial_state={
            "user_input": req.message,
            "pet_profile": req.pet_profile,
            "health_concerns": req.health_concerns,
            "allergies": req.allergies,
            "food_preferences": req.food_preferences,
            "user_id": req.user_id,
            "target_pet_id": req.target_pet_id,
        },
        config={
            "configurable": {"thread_id": req.thread_id},
            "metadata": metadata,
        },
    )
