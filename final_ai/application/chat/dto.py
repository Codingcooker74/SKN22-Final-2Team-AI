from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from final_ai.contracts.chat import ChatRequest
from final_ai.contracts.filters import normalize_search_filters


@dataclass(frozen=True)
class ChatExecutionRequest:
    initial_state: dict[str, Any]
    config: dict[str, Any]


def _history_to_messages(history: list[dict[str, Any]]) -> list:
    messages = []
    for item in history:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if item.get("role") == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def build_chat_execution_request(req: ChatRequest) -> ChatExecutionRequest:
    dialog_state = dict(req.dialog_state or {})
    resolved_target_pet_id = dialog_state.get("target_pet_id") or req.target_pet_id
    conversation_history = [
        item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
        for item in (req.conversation_history or [])
    ]
    summary_candidates = [
        item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)
        for item in (req.summary_candidates or [])
    ]
    metadata = {
        "request_id": getattr(req, "request_id", None),
        "session_id": req.thread_id,
        "user_id": req.user_id,
        "target_pet_id": resolved_target_pet_id,
    }
    initial_state = {
        **dialog_state,
        "user_input": req.message,
        "messages": _history_to_messages(conversation_history),
        "conversation_history": conversation_history,
        "summary_candidates": summary_candidates,
        "memory_summary": (req.memory_summary or "").strip(),
        "last_compacted_message_id": req.last_compacted_message_id,
        "user_id": req.user_id or dialog_state.get("user_id"),
        "target_pet_id": resolved_target_pet_id,
        "pet_profile": req.pet_profile if req.pet_profile is not None else dict(dialog_state.get("pet_profile") or {}),
        "health_concerns": req.health_concerns or list(dialog_state.get("health_concerns") or []),
        "allergies": req.allergies or list(dialog_state.get("allergies") or []),
        "food_preferences": req.food_preferences or list(dialog_state.get("food_preferences") or []),
        "intents": list(dialog_state.get("intents") or []),
        "decomposed_tasks": list(dialog_state.get("decomposed_tasks") or []),
        "pending_requests": list(dialog_state.get("pending_requests") or []),
        "pending_pet_ids": list(dialog_state.get("pending_pet_ids") or []),
        "pending_categories": list(dialog_state.get("pending_categories") or []),
        "clarification_count": int(dialog_state.get("clarification_count") or 0),
        "filter_relaxation_count": int(dialog_state.get("filter_relaxation_count") or 0),
        "recommend_retry_pending": bool(dialog_state.get("recommend_retry_pending")),
        "is_pet_override": bool(dialog_state.get("is_pet_override")),
        "pet_mismatch": bool(dialog_state.get("pet_mismatch")),
        "filters": normalize_search_filters(dialog_state.get("filters")),
    }
    return ChatExecutionRequest(
        initial_state=initial_state,
        config={
            "configurable": {"thread_id": req.thread_id},
            "metadata": metadata,
        },
    )
