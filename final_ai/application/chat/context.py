from final_ai.contracts.chat import ChatRequest
from final_ai.domain.profile.service import get_future_pet_profile, get_pet_full_profile
from final_ai.infrastructure.repositories.chat_context_repository import fetch_chat_context


class ChatContextLoadError(RuntimeError):
    pass


def _copy_request(req: ChatRequest, updates: dict) -> ChatRequest:
    if hasattr(req, "model_copy"):
        return req.model_copy(update=updates)
    return req.copy(update=updates)


def hydrate_chat_request(req: ChatRequest) -> ChatRequest:
    if not req.current_user_message_id:
        return req
    if not req.user_id:
        raise ChatContextLoadError("user_id is required to load chat context")
    if not req.thread_id or req.thread_id == "default":
        raise ChatContextLoadError("thread_id is required to load chat context")

    context = fetch_chat_context(req.user_id, req.thread_id, req.current_user_message_id)
    if not context:
        raise ChatContextLoadError("chat context not found")

    dialog_state = dict(context.get("dialog_state") or {})
    resolved_target_pet_id = dialog_state.get("target_pet_id") or req.target_pet_id or context.get("target_pet_id")

    profile_updates = {}
    should_fill_profile = not any(
        [
            req.pet_profile is not None,
            req.health_concerns,
            req.allergies,
            req.food_preferences,
            dialog_state.get("pet_profile"),
        ]
    )
    if should_fill_profile and resolved_target_pet_id:
        pet_context = get_pet_full_profile(resolved_target_pet_id)
        if pet_context:
            profile_updates["pet_profile"] = pet_context.get("pet_profile")
            profile_updates["health_concerns"] = pet_context.get("health_concerns") or []
            profile_updates["allergies"] = pet_context.get("allergies") or []
            profile_updates["food_preferences"] = pet_context.get("food_preferences") or []
    elif should_fill_profile and context.get("profile_context_type") == "future":
        future_profile = get_future_pet_profile(req.user_id)
        if future_profile:
            profile_updates["pet_profile"] = future_profile

    return _copy_request(
        req,
        {
            "message": context["message"],
            "target_pet_id": resolved_target_pet_id,
            "conversation_history": context.get("conversation_history") or [],
            "summary_candidates": context.get("summary_candidates") or [],
            "memory_summary": context.get("memory_summary") or "",
            "dialog_state": dialog_state,
            "last_compacted_message_id": context.get("last_compacted_message_id"),
            **profile_updates,
        },
    )
