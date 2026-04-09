from __future__ import annotations

from contextlib import ExitStack
from time import perf_counter
from typing import Any
from unittest.mock import patch

from final_ai.contracts.chat import ChatRequest
from final_ai.contracts.filters import normalize_search_filters

from .schemas import StateCase, StateRunResult, StateVariant


_STATE_SNAPSHOT_KEYS = (
    "user_input",
    "original_user_input",
    "normalized_user_input",
    "target_pet_id",
    "last_recommended_goods_ids",
    "allowed_goods_ids",
    "pet_profile",
    "health_concerns",
    "allergies",
    "food_preferences",
    "intents",
    "filters",
    "decomposed_tasks",
    "pending_requests",
    "clarification_count",
    "filter_relaxation_count",
    "recommend_retry_pending",
    "is_result_refinement",
    "pet_mismatch",
)


def _build_request(case: StateCase) -> ChatRequest:
    context = case.thread_context or {}
    return ChatRequest(
        message=case.message,
        thread_id=case.thread_id,
        user_id=case.user_id,
        target_pet_id=case.target_pet_id or context.get("target_pet_id"),
        profile_context_type=context.get("profile_context_type"),
        pet_profile=case.pet_profile if case.pet_profile is not None else context.get("pet_profile"),
        health_concerns=case.health_concerns or list(context.get("health_concerns") or []),
        allergies=case.allergies or list(context.get("allergies") or []),
        food_preferences=case.food_preferences or list(context.get("food_preferences") or []),
        conversation_history=list(context.get("conversation_history") or []),
        summary_candidates=list(context.get("summary_candidates") or []),
        memory_summary=context.get("memory_summary") or "",
        dialog_state=dict(context.get("dialog_state") or {}),
        last_compacted_message_id=context.get("last_compacted_message_id"),
        current_user_message_id=context.get("current_user_message_id"),
    )


def _snapshot_request(req: ChatRequest) -> dict[str, Any]:
    return {
        "message": req.message,
        "thread_id": req.thread_id,
        "user_id": req.user_id,
        "target_pet_id": req.target_pet_id,
        "pet_profile": req.pet_profile,
        "health_concerns": list(req.health_concerns or []),
        "allergies": list(req.allergies or []),
        "food_preferences": list(req.food_preferences or []),
        "dialog_state": dict(req.dialog_state or {}),
    }


def _snapshot_state(state: dict[str, Any]) -> dict[str, Any]:
    snapshot = {key: state.get(key) for key in _STATE_SNAPSHOT_KEYS}
    snapshot["filters"] = normalize_search_filters(snapshot.get("filters"))
    return snapshot


def _normalize_route(route_output: Any) -> str | list[str]:
    if isinstance(route_output, list):
        nodes: list[str] = []
        for item in route_output:
            node = getattr(item, "node", None)
            nodes.append(str(node) if node is not None else str(item))
        return nodes
    return str(route_output)


def _build_profile_lookup(case: StateCase) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for pet_id, profile in (case.pet_profiles or {}).items():
        if isinstance(profile, dict):
            lookup[str(pet_id)] = dict(profile)
    return lookup


def run_state_case(case: StateCase, variant: StateVariant | None = None) -> StateRunResult:
    active_variant = variant or StateVariant()
    start = perf_counter()
    result = StateRunResult(case=case, variant_name=active_variant.name)

    try:
        request = _build_request(case)
        hydrated = active_variant.hydrate_request_fn(request)
        execution = active_variant.build_execution_request_fn(hydrated)

        initial_state = dict(execution.initial_state)
        pet_profiles = _build_profile_lookup(case)
        with ExitStack() as stack:
            if case.registered_pets:
                stack.enter_context(
                    patch(
                        "final_ai.domain.intent.service.get_user_pets",
                        return_value=[dict(pet) for pet in case.registered_pets],
                    )
                )
            if pet_profiles:
                stack.enter_context(
                    patch(
                        "final_ai.domain.intent.service.get_pet_full_profile",
                        side_effect=lambda pet_id: dict(pet_profiles.get(str(pet_id), {})),
                    )
                )
                stack.enter_context(
                    patch(
                        "final_ai.application.chat.context.get_pet_full_profile",
                        side_effect=lambda pet_id: dict(pet_profiles.get(str(pet_id), {})),
                    )
                )

            extracted_updates = active_variant.intent_classifier_fn(dict(initial_state))
        merged_state = dict(initial_state)
        merged_state.update(extracted_updates)
        route_output = active_variant.route_intent_fn(merged_state)
        normalized_route = _normalize_route(route_output)

        resolved_pet_name = None
        pet_profile = merged_state.get("pet_profile") or {}
        if isinstance(pet_profile, dict):
            resolved_pet_name = pet_profile.get("name")
        if not resolved_pet_name:
            resolved_pet_name = merged_state.get("switched_pet_name")

        result.hydrated_request = _snapshot_request(hydrated)
        result.initial_state = _snapshot_state(initial_state)
        result.extracted_updates = _snapshot_state(extracted_updates)
        result.merged_state = _snapshot_state(merged_state)
        result.route = normalized_route
        result.should_clarify = normalized_route == "clarify"
        result.resolved_pet_name = resolved_pet_name
    except Exception as exc:
        result.error = str(exc)
    finally:
        result.duration_ms = round((perf_counter() - start) * 1000, 3)

    return result
