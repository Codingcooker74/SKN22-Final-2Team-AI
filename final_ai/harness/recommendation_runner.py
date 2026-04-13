from __future__ import annotations

from contextlib import ExitStack
from time import perf_counter
from typing import Any
from unittest.mock import patch

from final_ai.application.recommendation.service import serialize_product_card
from final_ai.contracts.filters import build_search_filters, normalize_search_filters
from final_ai.domain.recommendation.constants import MAX_FILTER_RELAXATION_COUNT
from final_ai.infrastructure.search.hybrid_search import normalize_pet_species

from .schemas import RecommendationCase, RecommendationRunResult, RecommendationVariant


_PREVIEW_KEYS = (
    "goods_id",
    "goods_name",
    "pet_type",
    "category",
    "subcategory",
    "_score",
    "rerank_score",
    "popularity_score",
    "sentiment_avg",
    "repeat_rate",
    "health_concern_tags",
)


def _preview_products(products: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for product in products[:limit]:
        preview.append({key: product.get(key) for key in _PREVIEW_KEYS if key in product})
    return preview


def _snapshot_profile_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "pet_profile": state.get("pet_profile"),
        "health_concerns": state.get("health_concerns") or [],
        "allergies": state.get("allergies") or [],
        "food_preferences": state.get("food_preferences") or [],
        "budget": state.get("budget"),
        "pet_mismatch": bool(state.get("pet_mismatch")),
        "age_group": state.get("age_group"),
    }


def _snapshot_query_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "search_query": state.get("search_query"),
        "original_filters": normalize_search_filters(state.get("original_filters")),
        "filters": normalize_search_filters(state.get("filters")),
        "effective_filters": normalize_search_filters(state.get("effective_filters")),
        "relaxed_filters": list(state.get("relaxed_filters") or []),
        "filter_relaxation_count": int(state.get("filter_relaxation_count") or 0),
    }


def _build_initial_state(case: RecommendationCase) -> dict[str, Any]:
    pet_profile: dict[str, Any] = {}
    if case.pet_type:
        normalized = normalize_pet_species(case.pet_type)
        if normalized == "강아지":
            pet_profile["species"] = "dog"
        elif normalized == "고양이":
            pet_profile["species"] = "cat"
        else:
            pet_profile["species"] = case.pet_type
    if case.breed:
        pet_profile["breed"] = case.breed
    if case.age:
        pet_profile["age"] = case.age

    return {
        "user_input": case.query,
        "user_id": case.user_id,
        "target_pet_id": case.target_pet_id,
        "last_recommended_goods_ids": list(case.last_recommended_goods_ids),
        "allowed_goods_ids": list(case.allowed_goods_ids),
        "pet_profile": pet_profile or None,
        "health_concerns": list(case.health_concerns),
        "allergies": list(case.allergies),
        "food_preferences": list(case.food_preferences),
        "filters": build_search_filters(
            pet_type=case.pet_type,
            category=case.category,
            subcategory=case.subcategory,
            brand=case.brand,
        ),
        "budget": case.budget,
        "intents": ["recommend"],
        "pending_pet_ids": [],
        "pending_categories": [],
        "is_pet_switched": False,
        "clarification_count": 0,
        "filter_relaxation_count": 0,
        "recommendation_limit": case.limit,
        "best_reranked_results": [],
        "candidate_count_by_stage": {},
        "effective_filters": {},
        "original_filters": {},
        "relaxed_filters": [],
        "is_pet_override": bool(pet_profile),
        "is_result_refinement": bool(case.is_result_refinement),
        "refinement_sort": case.refinement_sort,
        "pet_mismatch": False,
    }


def _build_pet_row_lookup(case: RecommendationCase) -> dict[str, dict[str, Any]]:
    return {str(pet_id): dict(row) for pet_id, row in (case.pet_rows or {}).items() if isinstance(row, dict)}


def _build_pet_preferences_lookup(case: RecommendationCase) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for pet_id, payload in (case.pet_preferences_by_pet_id or {}).items():
        if not isinstance(payload, dict):
            continue
        lookup[str(pet_id)] = {
            "health_concerns": list(payload.get("health_concerns") or []),
            "allergies": list(payload.get("allergies") or []),
            "food_preferences": list(payload.get("food_preferences") or []),
        }
    return lookup


def _select_pet_row(
    pet_rows: dict[str, dict[str, Any]],
    *,
    target_pet_id: str | None,
) -> dict[str, Any] | None:
    if not pet_rows:
        return None
    if target_pet_id:
        row = pet_rows.get(str(target_pet_id))
        return dict(row) if row else None

    last_pet_id = next(reversed(pet_rows), None)
    if last_pet_id is None:
        return None
    return dict(pet_rows[last_pet_id])


def run_recommendation_case(
    case: RecommendationCase,
    variant: RecommendationVariant | None = None,
) -> RecommendationRunResult:
    active_variant = variant or RecommendationVariant()
    state = _build_initial_state(case)
    start = perf_counter()

    result = RecommendationRunResult(
        case=case,
        variant_name=active_variant.name,
        initial_state={
            "user_input": state["user_input"],
            "user_id": state["user_id"],
            "target_pet_id": state["target_pet_id"],
            "last_recommended_goods_ids": list(state.get("last_recommended_goods_ids") or []),
            "allowed_goods_ids": list(state.get("allowed_goods_ids") or []),
            "is_result_refinement": bool(state.get("is_result_refinement")),
            "refinement_sort": state.get("refinement_sort"),
            "pet_profile": state["pet_profile"],
            "filters": normalize_search_filters(state["filters"]),
            "health_concerns": state["health_concerns"],
            "allergies": state["allergies"],
            "food_preferences": state["food_preferences"],
            "budget": state["budget"],
        },
    )

    try:
        pet_rows = _build_pet_row_lookup(case)
        pet_preferences = _build_pet_preferences_lookup(case)
        with ExitStack() as stack:
            if pet_rows:
                stack.enter_context(
                    patch(
                        "final_ai.domain.recommendation.profile_service.fetch_pet_for_user",
                        side_effect=lambda user_id, target_pet_id=None, auto_latest=True: _select_pet_row(
                            pet_rows,
                            target_pet_id=target_pet_id,
                        ),
                    )
                )
            if pet_preferences:
                stack.enter_context(
                    patch(
                        "final_ai.domain.recommendation.profile_service.fetch_pet_preferences",
                        side_effect=lambda pet_id: dict(
                            pet_preferences.get(
                                str(pet_id),
                                {
                                    "health_concerns": [],
                                    "allergies": [],
                                    "food_preferences": [],
                                },
                            )
                        ),
                    )
                )

            state.update(active_variant.build_profile_state_fn(state))
        result.profile_state = _snapshot_profile_state(state)

        for _ in range(MAX_FILTER_RELAXATION_COUNT + 1):
            state.update(active_variant.build_search_query_state_fn(state))
            result.query_state = _snapshot_query_state(state)
            result.search_query = state.get("search_query")

            state.update(active_variant.execute_search_state_fn(state))
            search_results = list(state.get("search_results") or [])
            result.search_results_count = len(search_results)
            result.search_results_preview = _preview_products(search_results)

            state.update(active_variant.rerank_search_results_fn(state))
            if not state.get("recommend_retry_pending"):
                break

        reranked_results = list(state.get("reranked_results") or [])
        top_products = reranked_results[: case.limit]

        result.reranked_results_preview = _preview_products(reranked_results)
        result.raw_top_products = top_products
        result.products = [serialize_product_card(product) for product in top_products]
        result.filter_relaxation_count = int(state.get("filter_relaxation_count") or 0)
        result.recommend_retry_pending = bool(state.get("recommend_retry_pending"))
        result.pet_mismatch = bool(state.get("pet_mismatch"))
    except Exception as exc:
        result.error = str(exc)
    finally:
        result.duration_ms = round((perf_counter() - start) * 1000, 3)

    return result
