from typing import Any

from final_ai.contracts.filters import SearchFilters, build_search_filters, normalize_search_filters
from final_ai.domain.recommendation.constants import (
    MAX_FILTER_RELAXATION_COUNT,
    MIN_RECOMMENDATION_RESULTS,
)


def clamp_relaxation_count(value: object) -> int:
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(count, MAX_FILTER_RELAXATION_COUNT))


def should_include_health_concerns(relaxation: int) -> bool:
    return clamp_relaxation_count(relaxation) < 1


def should_include_subcategory(relaxation: int) -> bool:
    return clamp_relaxation_count(relaxation) < 2


def should_include_profile_hints(relaxation: int) -> bool:
    return clamp_relaxation_count(relaxation) < 3


def build_effective_search_filters(filters: dict[str, Any] | None, *, relaxation: int) -> SearchFilters:
    normalized = normalize_search_filters(filters)
    count = clamp_relaxation_count(relaxation)

    return build_search_filters(
        pet_type=normalized.get("pet_type"),
        category=normalized.get("category"),
        subcategory=normalized.get("subcategory") if should_include_subcategory(count) else None,
        brand=normalized.get("brand"),
    )


def build_relaxed_filter_names(
    *,
    relaxation: int,
    filters: dict[str, Any] | None,
    health_concerns: list[str] | None,
    age_group: str | None,
    breed: str | None,
) -> list[str]:
    count = clamp_relaxation_count(relaxation)
    normalized = normalize_search_filters(filters)
    relaxed: list[str] = []

    if count >= 1 and health_concerns:
        relaxed.append("health_concern")
    if count >= 2 and normalized.get("subcategory"):
        relaxed.append("subcategory")
    if count >= 3 and age_group:
        relaxed.append("age_group")
    if count >= 3 and breed:
        relaxed.append("breed")

    return relaxed


def target_recommendation_count(state: dict[str, Any]) -> int:
    try:
        requested_limit = int(state.get("recommendation_limit") or MIN_RECOMMENDATION_RESULTS)
    except (TypeError, ValueError):
        requested_limit = MIN_RECOMMENDATION_RESULTS

    target = max(1, min(MIN_RECOMMENDATION_RESULTS, requested_limit))
    allowed_goods_ids = list(state.get("allowed_goods_ids") or [])
    if not allowed_goods_ids and state.get("is_result_refinement"):
        allowed_goods_ids = list(state.get("last_recommended_goods_ids") or [])
    if allowed_goods_ids:
        target = min(target, len(set(str(goods_id) for goods_id in allowed_goods_ids)))
    return target


def should_retry_recommendation(*, result_count: int, relaxation: int, target_count: int) -> bool:
    return (
        result_count < target_count
        and clamp_relaxation_count(relaxation) < MAX_FILTER_RELAXATION_COUNT
    )


def next_relaxation_count(relaxation: int) -> int:
    return min(clamp_relaxation_count(relaxation) + 1, MAX_FILTER_RELAXATION_COUNT)


__all__ = [
    "build_effective_search_filters",
    "build_relaxed_filter_names",
    "clamp_relaxation_count",
    "next_relaxation_count",
    "should_include_health_concerns",
    "should_include_profile_hints",
    "should_include_subcategory",
    "should_retry_recommendation",
    "target_recommendation_count",
]
