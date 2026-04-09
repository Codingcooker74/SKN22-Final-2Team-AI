from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from final_ai.application.chat.context import hydrate_chat_request
from final_ai.application.chat.dto import build_chat_execution_request
from final_ai.application.recommendation.service import (
    build_profile_state,
    build_search_query_state,
    rerank_search_results,
)
from final_ai.domain.intent.service import classify_intent
from final_ai.domain.recommendation.search_service import execute_search_state
from final_ai.graph.builder import route_intent


def _list_of_strings(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _mapping_of_mappings(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    items: dict[str, dict[str, Any]] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            items[str(key)] = dict(item)
    return items


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            items.append(dict(item))
    return items


def _string_or_list(value: Any) -> str | list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value)


@dataclass(slots=True)
class RecommendationExpectation:
    min_results: int | None = None
    max_results: int | None = None
    allowed_pet_types: list[str] = field(default_factory=list)
    allowed_categories: list[str] = field(default_factory=list)
    must_include_any_goods_ids: list[str] = field(default_factory=list)
    must_exclude_goods_ids: list[str] = field(default_factory=list)
    relevant_goods_ids: list[str] = field(default_factory=list)
    search_query_contains: list[str] = field(default_factory=list)
    max_filter_relaxation_count: int | None = None
    max_duplicate_base_products: int | None = None
    max_effective_price: int | None = None
    allowed_goods_ids: list[str] = field(default_factory=list)
    top_goods_ids_prefix: list[str] = field(default_factory=list)
    forbidden_ingredient_keywords: list[str] = field(default_factory=list)
    forbid_sample_products: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RecommendationExpectation":
        payload = data or {}
        return cls(
            min_results=payload.get("min_results"),
            max_results=payload.get("max_results"),
            allowed_pet_types=_list_of_strings(payload.get("allowed_pet_types")),
            allowed_categories=_list_of_strings(payload.get("allowed_categories")),
            must_include_any_goods_ids=_list_of_strings(payload.get("must_include_any_goods_ids")),
            must_exclude_goods_ids=_list_of_strings(payload.get("must_exclude_goods_ids")),
            relevant_goods_ids=_list_of_strings(payload.get("relevant_goods_ids")),
            search_query_contains=_list_of_strings(payload.get("search_query_contains")),
            max_filter_relaxation_count=payload.get("max_filter_relaxation_count"),
            max_duplicate_base_products=payload.get("max_duplicate_base_products"),
            max_effective_price=payload.get("max_effective_price"),
            allowed_goods_ids=_list_of_strings(payload.get("allowed_goods_ids")),
            top_goods_ids_prefix=_list_of_strings(payload.get("top_goods_ids_prefix")),
            forbidden_ingredient_keywords=_list_of_strings(payload.get("forbidden_ingredient_keywords")),
            forbid_sample_products=payload.get("forbid_sample_products", True),
        )


@dataclass(slots=True)
class RecommendationCase:
    case_id: str
    query: str
    user_id: str | None = None
    target_pet_id: str | None = None
    pet_type: str | None = None
    breed: str | None = None
    age: str | None = None
    category: str | None = None
    subcategory: str | None = None
    brand: str | None = None
    health_concerns: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    food_preferences: list[str] = field(default_factory=list)
    budget: int | None = None
    limit: int = 5
    last_recommended_goods_ids: list[str] = field(default_factory=list)
    allowed_goods_ids: list[str] = field(default_factory=list)
    is_result_refinement: bool = False
    refinement_sort: str | None = None
    pet_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    pet_preferences_by_pet_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    expect: RecommendationExpectation = field(default_factory=RecommendationExpectation)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecommendationCase":
        return cls(
            case_id=str(data["case_id"]),
            query=str(data["query"]),
            user_id=data.get("user_id"),
            target_pet_id=data.get("target_pet_id"),
            pet_type=data.get("pet_type"),
            breed=data.get("breed"),
            age=data.get("age"),
            category=data.get("category"),
            subcategory=data.get("subcategory"),
            brand=data.get("brand"),
            health_concerns=_list_of_strings(data.get("health_concerns")),
            allergies=_list_of_strings(data.get("allergies")),
            food_preferences=_list_of_strings(data.get("food_preferences")),
            budget=data.get("budget"),
            limit=int(data.get("limit", 5)),
            last_recommended_goods_ids=_list_of_strings(data.get("last_recommended_goods_ids")),
            allowed_goods_ids=_list_of_strings(data.get("allowed_goods_ids")),
            is_result_refinement=bool(data.get("is_result_refinement", False)),
            refinement_sort=data.get("refinement_sort"),
            pet_rows=_mapping_of_mappings(data.get("pet_rows")),
            pet_preferences_by_pet_id=_mapping_of_mappings(data.get("pet_preferences_by_pet_id")),
            expect=RecommendationExpectation.from_dict(data.get("expect")),
        )


@dataclass(slots=True)
class StateExpectation:
    intents: list[str] = field(default_factory=list)
    pet_type: str | None = None
    category: str | None = None
    subcategory: str | None = None
    brand: str | None = None
    health_concerns: list[str] = field(default_factory=list)
    target_pet_id: str | None = None
    target_pet_name: str | None = None
    route: str | list[str] | None = None
    min_decomposed_tasks: int | None = None
    should_clarify: bool | None = None
    is_result_refinement: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StateExpectation":
        payload = data or {}
        return cls(
            intents=_list_of_strings(payload.get("intents")),
            pet_type=payload.get("pet_type"),
            category=payload.get("category"),
            subcategory=payload.get("subcategory"),
            brand=payload.get("brand"),
            health_concerns=_list_of_strings(payload.get("health_concerns")),
            target_pet_id=payload.get("target_pet_id"),
            target_pet_name=payload.get("target_pet_name"),
            route=_string_or_list(payload.get("route")),
            min_decomposed_tasks=payload.get("min_decomposed_tasks"),
            should_clarify=payload.get("should_clarify"),
            is_result_refinement=payload.get("is_result_refinement"),
        )


@dataclass(slots=True)
class StateCase:
    case_id: str
    message: str
    user_id: str | None = None
    thread_id: str = "harness-thread"
    target_pet_id: str | None = None
    pet_profile: dict[str, Any] | None = None
    health_concerns: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    food_preferences: list[str] = field(default_factory=list)
    registered_pets: list[dict[str, Any]] = field(default_factory=list)
    pet_profiles: dict[str, Any] = field(default_factory=dict)
    thread_context: dict[str, Any] = field(default_factory=dict)
    expect_state: StateExpectation = field(default_factory=StateExpectation)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateCase":
        return cls(
            case_id=str(data["case_id"]),
            message=str(data["message"]),
            user_id=data.get("user_id"),
            thread_id=data.get("thread_id", "harness-thread"),
            target_pet_id=data.get("target_pet_id"),
            pet_profile=data.get("pet_profile"),
            health_concerns=_list_of_strings(data.get("health_concerns")),
            allergies=_list_of_strings(data.get("allergies")),
            food_preferences=_list_of_strings(data.get("food_preferences")),
            registered_pets=_list_of_mappings(data.get("registered_pets")),
            pet_profiles=_mapping(data.get("pet_profiles")),
            thread_context=_mapping(data.get("thread_context")),
            expect_state=StateExpectation.from_dict(data.get("expect_state")),
        )


@dataclass(slots=True)
class RecommendationVariant:
    name: str = "baseline"
    build_profile_state_fn: Callable[[dict[str, Any]], dict[str, Any]] = build_profile_state
    build_search_query_state_fn: Callable[[dict[str, Any]], dict[str, Any]] = build_search_query_state
    execute_search_state_fn: Callable[[dict[str, Any]], dict[str, Any]] = execute_search_state
    rerank_search_results_fn: Callable[[dict[str, Any]], dict[str, Any]] = rerank_search_results


@dataclass(slots=True)
class StateVariant:
    name: str = "baseline"
    hydrate_request_fn: Callable[..., Any] = hydrate_chat_request
    build_execution_request_fn: Callable[..., Any] = build_chat_execution_request
    intent_classifier_fn: Callable[[dict[str, Any]], dict[str, Any]] = classify_intent
    route_intent_fn: Callable[[dict[str, Any]], Any] = route_intent


@dataclass(slots=True)
class RecommendationRunResult:
    case: RecommendationCase
    variant_name: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    profile_state: dict[str, Any] = field(default_factory=dict)
    query_state: dict[str, Any] = field(default_factory=dict)
    search_query: str | None = None
    search_results_count: int = 0
    search_results_preview: list[dict[str, Any]] = field(default_factory=list)
    reranked_results_preview: list[dict[str, Any]] = field(default_factory=list)
    raw_top_products: list[dict[str, Any]] = field(default_factory=list)
    products: list[dict[str, Any]] = field(default_factory=list)
    filter_relaxation_count: int = 0
    recommend_retry_pending: bool = False
    pet_mismatch: bool = False
    duration_ms: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class StateRunResult:
    case: StateCase
    variant_name: str
    hydrated_request: dict[str, Any] = field(default_factory=dict)
    initial_state: dict[str, Any] = field(default_factory=dict)
    extracted_updates: dict[str, Any] = field(default_factory=dict)
    merged_state: dict[str, Any] = field(default_factory=dict)
    route: str | list[str] | None = None
    should_clarify: bool = False
    resolved_pet_name: str | None = None
    duration_ms: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class RecommendationScore:
    passed: bool
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    policy_failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StateScore:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
