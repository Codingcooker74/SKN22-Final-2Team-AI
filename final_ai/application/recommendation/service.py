from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from final_ai.contracts.filters import (
    build_search_exclusions,
    build_search_filters,
    normalize_search_exclusions,
    normalize_search_filters,
)
from final_ai.domain.recommendation.profile_service import build_profile_state
from final_ai.domain.recommendation.query_service import build_search_query_state
from final_ai.domain.recommendation.rerank_service import rerank_search_results
from final_ai.domain.recommendation.search_service import execute_search_state
from final_ai.domain.recommendation.constants import MAX_FILTER_RELAXATION_COUNT
from final_ai.infrastructure.repositories.product_repository import list_products as fetch_products
from final_ai.infrastructure.search.hybrid_search import hybrid_search_pg, normalize_pet_species


def _serialize_number(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return value


def _parse_decimal(value):
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None

    try:
        normalized = str(value).strip().replace(",", "")
        if not normalized or normalized == "-":
            return None
        numeric = Decimal(normalized)
    except (InvalidOperation, ValueError):
        return None

    return numeric if numeric.is_finite() else None


def _serialize_rating(value):
    numeric = _parse_decimal(value)
    if numeric is None:
        return _serialize_number(value)
    clamped = min(max(numeric, Decimal("0.0")), Decimal("5.0"))
    return float(clamped.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _serialize_review_count(value):
    numeric = _parse_decimal(value)
    if numeric is None:
        return _serialize_number(value)
    return max(int(numeric), 0)


def serialize_product_card(product: dict) -> dict:
    return {
        "goods_id": product.get("goods_id"),
        "product_name": product.get("goods_name"),
        "brand_name": product.get("brand_name"),
        "price": _serialize_number(product.get("price")),
        "discount_price": _serialize_number(product.get("discount_price")),
        "rating": _serialize_rating(product.get("rating")),
        "reviews": _serialize_review_count(product.get("review_count")),
        "thumbnail_url": product.get("thumbnail_url"),
        "product_url": product.get("product_url"),
    }


def serialize_product_summary(product: dict) -> dict:
    return {
        **serialize_product_card(product),
        "pet_type": product.get("pet_type") or [],
        "category": product.get("category") or [],
        "subcategory": product.get("subcategory") or [],
        "soldout_yn": bool(product.get("soldout_yn")),
        "popularity_score": _serialize_number(product.get("popularity_score")),
        "sentiment_avg": _serialize_number(product.get("sentiment_avg")),
        "repeat_rate": _serialize_number(product.get("repeat_rate")),
    }


def recommend_products(
    *,
    query: str,
    user_id: str | None = None,
    target_pet_id: str | None = None,
    pet_type: str | None = None,
    breed: str | None = None,
    age: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    brand: str | None = None,
    health_concerns: list[str] | None = None,
    allergies: list[str] | None = None,
    food_preferences: list[str] | None = None,
    exclude_brands: list[str] | None = None,
    exclude_categories: list[str] | None = None,
    exclude_subcategories: list[str] | None = None,
    exclude_health_concerns: list[str] | None = None,
    exclude_ingredients: list[str] | None = None,
    exclude_keywords: list[str] | None = None,
    exclude_goods_ids: list[str] | None = None,
    budget: int | None = None,
    limit: int = 5,
) -> dict:
    pet_profile = {}
    if pet_type:
        normalized = normalize_pet_species(pet_type)
        if normalized == "강아지":
            pet_profile["species"] = "dog"
        elif normalized == "고양이":
            pet_profile["species"] = "cat"
        else:
            pet_profile["species"] = pet_type
    if breed:
        pet_profile["breed"] = breed
    if age:
        pet_profile["age"] = age

    state = {
        "user_input": query,
        "user_id": user_id,
        "target_pet_id": target_pet_id,
        "pet_profile": pet_profile or None,
        "health_concerns": health_concerns or [],
        "allergies": allergies or [],
        "food_preferences": food_preferences or [],
        "filters": build_search_filters(
            pet_type=pet_type,
            category=category,
            subcategory=subcategory,
            brand=brand,
        ),
        "budget": budget,
        "intents": ["recommend"],
        "pending_pet_ids": [],
        "pending_categories": [],
        "is_pet_switched": False,
        "clarification_count": 0,
        "filter_relaxation_count": 0,
        "recommendation_limit": limit,
        "best_reranked_results": [],
        "candidate_count_by_stage": {},
        "effective_filters": {},
        "original_filters": {},
        "exclusions": build_search_exclusions(
            brands=exclude_brands,
            categories=exclude_categories,
            subcategories=exclude_subcategories,
            health_concerns=exclude_health_concerns,
            ingredients=exclude_ingredients,
            keywords=exclude_keywords,
            goods_ids=exclude_goods_ids,
        ),
        "relaxed_filters": [],
        "is_pet_override": bool(pet_profile),
        "pet_mismatch": False,
    }

    state.update(build_profile_state(state))
    # Initial strict search plus staged fallback retries, matching graph retry behavior.
    for _ in range(MAX_FILTER_RELAXATION_COUNT + 1):
        state.update(build_search_query_state(state))
        state.update(execute_search_state(state))
        state.update(rerank_search_results(state))
        if not state.get("recommend_retry_pending"):
            break

    top_products = (state.get("reranked_results") or [])[:limit]
    return {
        "query": query,
        "search_query": state.get("search_query"),
        "filters": normalize_search_filters(state.get("filters")),
        "products": [serialize_product_card(product) for product in top_products],
        "meta": {
            "candidate_count": len(state.get("search_results") or []),
            "reranked_count": len(top_products),
            "filter_relaxation_count": state.get("filter_relaxation_count", 0),
            "recommend_retry_pending": bool(state.get("recommend_retry_pending")),
            "pet_mismatch": bool(state.get("pet_mismatch")),
            "relaxed_filters": state.get("relaxed_filters") or [],
            "original_filters": normalize_search_filters(state.get("original_filters")),
            "effective_filters": normalize_search_filters(state.get("effective_filters")),
            "exclusions": normalize_search_exclusions(state.get("exclusions")),
            "candidate_count_by_stage": state.get("candidate_count_by_stage") or {},
        },
    }


def list_products(
    *,
    query: str | None = None,
    pet_type: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    brand: str | None = None,
    budget: int | None = None,
    include_soldout: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    normalized_pet_type = normalize_pet_species(pet_type) or pet_type

    if query:
        rows = hybrid_search_pg(
            query=query,
            top_k=limit,
            pet_type=normalized_pet_type,
            category=category,
            subcategory=subcategory,
            brand=brand,
            budget=budget,
        )
        products = [serialize_product_summary(row) for row in rows]
    else:
        rows = fetch_products(
            query=query,
            pet_type=normalized_pet_type,
            category=category,
            subcategory=subcategory,
            brand=brand,
            budget=budget,
            include_soldout=include_soldout,
            limit=limit,
            offset=offset,
        )
        products = [serialize_product_summary(row) for row in rows]

    return {
        "products": products,
        "meta": {
            "count": len(products),
            "limit": limit,
            "offset": offset,
            "query": query,
        },
    }

__all__ = [
    "build_profile_state",
    "build_search_query_state",
    "execute_search_state",
    "rerank_search_results",
    "recommend_products",
    "list_products",
    "serialize_product_card",
    "serialize_product_summary",
]
