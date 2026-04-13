from collections.abc import Iterable, Mapping
from typing import Literal

from typing_extensions import TypedDict

FilterKey = Literal["pet_type", "category", "subcategory", "brand"]
ExclusionKey = Literal[
    "brands",
    "categories",
    "subcategories",
    "health_concerns",
    "ingredients",
    "keywords",
    "goods_ids",
]


class SearchFilters(TypedDict, total=False):
    pet_type: str
    category: str
    subcategory: str
    brand: str


class SearchExclusions(TypedDict, total=False):
    brands: list[str]
    categories: list[str]
    subcategories: list[str]
    health_concerns: list[str]
    ingredients: list[str]
    keywords: list[str]
    goods_ids: list[str]


def normalize_filter_value(value: object) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None

    if isinstance(value, Mapping):
        return None

    if isinstance(value, Iterable):
        for item in value:
            normalized = normalize_filter_value(item)
            if normalized:
                return normalized
        return None

    normalized = str(value).strip()
    return normalized or None


def normalize_filter_list(value: object) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []

    if isinstance(value, Mapping):
        return []

    values: list[str] = []
    seen: set[str] = set()

    if isinstance(value, Iterable):
        for item in value:
            normalized = normalize_filter_value(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
        return values

    normalized = str(value).strip()
    return [normalized] if normalized else []


def build_search_filters(
    *,
    pet_type: object = None,
    category: object = None,
    subcategory: object = None,
    brand: object = None,
) -> SearchFilters:
    filters: SearchFilters = {}

    normalized_pet_type = normalize_filter_value(pet_type)
    normalized_category = normalize_filter_value(category)
    normalized_subcategory = normalize_filter_value(subcategory)
    normalized_brand = normalize_filter_value(brand)

    if normalized_pet_type:
        filters["pet_type"] = normalized_pet_type
    if normalized_category:
        filters["category"] = normalized_category
    if normalized_subcategory:
        filters["subcategory"] = normalized_subcategory
    if normalized_brand:
        filters["brand"] = normalized_brand

    return filters


def build_search_exclusions(
    *,
    brands: object = None,
    categories: object = None,
    subcategories: object = None,
    health_concerns: object = None,
    ingredients: object = None,
    keywords: object = None,
    goods_ids: object = None,
) -> SearchExclusions:
    exclusions: SearchExclusions = {}

    normalized_brands = normalize_filter_list(brands)
    normalized_categories = normalize_filter_list(categories)
    normalized_subcategories = normalize_filter_list(subcategories)
    normalized_health_concerns = normalize_filter_list(health_concerns)
    normalized_ingredients = normalize_filter_list(ingredients)
    normalized_keywords = normalize_filter_list(keywords)
    normalized_goods_ids = normalize_filter_list(goods_ids)

    if normalized_brands:
        exclusions["brands"] = normalized_brands
    if normalized_categories:
        exclusions["categories"] = normalized_categories
    if normalized_subcategories:
        exclusions["subcategories"] = normalized_subcategories
    if normalized_health_concerns:
        exclusions["health_concerns"] = normalized_health_concerns
    if normalized_ingredients:
        exclusions["ingredients"] = normalized_ingredients
    if normalized_keywords:
        exclusions["keywords"] = normalized_keywords
    if normalized_goods_ids:
        exclusions["goods_ids"] = normalized_goods_ids

    return exclusions


def normalize_search_filters(filters: Mapping[str, object] | None) -> SearchFilters:
    if not filters:
        return {}

    return build_search_filters(
        pet_type=filters.get("pet_type"),
        category=filters.get("category"),
        subcategory=filters.get("subcategory"),
        brand=filters.get("brand"),
    )


def normalize_search_exclusions(exclusions: Mapping[str, object] | None) -> SearchExclusions:
    if not exclusions:
        return {}

    return build_search_exclusions(
        brands=exclusions.get("brands") or exclusions.get("brand"),
        categories=exclusions.get("categories") or exclusions.get("category"),
        subcategories=exclusions.get("subcategories") or exclusions.get("subcategory"),
        health_concerns=exclusions.get("health_concerns") or exclusions.get("health_concern"),
        ingredients=exclusions.get("ingredients") or exclusions.get("ingredient"),
        keywords=exclusions.get("keywords") or exclusions.get("keyword"),
        goods_ids=exclusions.get("goods_ids") or exclusions.get("goods_id"),
    )


__all__ = [
    "SearchExclusions",
    "SearchFilters",
    "build_search_exclusions",
    "build_search_filters",
    "normalize_filter_list",
    "normalize_filter_value",
    "normalize_search_exclusions",
    "normalize_search_filters",
]
