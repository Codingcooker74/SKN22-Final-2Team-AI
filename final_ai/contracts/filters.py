from collections.abc import Iterable, Mapping
from typing import Literal

from typing_extensions import TypedDict

FilterKey = Literal["pet_type", "category", "subcategory", "brand"]


class SearchFilters(TypedDict, total=False):
    pet_type: str
    category: str
    subcategory: str
    brand: str


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


def normalize_search_filters(filters: Mapping[str, object] | None) -> SearchFilters:
    if not filters:
        return {}

    return build_search_filters(
        pet_type=filters.get("pet_type"),
        category=filters.get("category"),
        subcategory=filters.get("subcategory"),
        brand=filters.get("brand"),
    )


__all__ = [
    "SearchFilters",
    "build_search_filters",
    "normalize_filter_value",
    "normalize_search_filters",
]
