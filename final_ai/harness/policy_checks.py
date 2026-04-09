from __future__ import annotations

import re
from typing import Any

from final_ai.domain.recommendation.constants import SAMPLE_BLACKLIST_WORDS

from .schemas import RecommendationRunResult


def _as_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _base_product_name(name: str) -> str:
    weight_pattern = r"\b\d+(\.\d+)?\s*(kg|g|키로|그램|팩|p|입|개입|l|ml)\b"
    bracket_pattern = r"\[\d+(\.\d+)?\s*(kg|g|키로|그램|팩|p|입|개입|l|ml)\]"
    normalized = re.sub(weight_pattern, "", name or "", flags=re.IGNORECASE)
    normalized = re.sub(bracket_pattern, "", normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return re.sub(r"\s+", "", str(value)).lower()


def _effective_price(product: dict[str, Any]) -> float | None:
    price = product.get("discount_price")
    if price is None:
        price = product.get("price")
    if price in (None, ""):
        return None
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


def _ingredient_text(product: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in ("goods_name", "ingredient_text_ocr"):
        value = product.get(field)
        if value:
            parts.append(str(value))
    main_ingredients = product.get("main_ingredients") or []
    if isinstance(main_ingredients, list):
        parts.extend(str(item) for item in main_ingredients if str(item).strip())
    elif main_ingredients:
        parts.append(str(main_ingredients))
    return " ".join(parts)


def _contains_normalized_term(value: Any, term: str) -> bool:
    normalized_value = _normalize_text(value)
    normalized_term = _normalize_text(term)
    return bool(normalized_term) and normalized_term in normalized_value


def evaluate_recommendation_policy(result: RecommendationRunResult) -> list[str]:
    if result.error:
        return [f"runner_error: {result.error}"]

    failures: list[str] = []
    expectation = result.case.expect
    raw_products = result.raw_top_products or []
    goods_ids = [str(product.get("goods_id")) for product in raw_products if product.get("goods_id")]

    if expectation.min_results is not None and len(raw_products) < expectation.min_results:
        failures.append(f"min_results_failed: expected>={expectation.min_results} actual={len(raw_products)}")
    if expectation.max_results is not None and len(raw_products) > expectation.max_results:
        failures.append(f"max_results_failed: expected<={expectation.max_results} actual={len(raw_products)}")

    if expectation.must_include_any_goods_ids and not (set(goods_ids) & set(expectation.must_include_any_goods_ids)):
        failures.append("must_include_any_goods_ids_failed")

    if expectation.allowed_goods_ids:
        disallowed = sorted(set(goods_ids) - set(expectation.allowed_goods_ids))
        if disallowed:
            failures.append(f"allowed_goods_ids_failed: {', '.join(disallowed)}")

    if expectation.top_goods_ids_prefix:
        actual_prefix = goods_ids[: len(expectation.top_goods_ids_prefix)]
        if actual_prefix != expectation.top_goods_ids_prefix:
            failures.append(
                "top_goods_ids_prefix_failed: "
                f"expected={expectation.top_goods_ids_prefix} actual={actual_prefix}"
            )

    excluded = sorted(set(goods_ids) & set(expectation.must_exclude_goods_ids))
    if excluded:
        failures.append(f"must_exclude_goods_ids_failed: {', '.join(excluded)}")

    if result.case.brand:
        invalid_brand = [
            str(product.get("goods_id"))
            for product in raw_products
            if not _contains_normalized_term(product.get("brand_name"), result.case.brand)
        ]
        if invalid_brand:
            failures.append(f"brand_filter_failed: {', '.join(invalid_brand)}")

    if expectation.search_query_contains:
        search_query = result.search_query or ""
        missing_terms = [
            term
            for term in expectation.search_query_contains
            if _normalize_text(term) not in _normalize_text(search_query)
        ]
        if missing_terms:
            failures.append(f"search_query_contains_failed: {', '.join(missing_terms)}")

    if (
        expectation.max_filter_relaxation_count is not None
        and result.filter_relaxation_count > expectation.max_filter_relaxation_count
    ):
        failures.append(
            "filter_relaxation_failed: "
            f"expected<={expectation.max_filter_relaxation_count} actual={result.filter_relaxation_count}"
        )

    if expectation.forbid_sample_products:
        offenders = [
            str(product.get("goods_id"))
            for product in raw_products
            if any(word in (product.get("goods_name") or "") for word in SAMPLE_BLACKLIST_WORDS)
        ]
        if offenders:
            failures.append(f"sample_blacklist_failed: {', '.join(offenders)}")

    if expectation.allowed_pet_types:
        allowed_pet_types = set(expectation.allowed_pet_types)
        invalid = []
        for product in raw_products:
            product_pet_types = set(_as_list(product.get("pet_type")))
            if product_pet_types and not (product_pet_types & allowed_pet_types):
                invalid.append(str(product.get("goods_id")))
        if invalid:
            failures.append(f"allowed_pet_types_failed: {', '.join(invalid)}")

    if expectation.allowed_categories:
        allowed_categories = set(expectation.allowed_categories)
        invalid = []
        for product in raw_products:
            product_categories = set(_as_list(product.get("category")))
            if product_categories and not (product_categories & allowed_categories):
                invalid.append(str(product.get("goods_id")))
        if invalid:
            failures.append(f"allowed_categories_failed: {', '.join(invalid)}")

    if expectation.max_effective_price is not None:
        overpriced = [
            f"{product.get('goods_id')}={int(price) if price is not None and price.is_integer() else price}"
            for product in raw_products
            if (price := _effective_price(product)) is not None and price > expectation.max_effective_price
        ]
        if overpriced:
            failures.append(
                "max_effective_price_failed: "
                + ", ".join(overpriced)
            )

    if expectation.forbidden_ingredient_keywords:
        offending_products: list[str] = []
        normalized_keywords = [_normalize_text(keyword) for keyword in expectation.forbidden_ingredient_keywords]
        for product in raw_products:
            haystack = _normalize_text(_ingredient_text(product))
            if any(keyword and keyword in haystack for keyword in normalized_keywords):
                offending_products.append(str(product.get("goods_id")))
        if offending_products:
            failures.append(f"forbidden_ingredient_keywords_failed: {', '.join(offending_products)}")

    if expectation.max_duplicate_base_products is not None:
        counts: dict[str, int] = {}
        for product in raw_products:
            base_name = _base_product_name(product.get("goods_name") or "")
            if not base_name:
                continue
            counts[base_name] = counts.get(base_name, 0) + 1
        duplicated = {
            name: count
            for name, count in counts.items()
            if count > expectation.max_duplicate_base_products
        }
        if duplicated:
            failures.append(
                "duplicate_base_product_failed: "
                + ", ".join(f"{name}={count}" for name, count in sorted(duplicated.items()))
            )

    return failures
