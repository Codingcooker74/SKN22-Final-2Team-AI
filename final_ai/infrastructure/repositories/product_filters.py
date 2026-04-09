from collections.abc import Iterable


FilterValue = str | Iterable[str] | None


def normalize_filter_values(value: FilterValue) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []

    values: list[str] = []
    seen: set[str] = set()

    for item in value:
        if item is None:
            continue
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)

    return values


def build_product_filter_clauses(
    *,
    pet_type: FilterValue = None,
    category: FilterValue = None,
    subcategory: FilterValue = None,
    health_concerns: FilterValue = None,
    brand: str | None = None,
    budget: int | None = None,
    allowed_goods_ids: list[str] | None = None,
) -> tuple[list[str], list[object]]:
    filters: list[str] = []
    params: list[object] = []

    pet_types = normalize_filter_values(pet_type)
    categories = normalize_filter_values(category)
    subcategories = normalize_filter_values(subcategory)
    concerns = normalize_filter_values(health_concerns)

    if pet_types:
        if len(pet_types) == 1:
            filters.append("%s = ANY(pet_type)")
            params.append(pet_types[0])
        else:
            filters.append("pet_type && %s::text[]")
            params.append(pet_types)

    if categories:
        # 단일 카테고리인 경우 부분 일치(ILIKE) 지원
        if len(categories) == 1:
            cat_pattern = f"%{categories[0]}%"
            filters.append(
                "("
                "EXISTS (SELECT 1 FROM unnest(category) c WHERE c ILIKE %s) OR "
                "EXISTS (SELECT 1 FROM unnest(subcategory) s WHERE s ILIKE %s)"
                ")"
            )
            params.extend([cat_pattern, cat_pattern])
        else:
            filters.append("(category && %s::text[] OR subcategory && %s::text[])")
            params.extend([categories, categories])

    if subcategories:
        # 단일 서브카테고리인 경우 부분 일치(ILIKE) 지원
        if len(subcategories) == 1:
            sub_pattern = f"%{subcategories[0]}%"
            filters.append("EXISTS (SELECT 1 FROM unnest(subcategory) s WHERE s ILIKE %s)")
            params.append(sub_pattern)
        else:
            filters.append("subcategory && %s::text[]")
            params.append(subcategories)

    if concerns:
        if len(concerns) == 1:
            filters.append("%s = ANY(health_concern_tags)")
            params.append(concerns[0])
        else:
            filters.append("health_concern_tags && %s::text[]")
            params.append(concerns)

    if brand:
        filters.append("brand_name ILIKE %s")
        params.append(f"%{brand}%")

    if budget is not None:
        filters.append("price <= %s")
        params.append(budget)

    if allowed_goods_ids:
        goods_ids = normalize_filter_values(allowed_goods_ids)
        if goods_ids:
            if len(goods_ids) == 1:
                filters.append("goods_id = %s")
                params.append(goods_ids[0])
            else:
                filters.append("goods_id = ANY(%s::text[])")
                params.append(goods_ids)

    return filters, params
