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
    exclude_brands: FilterValue = None,
    exclude_categories: FilterValue = None,
    exclude_subcategories: FilterValue = None,
    exclude_health_concerns: FilterValue = None,
    exclude_goods_ids: FilterValue = None,
    budget: int | None = None,
    allowed_goods_ids: list[str] | None = None,
) -> tuple[list[str], list[object]]:
    filters: list[str] = []
    params: list[object] = []

    pet_types = normalize_filter_values(pet_type)
    categories = normalize_filter_values(category)
    subcategories = normalize_filter_values(subcategory)
    concerns = normalize_filter_values(health_concerns)
    excluded_brands = normalize_filter_values(exclude_brands)
    excluded_categories = normalize_filter_values(exclude_categories)
    excluded_subcategories = normalize_filter_values(exclude_subcategories)
    excluded_concerns = normalize_filter_values(exclude_health_concerns)
    excluded_goods = normalize_filter_values(exclude_goods_ids)

    if pet_types:
        if len(pet_types) == 1:
            filters.append("%s = ANY(pet_type)")
            params.append(pet_types[0])
        else:
            filters.append("pet_type && %s::varchar[]")
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
            filters.append("(category && %s::varchar[] OR subcategory && %s::varchar[])")
            params.extend([categories, categories])

    if subcategories:
        # 단일 서브카테고리인 경우 부분 일치(ILIKE) 지원
        if len(subcategories) == 1:
            sub_pattern = f"%{subcategories[0]}%"
            filters.append("EXISTS (SELECT 1 FROM unnest(subcategory) s WHERE s ILIKE %s)")
            params.append(sub_pattern)
        else:
            filters.append("subcategory && %s::varchar[]")
            params.append(subcategories)

    if concerns:
        if len(concerns) == 1:
            filters.append("%s = ANY(health_concern_tags)")
            params.append(concerns[0])
        else:
            filters.append("health_concern_tags && %s::varchar[]")
            params.append(concerns)

    if brand:
        filters.append("brand_name ILIKE %s")
        params.append(f"%{brand}%")

    for excluded_brand in excluded_brands:
        filters.append("brand_name NOT ILIKE %s")
        params.append(f"%{excluded_brand}%")

    for excluded_category in excluded_categories:
        category_pattern = f"%{excluded_category}%"
        filters.append(
            "NOT ("
            "EXISTS (SELECT 1 FROM unnest(category) c WHERE c ILIKE %s) OR "
            "EXISTS (SELECT 1 FROM unnest(subcategory) s WHERE s ILIKE %s)"
            ")"
        )
        params.extend([category_pattern, category_pattern])

    for excluded_subcategory in excluded_subcategories:
        filters.append("NOT EXISTS (SELECT 1 FROM unnest(subcategory) s WHERE s ILIKE %s)")
        params.append(f"%{excluded_subcategory}%")

    if excluded_concerns:
        if len(excluded_concerns) == 1:
            filters.append("NOT (%s = ANY(health_concern_tags))")
            params.append(excluded_concerns[0])
        else:
            filters.append("NOT (health_concern_tags && %s::varchar[])")
            params.append(excluded_concerns)

    if excluded_goods:
        if len(excluded_goods) == 1:
            filters.append("goods_id <> %s")
            params.append(excluded_goods[0])
        else:
            filters.append("NOT (goods_id = ANY(%s::text[]))")
            params.append(excluded_goods)

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
