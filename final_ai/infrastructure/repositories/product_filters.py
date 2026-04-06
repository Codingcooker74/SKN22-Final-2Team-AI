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
    budget: int | None = None,
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
        if len(categories) == 1:
            filters.append("(%s = ANY(category) OR %s = ANY(subcategory))")
            params.extend([categories[0], categories[0]])
        else:
            filters.append("(category && %s::text[] OR subcategory && %s::text[])")
            params.extend([categories, categories])

    if subcategories:
        if len(subcategories) == 1:
            filters.append("%s = ANY(subcategory)")
            params.append(subcategories[0])
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

    if budget is not None:
        filters.append("price <= %s")
        params.append(budget)

    return filters, params
