import json
import unicodedata
from functools import lru_cache
from pathlib import Path


def normalize_product_text(text) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text)).lower().replace(" ", "")


@lru_cache(maxsize=1)
def _load_category_tree() -> dict:
    path = Path(__file__).resolve().parents[2] / "pipeline" / "data" / "category.json"
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def _iter_subcategory_aliases(
    *,
    pet_type: str | None,
    category: str | None,
    subcategory: str | None,
) -> list[str]:
    category_tree = _load_category_tree()
    pet_nodes = []
    if pet_type and pet_type in category_tree:
        pet_nodes.append(category_tree[pet_type])
    else:
        pet_nodes.extend(category_tree.values())

    terms: list[str] = []
    for pet_node in pet_nodes:
        categories = []
        if category and category in pet_node:
            categories.append(pet_node[category])
        else:
            categories.extend(pet_node.values())

        for category_node in categories:
            subcategories = (category_node or {}).get("subcategories") or {}
            for canonical, aliases in subcategories.items():
                if subcategory and canonical != subcategory:
                    continue
                terms.extend(str(part) for part in str(canonical).split("/") if part)
                terms.extend(str(alias) for alias in aliases or [])

    if subcategory:
        terms.extend(str(part) for part in str(subcategory).split("/") if part)

    seen = set()
    unique_terms = []
    normalized_category = normalize_product_text(category)
    normalized_subcategory = normalize_product_text(subcategory)
    for term in terms:
        normalized = normalize_product_text(term)
        if len(normalized) < 2 or normalized in seen:
            continue
        if normalized and normalized in {normalized_category, normalized_subcategory}:
            continue
        seen.add(normalized)
        unique_terms.append(term)
    return unique_terms


def detect_requested_product_terms(
    user_input: str | None,
    *,
    pet_type: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
) -> list[str]:
    if category == "사료":
        return []

    normalized_input = normalize_product_text(user_input)
    if not normalized_input:
        return []

    matched: list[tuple[int, int, str]] = []
    for term in _iter_subcategory_aliases(
        pet_type=pet_type,
        category=category,
        subcategory=subcategory,
    ):
        normalized_term = normalize_product_text(term)
        position = normalized_input.find(normalized_term)
        if position >= 0:
            matched.append((position, -len(normalized_term), term))

    return [term for _, _, term in sorted(matched)]


def candidate_matches_requested_terms(candidate: dict, requested_product_terms: list[str]) -> bool:
    if not requested_product_terms:
        return False

    goods_name = normalize_product_text(candidate.get("goods_name"))
    if not goods_name:
        return False

    return any(
        normalize_product_text(term) in goods_name
        for term in requested_product_terms
        if normalize_product_text(term)
    )
