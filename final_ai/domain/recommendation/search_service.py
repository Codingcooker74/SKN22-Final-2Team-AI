import ast
import json
import unicodedata

from final_ai.contracts.filters import normalize_search_filters
from final_ai.domain.recommendation.constants import (
    AGE_EXCLUDE_KEYWORDS,
    AGE_MANDATORY_KEYWORDS,
    ALLERGY_SAFE_WORDS,
    ALLERGY_STOP_NOUNS,
    CORE_ANIMAL_PLANTS,
    FEED_CATEGORIES,
    MIXED_SUBS,
    PURE_CAN_SUBS,
    PURE_POUCH_SUBS,
    SAMPLE_BLACKLIST_WORDS,
    STRICT_SUBCATEGORIES,
)
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import get_logger
from final_ai.infrastructure.repositories.product_repository import list_gp_products
from final_ai.infrastructure.search.hybrid_search import hybrid_search_pg, normalize_pet_species

logger = get_logger(__name__)


def _parse_collection(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [raw]
        return [raw]
    return list(raw)

def _build_allergy_roots(allergies: list[str]) -> set[str]:
    if not allergies:
        return set()

    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    roots = set()
    for text in allergies:
        text_lower = str(text).lower()
        for token in kiwi.tokenize(text_lower):
            if token.tag.startswith("NN") and token.form not in ALLERGY_STOP_NOUNS:
                roots.add(token.form)

        cleaned_text = text_lower
        for safe_word in ALLERGY_SAFE_WORDS:
            cleaned_text = cleaned_text.replace(safe_word, " ")
        for animal in CORE_ANIMAL_PLANTS:
            if animal in cleaned_text:
                roots.add(animal)
        roots.add(text_lower)

    logger.debug("allergy roots=%s", roots)
    return roots


def _normalize_text(text) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text)).lower().replace(" ", "")


def _to_normalized_list(raw) -> list[str]:
    if isinstance(raw, str):
        return [_normalize_text(value) for value in raw.replace("{", "").replace("}", "").split(",")]
    return [_normalize_text(str(value)) for value in raw]


def _is_safe_candidate(
    candidate: dict,
    *,
    allergy_roots: set[str],
    target_age_group: str,
    mandatory_keywords: list[str],
    forbidden_age_keywords: list[str],
) -> bool:
    goods_name = _normalize_text(candidate.get("goods_name"))
    ingredient_ocr = _normalize_text(candidate.get("ingredient_text_ocr"))
    sub_list = _to_normalized_list(candidate.get("subcategory") or [])
    cat_list = _to_normalized_list(candidate.get("category") or [])

    is_feed = any(feed_category in value for feed_category in FEED_CATEGORIES for value in cat_list)
    
    # ── 연령별 필터링 (사료 카테고리에만 적용) ──────────────────────────────────
    if is_feed:
        # 1. 필수 키워드 검사 (키튼/퍼피/전연령 중 하나는 있어야 함)
        if mandatory_keywords:
            has_mandatory = any(_normalize_text(keyword) in value for keyword in mandatory_keywords for value in sub_list)
            has_mandatory = has_mandatory or any(_normalize_text(keyword) in goods_name for keyword in mandatory_keywords)
            if not has_mandatory:
                return False

        # 2. 금지 키워드 검사 (키튼인데 어덜트가 있으면 제외)
        for forbidden in forbidden_age_keywords:
            forbidden_normalized = _normalize_text(forbidden)
            if any(forbidden_normalized in value for value in sub_list) or forbidden_normalized in goods_name:
                logger.debug(
                    "age mismatch filtered product=%s forbidden=%s",
                    candidate["goods_name"],
                    forbidden_normalized,
                )
                return False

    # ── 알러지 필터링 (모든 카테고리에 적용) ────────────────────────────────────
    if allergy_roots:
        for allergy_root in allergy_roots:
            allergy_normalized = _normalize_text(allergy_root)
            if allergy_normalized in goods_name or allergy_normalized in ingredient_ocr:
                return False

        main_ingredients = candidate.get("main_ingredients") or []
        if isinstance(main_ingredients, str):
            try:
                main_ingredients = json.loads(main_ingredients)
            except Exception:
                main_ingredients = [main_ingredients]

        for ingredient in main_ingredients:
            ingredient_normalized = _normalize_text(ingredient)
            if any(_normalize_text(allergy_root) in ingredient_normalized for allergy_root in allergy_roots):
                return False

    return True


def execute_search_state(state: ChatState) -> dict:
    query = state.get("search_query") or state["user_input"]
    filters = normalize_search_filters(state.get("filters"))
    relaxation = state.get("filter_relaxation_count", 0)
    pet_type = filters.get("pet_type")
    category = filters.get("category")
    subcategory = filters.get("subcategory")
    is_strict = subcategory in STRICT_SUBCATEGORIES if subcategory else False
    if relaxation > 0 and not is_strict:
        subcategory = None
    budget = state.get("budget")
    health_concerns = state.get("health_concerns") or []

    pet_type_kr = normalize_pet_species(pet_type)
    if not pet_type_kr:
        pet_type_kr = normalize_pet_species((state.get("pet_profile") or {}).get("species"))

    candidates = hybrid_search_pg(
        query=query,
        top_k=50,
        pet_type=pet_type_kr,
        category=category,
        subcategory=subcategory,
        health_concerns=health_concerns,
        budget=budget,
    )
    logger.info(
        "search hybrid returned=%s subcategory=%s category=%s pet=%s health=%s",
        len(candidates),
        subcategory,
        category,
        pet_type_kr,
        health_concerns,
    )

    candidates = [
        candidate
        for candidate in candidates
        if not any(word in candidate.get("goods_name", "") for word in SAMPLE_BLACKLIST_WORDS)
    ]
    logger.debug("blacklist filter count=%s", len(candidates))

    target_age_group = state.get("age_group", "어덜트")
    forbidden_age_keywords = AGE_EXCLUDE_KEYWORDS.get(target_age_group, [])
    mandatory_keywords = AGE_MANDATORY_KEYWORDS.get(target_age_group, [])
    logger.info(
        "search age_group=%s forbidden=%s mandatory=%s",
        target_age_group,
        forbidden_age_keywords,
        mandatory_keywords,
    )

    allergy_roots = _build_allergy_roots(state.get("allergies") or [])
    candidates = [
        candidate
        for candidate in candidates
        if _is_safe_candidate(
            candidate,
            allergy_roots=allergy_roots,
            target_age_group=target_age_group,
            mandatory_keywords=mandatory_keywords,
            forbidden_age_keywords=forbidden_age_keywords,
        )
    ]

    logger.info("search candidates=%s relaxation=%s", len(candidates), relaxation)
    return {"search_results": candidates}
