import re

from final_ai.contracts.filters import (
    build_search_filters,
    normalize_search_exclusions,
    normalize_search_filters,
)
from final_ai.domain.recommendation.filter_relaxation import (
    build_relaxed_filter_names,
    clamp_relaxation_count,
    should_include_health_concerns,
    should_include_profile_hints,
    should_include_subcategory,
)
from final_ai.domain.recommendation.product_intent import detect_requested_product_terms
from final_ai.domain.profile.health_concerns import normalize_health_concerns
from final_ai.infrastructure.observability import get_logger
from final_ai.graph.state import ChatState

logger = get_logger(__name__)


def _sanitize_refinement_query(raw_query: str, exclusions: dict[str, list[str]]) -> str:
    sanitized = str(raw_query or "").strip()
    if not sanitized:
        return ""

    terms: list[str] = []
    for key in ("brands", "categories", "subcategories", "health_concerns", "ingredients", "keywords"):
        terms.extend(exclusions.get(key) or [])

    for term in sorted({term.strip() for term in terms if str(term or "").strip()}, key=len, reverse=True):
        sanitized = re.sub(re.escape(term), " ", sanitized, flags=re.IGNORECASE)

    for token in ("제외", "빼고", "빼줘", "말고", "삭제", "제거", "없는", "다른 거", "다른걸로", "다른상품"):
        sanitized = sanitized.replace(token, " ")

    for pattern in (
        r"\d+(?:\.\d+)?\s*만\s*원?\s*(?:이하|미만|까지|안쪽|선|이상|초과|넘는|부터)",
        r"\d+(?:\.\d+)?\s*원\s*(?:이하|미만|까지|안쪽|선|이상|초과|넘는|부터)",
    ):
        sanitized = re.sub(pattern, " ", sanitized, flags=re.IGNORECASE)

    for token in (
        "이중에서",
        "이중",
        "그중에서",
        "그중",
        "추천된 것 중",
        "추천된것중",
        "추천해준 것 중",
        "추천해준것중",
        "방금 추천한 것 중",
        "방금추천한것중",
        "보여줘",
        "보여주세요",
        "추천된",
        "추천해준",
        "방금",
        "제품들",
        "제품",
        "상품들",
        "상품",
        "것들",
        "것",
        "중",
        "만",
        "가격",
        "이하",
        "이상",
        "미만",
        "초과",
    ):
        sanitized = sanitized.replace(token, " ")

    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized


def build_search_query_state(state: ChatState) -> dict:
    """
    지정된 핵심 정보를 조합하여 검색 쿼리를 생성합니다.
    포함 정보: species(pet_type), breed, category, subcategory, health_concerns, age_group
    """
    filters = normalize_search_filters(state.get("original_filters") or state.get("filters"))
    exclusions = normalize_search_exclusions(state.get("exclusions"))
    relaxation = clamp_relaxation_count(state.get("filter_relaxation_count", 0))
    pet_profile = state.get("pet_profile") or {}
    brand = filters.get("brand") or ""

    # 1. 정보 수집 및 정규화
    # 종(species) 정보: filters에 없으면 프로필에서 가져옴
    raw_species = filters.get("pet_type") or pet_profile.get("species") or ""
    
    # 검색어용 한국어 변환 (dog -> 강아지, cat -> 고양이)
    if isinstance(raw_species, str):
        lowered = raw_species.lower()
        if lowered in ["dog", "강아지"]:
            pet_type = "강아지"
        elif lowered in ["cat", "고양이"]:
            pet_type = "고양이"
        else:
            pet_type = raw_species
    else:
        pet_type = ""
    
    # 품종(breed)
    raw_breed = pet_profile.get("breed") or ""
    breed = raw_breed if should_include_profile_hints(relaxation) else ""
    
    # 카테고리 / 소분류
    category_hint = filters.get("category") or ""
    raw_sub = filters.get("subcategory") or ""
    subcategory_hint = raw_sub if should_include_subcategory(relaxation) else ""

    # 건강 고민 및 연령대
    all_concerns = normalize_health_concerns(state.get("health_concerns") or [])
    concerns = all_concerns if should_include_health_concerns(relaxation) else []
    age_group = (state.get("age_group") or "") if should_include_profile_hints(relaxation) else ""

    # 2. 쿼리 구성 요소 수집 (순서: 종 -> 품종 -> 카테고리 -> 소분류 -> 건강고민 -> 연령대)
    query_parts = []
    
    if pet_type:
        query_parts.append(pet_type)
    
    # 품종이 종 이름과 겹치지 않을 때만 추가 (예: "강아지 강아지" 방지)
    if breed and breed != pet_type:
        query_parts.append(breed)
        
    if category_hint:
        query_parts.append(category_hint)
    if subcategory_hint and subcategory_hint != category_hint:
        query_parts.append(subcategory_hint)
    requested_product_terms = detect_requested_product_terms(
        state.get("user_input"),
        pet_type=pet_type,
        category=category_hint,
        subcategory=raw_sub,
    )
    for term in requested_product_terms:
        if term not in query_parts:
            query_parts.append(term)
    if brand and brand not in query_parts:
        query_parts.append(brand)
    
    # 건강 고민 키워드 추가
    for concern in concerns:
        if concern not in query_parts:
            query_parts.append(concern)
            
    # 연령대 정보 추가
    # [수정] 규칙 적용:
    # 1. 카테고리가 '사료'인 경우: '키튼', '퍼피'만 검색어에 포함
    # 2. 그 외 카테고리: 연령대 정보 미포함
    if category_hint == "사료":
        if age_group in ["키튼", "퍼피"]:
            if age_group not in query_parts:
                query_parts.append(age_group)

    # 3. 최종 검색어 조합
    if not query_parts:
        search_query = state.get("user_input", "").strip()
    else:
        search_query = " ".join(query_parts).strip()

    if state.get("is_result_refinement"):
        refinement_query = _sanitize_refinement_query(state.get("user_input") or "", exclusions)
        if refinement_query:
            if search_query:
                search_query = f"{refinement_query} {search_query}".strip()
            else:
                search_query = refinement_query

    logger.info(
        "search query built (Deterministic) query=%r relaxation=%s relaxed=%s refinement=%s",
        search_query,
        relaxation,
        build_relaxed_filter_names(
            relaxation=relaxation,
            filters=filters,
            health_concerns=all_concerns,
            age_group=state.get("age_group"),
            breed=raw_breed,
        ),
        bool(state.get("is_result_refinement")),
    )
    
    effective_filters = build_search_filters(
        pet_type=pet_type,
        category=category_hint,
        subcategory=subcategory_hint,
        brand=brand,
    )
    return {
        "search_query": search_query,
        "filters": effective_filters,
        "original_filters": filters,
        "effective_filters": effective_filters,
        "relaxed_filters": build_relaxed_filter_names(
            relaxation=relaxation,
            filters=filters,
            health_concerns=all_concerns,
            age_group=state.get("age_group"),
            breed=raw_breed,
        ),
        "requested_product_terms": requested_product_terms,
    }
