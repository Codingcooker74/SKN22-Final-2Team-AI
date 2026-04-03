from final_ai.api.dependencies.request_context import ensure_request_active
from final_ai.contracts.filters import build_search_filters, normalize_search_filters
from final_ai.domain.profile.service import build_pet_context
from final_ai.domain.recommendation.constants import STRICT_SUBCATEGORIES
from final_ai.infrastructure.llm.openai_client import LLM_MODEL, llm
from final_ai.infrastructure.observability import get_logger
from final_ai.graph.state import ChatState

logger = get_logger(__name__)


def build_search_query_state(state: ChatState) -> dict:
    pet_ctx = build_pet_context(state)
    filters = normalize_search_filters(state.get("filters"))
    relaxation = state.get("filter_relaxation_count", 0)

    category_hint = filters.get("category") or ""
    raw_sub = filters.get("subcategory") or ""
    is_strict = raw_sub in STRICT_SUBCATEGORIES if raw_sub else False
    subcategory_hint = raw_sub if (relaxation == 0 or is_strict) else ""

    prominent_concerns = ", ".join(state.get("health_concerns") or [])
    if prominent_concerns:
        concern_clause = (
            f"- **특히 다음 건강 고민사항을 반드시 해결할 수 있는 상품 위주로 검색어를 구성하세요: {prominent_concerns}**\n"
            f"- **사료(주식)와 간식(보상용)을 엄격히 구분하세요.**\n"
            f"- **캔(Can)과 파우치(Pouch)는 서로 다른 제형입니다. 사용자가 '캔'을 언급하면 반드시 '캔'이 포함된 소분류를, "
            f"'파우치'를 언급하면 '파우치'가 포함된 소분류를 선택하세요.**"
        )
    else:
        concern_clause = (
            f"- **사료(주식)와 간식(보상용)을 엄격히 구분하세요.**\n"
            f"- **캔(Can)과 파우치(Pouch)는 서로 다른 제형입니다. 사용자가 '캔'을 언급하면 반드시 '캔'이 포함된 소분류를, "
            f"'파우치'를 언급하면 '파우치'가 포함된 소분류를 선택하세요.**"
        )

    prompt = (
        "반려동물 상품 검색을 위한 최적화된 한국어 검색어를 한 문장으로만 반환하세요.\n"
        "중요: 검색어에는 '어덜트', '시니어' 단어를 직접 포함하지 마세요.\n"
        f"펫 정보: {pet_ctx}\n"
        f"연령대: {state.get('age_group') or '없음'}\n"
        f"품종 특성 지식:\n{state.get('breed_context') or '없음'}\n"
        f"제외 성분: {', '.join(state.get('allergies') or []) or '없음'}\n"
        f"카테고리: {category_hint} / 세부: {subcategory_hint}\n"
        f"{concern_clause}\n"
        f"원래 질문: {state['user_input']}"
    )
    try:
        ensure_request_active()
        search_query = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        ).choices[0].message.content.strip()
    except Exception as exc:
        fallback_parts = [category_hint, subcategory_hint, state["user_input"]]
        search_query = " ".join(part for part in fallback_parts if part).strip() or state["user_input"]
        logger.warning("query llm fallback used: %s", exc)

    query_age_group = state.get("age_group")
    if query_age_group == "키튼" and "키튼" not in search_query:
        search_query = f"{search_query} 키튼"
    elif query_age_group == "퍼피" and "퍼피" not in search_query:
        search_query = f"{search_query} 퍼피"

    logger.info("search query built query=%r relaxation=%s", search_query, relaxation)
    return {
        "search_query": search_query,
        "filters": build_search_filters(
            pet_type=filters.get("pet_type"),
            category=category_hint,
            subcategory=subcategory_hint,
        ),
    }
