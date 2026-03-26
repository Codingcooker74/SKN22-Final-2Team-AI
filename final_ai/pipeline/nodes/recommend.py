import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.utils import llm, LLM_MODEL, hybrid_search_pg, build_pet_context
from pipeline.state import ChatState


# ── profile_node ──────────────────────────────────────────────────────────────

def profile_node(state: ChatState) -> dict:
    """
    breed_meta 조회 대신 pet_profile에서 species 및 health_concerns를 그대로 활용.
    (Qdrant 제거로 breed_meta 검색 불필요)
    """
    pet_profile = dict(state.get("pet_profile") or {})
    health_concerns = list(state.get("health_concerns") or [])
    print(f"[PROFILE] species={pet_profile.get('species')}, concerns={health_concerns}")
    return {"health_concerns": health_concerns}


# ── query_node ────────────────────────────────────────────────────────────────

def query_node(state: ChatState) -> dict:
    """
    LLM을 통해 검색 쿼리를 생성합니다.
    filters에서 pet_type, category, subcategory를 추출합니다.
    """
    pet_ctx  = build_pet_context(state)
    filters  = state.get("filters") or {}
    relaxation = state.get("filter_relaxation_count", 0)

    category_hint    = filters.get("category") or ""
    subcategory_hint = filters.get("subcategory") or "" if relaxation == 0 else ""

    prompt = (
        f"반려동물 상품 검색을 위한 최적화된 한국어 검색어를 한 문장으로만 반환하세요.\n"
        f"펫 정보: {pet_ctx}\n"
        f"카테고리: {category_hint} / 세부: {subcategory_hint}\n"
        f"원래 질문: {state['user_input']}"
    )
    search_query = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ).choices[0].message.content.strip()

    print(f"[QUERY] query={search_query!r}, relaxation={relaxation}")
    return {
        "search_query": search_query,
        "filters": {
            "pet_type":    filters.get("pet_type"),
            "category":    category_hint,
            "subcategory": subcategory_hint,
        },
    }


# ── search_node ───────────────────────────────────────────────────────────────

def search_node(state: ChatState) -> dict:
    """
    PostgreSQL hybrid_search_pg 실행:
    pgvector(코사인 유사도) + tsvector(FTS) + RRF 결합
    """
    query      = state.get("search_query") or state["user_input"]
    filters    = state.get("filters") or {}
    relaxation = state.get("filter_relaxation_count", 0)

    pet_type    = filters.get("pet_type")
    category    = filters.get("category")
    subcategory = filters.get("subcategory") if relaxation == 0 else None
    budget      = state.get("budget")

    # pet_type 변환 (dog/cat → 강아지/고양이)
    pt_kr = None
    if pet_type == "dog":
        pt_kr = "강아지"
    elif pet_type == "cat":
        pt_kr = "고양이"
    elif pet_type:  # 이미 한글인 경우
        pt_kr = pet_type

    candidates = hybrid_search_pg(
        query=query,
        top_k=20,
        pet_type=pt_kr,
        category=category,
        subcategory=subcategory,
        budget=budget,
    )

    # 알레르기 post-filter
    allergies = state.get("allergies") or []
    if allergies:
        def is_safe(c):
            ingredients = str(c.get("ingredient_text_ocr") or "").lower()
            return not any(a.lower() in ingredients for a in allergies)
        candidates = [c for c in candidates if is_safe(c)]

    print(f"[SEARCH] {len(candidates)}개 후보 (relaxation={relaxation})")
    return {"search_results": candidates}


# ── rerank_node ───────────────────────────────────────────────────────────────

_ALPHA   = 0.50
_BETA    = 0.25
_GAMMA   = 0.15
_DELTA   = 0.10
_EPSILON = 0.10
_TOP_K   = 5


def _normalize(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def rerank_node(state: ChatState) -> dict:
    """재랭킹: RRF 점수 + 인기도·감성·재구매율 가중치"""
    candidates      = state.get("search_results") or []
    detected_aspect = state.get("detected_aspect")
    relaxation      = state.get("filter_relaxation_count", 0)

    if not candidates:
        print("[RERANK] 후보 없음")
        return {
            "reranked_results":        [],
            "filter_relaxation_count": relaxation + 1 if relaxation < 1 else relaxation,
        }

    rrf_scores  = [c.get("_score", 0.0)        for c in candidates]
    pop_scores  = [c.get("popularity_score")    for c in candidates]
    sent_scores = [c.get("sentiment_avg")       for c in candidates]
    rep_scores  = [c.get("repeat_rate")         for c in candidates]

    norm_rrf = _normalize(rrf_scores)
    norm_pop = _normalize([v if v is not None else 0.0 for v in pop_scores])

    scored = []
    for i, c in enumerate(candidates):
        has_pop       = pop_scores[i]  is not None
        has_sentiment = sent_scores[i] is not None
        has_repeat    = rep_scores[i]  is not None

        if not has_pop and not has_sentiment and not has_repeat:
            score = float(norm_rrf[i])
        elif not has_sentiment and not has_repeat:
            score = _ALPHA * float(norm_rrf[i]) + 0.35 * float(norm_pop[i])
        else:
            # Decimal → float 변환
            gamma_v = float(sent_scores[i]) if has_sentiment else 0.0
            delta_v = float(rep_scores[i])  if has_repeat    else 0.0
            score   = (
                _ALPHA * float(norm_rrf[i])
                + _BETA  * float(norm_pop[i])
                + _GAMMA * gamma_v
                + _DELTA * delta_v
            )

        if detected_aspect and c.get("sentiment_avg") is not None:
            score += _EPSILON * float(c["sentiment_avg"])

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:_TOP_K]]

    new_relaxation = relaxation
    if len(top) < 3 and relaxation < 1:
        new_relaxation = relaxation + 1
        print(f"[RERANK] 결과 부족 ({len(top)}개) → 필터 완화 예정")
    else:
        print(f"[RERANK] 최종 {len(top)}개")

    return {
        "reranked_results":        top,
        "filter_relaxation_count": new_relaxation,
    }
