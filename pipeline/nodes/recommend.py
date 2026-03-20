import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range
from pipeline.utils import llm, LLM_MODEL, qdrant, hybrid_search, build_pet_context
from pipeline.state import ChatState


# ── profile_node ──────────────────────────────────────────────────────────────

def profile_node(state: ChatState) -> dict:
    """breed_meta 조회: 품종 기반 health_concern_tags 보완"""
    pet_profile = dict(state.get("pet_profile") or {})
    breed = pet_profile.get("breed")
    species = pet_profile.get("species")

    extra_concerns: list[str] = []

    if breed:
        f = None
        if species:
            f = Filter(must=[FieldCondition(key="pet_type", match=MatchValue(value=species))])
        points = hybrid_search("breed_meta", breed, top_k=1, qdrant_filter=f)
        if points:
            p = points[0].payload
            extra_concerns = p.get("health_keywords") or []
            print(f"[PROFILE] breed={breed} → health_keywords={extra_concerns}")
        else:
            print(f"[PROFILE] breed_meta 미검색: {breed}")
    else:
        print("[PROFILE] 품종 정보 없음 — 스킵")

    # state의 health_concerns와 합산 (중복 제거)
    existing = list(state.get("health_concerns") or [])
    merged = list(dict.fromkeys(existing + extra_concerns))

    return {"health_concerns": merged}


# ── query_node ────────────────────────────────────────────────────────────────

def query_node(state: ChatState) -> dict:
    """검색 쿼리 생성 + Qdrant 필터 빌드"""
    pet_ctx        = build_pet_context(state)
    filters        = state.get("filters") or {}
    health_concerns = state.get("health_concerns") or []
    relaxation     = state.get("filter_relaxation_count", 0)

    # ── 검색 쿼리 ──────────────────────────────────────────────────────────────
    category_hint = filters.get("category") or ""
    subcategory_hint = filters.get("subcategory") or ""

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

    # ── Qdrant 필터 ────────────────────────────────────────────────────────────
    must = [FieldCondition(key="sold_out", match=MatchValue(value=False))]

    pet_type = filters.get("pet_type")
    if pet_type:
        must.append(FieldCondition(key="pet_type", match=MatchAny(any=[pet_type])))

    # 필터 완화 전: category / subcategory 적용
    if relaxation == 0:
        if category_hint:
            must.append(FieldCondition(key="category", match=MatchAny(any=[category_hint])))
        if subcategory_hint:
            must.append(FieldCondition(key="subcategory", match=MatchAny(any=[subcategory_hint])))
    else:
        # 완화: category만 유지, subcategory 제거
        if category_hint:
            must.append(FieldCondition(key="category", match=MatchAny(any=[category_hint])))
        print(f"[QUERY] 필터 완화 (relaxation={relaxation}): subcategory 제거")

    # 예산
    budget = state.get("budget")
    if budget:
        must.append(FieldCondition(key="price", range=Range(lte=budget)))

    qdrant_filter = Filter(must=must)

    # 알레르기 제외 (must_not)
    allergies = state.get("allergies") or []
    if allergies:
        must_not = [
            FieldCondition(key="main_ingredients", match=MatchAny(any=[a]))
            for a in allergies
        ]
        qdrant_filter = Filter(must=must, must_not=must_not)

    print(f"[QUERY] query={search_query!r}, relaxation={relaxation}")
    return {"search_query": search_query, "filters": {**filters, "_qdrant_filter": None},
            "_qdrant_filter_obj": qdrant_filter}


# ── search_node ───────────────────────────────────────────────────────────────

def search_node(state: ChatState) -> dict:
    """products Hybrid Search"""
    query  = state.get("search_query") or state["user_input"]
    # _qdrant_filter_obj는 직접 꺼낼 수 없으니 rebuild (query_node와 동일 로직)
    # → query_node에서 state에 직렬화 불가 객체를 넣을 수 없으므로, 여기서 재빌드
    filters    = state.get("filters") or {}
    relaxation = state.get("filter_relaxation_count", 0)
    allergies  = state.get("allergies") or []
    budget     = state.get("budget")

    must = [FieldCondition(key="sold_out", match=MatchValue(value=False))]

    pet_type = filters.get("pet_type")
    if pet_type:
        must.append(FieldCondition(key="pet_type", match=MatchAny(any=[pet_type])))

    category = filters.get("category")
    subcategory = filters.get("subcategory")
    if relaxation == 0:
        if category:
            must.append(FieldCondition(key="category", match=MatchAny(any=[category])))
        if subcategory:
            must.append(FieldCondition(key="subcategory", match=MatchAny(any=[subcategory])))
    else:
        if category:
            must.append(FieldCondition(key="category", match=MatchAny(any=[category])))

    if budget:
        must.append(FieldCondition(key="price", range=Range(lte=budget)))

    must_not = (
        [FieldCondition(key="main_ingredients", match=MatchAny(any=[a])) for a in allergies]
        if allergies else []
    )
    f = Filter(must=must, must_not=must_not) if must_not else Filter(must=must)

    points = hybrid_search("products", query, top_k=20, qdrant_filter=f)
    candidates = [p.payload | {"_score": p.score} for p in points]

    # 알레르기 ingredient_text_ocr 추가 필터링 (post-filter)
    if allergies:
        def safe(c):
            ocr = (c.get("ingredient_text_ocr") or "").lower()
            return not any(a.lower() in ocr for a in allergies)
        candidates = [c for c in candidates if safe(c)]

    print(f"[SEARCH] {len(candidates)}개 후보 (relaxation={relaxation})")
    return {"search_results": candidates}


# ── rerank_node ───────────────────────────────────────────────────────────────

_ALPHA = 0.50
_BETA  = 0.25
_GAMMA = 0.15
_DELTA = 0.10
_EPSILON = 0.10

_TOP_K = 5


def _normalize(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def rerank_node(state: ChatState) -> dict:
    """재랭킹: α·β·γ·δ·ε 가중치 + Fallback A/B/C/D"""
    candidates     = state.get("search_results") or []
    detected_aspect = state.get("detected_aspect")
    relaxation     = state.get("filter_relaxation_count", 0)

    if not candidates:
        print("[RERANK] 후보 없음")
        return {
            "reranked_results":        [],
            "filter_relaxation_count": relaxation + 1 if relaxation < 1 else relaxation,
        }

    rrf_scores  = [c.get("_score", 0.0) for c in candidates]
    pop_scores  = [c.get("popularity_score") for c in candidates]
    sent_scores = [c.get("sentiment_avg")    for c in candidates]
    rep_scores  = [c.get("repeat_rate")      for c in candidates]

    norm_rrf = _normalize(rrf_scores)
    norm_pop = _normalize([v if v is not None else 0.0 for v in pop_scores])

    scored = []
    for i, c in enumerate(candidates):
        has_sentiment = sent_scores[i] is not None
        has_repeat    = rep_scores[i]  is not None
        has_pop       = pop_scores[i]  is not None

        # Fallback 결정
        if not has_pop and not has_sentiment and not has_repeat:
            # Case C: 신상품 — RRF만
            score = norm_rrf[i]
        elif not has_sentiment and not has_repeat:
            # Case B: 리뷰 없음 — β 상향
            score = _ALPHA * norm_rrf[i] + 0.35 * norm_pop[i]
        else:
            # Case A: 풀 스코어링
            gamma_v = sent_scores[i] if has_sentiment else 0.0
            delta_v = rep_scores[i]  if has_repeat    else 0.0
            score   = (
                _ALPHA * norm_rrf[i]
                + _BETA  * norm_pop[i]
                + _GAMMA * gamma_v
                + _DELTA * delta_v
            )

        # Case D: ABSA 속성 감지 시 ε 추가
        if detected_aspect and c.get("sentiment_avg") is not None:
            score += _EPSILON * c["sentiment_avg"]

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:_TOP_K]]

    new_relaxation = relaxation
    if len(top) < 3 and relaxation < 1:
        new_relaxation = relaxation + 1
        print(f"[RERANK] 결과 부족 ({len(top)}개) → 필터 완화 예정 (relaxation → {new_relaxation})")
    else:
        print(f"[RERANK] 최종 {len(top)}개")

    return {
        "reranked_results":        top,
        "filter_relaxation_count": new_relaxation,
    }
