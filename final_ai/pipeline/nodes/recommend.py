import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.utils import llm, LLM_MODEL, hybrid_search_pg, build_pet_context
from pipeline.state import ChatState


# ── profile_node ──────────────────────────────────────────────────────────────

def profile_node(state: ChatState) -> dict:
    """
    user_id가 있으면 DB에서 pet 정보를 조회하고, 해당 품종의 breed_meta 정보를 결합합니다.
    """
    from pipeline.utils import get_db_connection
    
    user_id = state.get("user_id")
    pet_profile = dict(state.get("pet_profile") or {})
    health_concerns = list(state.get("health_concerns") or [])
    allergies = list(state.get("allergies") or [])
    food_prefs = list(state.get("food_preferences") or [])
    breed_context = ""
    health_traits = ""

    if user_id:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        try:
            # 1. 가장 최근 등록된 펫 정보 가져오기 (또는 특정 pet_id가 있다면 반영 가능)
            cur.execute("""
                SELECT pet_id, name, species, breed, age_years, age_months, weight_kg, gender, budget_range
                FROM pet 
                WHERE user_id = %s 
                ORDER BY created_at DESC LIMIT 1
            """, (user_id,))
            pet_row = cur.fetchone()

            if pet_row:
                pet_id = pet_row["pet_id"]
                # 펫 기본 정보 업데이트 (pet_profile 딕셔너리에 name 추가)
                pet_profile.update({
                    "name":   pet_row["name"],
                    "species": pet_row["species"],
                    "breed":  pet_row["breed"],
                    "age":    f"{pet_row['age_years']}세 {pet_row['age_months']}개월",
                    "weight": f"{pet_row['weight_kg']}kg",
                    "gender": pet_row["gender"],
                })

                # 2. 건강 관심사, 알레르기, 선호 사료 타입 가져오기
                cur.execute("SELECT concern FROM pet_health_concern WHERE pet_id = %s", (pet_id,))
                health_concerns = [r[0] for r in cur.fetchall()] or health_concerns

                cur.execute("SELECT ingredient FROM pet_allergy WHERE pet_id = %s", (pet_id,))
                allergies = [r[0] for r in cur.fetchall()] or allergies

                cur.execute("SELECT food_type FROM pet_food_preference WHERE pet_id = %s", (pet_id,))
                food_prefs = [r[0] for r in cur.fetchall()] or food_prefs

                # 3. 품종 메타 정보 가져오기 (breed_meta)
                if pet_row["breed"]:
                    cur.execute("""
                        SELECT preferred_food, health_products, chunk_text
                        FROM breed_meta 
                        WHERE breed_name = %s OR breed_name_en ILIKE %s
                        LIMIT 1
                    """, (pet_row["breed"], f"%{pet_row['breed']}%"))
                    bm = cur.fetchone()
                    if bm:
                        parts = []
                        if bm["preferred_food"]:  parts.append(f"선호 사료: {bm['preferred_food']}")
                        if bm["health_products"]: parts.append(f"추천 건강제품: {bm['health_products']}")
                        if bm["chunk_text"]:      parts.append(f"품종 특성: {bm['chunk_text']}")
                        breed_context = "\n".join(parts)
                        
                        # 4. 건강 특징 섹션만 추출 (이미지 요구사항 반영)
                        chunk = bm["chunk_text"] or ""
                        if "[건강 특징]" in chunk:
                            # [건강 특징] 이후부터 [좋아하는 사료] 전까지
                            after_health = chunk.split("[건강 특징]")[1]
                            if "[좋아하는 사료]" in after_health:
                                health_traits = after_health.split("[좋아하는 사료]")[0].strip()
                            else:
                                health_traits = after_health.strip()

        except Exception as e:
            print(f"[PROFILE] DB 조회 중 오류: {e}")
        finally:
            cur.close()
            conn.close()

    print(f"[PROFILE] user={user_id}, pet={pet_profile.get('name')}, breed={pet_profile.get('breed')}")
    return {
        "pet_profile":      pet_profile,
        "health_concerns":  health_concerns,
        "allergies":        allergies,
        "food_preferences": food_prefs,
        "breed_context":    breed_context,
        "health_traits":    health_traits
    }


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
        f"품종 특성 지식:\n{state.get('breed_context') or '없음'}\n"
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

    rrf_scores  = [float(c.get("_score", 0.0)) for c in candidates]
    pop_scores  = [c.get("popularity_score")    for c in candidates]
    sent_scores = [c.get("sentiment_avg")       for c in candidates]
    rep_scores  = [c.get("repeat_rate")         for c in candidates]

    # float 변환 후 normalize (None은 0.0으로 처리)
    norm_rrf = _normalize(rrf_scores)
    norm_pop = _normalize([float(v) if v is not None else 0.0 for v in pop_scores])

    scored = []
    for i, c in enumerate(candidates):
        # 원본 값이 None 인지 확인 (가중치 로직용)
        has_pop       = pop_scores[i]  is not None
        has_sentiment = sent_scores[i] is not None
        has_repeat    = rep_scores[i]  is not None

        if not has_pop and not has_sentiment and not has_repeat:
            score = norm_rrf[i]
        elif not has_sentiment and not has_repeat:
            # 원본이 Decimal일 수 있으므로 math할 때 float() 보장
            v_pop = float(pop_scores[i]) if has_pop else 0.0
            score = _ALPHA * norm_rrf[i] + 0.35 * float(norm_pop[i])
        else:
            gamma_v = float(sent_scores[i]) if has_sentiment else 0.0
            delta_v = float(rep_scores[i])  if has_repeat    else 0.0
            score   = (
                _ALPHA * norm_rrf[i]
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
