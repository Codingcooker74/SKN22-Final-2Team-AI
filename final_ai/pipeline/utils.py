import os
import psycopg2
from dotenv import load_dotenv, find_dotenv

# .env 파일 로드 (find_dotenv로 상위 디렉토리까지 탐색)
load_dotenv(find_dotenv())

from fastembed import TextEmbedding
from final_ai.pipeline.state import ChatState

# ── 클라이언트 (lazy init) ──────────────────────────────────────────────────────
_llm = None
LLM_MODEL = "gpt-4o-mini"

def get_llm():
    global _llm
    if _llm is None:
        from openai import OpenAI
        _llm = OpenAI()
    return _llm

# 기존 코드 호환을 위한 별칭 (노드들이 llm.chat.completions.create 형식으로 호출)
class _LazyLLM:
    def __getattr__(self, name):
        return getattr(get_llm(), name)

llm = _LazyLLM()

# ── DB 연결 설정 ────────────────────────────────────────────────────────────────
def get_db_connection():
    """PostgreSQL 연결 반환"""
    host = os.getenv("POSTGRES_HOST", "localhost")
    if host == "postgres":  # Docker 내부 호스트명 → 로컬 실행 시 localhost로 변환
        host = "localhost"
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "tailtalk_db"),
        user=os.getenv("POSTGRES_USER", "mungnyang"),
        password=os.getenv("POSTGRES_PASSWORD", "finalprojectljs1908"),
        host=host,
        port=os.getenv("POSTGRES_PORT", "5432"),
    )

# ── 임베딩 모델 (lazy loading) ──────────────────────────────────────────────────
_embed_model = None

def get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        print("임베딩 모델 로드 중 (intfloat/multilingual-e5-large)...")
        _embed_model = TextEmbedding("intfloat/multilingual-e5-large")
        print("임베딩 모델 로드 완료")
    return _embed_model


def embed_query(query: str) -> list[float]:
    """E5 모델로 쿼리 임베딩 생성"""
    model = get_embed_model()
    return list(model.embed([f"query: {query}"]))[0].tolist()


# ── Hybrid Search (Vector + Keyword + RRF) ──────────────────────────────────────

def hybrid_search_pg(query: str, top_k: int = 20,
                     pet_type: str | None = None,
                     category: str | None = None,
                     subcategory: str | None = None,
                     budget: int | None = None) -> list[dict]:
    """
    PostgreSQL의 pgvector(embedding 컬럼)와 tsvector(search_vector 컬럼)를
    이용한 Hybrid Search 후 RRF(Reciprocal Rank Fusion)로 상위 결과 반환.
    """
    query_vec = embed_query(query)
    k = 60  # RRF 상수

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # [A] 벡터 검색 (코사인 유사도 기반)
        vec_sql = """
            SELECT goods_id, goods_name, pet_type, category, subcategory,
                   price, thumbnail_url, product_url, brand_name, discount_price,
                   popularity_score, sentiment_avg, repeat_rate
            FROM product
            WHERE 1=1 {filters}
            ORDER BY embedding <=> %s::vector
            LIMIT 100
        """
        keyword_sql = """
            SELECT goods_id, goods_name, pet_type, category, subcategory,
                   price, thumbnail_url, product_url, brand_name, discount_price,
                   popularity_score, sentiment_avg, repeat_rate
            FROM product
            WHERE search_vector @@ plainto_tsquery('simple', %s) {filters}
            ORDER BY ts_rank(search_vector, plainto_tsquery('simple', %s)) DESC
            LIMIT 100
        """

        # 공통 필터 조건 구성
        filter_parts = []
        filter_params_shared = []

        if pet_type:
            filter_parts.append("AND %s = ANY(pet_type)")
            filter_params_shared.append(pt_kr if 'pt_kr' in locals() else pet_type)
        if category:
            filter_parts.append("AND (%s = ANY(subcategory) OR goods_name ILIKE %s)")
            filter_params_shared.extend([category, f"%{category}%"])
        if subcategory:
            filter_parts.append("AND %s = ANY(subcategory)")
            filter_params_shared.append(subcategory)
        if budget:
            filter_parts.append("AND price <= %s")
            filter_params_shared.append(budget)

        filter_str = " ".join(filter_parts)

        # [A] 벡터 검색 파라미터: 필터들... 그 다음 벡터
        cur.execute(vec_sql.format(filters=filter_str), filter_params_shared + [query_vec])
        cols = [d[0] for d in cur.description]
        vec_rows = [dict(zip(cols, row)) for row in cur.fetchall()]

        # [B] 키워드 검색 파라미터: 검색어들... 그 다음 필터들
        # keyword_sql은 %s가 검색어용으로 앞단에 2개 있음
        cur.execute(keyword_sql.format(filters=filter_str), [query, query] + filter_params_shared)
        kw_rows = [dict(zip(cols, row)) for row in cur.fetchall()]

        # [B] RRF 점수 계산
        scores: dict[str, float] = {}
        rows_by_id: dict[str, dict] = {}

        for rank, row in enumerate(vec_rows):
            gid = row["goods_id"]
            scores[gid] = scores.get(gid, 0) + 1 / (k + rank + 1)
            rows_by_id[gid] = row

        for rank, row in enumerate(kw_rows):
            gid = row["goods_id"]
            scores[gid] = scores.get(gid, 0) + 1 / (k + rank + 1)
            rows_by_id[gid] = row

        # [C] 정렬 후 상위 반환
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        results = []
        for gid in sorted_ids[:top_k]:
            row = rows_by_id[gid]
            row["_score"] = scores[gid]
            results.append(row)

        return results

    except Exception as e:
        print(f"[hybrid_search_pg] 오류: {e}")
        return []
    finally:
        cur.close()
        conn.close()


# ── 공통 헬퍼 ───────────────────────────────────────────────────────────────────

DOMAIN_INTENT_TO_CATEGORY = {
    "health_disease":      "건강 및 질병",
    "care_management":     "사육 및 관리",
    "nutrition_diet":      "영양 및 식단",
    "behavior_psychology": "행동 및 심리",
    "travel":              "여행 및 이동",
}


def build_pet_context(state: ChatState) -> str:
    p = state.get("pet_profile") or {}
    parts = []
    if p.get("species"):
        parts.append(f"종: {'강아지' if p['species'] == 'dog' else '고양이'}")
    if p.get("breed"):   parts.append(f"품종: {p['breed']}")
    if p.get("age"):     parts.append(f"나이: {p['age']}")
    if state.get("health_concerns"):
        parts.append(f"건강관심사: {', '.join(state['health_concerns'])}")
    if state.get("allergies"):
        parts.append(f"알레르기: {', '.join(state['allergies'])}")
    if state.get("food_preferences"):
        parts.append(f"선호사료타입: {', '.join(state['food_preferences'])}")
    return " / ".join(parts) if parts else "펫 프로필 없음"
