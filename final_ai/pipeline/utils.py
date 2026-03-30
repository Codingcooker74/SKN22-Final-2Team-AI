import os
import re
import psycopg2
from dotenv import load_dotenv, find_dotenv

# .env 파일 로드 (find_dotenv로 상위 디렉토리까지 탐색)
load_dotenv(find_dotenv())

from fastembed import TextEmbedding
from final_ai.observability import wrap_openai
from final_ai.pipeline.state import ChatState

# ── 클라이언트 (lazy init) ──────────────────────────────────────────────────────
_llm = None
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL_NAME = os.getenv("FASTEMBED_MODEL", "intfloat/multilingual-e5-large")
_SUPPORTED_EMBED_MODELS = {
    model["model"]: model for model in TextEmbedding.list_supported_models()
}
_embed_model_unavailable_reason = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_memory_limit_bytes() -> int | None:
    paths = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        if limit > 0 and limit < (1 << 60):
            return limit
    return None


def _read_host_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or total_pages <= 0:
        return None
    return page_size * total_pages


def _total_memory_gb() -> float | None:
    limits = [value for value in (_read_memory_limit_bytes(), _read_host_memory_bytes()) if value]
    if not limits:
        return None
    return min(limits) / (1024 ** 3)


def _embed_model_size_gb() -> float | None:
    model = _SUPPORTED_EMBED_MODELS.get(EMBED_MODEL_NAME)
    if not model:
        return None
    size = model.get("size_in_GB")
    try:
        return float(size)
    except (TypeError, ValueError):
        return None


def _dense_model_allowed() -> tuple[bool, str | None]:
    if not _env_flag("FASTEMBED_ENABLED", default=True):
        return False, "FASTEMBED_ENABLED=false"

    if not _env_flag("FASTEMBED_AUTO_DISABLE_LOW_MEMORY", default=True):
        return True, None

    total_memory_gb = _total_memory_gb()
    model_size_gb = _embed_model_size_gb()
    headroom_gb = float(os.getenv("FASTEMBED_MEMORY_HEADROOM_GB", "0.75"))

    if total_memory_gb is None or model_size_gb is None:
        return True, None

    required_memory_gb = model_size_gb + headroom_gb
    if total_memory_gb < required_memory_gb:
        return (
            False,
            "insufficient memory for dense embedding "
            f"(available={total_memory_gb:.2f}GB, required~={required_memory_gb:.2f}GB, model={EMBED_MODEL_NAME})",
        )

    return True, None

def get_llm():
    global _llm
    if _llm is None:
        from openai import OpenAI
        _llm = wrap_openai(OpenAI())
    return _llm

# 기존 코드 호환을 위한 별칭 (노드들이 llm.chat.completions.create 형식으로 호출)
class _LazyLLM:
    def __getattr__(self, name):
        return getattr(get_llm(), name)

llm = _LazyLLM()

# ── DB 연결 설정 ────────────────────────────────────────────────────────────────
_PET_SPECIES_KR = {
    "dog": "강아지",
    "cat": "고양이",
    "강아지": "강아지",
    "고양이": "고양이",
}

_SEARCH_STOPWORDS = {
    "추천",
    "추천해줘",
    "추천해주세요",
    "추천해",
    "알려줘",
    "알려주세요",
    "좋은",
    "좋아요",
    "우리",
    "아이",
    "반려동물",
    "반려견",
    "반려묘",
}

_TOKEN_SUFFIXES = (
    "추천해주세요",
    "추천해줘",
    "알려주세요",
    "알려줘",
    "입니다",
    "이에요",
    "예요",
    "에요",
    "으로",
    "에서",
    "한테",
    "용",
    "에",
    "의",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "과",
    "와",
    "도",
    "로",
)


def normalize_pet_species(species: str | None) -> str | None:
    if not species:
        return None
    return _PET_SPECIES_KR.get(str(species).strip())


def _extract_search_terms(query: str, *extra_terms: str | None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add_token(raw: str):
        token = raw.strip()
        if not token:
            return
        if token in _SEARCH_STOPWORDS:
            return
        if token not in seen:
            seen.add(token)
            terms.append(token)

    for source in (query, *extra_terms):
        if not source:
            continue
        for raw in re.findall(r"[0-9A-Za-z가-힣/]+", str(source)):
            pieces = [piece for piece in raw.split("/") if piece]
            for piece in pieces:
                add_token(piece)
                for suffix in _TOKEN_SUFFIXES:
                    if len(piece) <= len(suffix) + 1:
                        continue
                    if piece.endswith(suffix):
                        trimmed = piece[: -len(suffix)].strip()
                        if len(trimmed) >= 2:
                            add_token(trimmed)
    return terms


def get_db_connection():
    """PostgreSQL 연결 반환.

    배포/로컬 실행 모두 `POSTGRES_HOST`에 지정된 값을 그대로 사용한다.
    - Docker compose 내부: `postgres`
    - 호스트 로컬 실행: `localhost`
    - 배포 환경: RDS endpoint
    """
    host = (os.getenv("POSTGRES_HOST") or "localhost").strip() or "localhost"
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "tailtalk_db"),
        user=os.getenv("POSTGRES_USER", "mungnyang"),
        password=os.getenv("POSTGRES_PASSWORD", "finalprojectljs1908"),
        host=host,
        port=os.getenv("POSTGRES_PORT", "5432"),
    )

# ── 임베딩 모델 (lazy loading) ──────────────────────────────────────────────────
_embed_model = None

def get_embed_model() -> TextEmbedding | None:
    global _embed_model, _embed_model_unavailable_reason
    if _embed_model is not None:
        return _embed_model

    if _embed_model_unavailable_reason is not None:
        return None

    allowed, reason = _dense_model_allowed()
    if not allowed:
        _embed_model_unavailable_reason = reason
        print(f"[get_embed_model] dense embedding disabled: {reason}")
        return None

    try:
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        local_files_only = _env_flag("FASTEMBED_LOCAL_FILES_ONLY", default=False)
        kwargs = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            kwargs["cache_dir"] = cache_dir
        if local_files_only:
            kwargs["local_files_only"] = True

        print(
            f"임베딩 모델 로드 중 ({EMBED_MODEL_NAME}, "
            f"local_only={local_files_only}, cache_dir={cache_dir or 'default'})..."
        )
        _embed_model = TextEmbedding(EMBED_MODEL_NAME, **kwargs)
        print("임베딩 모델 로드 완료")
    except Exception as e:
        _embed_model_unavailable_reason = str(e)
        raise
    return _embed_model


def embed_query(query: str) -> list[float] | None:
    """E5 모델로 쿼리 임베딩 생성.

    배포 환경에서 모델 로드에 실패하더라도 FTS 검색으로 폴백할 수 있게
    예외를 삼키고 `None`을 반환한다.
    """
    try:
        model = get_embed_model()
        if model is None:
            if _embed_model_unavailable_reason:
                print(f"[embed_query] dense embedding unavailable: {_embed_model_unavailable_reason}")
            return None
        return list(model.embed([f"query: {query}"]))[0].tolist()
    except Exception as e:
        print(f"[embed_query] dense embedding unavailable: {e}")
        return None


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
    pet_type_kr = normalize_pet_species(pet_type) or pet_type
    k = 60  # RRF 상수

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # [A] 벡터 검색 (코사인 유사도 기반)
        vec_sql = """
            SELECT goods_id, goods_name, pet_type, category, subcategory,
                   price, thumbnail_url, product_url, brand_name, discount_price,
                   popularity_score, sentiment_avg, repeat_rate, health_concern_tags,
                   rating, review_count
            FROM product
            WHERE 1=1 {filters}
            ORDER BY embedding <=> %s::vector
            LIMIT 100
        """
        keyword_sql = """
            SELECT goods_id, goods_name, pet_type, category, subcategory,
                   price, thumbnail_url, product_url, brand_name, discount_price,
                   popularity_score, sentiment_avg, repeat_rate, health_concern_tags,
                   rating, review_count
            FROM product
            WHERE search_vector @@ plainto_tsquery('simple', %s) {filters}
            ORDER BY ts_rank(search_vector, plainto_tsquery('simple', %s)) DESC
            LIMIT 100
        """

        # 공통 필터 조건 구성
        filter_parts = []
        filter_params_shared = []

        if pet_type_kr:
            filter_parts.append("AND %s = ANY(pet_type)")
            filter_params_shared.append(pet_type_kr)
        if category:
            filter_parts.append("AND (%s = ANY(category) OR %s = ANY(subcategory) OR goods_name ILIKE %s)")
            filter_params_shared.extend([category, category, f"%{category}%"])
        if subcategory:
            filter_parts.append("AND %s = ANY(subcategory)")
            filter_params_shared.append(subcategory)
        if budget:
            filter_parts.append("AND price <= %s")
            filter_params_shared.append(budget)

        filter_str = " ".join(filter_parts)

        cols = []
        vec_rows = []
        if query_vec is not None:
            # [A] 벡터 검색 파라미터: 필터들... 그 다음 벡터
            cur.execute(vec_sql.format(filters=filter_str), filter_params_shared + [query_vec])
            cols = [d[0] for d in cur.description]
            vec_rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        else:
            print("[hybrid_search_pg] dense search skipped; falling back to keyword-only search")

        # [B] 키워드 검색 파라미터: 검색어들... 그 다음 필터들
        # keyword_sql은 %s가 검색어용으로 앞단에 2개 있음
        cur.execute(keyword_sql.format(filters=filter_str), [query, query] + filter_params_shared)
        kw_cols = [d[0] for d in cur.description]
        if not cols:
            cols = kw_cols
        kw_rows = [dict(zip(kw_cols, row)) for row in cur.fetchall()]

        # 자연어 질문이 search_vector와 정확히 맞지 않는 경우를 위한 느슨한 폴백 검색.
        if not vec_rows and not kw_rows:
            loose_terms = _extract_search_terms(query, category, subcategory)
            if loose_terms:
                score_parts = []
                score_params = []
                where_parts = []
                where_params = []

                for term in loose_terms:
                    like = f"%{term}%"
                    score_parts.append(
                        "("
                        "CASE WHEN goods_name ILIKE %s THEN 5 ELSE 0 END + "
                        "CASE WHEN brand_name ILIKE %s THEN 2 ELSE 0 END + "
                        "CASE WHEN COALESCE(array_to_string(category, ' '), '') ILIKE %s THEN 3 ELSE 0 END + "
                        "CASE WHEN COALESCE(array_to_string(subcategory, ' '), '') ILIKE %s THEN 4 ELSE 0 END + "
                        "CASE WHEN COALESCE(array_to_string(health_concern_tags, ' '), '') ILIKE %s THEN 4 ELSE 0 END"
                        ")"
                    )
                    score_params.extend([like, like, like, like, like])
                    where_parts.append(
                        "("
                        "goods_name ILIKE %s OR "
                        "brand_name ILIKE %s OR "
                        "COALESCE(array_to_string(category, ' '), '') ILIKE %s OR "
                        "COALESCE(array_to_string(subcategory, ' '), '') ILIKE %s OR "
                        "COALESCE(array_to_string(health_concern_tags, ' '), '') ILIKE %s"
                        ")"
                    )
                    where_params.extend([like, like, like, like, like])

                loose_sql = f"""
                    SELECT goods_id, goods_name, pet_type, category, subcategory,
                           price, thumbnail_url, product_url, brand_name, discount_price,
                           popularity_score, sentiment_avg, repeat_rate, health_concern_tags,
                           rating, review_count,
                           ({' + '.join(score_parts)}) AS loose_score
                    FROM product
                    WHERE 1=1 {filter_str}
                      AND ({' OR '.join(where_parts)})
                    ORDER BY loose_score DESC,
                             popularity_score DESC NULLS LAST,
                             review_count DESC NULLS LAST,
                             price ASC
                    LIMIT 100
                """
                cur.execute(loose_sql, score_params + filter_params_shared + where_params)
                loose_cols = [d[0] for d in cur.description]
                kw_rows = [dict(zip(loose_cols, row)) for row in cur.fetchall()]
                if kw_rows:
                    print(
                        "[hybrid_search_pg] loose fallback search matched "
                        f"{len(kw_rows)} rows for query={query!r}, terms={loose_terms}"
                    )

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
        if cur is not None:
            cur.close()
        if conn is not None:
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
        parts.append(f"종: {normalize_pet_species(p['species']) or p['species']}")
    if p.get("breed"):   parts.append(f"품종: {p['breed']}")
    if p.get("age"):     parts.append(f"나이: {p['age']}")
    if state.get("health_concerns"):
        parts.append(f"건강관심사: {', '.join(state['health_concerns'])}")
    if state.get("allergies"):
        parts.append(f"알레르기: {', '.join(state['allergies'])}")
    if state.get("food_preferences"):
        parts.append(f"선호사료타입: {', '.join(state['food_preferences'])}")
    return " / ".join(parts) if parts else "펫 프로필 없음"
