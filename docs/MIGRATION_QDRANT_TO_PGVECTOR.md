# Migration: Qdrant → PostgreSQL + pgvector

> 작성일: 2026-03-23
> 목적: Qdrant 벡터 DB를 제거하고 PostgreSQL + pgvector로 전환

---

## 1. 현재 Qdrant 검색 구조 (AS-IS)

모든 검색은 `pipeline/utils.py`의 `hybrid_search()` 함수를 공유한다.
Dense(의미 유사도) + Sparse(키워드 매칭)을 RRF Fusion으로 합산하는 Hybrid Search 방식이며, 컬렉션별로 필터와 용도만 다르다.

```
쿼리 텍스트
  ├─ Dense : multilingual-e5-large (1024d) → cosine similarity
  ├─ Sparse: Qdrant/bm25 → 키워드 매칭
  └─ RRF Fusion: 두 결과를 Reciprocal Rank Fusion으로 합산
```

### 1.1 `products` 컬렉션 — 상품 추천 검색

| 항목 | 내용 |
|------|------|
| 호출 위치 | `search_node()` — `pipeline/nodes/recommend.py:145` |
| 검색 방식 | Dense + Sparse → RRF Fusion, `top_k=20` |
| 필터 | `sold_out=False`, `pet_type`, `category`, `subcategory`, `price <= budget`, `main_ingredients NOT IN allergies` |
| 반환 payload | goods_id, product_name, brand_name, price, discount_price, thumbnail_url, product_url, popularity_score, sentiment_avg, repeat_rate, ingredient_text_ocr |
| 용도 | 사용자 질문에 맞는 상품 후보 검색 → `rerank_node`에서 5가지 가중치로 재랭킹 후 상위 5개 선정 |

### 1.2 `domain_qna` 컬렉션 — 반려동물 전문 지식 RAG

| 항목 | 내용 |
|------|------|
| 호출 위치 | `rag_node()` — `pipeline/nodes/domain_qa.py:44` |
| 검색 방식 | Dense + Sparse → RRF Fusion, `top_k=5` |
| 필터 | `species IN [dog/cat, "both"]`, `category` (건강 및 질병, 사육 및 관리, 영양 및 식단, 행동 및 심리, 여행 및 이동) |
| 반환 payload | question, answer |
| 용도 | 건강/영양/행동 등 도메인 질문에 대한 RAG 컨텍스트 → `respond_node`에서 LLM 응답 생성에 활용 |

### 1.3 `breed_meta` 컬렉션 — 품종별 건강 메타데이터

| 항목 | 내용 |
|------|------|
| 호출 위치 | `profile_node()` — `pipeline/nodes/recommend.py:25` |
| 검색 방식 | Dense + Sparse → RRF Fusion, `top_k=1` |
| 필터 | `pet_type` (dog/cat) |
| 반환 payload | health_keywords (품종 특이 건강 키워드 리스트) |
| 용도 | 품종명으로 검색 → 해당 품종의 건강 취약점 키워드를 `health_concerns`에 자동 병합 → 상품 검색 품질 향상 |

### 1.4 pgvector 전환 가능성

> **결론: 기존 PG 테이블에 `dense_embedding vector(1024)` + `text_search tsvector` 컬럼만 추가하면 동일 기능 구현 가능.**

| Qdrant 기능 | pgvector 대체 | 비고 |
|-------------|--------------|------|
| Dense 벡터 검색 (cosine) | `<=>` 연산자 + HNSW 인덱스 | 동일 임베딩 모델 사용 |
| Sparse 벡터 검색 (BM25) | `tsvector` + `ts_rank()` + GIN 인덱스 | 별도 임베딩 불필요, PG 내장 |
| RRF Fusion | SQL CTE에서 `1/(k+rank)` 직접 계산 | k=60 기본값 |
| FieldCondition 필터 | SQL `WHERE` 절 | 더 유연함 |
| must_not (알레르기 제외) | `NOT (main_ingredients && ARRAY[:allergies])` | PG 배열 연산자 |
| Range (price <= budget) | `WHERE price <= :budget` | 그대로 |

핵심 포인트:
- **데이터가 이미 PG에 있다면** → 임베딩 컬럼 2개 추가 + 인덱스 생성만으로 충분
- **데이터가 PG에 없다면** → 테이블 생성 + 데이터 적재 + 임베딩 생성이 필요
- Qdrant 전용 컬렉션 3개는 PG 테이블 3개로 1:1 대응
- `hybrid_search()` 함수 시그니처를 유지하면 **노드 코드 변경을 최소화**할 수 있음

---

## 2. 영향 범위 요약

| 파일 | 수정 유형 | 영향도 |
|------|----------|--------|
| `requirements.txt` | 의존성 교체 | 낮음 |
| `core/qdrant_setup.py` | **전체 재작성** → `core/db_setup.py` | 높음 |
| `pipeline/utils.py` | **전체 재작성** (클라이언트 + 검색 함수) | 높음 |
| `pipeline/nodes/domain_qa.py` | 필터 빌드 로직 변경 | 중간 |
| `pipeline/nodes/recommend.py` | 필터 빌드 로직 변경 (4개 함수) | 높음 |
| `routers/recommend.py` | TODO → pgvector 연결 | 낮음 |
| `routers/products.py` | TODO → PostgreSQL 연결 | 낮음 |
| `Dockerfile` | 시스템 패키지 확인 | 낮음 |
| `pipeline/state.py` | 변경 없음 | — |
| `pipeline/nodes/intent.py` | 변경 없음 | — |
| `pipeline/nodes/clarify.py` | 변경 없음 | — |
| `pipeline/nodes/merge.py` | 변경 없음 | — |
| `pipeline/nodes/respond.py` | 변경 없음 | — |
| `pipeline/chatbot_graph.py` | 변경 없음 | — |
| `main.py` | 변경 없음 | — |

**변경 필요: 7개 파일 / 변경 없음: 8개 파일**

---

## 3. 파일별 상세 수정 사항

### 3.1`requirements.txt`

```diff
- qdrant-client==1.17.1
- fastembed==0.4.1
+ pgvector==0.3.6
+ asyncpg==0.30.0          # (선택) async 사용 시
```

> `sqlalchemy`, `psycopg2-binary`는 이미 존재. pgvector SQLAlchemy 통합 사용 가능.
> `fastembed`는 Dense 임베딩용이므로 유지하거나 sentence-transformers로 교체 가능.
> Sparse(BM25) 임베딩은 pgvector에서 불필요 — PostgreSQL `tsvector` 풀텍스트로 대체.

### 3.2`core/qdrant_setup.py` → `core/db_setup.py` (전체 재작성)

현재 역할: Qdrant 컬렉션 3개 (`products`, `domain_qna`, `breed_meta`) 생성

**변경 내용:**
- [ ] Qdrant 컬렉션 생성 → PostgreSQL 테이블 + pgvector 컬럼 DDL로 교체
- [ ] Dense 벡터: `vector(1024)` 컬럼 (HNSW 인덱스)
- [ ] Sparse 벡터: `tsvector` 컬럼 (GIN 인덱스) — BM25 대체
- [ ] 3개 테이블 DDL 정의: `products`, `domain_qna`, `breed_meta`
- [ ] `--recreate` 옵션 → `DROP TABLE IF EXISTS` + `CREATE TABLE`

**새 테이블 스키마 예시:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- products 테이블
CREATE TABLE products (
    id            SERIAL PRIMARY KEY,
    goods_id      TEXT,
    product_name  TEXT,
    brand_name    TEXT,
    price         INTEGER,
    discount_price INTEGER,
    thumbnail_url TEXT,
    product_url   TEXT,
    pet_type      TEXT,
    category      TEXT,
    subcategory   TEXT,
    sold_out      BOOLEAN DEFAULT FALSE,
    main_ingredients TEXT[],
    ingredient_text_ocr TEXT,
    popularity_score  FLOAT,
    sentiment_avg     FLOAT,
    repeat_rate       FLOAT,
    dense_embedding   vector(1024),
    text_search       tsvector
);

CREATE INDEX idx_products_dense ON products USING hnsw (dense_embedding vector_cosine_ops);
CREATE INDEX idx_products_tsv   ON products USING gin (text_search);

-- domain_qna 테이블
CREATE TABLE domain_qna (
    id             SERIAL PRIMARY KEY,
    question       TEXT,
    answer         TEXT,
    species        TEXT,
    category       TEXT,
    dense_embedding vector(1024),
    text_search     tsvector
);

CREATE INDEX idx_domain_qna_dense ON domain_qna USING hnsw (dense_embedding vector_cosine_ops);
CREATE INDEX idx_domain_qna_tsv   ON domain_qna USING gin (text_search);

-- breed_meta 테이블
CREATE TABLE breed_meta (
    id               SERIAL PRIMARY KEY,
    breed            TEXT,
    pet_type         TEXT,
    health_keywords  TEXT[],
    dense_embedding  vector(1024),
    text_search      tsvector
);

CREATE INDEX idx_breed_meta_dense ON breed_meta USING hnsw (dense_embedding vector_cosine_ops);
CREATE INDEX idx_breed_meta_tsv   ON breed_meta USING gin (text_search);
```

### 3.3`pipeline/utils.py` (전체 재작성)

현재 역할: Qdrant 클라이언트 + `hybrid_search()` + `embed()` + 헬퍼

| 현재 (Qdrant) | 변경 후 (pgvector) |
|---|---|
| `QdrantClient` 인스턴스 | SQLAlchemy `engine` / `Session` |
| `QDRANT_URL`, `QDRANT_HOST`, `QDRANT_PORT` 환경변수 | `DATABASE_URL` 환경변수 |
| `embed()` → Dense + Sparse 벡터 반환 | `embed()` → Dense 벡터만 반환 |
| `SparseTextEmbedding("Qdrant/bm25")` | 제거 — `tsvector` + `ts_rank` 사용 |
| `hybrid_search()` → Prefetch + RRF Fusion | `hybrid_search()` → SQL: cosine similarity + ts_rank → RRF 합산 |

**수정 체크리스트:**
- [ ] `qdrant_client` import 전체 제거
- [ ] `SparseVector`, `Prefetch`, `FusionQuery`, `Fusion` 제거
- [ ] SQLAlchemy engine + sessionmaker 설정 추가
- [ ] `embed()`: Sparse 반환 제거, Dense만 반환
- [ ] `hybrid_search()` 재구현:
  - Dense: `ORDER BY dense_embedding <=> :query_vector LIMIT N` (cosine distance)
  - Sparse: `ts_rank(text_search, plainto_tsquery('korean', :query)) DESC LIMIT N`
  - RRF 합산: `1/(k+rank_dense) + 1/(k+rank_sparse)` (k=60)
  - 필터: SQL WHERE 절로 변환
- [ ] 반환값을 기존 `points` 형태와 호환되는 객체로 래핑 (`.payload`, `.score`)

**hybrid_search SQL 예시:**

```sql
WITH dense_ranked AS (
    SELECT *, dense_embedding <=> :qvec AS dist,
           ROW_NUMBER() OVER (ORDER BY dense_embedding <=> :qvec) AS rank_d
    FROM {table}
    WHERE {filters}
    LIMIT :limit
),
sparse_ranked AS (
    SELECT *, ts_rank(text_search, plainto_tsquery(:query)) AS ts_score,
           ROW_NUMBER() OVER (ORDER BY ts_rank(text_search, plainto_tsquery(:query)) DESC) AS rank_s
    FROM {table}
    WHERE {filters}
    LIMIT :limit
)
SELECT *, (1.0/(60+rank_d) + 1.0/(60+rank_s)) AS rrf_score
FROM dense_ranked d
FULL OUTER JOIN sparse_ranked s USING (id)
ORDER BY rrf_score DESC
LIMIT :top_k;
```

### 3.4`pipeline/nodes/domain_qa.py`

| 라인 | 현재 | 변경 |
|------|------|------|
| 6 | `from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue` | 제거 |
| 7 | `from pipeline.utils import ... qdrant, hybrid_search ...` | `qdrant` 제거 |
| 36-42 | Qdrant `FieldCondition` + `Filter` 필터 빌드 | `dict` 기반 필터로 변환 |
| 44 | `hybrid_search("domain_qna", query, top_k=5, qdrant_filter=f)` | 시그니처 유지, 내부 구현만 변경 |

**수정 체크리스트:**
- [ ] `qdrant_client.models` import 제거
- [ ] `rag_node()`: 필터를 `dict`로 변환

```python
# Before (Qdrant)
must = []
if species:
    must.append(FieldCondition(key="species", match=MatchAny(any=[species, "both"])))
f = Filter(must=must) if must else None

# After (pgvector)
filters = {}
if species:
    filters["species"] = [species, "both"]  # IN 조건
if domain_intent in DOMAIN_INTENT_TO_CATEGORY:
    filters["category"] = DOMAIN_INTENT_TO_CATEGORY[domain_intent]
```

### 3.5`pipeline/nodes/recommend.py` (수정량 가장 많음)

4개 함수 모두 Qdrant 의존:

#### `profile_node()` (라인 13-39)
- [ ] 라인 6: `qdrant_client.models` import 제거
- [ ] 라인 24: `Filter(must=[FieldCondition(...)])` → `dict` 필터
- [ ] 라인 25: `hybrid_search("breed_meta", ...)` → 시그니처 유지

#### `query_node()` (라인 44-104)
- [ ] 라인 67-100: Qdrant `FieldCondition` 필터 빌드 전체를 `dict` 기반으로 변환
- [ ] `must` 리스트 → `filters dict`
- [ ] `must_not` 리스트 → `exclude dict`
- [ ] `_qdrant_filter_obj` 반환 제거

```python
# Before (Qdrant)
must = [FieldCondition(key="sold_out", match=MatchValue(value=False))]
if pet_type:
    must.append(FieldCondition(key="pet_type", match=MatchAny(any=[pet_type])))
if budget:
    must.append(FieldCondition(key="price", range=Range(lte=budget)))
qdrant_filter = Filter(must=must, must_not=must_not)

# After (pgvector)
filters = {"sold_out": False}
if pet_type:
    filters["pet_type"] = pet_type
if relaxation == 0:
    if category_hint:
        filters["category"] = category_hint
    if subcategory_hint:
        filters["subcategory"] = subcategory_hint
else:
    if category_hint:
        filters["category"] = category_hint

exclude = {}
if budget:
    filters["price_lte"] = budget
if allergies:
    exclude["main_ingredients"] = allergies
```

#### `search_node()` (라인 109-156)
- [ ] 라인 112-143: Qdrant 필터 rebuild 로직 전체를 `dict` 기반으로 변환
- [ ] `query_node`에서 `dict` 필터를 state에 저장하면, `search_node`에서 그대로 사용 가능 → **중복 필터 빌드 제거 가능**
- [ ] 라인 145: `hybrid_search("products", ...)` → 시그니처 유지

#### `rerank_node()` (라인 160-241)
- [ ] 변경 없음 — Qdrant 의존 없음 (state에서 `search_results` 받아서 처리)

### 3.6`routers/recommend.py` (TODO → 구현)
- [ ] `hybrid_search()` 호출로 직접 추천 API 구현 (기존 TODO)

### 3.7`routers/products.py` (TODO → 구현)
- [ ] SQLAlchemy로 products 테이블 조회 (기존 TODO)

### 3.8`Dockerfile`
- [ ] `libpq-dev` 이미 포함 — 추가 패키지 불필요
- [ ] Qdrant 관련 환경변수 제거, `DATABASE_URL` 추가

---

## 4. 환경변수 변경

| 제거 | 추가 |
|------|------|
| `QDRANT_HOST` | `DATABASE_URL` (예: `postgresql://user:pass@localhost:5432/tailtalk`) |
| `QDRANT_PORT` | — |
| `QDRANT_URL` | — |
| `QDRANT_API_KEY` | — |

---

## 5. 임베딩 전략 변경

| 항목 | Qdrant (현재) | pgvector (변경 후) |
|------|--------------|-------------------|
| Dense 벡터 | `intfloat/multilingual-e5-large` (1024d) | 동일 유지 |
| Dense 인덱스 | HNSW (Qdrant 내장) | HNSW (`vector_cosine_ops`) |
| Sparse 벡터 | `Qdrant/bm25` (fastembed) | **제거** → `tsvector` + `ts_rank` |
| Hybrid 합산 | Qdrant RRF Fusion (내장) | **SQL에서 RRF 직접 계산** |
| 필터링 | `FieldCondition` + `Filter` | SQL `WHERE` 절 |

---

## 6. 마이그레이션 순서

```
Phase 1: 인프라 준비
  ├─ [1] PostgreSQL에 pgvector 확장 설치
  ├─ [2] 테이블 DDL 실행 (core/db_setup.py)
  └─ [3] 환경변수 전환 (QDRANT_* → DATABASE_URL)

Phase 2: 핵심 모듈 교체
  ├─ [4] pipeline/utils.py 재작성 (hybrid_search SQL 구현)
  └─ [5] embed() 함수에서 Sparse 제거, Dense만 반환

Phase 3: 노드 필터 로직 변환
  ├─ [6] pipeline/nodes/domain_qa.py 필터 변환
  ├─ [7] pipeline/nodes/recommend.py 필터 변환 (4개 함수)
  └─ [8] search_node 중복 필터 빌드 제거 (query_node에서 dict 전달)

Phase 4: 데이터 적재 + 검증
  ├─ [9] 데이터 적재 스크립트 작성 (embed → INSERT)
  ├─ [10] HTTP 테스트 전체 경로 검증
  └─ [11] requirements.txt / Dockerfile 정리

Phase 5: 정리
  ├─ [12] core/qdrant_setup.py 삭제
  ├─ [13] qdrant-client, fastembed(선택) 제거
  └─ [14] routers/recommend.py, products.py TODO 구현
```

---

## 7. 리스크 및 주의사항

| 리스크 | 설명 | 대응 |
|--------|------|------|
| 검색 품질 저하 | Qdrant 내장 RRF vs SQL 수동 RRF 차이 | RRF k값 튜닝 (k=60 기본), A/B 비교 |
| BM25 → tsvector 차이 | fastembed BM25와 PostgreSQL tsvector 토큰화 차이 | 한국어 검색 시 `pg_bigm` 또는 mecab 고려 |
| 성능 | pgvector HNSW는 Qdrant 대비 대규모 데이터에서 느릴 수 있음 | 현재 데이터 규모에서는 문제 없을 것으로 예상 |
| 반환값 호환 | `.payload`, `.score` 인터페이스 | `hybrid_search()`에서 호환 래퍼 제공 |
| 한국어 풀텍스트 | PostgreSQL 기본 파서는 한국어 토큰화 미지원 | `textsearch_ko` 또는 `pg_bigm` 확장 필요 |
