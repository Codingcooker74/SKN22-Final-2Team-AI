# TailTalk AI 구현 계획서

> 작성일: 2026-03-24
> 시작점: petchat (기본 RAG 챗봇)
> 목표: SKN22-Final-2Team-AI (고급 반려동물 쇼핑 챗봇)

---

## 1. 두 프로젝트 비교 (AS-IS → TO-BE)

| 항목 | petchat (시작점) | TailTalk AI (목표) |
|------|---------------------------|-------------------|
| **그래프 노드** | 4개 (intent, pet_qa, product_search, rag_response) | 10개 (+clarify, general, rag, profile, query, search, rerank, merge, respond) |
| **의도 분류** | 2가지 (pet_qa / product_recommend) | 4가지 (recommend / domain_qa / unclear / 복합) |
| **검색 방식** | SQL ILIKE + pgvector cosine → RRF | Dense + Sparse(BM25) → RRF + 5-factor 재랭킹 |
| **임베딩 모델** | OpenAI text-embedding-3-small (1536d) | multilingual-e5-large (1024d, 로컬) |
| **벡터 DB** | PostgreSQL + pgvector | Qdrant (→ pgvector로 전환 예정) |
| **데이터** | Product 50건, Review 50건 | products + domain_qna + breed_meta 3개 컬렉션 |
| **필터링** | 없음 (검색 결과 그대로) | pet_type, category, subcategory, 예산, 알레르기 |
| **재랭킹** | RRF만 | RRF + 인기도 + 감정분석 + 재구매율 + ABSA |
| **응답** | 동기 JSON | SSE 스트리밍 (단어 단위) |
| **대화 관리** | 단일 턴 (상태 없음) | MemorySaver 멀티턴 |
| **펫 프로필** | 없음 | species, breed, age, weight, health_concerns, allergies |
| **도메인 QA** | LLM 직답 (RAG 없음) | domain_qna 컬렉션 RAG 검색 |
| **테스트** | 46건 | 없음 (→ 18건 HTTP 테스트 작성 완료) |

---

## 2. 구현 단계 (7단계)

학습곡선을 고려하여, 이미 동작하는 petchat 위에 점진적으로 기능을 확장하는 방식으로 설계했다.
각 단계가 완료될 때마다 **독립적으로 동작하는 서비스** 상태를 유지한다.

```
Phase 1: 데이터 확장 (DB 스키마)
    │  ← 기존 검색/그래프 그대로 동작
    ▼
Phase 2: 검색 고도화 (tsvector + 필터링)
    │  ← 검색 품질 향상 체감
    ▼
Phase 3: 의도 분류 고도화
    │  ← 3가지 의도 + clarify 분기
    ▼
Phase 4: 도메인 QA RAG 구축
    │  ← pet_qa가 RAG 기반으로 전환
    ▼
Phase 5: 추천 파이프라인 고도화
    │  ← 재랭킹 + 필터 완화 루프
    ▼
Phase 6: 사용자 경험 개선
    │  ← SSE 스트리밍 + 멀티턴 + 프로필
    ▼
Phase 7: RAG 성능 최적화
       ← 임베딩 모델 교체 + 고급 기법
```

---

### Phase 1: 데이터 확장 (DB 스키마)

**목표**: TailTalk에 필요한 3개 테이블 구조를 pgvector 기반으로 구축

**학습 포인트**: SQLModel 관계, pgvector 인덱스, 시드 스크립트 확장

| 작업 | 상세 | 난이도 |
|------|------|--------|
| Product 테이블 확장 | `subcategory`, `sold_out`, `main_ingredients[]`, `ingredient_text_ocr`, `popularity_score`, `sentiment_avg`, `repeat_rate`, `thumbnail_url`, `product_url`, `goods_id` 컬럼 추가 | 낮음 |
| DomainQnA 테이블 신규 | `question`, `answer`, `species`, `category`, `embedding vector(1024)`, `text_search tsvector` | 낮음 |
| BreedMeta 테이블 신규 | `breed`, `pet_type`, `health_keywords[]`, `embedding vector(1024)`, `text_search tsvector` | 낮음 |
| 카테고리 체계 적용 | `pipeline/data/category.json` (강아지 5 + 고양이 5 대분류) | 낮음 |
| HNSW + GIN 인덱스 생성 | Dense 검색 + 풀텍스트 검색 인덱스 | 낮음 |
| 시드 스크립트 확장 | 3개 테이블 샘플 데이터 + 임베딩 생성 | 중간 |

**완료 기준**: 기존 테스트 46건 + 새 테이블 CRUD 테스트 통과

---

### Phase 2: 검색 고도화 (tsvector Hybrid + 필터링)

**목표**: SQL ILIKE → tsvector + pgvector Hybrid Search로 전환, 메타데이터 필터 추가

**학습 포인트**: PostgreSQL tsvector/ts_rank, RRF SQL CTE, WHERE 절 필터

| 작업 | 상세 | 난이도 |
|------|------|--------|
| tsvector 컬럼 + GIN 인덱스 | Product, DomainQnA, BreedMeta에 tsvector 자동 생성 트리거 | 중간 |
| `sql_search.py` 리팩토링 | ILIKE → `ts_rank(text_search, plainto_tsquery(:query))` | 중간 |
| `hybrid.py` SQL CTE 방식 전환 | 단일 SQL로 Dense + Sparse 결과를 RRF 합산 | 중간 |
| 메타데이터 필터 추가 | `pet_type`, `category`, `subcategory`, `sold_out`, `price` 필터 | 중간 |
| 알레르기 제외 필터 | `NOT (main_ingredients && ARRAY[:allergies])` + OCR post-filter | 중간 |
| `hybrid_search()` 시그니처 통합 | `hybrid_search(table, query, top_k, filters, exclude)` | 중간 |

**완료 기준**: 기존 검색 테스트 통과 + 필터링 테스트 추가

---

### Phase 3: 의도 분류 고도화

**목표**: 2가지 → 4가지 의도 분류 + clarify 분기 + 복합 의도 처리

**학습 포인트**: LLM JSON 구조화 출력, LangGraph 조건부 엣지, Send API (병렬 fan-out)

| 작업 | 상세 | 난이도 |
|------|------|--------|
| ChatState 확장 | `intents[]`, `domain_intent`, `filters`, `clarification_count`, `detected_aspect`, `budget` 추가 | 낮음 |
| intent_classifier 고도화 | JSON 응답 형식으로 전환 (intents, domain_intent, pet_type, category, subcategory, budget 추출) | 중간 |
| clarify_node 추가 | pet_type/category 누락 시 재질문 생성, clarification_count 관리 | 낮음 |
| 그래프 분기 확장 | `route_intent()` — unclear/domain_qa/recommend/복합(Send) | 높음 |
| Send API 병렬 fan-out | `domain_qa + recommend` 복합 의도 시 두 서브플로우 병렬 실행 | 높음 |

**완료 기준**: 4가지 의도 분류 테스트 통과, clarify 재질문 동작

---

### Phase 4: 도메인 QA RAG 구축

**목표**: pet_qa LLM 직답 → domain_qna 테이블 RAG 기반 응답으로 전환

**학습 포인트**: RAG 파이프라인 설계, 쿼리 정제, 컨텍스트 조합 응답 생성

| 작업 | 상세 | 난이도 |
|------|------|--------|
| general_node | 펫 프로필 반영한 검색 쿼리 정제 (LLM) | 중간 |
| rag_node | domain_qna 테이블 Hybrid Search (species + category 필터) | 중간 |
| respond_node 리팩토링 | domain_contexts + product_cards 통합 응답 생성 | 중간 |
| merge_node | domain_qa + recommend 결과 병합 로직 | 낮음 |
| DomainQnA 데이터 수집/적재 | 반려동물 건강/영양/행동/사육/여행 QnA 데이터 | 높음 |

**완료 기준**: 도메인 질문에 RAG 기반 응답 생성, 컨텍스트 활용도 테스트

---

### Phase 5: 추천 파이프라인 고도화

**목표**: 단순 검색 → 프로필 보완 + 다단계 검색 + 5-factor 재랭킹 + 필터 완화 루프

**학습 포인트**: 다단계 파이프라인, 가중치 스코어링, 조건부 루프

| 작업 | 상세 | 난이도 |
|------|------|--------|
| profile_node | breed_meta 조회로 품종별 health_keywords 보완 | 중간 |
| query_node | LLM 쿼리 생성 + 필터 dict 빌드 (pet_type, category, subcategory, 예산, 알레르기) | 중간 |
| search_node 분리 | product_search_node → query_node + search_node 분리 | 낮음 |
| rerank_node | 5-factor 재랭킹 (RRF α=0.50, 인기도 β=0.25, 감정 γ=0.15, 재구매 δ=0.10, ABSA ε=0.10) | 높음 |
| 필터 완화 루프 | `route_rerank()` — 결과 < 3개 시 subcategory 제거 후 QUERY 재시도 | 중간 |
| product_cards 생성 | merge_node에서 상품 카드 데이터 추출 (thumbnail_url, product_url 등) | 낮음 |

**완료 기준**: 재랭킹 품질 테스트, 필터 완화 루프 동작 확인

---

### Phase 6: 사용자 경험 개선

**목표**: SSE 스트리밍, 멀티턴 대화, 펫 프로필 입력

**학습 포인트**: Server-Sent Events, LangGraph MemorySaver, 대화 상태 관리

| 작업 | 상세 | 난이도 |
|------|------|--------|
| SSE 스트리밍 응답 | `StreamingResponse` + 단어 단위 토큰 전송 + 상품 카드 이벤트 | 중간 |
| ChatRequest 확장 | `thread_id`, `pet_profile`, `health_concerns`, `allergies`, `food_preferences` | 낮음 |
| MemorySaver 통합 | `build_graph(checkpointer=MemorySaver())` — thread_id 기반 대화 히스토리 유지 | 중간 |
| 멀티턴 clarify 플로우 | CLARIFY → END → 같은 thread_id로 재호출 → INTENT 재분류 | 중간 |
| CORS 설정 | 프론트엔드 연동 대비 CORS 미들웨어 | 낮음 |

**완료 기준**: SSE 클라이언트로 스트리밍 확인, 멀티턴 재질문 동작

---

### Phase 7: RAG 성능 최적화 (추가 발전)

**목표**: 검색/생성 품질을 한 단계 더 끌어올리는 고급 기법 적용

이 단계는 서비스가 동작하는 상태에서 A/B 테스트로 점진적으로 적용한다.

| 작업 | 상세 | 난이도 |
|------|------|--------|
| 임베딩 모델 교체 | OpenAI → `intfloat/multilingual-e5-large` (로컬, 1024d) via fastembed | 중간 |
| tsvector 한국어 최적화 | `pg_bigm` 또는 `textsearch_ko` 확장 적용 | 중간 |
| HNSW 파라미터 튜닝 | `m=16, ef_construct=100` → Recall vs Latency 최적점 탐색 | 중간 |
| 리뷰 기반 감정분석 스코어 | Review 테이블의 rating/content → sentiment_avg, repeat_rate 산출 | 높음 |
| ABSA (Aspect-Based Sentiment) | 속성별 감정분석 → detected_aspect 반영 | 높음 |

**완료 기준**: 검색 품질 메트릭 비교 (Recall@K, MRR, NDCG)

---

## 3. 전체 구현 기능 목록

### 이미 구현된 기능 (petchat)

- [x] FastAPI 서버 (POST /chat, GET /health)
- [x] SQLModel ORM (Product, Review)
- [x] pgvector 코사인 유사도 검색
- [x] SQL ILIKE 키워드 검색
- [x] RRF 하이브리드 검색
- [x] LangGraph 2-way 의도 분류 (pet_qa / product_recommend)
- [x] RAG 응답 생성 (검색 결과 기반 LLM)
- [x] 테스트 46건 (모델/검색/그래프/API/RAG)
- [x] Docker Compose (pgvector/pg16)
- [x] 샘플 데이터 시드 (50건 + 임베딩)

### 추가 구현 필요 기능

#### 검색/데이터 레이어
- [ ] DomainQnA 테이블 + RAG 검색
- [ ] BreedMeta 테이블 + 품종별 건강 메타데이터
- [ ] tsvector 풀텍스트 검색 (ILIKE 대체)
- [ ] 카테고리/서브카테고리 체계 (강아지 5 + 고양이 5)
- [ ] 메타데이터 필터링 (pet_type, category, subcategory, price, sold_out)
- [ ] 알레르기 제외 필터 (배열 연산 + OCR post-filter)
- [ ] 예산 범위 필터 (price <= budget)

#### LangGraph 파이프라인
- [ ] 4-way 의도 분류 (recommend / domain_qa / unclear / 복합)
- [ ] clarify_node (재질문 생성)
- [ ] Send API 병렬 fan-out (domain_qa + recommend 동시 처리)
- [ ] general_node (펫 프로필 기반 쿼리 정제)
- [ ] rag_node (domain_qna Hybrid Search)
- [ ] profile_node (breed_meta 건강 키워드 보완)
- [ ] query_node (LLM 검색어 생성 + 필터 빌드)
- [ ] rerank_node (5-factor 가중치 재랭킹)
- [ ] 필터 완화 루프 (route_rerank → query 재시도)
- [ ] merge_node (domain + recommend 결과 병합)
- [ ] product_cards 생성 (상품 카드 데이터)

#### 사용자 경험
- [ ] SSE 스트리밍 응답 (단어 단위 + 상품 카드)
- [ ] 펫 프로필 입력 (species, breed, age, weight)
- [ ] 건강관심사/알레르기/식품선호 입력
- [ ] MemorySaver 멀티턴 대화
- [ ] thread_id 기반 세션 관리

#### 인프라
- [ ] Docker Compose 업데이트 (앱 컨테이너 추가)
- [ ] 환경변수 통합 (.env)
- [ ] 구조화된 로깅 (print → logging)
- [ ] 글로벌 에러 핸들링

---

## 4. RAG 검색/생성 성능 개선 방법론

현재 구현 이후 추가로 적용할 수 있는 고급 기법들을 단계별로 정리한다.

### Level 1: 검색 품질 개선 (Retrieval)

| 기법 | 설명 | 효과 |
|------|------|------|
| **Query Expansion** | 사용자 쿼리를 LLM으로 확장 (동의어, 관련어 추가) | Recall 향상 |
| **HyDE (Hypothetical Document Embeddings)** | 쿼리 대신 "가상 답변"을 생성하여 임베딩 → 검색 | 의미 매칭 정확도 향상 |
| **Multi-Query Retrieval** | 하나의 질문을 여러 관점에서 재작성 → 각각 검색 → 합산 | 다양한 관련 문서 발굴 |
| **Parent-Child Chunking** | 작은 청크로 검색, 큰 청크로 컨텍스트 제공 | 정밀 검색 + 풍부한 컨텍스트 |
| **Cross-Encoder Reranking** | 초기 검색 후 Cross-Encoder 모델로 정밀 재랭킹 | 정밀도(Precision) 대폭 향상 |
| **Metadata-Augmented Embedding** | 임베딩 생성 시 메타데이터(카테고리, 태그) 포함 | 필터링 없이도 관련도 향상 |

### Level 2: 생성 품질 개선 (Generation)

| 기법 | 설명 | 효과 |
|------|------|------|
| **Chain-of-Thought Prompting** | 추천 이유를 단계적으로 추론하도록 유도 | 응답 논리성 향상 |
| **Few-Shot Examples** | 시스템 프롬프트에 좋은 응답 예시 포함 | 응답 형식/품질 일관성 |
| **Self-RAG (Self-Reflective RAG)** | 검색 결과의 관련성을 LLM이 자체 평가 후 필터링 | 환각(Hallucination) 감소 |
| **Corrective RAG (CRAG)** | 검색 결과가 부족하면 웹 검색으로 보완 | 커버리지 향상 |
| **Context Compression** | 긴 컨텍스트를 LLM으로 요약 후 전달 | 토큰 절약 + 노이즈 감소 |
| **Structured Output (JSON Mode)** | 상품 추천을 구조화된 JSON으로 생성 | 파싱 안정성 + UI 연동 용이 |

### Level 3: 시스템 레벨 개선

| 기법 | 설명 | 효과 |
|------|------|------|
| **Embedding Cache** | 자주 쿼리되는 임베딩 결과 캐싱 (Redis) | 응답 속도 향상 |
| **Async Pipeline** | LangGraph 노드를 async로 전환, DB I/O 비동기화 | 처리량(throughput) 향상 |
| **Streaming RAG** | 검색과 생성을 파이프라인으로 연결하여 동시 진행 | 체감 응답 시간 단축 |
| **Evaluation Pipeline** | RAGAS / LangSmith로 자동 품질 평가 | 지속적 품질 모니터링 |
| **A/B Testing Framework** | 검색/생성 전략 변경 시 비교 실험 | 데이터 기반 의사결정 |
| **Fine-tuned Embedding** | 도메인 데이터로 임베딩 모델 파인튜닝 | 도메인 특화 검색 품질 |

### Level 4: 도메인 특화 개선

| 기법 | 설명 | 효과 |
|------|------|------|
| **Review-Augmented Retrieval** | 상품 검색 시 리뷰 임베딩도 함께 검색 → 실사용 관점 매칭 | 사용자 관점 추천 |
| **Collaborative Filtering 하이브리드** | 벡터 검색 + 유사 사용자 구매 패턴 결합 | 개인화 추천 |
| **Knowledge Graph RAG** | 품종→건강이슈→성분→상품 관계 그래프 구축 | 연관 추천 정확도 |
| **Temporal Relevance** | 계절/시기별 상품 가중치 조정 (여름→쿨링, 겨울→패딩) | 시의적절한 추천 |
| **Feedback Loop** | 사용자 클릭/구매 데이터로 재랭킹 가중치 자동 조정 | 지속적 품질 향상 |

---

## 5. 단계별 학습 로드맵

```
[Phase 1] SQLModel / pgvector 확장
  └─ 기존 지식으로 충분. 테이블 추가만.

[Phase 2] tsvector + SQL CTE
  └─ 새로운 학습: PostgreSQL 풀텍스트 검색, CTE 문법
  └─ 핵심: hybrid_search()를 단일 SQL로 재작성

[Phase 3] LangGraph 고급 패턴
  └─ 새로운 학습: JSON 구조화 출력, Send API, 조건부 엣지
  └─ 핵심: 4-way 분기 + 병렬 fan-out

[Phase 4] RAG 파이프라인 설계
  └─ 새로운 학습: 쿼리 정제, 컨텍스트 조합, 프롬프트 엔지니어링
  └─ 핵심: domain_qna RAG + 프로필 기반 검색

[Phase 5] 재랭킹 알고리즘
  └─ 새로운 학습: 가중치 스코어링, Fallback 전략, 조건부 루프
  └─ 핵심: 5-factor rerank + 필터 완화 루프

[Phase 6] 실시간 스트리밍
  └─ 새로운 학습: SSE, MemorySaver, 비동기 패턴
  └─ 핵심: 사용자 경험 완성

[Phase 7] 성능 최적화 (심화)
  └─ 새로운 학습: HyDE, Cross-Encoder, RAGAS 평가
  └─ 핵심: 측정 → 개선 → 측정 반복 사이클
```

### 난이도 곡선

```
난이도
  ▲
  │            ┌───┐
  │        ┌───┤ 5 │
  │    ┌───┤ 4 ├───┘
  │    │ 3 ├───┘       ┌───┐
  │ ┌──┤   │       ┌───┤ 7 │
  │ │2 ├───┘       │ 6 ├───┘
  │ ├──┘           ├───┘
  │ │1             │
  └─┴──────────────┴──────────► Phase
```

Phase 1-2는 기존 지식 기반으로 수월하고, Phase 3-5에서 LangGraph 고급 패턴과 재랭킹에서 학습량이 증가한다. Phase 6은 구현 난이도보다 통합 테스트에 시간이 들고, Phase 7은 측정 기반으로 점진적 적용이므로 부담이 분산된다.
