# RAG 검색/생성 분리 평가 계획

> 작성일: 2026-03-24
> 대상: petchat (기본 RAG 챗봇) → TailTalk AI 진화 과정에서의 품질 측정
> 목적: 검색(Retrieval)과 생성(Generation)을 분리 평가하여 병목을 정확히 진단

---

## 1. 왜 분리 평가인가

현재 `test_rag.py`는 파이프라인 전체를 한번에 테스트한다.
이 방식으로는 "검색이 나쁜 건지, 생성이 나쁜 건지" 구분할 수 없다.

```
사용자 쿼리
    │
    ▼
┌─────────────┐     검색 평가 (Retrieval Evaluation)
│  Retrieval  │──── 정답 문서를 잘 찾았는가?
│  (검색)      │     → 정답 셋 vs 검색 결과 비교
└──────┬──────┘
       │ search_results
       ▼
┌─────────────┐     생성 평가 (Generation Evaluation)
│ Generation  │──── 검색 결과를 잘 활용했는가?
│  (생성)      │     → 컨텍스트 vs 응답 비교
└─────────────┘
```

| 문제 상황 | 원인 진단 | 개선 방향 |
|-----------|----------|----------|
| 엉뚱한 상품 추천 | Recall 낮음 → **검색 문제** | HyDE, Query Expansion, 임베딩 교체 |
| 맞는 상품 찾았는데 응답이 이상 | Faithfulness 낮음 → **생성 문제** | 프롬프트 개선, Few-Shot, Self-RAG |
| 가격/성분 틀림 | Groundedness 낮음 → **생성 문제** | Structured Output, 사실 검증 단계 추가 |
| 고양이 질문에 강아지 상품 | Precision 낮음 → **검색 문제** | 메타데이터 필터링 강화 |

---

## 2. 검색 평가 (Retrieval Evaluation)

### 2.1 핵심 질문

> "쿼리에 맞는 상품을 정확히 찾아왔는가?"

### 2.2 평가 데이터셋 형식

쿼리마다 정답 상품 ID를 사전에 정의한다. petchat의 시드 데이터 50건 기준.

```python
RETRIEVAL_EVAL_SET = [
    {
        "query": "고양이 사료 추천해주세요",
        "relevant_ids": [4, 5, 7, 9],       # 고양이 사료 상품 ID
        "must_include_top3": [4, 5],          # 상위 3개에 반드시 포함
    },
    {
        "query": "피부와 털이 좋아지는 영양제",
        "relevant_ids": [43, 6],              # 오메가3 피모 영양제, 에코 오가닉
        "relevant_keywords": ["피모", "오메가", "피부"],
    },
    {
        "query": "강아지 장난감 추천",
        "relevant_ids": [29, 30, 35, 37],     # 강아지 장난감
        "exclude_pet_type": "고양이",          # 고양이 전용 제외
    },
    {
        "query": "저렴한 고양이 간식",
        "relevant_ids": [23, 25, 28],         # 고양이 간식 (가격 낮은 순)
        "price_upper_bound": 15000,
    },
    # ... 총 20~30건 구축
]
```

### 2.3 데이터셋 구축 가이드

| 항목 | 권장 |
|------|------|
| 총 건수 | 20~30건 |
| 카테고리 분포 | 사료/간식/장난감/건강/위생/의류/용품 각 3~4건 |
| 쿼리 유형 분포 | 정확 매칭 5건, 의미 매칭 5건, 교차 카테고리 5건, 가격 관련 3건, 모호한 쿼리 3건, 엣지 케이스 3건 |
| 정답 기준 | 2명 이상이 독립적으로 정답 라벨링 → 합의 |

### 2.4 메트릭

| 메트릭 | 측정 대상 | 계산법 | 기준선 목표 |
|--------|----------|--------|-----------|
| **Recall@K** | K개 결과 중 정답 포함 비율 | `len(retrieved ∩ relevant) / len(relevant)` | >= 0.7 |
| **Precision@K** | K개 결과 중 정답 비율 | `len(retrieved ∩ relevant) / K` | >= 0.5 |
| **MRR** | 첫 정답의 순위 역수 평균 | `mean(1 / rank_of_first_relevant)` | >= 0.5 |
| **NDCG@K** | 순서 고려한 랭킹 품질 | DCG / IDCG | >= 0.6 |
| **Hit Rate** | 정답 1개 이상 포함 비율 | `1 if any relevant in retrieved else 0` | >= 0.9 |

### 2.5 비교 실험 매트릭스

같은 평가 데이터셋으로 검색 방식별 성능을 비교한다.

```
┌─────────────────┬────────────┬──────────┬──────────┬──────────┐
│ Method          │ Recall@5   │ MRR      │ Prec@5   │ Hit Rate │
├─────────────────┼────────────┼──────────┼──────────┼──────────┤
│ SQL (ILIKE)     │            │          │          │          │
│ Vector (cosine) │            │          │          │          │
│ Hybrid (RRF)    │            │          │          │          │
│ + tsvector      │            │          │          │          │
│ + 필터링        │            │          │          │          │
│ + 재랭킹        │            │          │          │          │
└─────────────────┴────────────┴──────────┴──────────┴──────────┘
```

각 Phase에서 검색 방식을 변경할 때마다 이 표를 채워서 개선 효과를 수치로 확인한다.

### 2.6 테스트 코드 구조

```python
# tests/eval_retrieval.py
"""검색 품질만 독립 평가. LLM 생성 호출 없음 → 빠르고 저렴."""

class TestRetrievalMetrics:

    def test_recall_at_5(self):
        """전체 평가셋의 평균 Recall@5 >= 0.7"""

    def test_mrr(self):
        """전체 평가셋의 MRR >= 0.5"""

    def test_category_precision(self):
        """쿼리 카테고리와 검색 결과 카테고리 일치율 >= 0.6"""

    def test_pet_type_precision(self):
        """pet_type 일치율 >= 0.8 (강아지 질문 → 고양이 전용 과반 미만)"""

    def test_hit_rate(self):
        """정답 1개 이상 포함 비율 >= 0.9"""


class TestRetrievalComparison:

    def test_hybrid_beats_sql_only(self):
        """Hybrid RRF의 MRR > SQL-only의 MRR"""

    def test_hybrid_beats_vector_only(self):
        """Hybrid RRF의 MRR > Vector-only의 MRR"""

    def test_semantic_query_vector_advantage(self):
        """의미 검색 쿼리에서 Vector가 SQL보다 Recall 높음"""

    def test_keyword_query_sql_advantage(self):
        """키워드 검색 쿼리에서 SQL이 Vector보다 Precision 높음"""
```

**핵심**: `product_search_node`만 독립 호출하므로 LLM 생성 비용이 들지 않고, 빠르게 반복 실험 가능.

---

## 3. 생성 평가 (Generation Evaluation)

### 3.1 핵심 질문

> "검색 결과를 기반으로 정확하고 유용한 응답을 만들었는가?"

### 3.2 평가 4축

| 축 | 질문 | 자동 평가 | 측정법 |
|---|---|---|---|
| **Faithfulness** (충실도) | 검색 결과에 있는 정보만 사용했는가? | O | 환각 탐지 (상품명/가격 매칭) |
| **Relevance** (관련성) | 사용자 질문에 적절한 응답인가? | △ | LLM-as-Judge 또는 키워드 매칭 |
| **Completeness** (완전성) | 관련 상품을 빠뜨리지 않았는가? | O | 상위 상품 언급 비율 |
| **Groundedness** (근거성) | 가격, 성분 등 사실이 정확한가? | O | 정규식으로 수치 비교 |

### 3.3 고정 입력 방식

검색 품질 변동을 제거하고 **생성만 독립 평가**하기 위해, 고정된 search_results를 직접 주입한다.

```python
FIXED_SEARCH_RESULTS = [
    {
        "id": 4,
        "name": "로얄캐닌 인도어 캣",
        "category": "사료",
        "pet_type": "고양이",
        "price": 42000,
        "description": "실내 고양이 전용 사료. 헤어볼 배출과 체중 관리에 효과적입니다.",
    },
    {
        "id": 5,
        "name": "오리젠 캣 앤 키튼",
        "category": "사료",
        "pet_type": "고양이",
        "price": 82000,
        "description": "고단백 그레인프리 고양이 사료. 신선한 생선과 가금류 사용.",
    },
    # ...
]

# 검색 결과를 고정하고 생성만 평가
fixed_state = {
    "query": "고양이 사료 추천",
    "intent": "product_recommend",
    "search_results": FIXED_SEARCH_RESULTS,
}
result = rag_response_node(fixed_state)
# → 생성 품질만 측정
```

### 3.4 테스트 코드 구조

```python
# tests/eval_generation.py
"""생성 품질만 독립 평가. 검색 결과는 고정 입력."""

class TestFaithfulness:
    """컨텍스트에 없는 정보를 생성하지 않는지 (환각 방지)."""

    def test_no_hallucinated_products(self):
        """search_results에 없는 상품명이 응답에 등장하지 않는가"""

    def test_price_accuracy(self):
        """응답에 포함된 가격이 search_results의 price와 일치하는가"""

    def test_no_fabricated_features(self):
        """description에 없는 기능/성분을 언급하지 않는가"""


class TestRelevance:
    """응답이 질문 의도에 부합하는지."""

    def test_recommendation_has_reason(self):
        """추천 시 이유가 포함되는가 (100자 이상)"""

    def test_answer_addresses_query(self):
        """질문의 핵심 키워드가 응답에 반영되는가"""

    def test_korean_response(self):
        """영어 질문에도 한국어로 응답하는가"""


class TestCompleteness:
    """검색된 관련 상품을 적절히 활용했는지."""

    def test_top_result_mentioned(self):
        """검색 1위 상품이 응답에 포함되는가"""

    def test_multiple_options_provided(self):
        """2개 이상 상품을 비교/추천하는가"""

    def test_price_info_included(self):
        """가격 정보가 응답에 포함되는가"""


class TestGroundedness:
    """사실 관계가 정확한지."""

    def test_category_consistency(self):
        """'사료 추천' 질문에 사료가 아닌 장난감을 추천하지 않는가"""

    def test_pet_type_consistency(self):
        """'고양이' 질문에 강아지 전용 상품을 추천하지 않는가"""

    def test_description_accuracy(self):
        """상품 설명이 원본 description과 일치하는가"""
```

### 3.5 LLM-as-Judge (선택적 심화)

자동 평가로 측정하기 어려운 "응답의 자연스러움", "추천 논리성"은 LLM을 평가자로 사용할 수 있다.

```python
JUDGE_PROMPT = """
당신은 RAG 챗봇 응답 품질 평가자입니다.
아래 기준으로 1~5점을 매기세요.

[평가 기준]
- Faithfulness (1~5): 검색 결과에 기반한 정보만 사용했는가?
- Relevance (1~5): 사용자 질문에 적절히 답변했는가?
- Helpfulness (1~5): 실제로 구매 결정에 도움이 되는가?

[사용자 질문]
{query}

[검색 결과]
{search_results}

[챗봇 응답]
{response}

JSON으로만 답하세요:
{{"faithfulness": N, "relevance": N, "helpfulness": N, "reason": "..."}}
"""
```

비용이 들므로 전체 평가셋이 아닌 **샘플 10건** 정도에만 적용 권장.

---

## 4. 평가 실행 계획

### Step 1: 평가 데이터셋 구축

| 작업 | 상세 |
|------|------|
| 검색 평가셋 | 쿼리 20~30건 + 정답 상품 ID 매핑 |
| 생성 평가 고정입력 | 대표 시나리오 5~10건의 고정 search_results |
| 카테고리 분포 | 7개 카테고리 각 3~4건 균등 배분 |
| 쿼리 유형 분포 | 정확 매칭, 의미 매칭, 교차 카테고리, 가격, 모호, 엣지 케이스 |

### Step 2: eval_retrieval.py 작성

| 작업 | 상세 |
|------|------|
| 메트릭 함수 | `recall_at_k()`, `precision_at_k()`, `mrr()`, `ndcg_at_k()`, `hit_rate()` |
| 검색 방식별 비교 | SQL-only / Vector-only / Hybrid 3가지 |
| 기준선 기록 | 현재 petchat의 수치를 baseline으로 저장 |
| 실행 비용 | LLM 호출 없음 (임베딩 API만) → 빠르고 저렴 |

### Step 3: eval_generation.py 작성

| 작업 | 상세 |
|------|------|
| Faithfulness 테스트 | 환각 탐지, 가격 정확성, 사실 검증 |
| Relevance 테스트 | 추천 이유 포함, 질문 키워드 반영 |
| Completeness 테스트 | 상위 상품 언급, 복수 추천, 가격 포함 |
| Groundedness 테스트 | 카테고리/pet_type 일관성 |
| 실행 비용 | 고정 입력 사용 시 건당 LLM 1회 호출 |

### Step 4: CI/리포트 통합 (선택)

| 작업 | 상세 |
|------|------|
| pytest 플러그인 | `--eval` 플래그로 평가 테스트만 실행 |
| 결과 테이블 출력 | 메트릭을 markdown 테이블로 출력 |
| 기준선 비교 | 이전 수치 대비 개선/퇴보 표시 |
| Phase별 추적 | 구현 단계마다 수치 기록하여 진화 과정 추적 |

---

## 5. Phase별 평가 적용 시점

| Phase | 검색 평가 | 생성 평가 | 측정 포인트 |
|-------|----------|----------|------------|
| Phase 1 (데이터 확장) | baseline 측정 | baseline 측정 | 현재 petchat 수치 기록 |
| Phase 2 (tsvector + 필터) | **핵심** | - | ILIKE → tsvector 전환 효과 측정 |
| Phase 3 (의도 분류 고도화) | - | 분류 정확도 | intent 분류 Accuracy 측정 |
| Phase 4 (도메인 QA RAG) | domain_qna 검색 품질 | RAG 충실도 | 새 RAG 파이프라인 품질 |
| Phase 5 (재랭킹) | **핵심** | - | 5-factor rerank 전후 비교 |
| Phase 6 (UX 개선) | - | 스트리밍 응답 품질 | 멀티턴 컨텍스트 활용도 |
| Phase 7 (최적화) | **핵심** | **핵심** | 임베딩 교체, HyDE 등 A/B 비교 |

---

## 6. 결과 해석 가이드

### Recall은 높은데 Faithfulness가 낮을 때

→ 검색은 잘 되지만 LLM이 검색 결과를 제대로 활용하지 못함
→ **프롬프트 개선**, Few-Shot 예시 추가, Context Compression

### Recall이 낮을 때

→ 검색 자체가 관련 문서를 못 찾음
→ **검색 개선** 우선: HyDE, Query Expansion, 임베딩 모델 교체, 필터링 추가

### Precision은 높은데 Recall이 낮을 때

→ 찾은 건 정확한데 놓치는 게 많음
→ **top_k 증가**, 필터 완화, Multi-Query Retrieval

### Hit Rate가 낮을 때

→ 정답을 아예 못 찾는 쿼리가 존재
→ 해당 쿼리 분석 → 임베딩 품질 문제 or 데이터 커버리지 부족

### Groundedness가 낮을 때

→ 가격, 성분 등 사실 정보 오류
→ **Structured Output** (JSON 응답), 사실 검증 후처리 단계 추가
