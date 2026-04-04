import sys
from pathlib import Path

# 프로젝트 루트 및 FastAPI 경로 추가
# 스크립트 위치: services/fastapi/tests/test_scoring.py
# 필요한 경로: services/fastapi/
FASTAPI_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(FASTAPI_DIR))

import os
os.environ["POSTGRES_HOST"] = "localhost"

from final_ai.domain.recommendation.rerank_service import (
    ALPHA as _ALPHA,
    BETA as _BETA,
    DELTA as _DELTA,
    EPSILON as _EPSILON,
    GAMMA as _GAMMA,
    _normalize,
)
from final_ai.graph.nodes import search_node
import ast

def print_scoring_details(state):
    """
    rerank_node의 로직을 시뮬레이션하여 상세 점수 내역을 출력합니다.
    """
    candidates = state.get("search_results") or []
    if not candidates:
        print("후보 상품이 없습니다.")
        return

    rrf_scores  = [float(c.get("_score", 0.0)) for c in candidates]
    pop_scores  = [c.get("popularity_score")    for c in candidates]
    sent_scores = [c.get("sentiment_avg")       for c in candidates]
    rep_scores  = [c.get("repeat_rate")         for c in candidates]

    norm_rrf = _normalize(rrf_scores)
    norm_pop = _normalize([float(v) if v is not None else 0.0 for v in pop_scores])

    user_concerns = state.get("health_concerns") or []
    health_traits = state.get("health_traits") or ""

    scored = []
    for i, c in enumerate(candidates):
        has_pop = pop_scores[i] is not None
        has_sentiment = sent_scores[i] is not None
        has_repeat = rep_scores[i] is not None

        gamma_v = float(sent_scores[i]) if has_sentiment else 0.0
        delta_v = float(rep_scores[i])  if has_repeat    else 0.0

        # 기본 점수 계산
        if not has_pop and not has_sentiment and not has_repeat:
            base_score = norm_rrf[i]
        elif not has_sentiment and not has_repeat:
            base_score = _ALPHA * norm_rrf[i] + 0.35 * float(norm_pop[i])
        else:
            base_score = (
                _ALPHA * norm_rrf[i]
                + _BETA  * float(norm_pop[i])
                + _GAMMA * gamma_v
                + _DELTA * delta_v
            )
        
        score = base_score

        # Aspect sentiment 보너스
        # (detected_aspect 생략)

        # 건강/품종 가산점
        bonus = 0.0
        product_tags = c.get("health_concern_tags") or []
        if isinstance(product_tags, str):
            try: product_tags = ast.literal_eval(product_tags)
            except: product_tags = [product_tags]

        health_bonus = 0.0
        if any(h in product_tags for h in user_concerns):
            health_bonus = 0.20
            bonus += health_bonus
            
        trait_bonus = 0.0
        if health_traits:
            trait_keywords = [k for k in ["슬개골", "기관허탈", "눈물", "피부", "관절", "체중", "소화", "신장", "심장"] if k in health_traits]
            if any(k in product_tags for k in trait_keywords):
                trait_bonus = 0.10
                bonus += trait_bonus
        
        score += bonus

        print(f"\n[SCORE BREAKDOWN] {c.get('goods_name')}")
        print(f"  - 리뷰 수: {c.get('review_count', 0)}개")
        print(f"  - RRF(norm): {norm_rrf[i]:.4f} (w={_ALPHA}) -> {_ALPHA * norm_rrf[i]:.4f}")
        print(f"  - Pop(norm): {float(norm_pop[i]):.4f} (w={_BETA}) -> {_BETA * float(norm_pop[i]):.4f}")
        print(f"  - Sent: {gamma_v:.4f} (w={_GAMMA}) -> {_GAMMA * gamma_v:.4f}")
        if bonus > 0:
            print(f"  - 가산점: {bonus:.2f} (건강:{health_bonus}, 품종:{trait_bonus})")
        print(f"  - 최종 점수: {score:.4f}")

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

def test_scoring_breakdown(query="러시안 블루 간식 추천"):
    print(f"\n=== SCORING TEST: {query} ===")
    
    state = {
        "user_input": query,
        "filters": {"pet_type": "고양이", "category": "간식"},
        "pet_profile": {"species": "cat", "breed": "러시안 블루", "age": "4살"},
        "health_concerns": [],
        "allergies": [],
        "filter_relaxation_count": 0,
        "health_traits": ""
    }
    
    print("\n[STEP 1] 검색(search_node) 실행 중...")
    search_res = search_node(state)
    state.update(search_res)
    
    print("\n[STEP 2] 스코어 세부 내역 분석 중...")
    results = print_scoring_details(state)
    
    print("\n=== 최종 추천 TOP 5 ===")
    for i, (score, product) in enumerate(results[:5]):
        print(f"{i+1}. {product['goods_name']} (Total: {score:.4f})")

if __name__ == "__main__":
    test_scoring_breakdown()
