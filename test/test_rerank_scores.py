import sys
import os
import json
from decimal import Decimal

# 프로젝트 루트 및 서비스 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from final_ai.pipeline.nodes.recommend import profile_node, query_node, search_node, rerank_node, _normalize

def run_rerank_test(user_input, pet_profile=None, user_id=None, health_concerns=None):
    print(f"\n{'='*20} 리랭킹 점수 분석 테스트 {'='*20}")
    print(f"입력어: {user_input}")
    print(f"건강 고민: {health_concerns or '없음'}")
    
    initial_state = {
        "user_input": user_input,
        "pet_profile": pet_profile or {},
        "user_id": user_id,
        "messages": [],
        "health_concerns": health_concerns or [],
        "allergies": [],
        "food_preferences": [],
        "filter_relaxation_count": 0,
        "intents": ["recommend"],
    }

    # 1. 프로필 분석
    print(f"\n[STEP 1] 프로필 및 펫 정보 분석 중...")
    state = profile_node(initial_state)
    initial_state.update(state)
    
    # 2. 쿼리 생성
    print(f"[STEP 2] LLM 검색 쿼리 생성 중...")
    state = query_node(initial_state)
    initial_state.update(state)
    print(f"생성된 검색어: {initial_state.get('search_query')}")
    
    # 3. 검색 수행
    print(f"[STEP 3] 하이브리드 검색 수행 중...")
    state = search_node(initial_state)
    initial_state.update(state)
    candidates = initial_state.get("search_results", [])
    print(f"검색 결과: {len(candidates)}개 후보 발견")
    
    if not candidates:
        print("검색 결과가 없어 리랭킹을 수행할 수 없습니다.")
        return

    # 4. 리랭킹 상세 점수 출력 (rerank_node 로직 재현)
    print(f"\n[STEP 4] 리랭킹 전/후 비교 및 상세 점수:")
    print("-" * 155)
    print(f"{'순위(전->후)':<12} | {'상품명':<40} | {'RRF(초기)':<10} | {'인기도':<8} | {'감성':<8} | {'재구매':<8} | {'H_Boost':<8} | {'최종 점수':<10}")
    print("-" * 155)

    initial_ranking = {c.get("goods_id"): idx + 1 for idx, c in enumerate(candidates)}
    
    rrf_scores  = [float(c.get("_score", 0.0)) for c in candidates]
    pop_scores  = [c.get("popularity_score")    for c in candidates]
    sent_scores = [c.get("sentiment_avg")       for c in candidates]
    rep_scores  = [c.get("repeat_rate")         for c in candidates]

    norm_rrf = _normalize(rrf_scores)
    norm_pop = _normalize([float(v) if v is not None else 0.0 for v in pop_scores])

    _ALPHA, _BETA, _GAMMA, _DELTA = 0.50, 0.25, 0.15, 0.10

    scored = []
    for i, c in enumerate(candidates):
        has_pop       = pop_scores[i]  is not None
        has_sentiment = sent_scores[i] is not None
        has_repeat    = rep_scores[i]  is not None

        v_rrf = norm_rrf[i]
        v_pop = float(norm_pop[i])
        v_sent = float(sent_scores[i]) if has_sentiment else 0.0
        v_rep = float(rep_scores[i]) if has_repeat else 0.0

        if not has_pop and not has_sentiment and not has_repeat:
            score = v_rrf
        elif not has_sentiment and not has_repeat:
            score = _ALPHA * v_rrf + 0.35 * v_pop
        else:
            score = (_ALPHA * v_rrf + _BETA * v_pop + _GAMMA * v_sent + _DELTA * v_rep)

        # [추가] 건강관심사 가산점 로직 재현
        v_hboost = 0.0
        product_tags = c.get("health_concern_tags") or []
        if isinstance(product_tags, str):
            import ast
            try: product_tags = ast.literal_eval(product_tags)
            except: product_tags = [product_tags]
            
        if any(h in product_tags for h in initial_state["health_concerns"]):
            v_hboost = 0.20
            score += v_hboost

        scored.append({
            "score": score,
            "name": c.get("name") or c.get("goods_name") or "이름 없음",
            "gid": c.get("goods_id"),
            "v_rrf": v_rrf,
            "v_pop": v_pop,
            "v_sent": v_sent,
            "v_rep": v_rep,
            "v_hboost": v_hboost
        })

    # 최종 점수 기반 재정렬
    scored.sort(key=lambda x: x["score"], reverse=True)

    for idx, item in enumerate(scored):
        new_rank = idx + 1
        old_rank = initial_ranking.get(item["gid"])
        
        # 순위 변동 표시
        rank_change = f"{old_rank:>2} -> {new_rank:<2}"
        if old_rank > new_rank: rank_change += " ▲"
        elif old_rank < new_rank: rank_change += " ▼"
        else: rank_change += " --"

        name = item["name"]
        short_name = name[:30] + "..." if len(name) > 30 else name
        
        print(f"{rank_change:<12} | {short_name:<40} | {item['v_rrf']:10.4f} | {item['v_pop']:8.3f} | {item['v_sent']:8.3f} | {item['v_rep']:8.3f} | {item['v_hboost']:8.2f} | {item['score']:10.4f}")

    print("-" * 155)
    print("\n[최종 TOP 5 추천 상품 및 순위 변동]")
    for i, item in enumerate(scored[:5], 1):
        old_rank = initial_ranking.get(item["gid"])
        change = f"(기존 {old_rank}위)" if old_rank != i else "(순위 유지)"
        print(f"{i}. {item['name']} {change} - 최종 점수: {item['score']:0.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
    else:
        user_query = input("분석할 검색어를 입력하세요: ")
    
    # 펫 정보 및 건강 고민 시뮬레이션
    sample_pet = {
        "species": "dog",
        "breed": "말티즈",
        "age": "3살",
    }
    sample_concerns = ["눈물", "관절"] # 테스트용 건강 고민
    
    run_rerank_test(user_query, pet_profile=sample_pet, health_concerns=sample_concerns)
