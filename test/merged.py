import json
import os
import sys

# 같은 폴더의 파일들을 임포트하기 위해 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classifier import analyze_intent, process_node
from domain_rag import get_domain_answer

def main():
    print("=" * 50)
    print("반려동물 통합 챗봇 테스트 (의도 분류 + 도메인 RAG)")
    print("종료하려면 'q' 또는 'exit'를 입력하세요.")
    print("=" * 50)

    while True:
        user_input = input("\n사용자: ").strip()
        
        if user_input.lower() in ['q', 'exit', 'quit', '종료']:
            print("테스트를 종료합니다.")
            break
            
        if not user_input:
            continue

        # 1. 의도 분류 수행 (상태 기반 정제)
        print("[STEP 1] 의도 분석 중...")
        from classifier import get_refined_intent, session_state
        result = get_refined_intent(user_input)
        
        # 의도 및 파라미터 추출
        intents = result.get("intents") or ["unclear"]
        pet_type = result.get("pet_type")
        category = result.get("category")
        subcategory = result.get("subcategory")
        
        # 분석 상태 출력
        formatted_result = {
            "intents": intents,
            "pet_type": pet_type,
            "category": category,
            "subcategory": subcategory,
            "domain_qa_sub": result.get("domain_qa_sub") if "domain_qa" in intents else None
        }
        print(f"\n[분석 결과]")
        print(json.dumps(formatted_result, ensure_ascii=False, indent=2))

        # 2. 의도에 따른 처리
        # domain_qa가 포함되어 있고 아직 처리되지 않았다면 RAG 수행
        if "domain_qa" in intents:
            print("\n[STEP 2] 도메인 Q&A 로직 실행 (RAG)...")
            answer = get_domain_answer(user_input, pet_type)
            print("-" * 50)
            print(f"도메인 답변:\n{answer}")
            print("-" * 50)
            # 처리가 완료되면 handled_intents에 추가
            session_state["handled_intents"].add("domain_qa")
        
        # recommend가 포함되어 있으면 추천 가이드 메시지
        if "recommend" in intents:
            print("\n[STEP 3] 추천 로직 실행 (상태 기반 안내)...")
            recommend_msg = process_node(result)
            print(f"{recommend_msg}")
            
            # 만약 모든 정보가 수집되어 추천이 완료되는 단계라면 (process_node의 특정 메시지 체크)
            if "찾아드릴게요" in recommend_msg:
                # 여기서 실제 추천 검색 로직(Search Node 등)을 호출할 수 있습니다.
                print(f"\n[결과] '{pet_type}'의 '{subcategory or category}' 관련 추천 상품 리스트를 생성합니다...")
                # session_state["handled_intents"].add("recommend") # 필요시

        # 그 외 (unclear 등)
        if "unclear" in intents and not any(i in ["recommend", "domain_qa"] for i in intents):
            print("\n[STEP 2] 일반 응답...")
            unclear_msg = process_node(result)
            print(f"{unclear_msg}")

if __name__ == "__main__":
    main()
