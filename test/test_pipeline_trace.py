import os
import sys
import uuid
from typing import Dict, Any

# 현 위치(services/fastapi/test)에서 상위 경로 추가하여 module 로드 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_ai.pipeline.chatbot_graph import build_graph
from langgraph.checkpoint.memory import MemorySaver

def main():
    print("=" * 60)
    print(" TailTalk AI Pipeline Trace Tool")
    print("=" * 60)
    print("이 도구는 입력에 대해 어떤 Node들이 실행되는지 순차적으로 보여줍니다.")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.\n")

    # 그래프 빌드
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 펫 프로필 없이 순수 입력만으로 테스트
    pet_profile = {}
    health_concerns = []
    
    print("펫 프로필 정보 없이 순수 입력값 기반으로 테스트를 진행합니다.\n")

    while True:
        try:
            user_input = input("USER > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            print(f"\n[실행 시작] 질문: '{user_input}'")
            print("-" * 40)

            initial_state = {
                "user_input": user_input,
                "pet_profile": pet_profile,
                "health_concerns": health_concerns,
                "messages": [],
                "breed_context": None,
                "search_results": [],
                "reranked_results": [],
                "domain_contexts": [],
                "product_cards": [],
                "filter_relaxation_count": 0,
                "clarification_count": 0,
                "intents": [],
            }

            # stream()을 사용하여 노드 실행 순서 추적
            for event in graph.stream(initial_state, config=config):
                for node_name, state_update in event.items():
                    print(f"▶ NODE: [{node_name}] 실행 완료")
                    
                    # 주요 상태 변화 출력
                    if node_name == "intent":
                        cur_pet = state_update.get('pet_profile', {})
                        print(f"   - 감지된 의도: {state_update.get('intents')}")
                        print(f"   - 필터: {state_update.get('filters')}")
                        print(f"   - 추출된 펫 정보: {cur_pet}")
                    elif node_name == "profile":
                        print(f"   - 펫 프로필 상태: {state_update.get('pet_profile')}")
                        print(f"   - 건강 특징(health_traits): {state_update.get('health_traits')}")
                        if not state_update.get('health_traits'):
                            print("     (!) 건강 특징을 찾지 못했습니다. 품종명이 DB(breed_meta)와 일치하는지 확인하십시오.")
                    elif node_name == "query":
                        print(f"   - 생성된 검색어: {state_update.get('query')}")
                    elif node_name == "search":
                        res = state_update.get('search_results', [])
                        print(f"   - 검색 결과: {len(res)}개 발견")
                    elif node_name == "rerank":
                        res = state_update.get('reranked_results', [])
                        print(f"   - 리랭킹 완료: TOP 상품 '{res[0].get('goods_name') if res else '없음'}'")
                    elif node_name == "respond":
                        print(f"\n[최종 답변]\n{state_update.get('response')}\n")

            print("-" * 40)
            print("[실행 종료]\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[오류 발생] {e}")

    print("\n도구를 종료합니다.")

if __name__ == "__main__":
    main()
