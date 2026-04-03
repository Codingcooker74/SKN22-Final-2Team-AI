import json
from final_ai.graph.nodes import intent_node, respond_node
from final_ai.graph.state import ChatState

def test_multi_pet_flow():
    # ── 1. 초기 상태 설정 (바나나, 초코를 가진 유저 2번) ───────────────────────
    state: ChatState = {
        "messages": [],
        "user_input": "바나나랑 초코 사료 추천해줘",
        "user_id": "2",  # DB의 실제 user_id (정수형일 수도 있으나 문자열로 테스트)
        "target_pet_id": None,
        "pending_pet_ids": [],
        "is_pet_switched": False,
        "switched_pet_name": None,
        "pet_profile": {},
        "health_concerns": [],
        "allergies": [],
        "food_preferences": [],
        "intents": [],
        "filters": {},
        "clarification_count": 0,
        "search_results": [],
        "reranked_results": [],
        "response": "",
        "product_cards": []
    }

    print("\n[STEP 1] '바나나랑 초코 사료 추천해줘' 입력")
    # ── 2. 의도 분류 (첫 번째 펫 '바나나' 전환 및 '초코' 큐잉) ─────────────
    state.update(intent_node(state))
    
    print(f"Target Pet: {state.get('switched_pet_name')} (ID: {state.get('target_pet_id')})")
    print(f"Pending Pets: {state.get('pending_pet_ids')}")
    print(f"Filters: {state.get('filters')}")

    assert state.get('switched_pet_name') in ["바나나", "초코"]
    assert len(state.get('pending_pet_ids', [])) > 0

    # ── 3. 응답 생성 (브릿지 질문 확인) ───────────────────────────────────────
    state.update(respond_node(state))
    print(f"Assistant Response: {state['response']}")
    assert "보여드릴까요" in state['response']

    print("\n[STEP 2] '응 보여줘' 입력 (다음 펫으로 전환)")
    # ── 4. 다음 펫 전환 요청 ───────────────────────────────────────────────────
    state["user_input"] = "응 보여줘"
    state.update(intent_node(state))

    print(f"Target Pet: {state.get('switched_pet_name')} (ID: {state.get('target_pet_id')})")
    print(f"Pending Pets: {state.get('pending_pet_ids')}")

    assert state.get('is_pet_switched') is True
    assert len(state.get('pending_pet_ids', [])) == 0

if __name__ == "__main__":
    try:
        test_multi_pet_flow()
        print("\n✅ Multi-pet switching and queuing test passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
