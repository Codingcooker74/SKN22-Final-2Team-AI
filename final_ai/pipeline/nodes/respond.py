from langchain_core.messages import AIMessage
from final_ai.observability import traceable
from final_ai.pipeline.state import ChatState
from final_ai.pipeline.utils import LLM_MODEL, build_pet_context, llm

RESPOND_SYSTEM = """\
당신은 반려동물 쇼핑 서비스의 친절한 AI 어시스턴트입니다.
사용자의 펫 정보와 품종별 건강 지식을 결합하여 개인화된 답변을 제공합니다.

[답변 형식 규칙]
1. 추천 의도(recommend)가 포함된 경우 반드시 **섹션 사이마다 빈 줄(Enter 2번)**을 넣어 다음 형식을 엄격히 준수하세요:

   "{펫이름}에 어울리는 상품을 추천드릴게요.
   
   등록된 건강 관심사 : {건강관심사} (값이 있는 경우에만 표시)
   
   종에 해당하는 건강특징: {제공된_건강특징}
   
   상품 추천 이유:
   1. {첫 번째 추천 이유: 건강특징과 상품 간의 연관성 설명}
   2. {두 번째 추천 이유}
   
   추천 상품을 확인해 주세요!"

2. 도메인 지식 답변(domain_qa)이 포함된 경우:
   - 관련 지식을 친절하게 설명하고 섹션 구분 시 반드시 빈 줄(Enter 2번)을 활용하세요.
   - 답변 서두에 등록된 건강 관심사가 있다면 "등록된 건강 관심사 : {건강관심사}" 형식을 포함하세요.

3. 일반 지점:
   - 답변의 각 주요 단락 사이에는 **반드시 한 줄의 빈 줄**을 넣어 가독성을 높이세요.
   - 상품 목록 자체는 언급하지 마세요 (우측 패널에 표시됨).
"""


@traceable(name="respond_node", run_type="chain")
def respond_node(state: ChatState) -> dict:
    """최종 응답 생성 (LLM)"""
    domain_contexts  = state.get("domain_contexts")  or []
    reranked_results = state.get("reranked_results") or []
    pet_ctx          = build_pet_context(state)
    user_input       = state["user_input"]
    health_concerns  = state.get("health_concerns") or []
    
    # 펫 이름 및 카테고리 추출
    pet_name = (state.get("pet_profile") or {}).get("name") or "우리 아이"
    category = (state.get("filters") or {}).get("category") or "상품"
    health_traits = state.get("health_traits") or "특별한 데이터가 없습니다."
    
    # ── 컨텍스트 조합 ───────────────────────────────────────────────────────────
    context_parts = []

    if domain_contexts:
        joined = "\n\n".join(domain_contexts[:2])
        context_parts.append(f"[도메인 지식]\n{joined}")

    if reranked_results:
        # 추천 상품들의 특징(브랜드 등)을 LLM이 알 수 있도록 전달
        products_info = "\n".join([f"- {p.get('brand_name')} {p.get('goods_name')}" for p in reranked_results[:3]])
        context_parts.append(f"[추천 상품 후보]\n{products_info}")

    context_block = "\n\n".join(context_parts) if context_parts else "검색된 정보가 없습니다."

    user_msg = (
        f"현재 상황 정보:\n"
        f"- 펫 이름: {pet_name}\n"
        f"- 카테고리: {category}\n"
        f"- 등록된 건강 관심사: {', '.join(health_concerns) if health_concerns else '없음'}\n"
        f"- 건강 특징: {health_traits}\n"
        f"- 전체 펫 정보: {pet_ctx}\n\n"
        f"사용자 질문: {user_input}\n\n"
        f"참고 데이터:\n{context_block}"
    )

    try:
        response = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": RESPOND_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
        ).choices[0].message.content.strip()
    except Exception as e:
        if reranked_results:
            response = f"{pet_name}에 어울리는 {category} 후보를 찾았어요.\n\n추천 상품을 확인해 주세요!"
        elif domain_contexts:
            response = "관련 정보를 찾았지만 답변 생성 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
        else:
            response = "지금은 추천 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."
        print(f"[RESPOND] LLM 실패로 폴백 응답 사용: {e}")

    print(f"[RESPOND] {response[:80]}...")
    return {
        "messages": [AIMessage(content=response)],
        "response": response,
    }
