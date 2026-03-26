from langchain_core.messages import AIMessage
from final_ai.pipeline.state import ChatState
from final_ai.pipeline.utils import LLM_MODEL, build_pet_context, llm

RESPOND_SYSTEM = """\
당신은 반려동물 쇼핑 서비스의 친절한 AI 어시스턴트입니다.
사용자의 펫 정보와 품종별 건강 지식을 결합하여 개인화된 답변을 제공합니다.

[답변 형식 규칙]
1. 추천 의도(recommend)가 포함된 경우 반드시 다음 형식을 준수하세요 (가독성을 위해 빈 줄 삽입 필수):
   - "{펫이름}에 어울리는 상품을 추천드릴게요."
   - (빈 줄)
   - "종에 해당하는 건강특징: {제공된_건강특징}"
   - "상품 추천 이유: {건강특징과 추천 상품 간의 연관성 설명. ~에 도움이 된다는 점 강조}"
   - (빈 줄)
   - "추천 상품을 확인해 주세요!"

2. 도메인 지식 답변(domain_qa)이 포함된 경우:
   - 관련 지식을 친절하게 설명하고 상품 추천으로 자연스럽게 연결하세요. 섹션 구분 시 줄바꿈을 활용하세요.

3. 일반적인 규칙:
   - 답변의 각 주요 단락 사이에는 반드시 빈 줄(double newline)을 넣어 가독성을 높이세요.
   - 답변은 5문장 내외로 간결하게 작성하되 줄바꿈을 적절히 사용하세요.
   - 답변은 5문장 내외로 간결하게 작성하세요.
   - 상품 목록 자체는 언급하지 마세요 (우측 패널에 표시됨).
"""


def respond_node(state: ChatState) -> dict:
    """최종 응답 생성 (LLM)"""
    domain_contexts  = state.get("domain_contexts")  or []
    reranked_results = state.get("reranked_results") or []
    pet_ctx          = build_pet_context(state)
    user_input       = state["user_input"]
    clarification_count = state.get("clarification_count", 0)
    
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
        products_info = "\n".join([f"- {p.get('brand_name')} {p.get('product_name')}" for p in reranked_results[:3]])
        context_parts.append(f"[추천 상품 후보]\n{products_info}")

    context_block = "\n\n".join(context_parts) if context_parts else "검색된 정보가 없습니다."

    user_msg = (
        f"현재 상황 정보:\n"
        f"- 펫 이름: {pet_name}\n"
        f"- 카테고리: {category}\n"
        f"- 건강 특징: {health_traits}\n"
        f"- 전체 펫 정보: {pet_ctx}\n\n"
        f"사용자 질문: {user_input}\n\n"
        f"참고 데이터:\n{context_block}"
    )

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RESPOND_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.3,
    ).choices[0].message.content.strip()

    print(f"[RESPOND] {response[:80]}...")
    return {
        "messages": [AIMessage(content=response)],
        "response": response,
    }
