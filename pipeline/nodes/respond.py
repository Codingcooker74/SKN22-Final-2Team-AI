import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage
from pipeline.utils import llm, LLM_MODEL, build_pet_context
from pipeline.state import ChatState

RESPOND_SYSTEM = """\
당신은 반려동물 쇼핑 서비스의 친절한 AI 어시스턴트입니다.
사용자에게 반려동물 건강 정보와 상품 추천을 함께 제공합니다.

규칙:
- 전문적이지만 친근한 말투로 답변하세요.
- 도메인 지식이 있으면 먼저 설명하고, 관련 상품을 자연스럽게 연결하세요.
- 상품 목록은 "추천 상품을 확인해 주세요!" 같은 간결한 문장으로 안내하세요 (상세 내용은 우측 패널에 표시됩니다).
- 정보가 없는 부분은 솔직하게 안내하세요.
- 답변은 3~5문장 이내로 간결하게 작성하세요.
"""


def respond_node(state: ChatState) -> dict:
    """최종 응답 생성 (LLM)"""
    domain_contexts  = state.get("domain_contexts")  or []
    reranked_results = state.get("reranked_results") or []
    pet_ctx          = build_pet_context(state)
    user_input       = state["user_input"]
    clarification_count = state.get("clarification_count", 0)

    # ── 컨텍스트 조합 ───────────────────────────────────────────────────────────
    context_parts = []

    if domain_contexts:
        joined = "\n\n".join(domain_contexts[:3])
        context_parts.append(f"[도메인 지식]\n{joined}")

    if reranked_results:
        names = [p.get("product_name", "") for p in reranked_results[:5]]
        context_parts.append(f"[추천 상품 목록]\n" + "\n".join(f"- {n}" for n in names if n))

    # 재질문 2회 초과 → best-effort
    if clarification_count > 2 and not context_parts:
        context_parts.append("[참고] 의도가 명확하지 않아 일반적인 안내를 드립니다.")

    context_block = "\n\n".join(context_parts) if context_parts else "검색된 정보가 없습니다."

    user_msg = (
        f"펫 정보: {pet_ctx}\n\n"
        f"사용자 질문: {user_input}\n\n"
        f"{context_block}"
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
