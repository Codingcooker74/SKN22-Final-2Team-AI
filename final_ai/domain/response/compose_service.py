from langchain_core.messages import AIMessage

from final_ai.api.dependencies.request_context import ensure_request_active
from final_ai.application.chat.memory import format_conversation_history
from final_ai.contracts.filters import normalize_search_filters
from final_ai.domain.profile.service import (
    build_pet_context,
    get_user_pets,
    translate_health_concerns,
)
from final_ai.domain.response.prompts import RESPOND_SYSTEM
from final_ai.infrastructure.llm.openai_client import LLM_MODEL, llm
from final_ai.infrastructure.observability import get_logger
from final_ai.graph.state import ChatState

logger = get_logger(__name__)


def _build_context_block(domain_contexts: list[str], reranked_results: list[dict]) -> str:
    context_parts = []
    if domain_contexts:
        domain_context_block = "\n\n".join(domain_contexts[:2])
        context_parts.append(f"[도메인 지식]\n{domain_context_block}")
    if reranked_results:
        products_info = "\n".join(
            f"- {product.get('brand_name')} {product.get('goods_name')}"
            for product in reranked_results[:3]
        )
        context_parts.append(f"[추천 상품 후보]\n{products_info}")
    return "\n\n".join(context_parts) if context_parts else "검색된 정보가 없습니다."


def _get_pending_info(state: ChatState) -> list[dict]:
    """pending_requests 및 decomposed_tasks 큐에서 대기 중인 펫이름+카테고리 정보를 추출합니다."""
    decomposed_tasks = state.get("decomposed_tasks") or []
    pending_requests = state.get("pending_requests") or []
    
    result = []
    
    # 1. decomposed_tasks 먼저 추가 (우선순위 높음)
    for task in decomposed_tasks:
        result.append({
            "pet_name": task.get("pet_name"),
            "category": task.get("category")
        })

    # 2. pending_requests 추가
    if pending_requests:
        user_id = state.get("user_id")
        all_pets = get_user_pets(user_id) if user_id else []
        pet_id_to_name = {str(pet["pet_id"]): pet["name"] for pet in all_pets}

        for req in pending_requests:
            pet_id = req.get("pet_id")
            category = req.get("category")
            pet_name = pet_id_to_name.get(str(pet_id)) if pet_id else None
            result.append({"pet_name": pet_name, "category": category})
            
    return result


def _build_user_message(
    *,
    state: ChatState,
    pet_name: str,
    category: str,
    translated_concerns: list[str],
    health_traits: str,
    pet_context: str,
    context_block: str,
    pending_info: list[dict],
    memory_summary: str,
    summary_candidates_text: str,
    conversation_history_text: str,
) -> str:
    # 다음 대기 항목 정보 구성
    if pending_info:
        next_item = pending_info[0]
        next_pet = next_item.get("pet_name") or "(현재 펫)"
        next_cat = next_item.get("category") or "상품"
        next_queue_desc = f"{next_pet}의 {next_cat}"
    else:
        next_queue_desc = "없음"

    pending_desc = ", ".join(
        f"{item.get('pet_name') or '현재 펫'}의 {item.get('category') or '상품'}"
        for item in pending_info
    ) if pending_info else "없음"

    return (
        "현재 상황 정보:\n"
        f"- 펫 이름: {pet_name}\n"
        f"- 펫 전환 발생: {'YES' if state.get('is_pet_switched') else 'NO'}\n"
        f"- 전환된 펫 이름: {state.get('switched_pet_name') or 'N/A'}\n"
        f"- 대기 중인 추천 목록: {pending_desc}\n"
        f"- 다음_추천: {next_queue_desc}\n"
        f"- 카테고리: {category}\n"
        f"- 등록된 건강 관심사: {', '.join(translated_concerns) if translated_concerns else '없음'}\n"
        f"- 건강 특징: {health_traits}\n"
        f"- 전체 펫 정보: {pet_context}\n\n"
        f"누적 대화 요약:\n{memory_summary or '없음'}\n\n"
        f"이번 턴에 메모리로 편입할 이전 대화:\n{summary_candidates_text}\n\n"
        f"최근 대화 기록:\n{conversation_history_text}\n\n"
        f"사용자 질문: {state['user_input']}\n\n"
        f"참고 데이터:\n{context_block}"
    )


def _build_fallback_response(
    *,
    pet_name: str,
    category: str,
    reranked_results: list[dict],
    domain_contexts: list[str],
    error: Exception,
) -> str:
    if reranked_results:
        response = f"{pet_name}에 어울리는 {category} 후보를 찾았어요.\n\n추천 상품을 확인해 주세요!"
    elif domain_contexts:
        response = "관련 정보를 찾았지만 답변 생성 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."
    else:
        response = "지금은 추천 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."

    logger.warning("response llm fallback used: %s", error)
    return response


def build_response_state(state: ChatState) -> dict:
    if state.get("pet_mismatch"):
        return {
            "response": "펫 프로필과 다른 반려동물입니다. 펫 프로필 등록 먼저 해주세요.",
            "product_cards": [],
        }

    domain_contexts = state.get("domain_contexts") or []
    reranked_results = state.get("reranked_results") or []
    pet_context = build_pet_context(state)
    health_concerns = state.get("health_concerns") or []

    pet_profile = state.get("pet_profile") or {}
    pet_name = pet_profile.get("name")
    if not pet_name:
        breed = pet_profile.get("breed")
        pet_name = f"{breed} 아이" if breed else "우리 아이"

    filters = normalize_search_filters(state.get("filters"))
    category = filters.get("category") or "상품"
    health_traits = state.get("health_traits") or "특별한 데이터가 없습니다."
    translated_concerns = translate_health_concerns(health_concerns)
    pending_info = _get_pending_info(state)
    context_block = _build_context_block(domain_contexts, reranked_results)
    summary_candidates_text = format_conversation_history(state.get("summary_candidates"), limit=8)
    conversation_history_text = format_conversation_history(state.get("conversation_history"), limit=10)
    user_message = _build_user_message(
        state=state,
        pet_name=pet_name,
        category=category,
        translated_concerns=translated_concerns,
        health_traits=health_traits,
        pet_context=pet_context,
        context_block=context_block,
        pending_info=pending_info,
        memory_summary=(state.get("memory_summary") or "").strip(),
        summary_candidates_text=summary_candidates_text,
        conversation_history_text=conversation_history_text,
    )

    try:
        ensure_request_active()
        response = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": RESPOND_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        ).choices[0].message.content.strip()
    except Exception as exc:
        response = _build_fallback_response(
            pet_name=pet_name,
            category=category,
            reranked_results=reranked_results,
            domain_contexts=domain_contexts,
            error=exc,
        )

    logger.info("response generated preview=%s", response[:80])
    return {
        "messages": [AIMessage(content=response)],
        "response": response,
    }
