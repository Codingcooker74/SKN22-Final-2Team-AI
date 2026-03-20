import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from pipeline.utils import llm, LLM_MODEL, qdrant, hybrid_search, build_pet_context, DOMAIN_INTENT_TO_CATEGORY
from pipeline.state import ChatState


def general_node(state: ChatState) -> dict:
    """쿼리 정제: 모호한 질문을 펫 프로필 기반으로 검색 최적화"""
    pet_ctx = build_pet_context(state)
    prompt  = (
        f"다음 질문을 반려동물 정보를 반영해 검색에 최적화된 한 문장으로 재작성하세요.\n"
        f"펫 정보: {pet_ctx}\n질문: {state['user_input']}"
    )
    refined = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ).choices[0].message.content.strip()

    print(f"[GENERAL] 정제 쿼리: {refined}")
    return {"search_query": refined}


def rag_node(state: ChatState) -> dict:
    """domain_qna Hybrid Search (species + category 필터)"""
    query         = state.get("search_query") or state["user_input"]
    domain_intent = state.get("domain_intent")
    species       = (state.get("pet_profile") or {}).get("species")

    must = []
    if species:
        must.append(FieldCondition(key="species", match=MatchAny(any=[species, "both"])))
    if domain_intent in DOMAIN_INTENT_TO_CATEGORY:
        must.append(FieldCondition(
            key="category",
            match=MatchValue(value=DOMAIN_INTENT_TO_CATEGORY[domain_intent]),
        ))
    f = Filter(must=must) if must else None

    points   = hybrid_search("domain_qna", query, top_k=5, qdrant_filter=f)
    contexts = [
        f"{p.payload.get('question', '')}\n{p.payload.get('answer', '')}".strip()
        for p in points
    ]
    print(f"[RAG] {len(contexts)}개 컨텍스트 (domain_intent={domain_intent})")
    return {"domain_contexts": contexts}
