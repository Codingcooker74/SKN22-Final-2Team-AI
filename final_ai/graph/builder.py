from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from final_ai.contracts.filters import normalize_search_filters
from final_ai.graph.nodes import (
    clarify_node,
    general_node,
    intent_node,
    merge_node,
    profile_node,
    query_node,
    rag_node,
    rerank_node,
    respond_node,
    search_node,
)
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import get_logger

logger = get_logger(__name__)


def route_intent(state: ChatState):
    intents = state.get("intents") or ["unclear"]
    filters = normalize_search_filters(state.get("filters"))
    pet_profile = state.get("pet_profile") or {}
    relaxation = state.get("filter_relaxation_count", 0)

    logger.info(
        "route_intent intents=%s pet_type=%s species=%s category=%s",
        intents,
        filters.get("pet_type"),
        pet_profile.get("species"),
        filters.get("category"),
    )

    if "unclear" in intents:
        return "clarify"

    has_domain = "domain_qa" in intents
    has_recommend = "recommend" in intents or "popularity" in intents

    if has_recommend:
        if not filters.get("pet_type") and not pet_profile.get("species"):
            return "clarify"
        if not filters.get("category") and relaxation == 0:
            return "clarify"

    if has_domain and has_recommend:
        # [수정] 복합 의도인 경우에도 펫 미선택 시 profile 노드를 건너뛰도록 분기 처리
        rec_node = "profile" if state.get("target_pet_id") else "query"
        return [Send("general", state), Send(rec_node, state)]
    if has_domain:
        # [수정] 건강 상담(domain_qa)만 있는 경우에도 펫 정보가 있다면 profile 노드를 거쳐 정보를 동기화함
        if state.get("target_pet_id") or state.get("is_pet_switched"):
            return [Send("general", state), Send("profile", state)]
        return "general"
    if has_recommend:
        # [수정] 특정 펫을 선택하지 않은 경우(target_pet_id가 없는 경우) profile 노드를 건너뜀
        if not state.get("target_pet_id"):
            return "query"
        return "profile"
    return "clarify"


def route_rerank(state: ChatState) -> str:
    results = state.get("reranked_results") or []
    relaxation = state.get("filter_relaxation_count", 0)
    retry_pending = bool(state.get("recommend_retry_pending"))

    if retry_pending:
        logger.info("route_rerank retry results=%s relaxation=%s", len(results), relaxation)
        return "query"
    return "merge"


def route_profile_node(state: ChatState) -> str:
    if state.get("pet_mismatch"):
        return "merge"

    intents = state.get("intents") or []
    has_recommend = "recommend" in intents or "popularity" in intents
    if not has_recommend:
        return "merge"
    return "query"


def build_graph(checkpointer=None):
    graph_builder = StateGraph(ChatState)

    graph_builder.add_node("intent", intent_node)
    graph_builder.add_node("clarify", clarify_node)
    graph_builder.add_node("general", general_node)
    graph_builder.add_node("rag", rag_node)
    graph_builder.add_node("profile", profile_node)
    graph_builder.add_node("query", query_node)
    graph_builder.add_node("search", search_node)
    graph_builder.add_node("rerank", rerank_node)
    graph_builder.add_node("merge", merge_node)
    graph_builder.add_node("respond", respond_node)

    graph_builder.add_edge(START, "intent")

    graph_builder.add_conditional_edges(
        "intent",
        route_intent,
        {
            "clarify": "clarify",
            "general": "general",
            "profile": "profile",
            "query": "query", # [추가] query 경로 명시
        },
    )

    graph_builder.add_edge("clarify", END)
    graph_builder.add_edge("general", "rag")
    graph_builder.add_edge("rag", "merge")

    graph_builder.add_conditional_edges(
        "profile",
        route_profile_node,
        {
            "merge": "merge",
            "query": "query",
        },
    )
    graph_builder.add_edge("query", "search")
    graph_builder.add_edge("search", "rerank")

    graph_builder.add_conditional_edges(
        "rerank",
        route_rerank,
        {
            "query": "query",
            "merge": "merge",
        },
    )

    graph_builder.add_edge("merge", "respond")
    graph_builder.add_edge("respond", END)

    if checkpointer is not None:
        return graph_builder.compile(checkpointer=checkpointer)
    return graph_builder.compile()


graph = build_graph()


def chat(
    user_input: str,
    thread_id: str = "default",
    pet_profile: dict | None = None,
    health_concerns: list[str] | None = None,
    allergies: list[str] | None = None,
    food_preferences: list[str] | None = None,
    user_id: str | None = None,
    target_pet_id: str | None = None,
) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "user_input": user_input,
        "messages": [],
        "conversation_history": [],
        "summary_candidates": [],
        "memory_summary": "",
        "last_compacted_message_id": None,
        "pet_profile": pet_profile,
        "health_concerns": health_concerns or [],
        "allergies": allergies or [],
        "food_preferences": food_preferences or [],
        "user_id": user_id,
        "target_pet_id": target_pet_id,
        "last_recommended_goods_ids": [],
        "last_search_goods_ids": [],
        "allowed_goods_ids": [],
        "pending_requests": [],
        "breed_context": None,
        "budget": None,
        "min_budget": None,
        "detected_aspect": None,
        "domain_intent": None,
        "health_traits": None,
        "age_group": None,
        "search_results": [],
        "reranked_results": [],
        "best_reranked_results": [],
        "domain_contexts": [],
        "product_cards": [],
        "response_mode": "",
        "filter_relaxation_count": 0,
        "recommend_retry_pending": False,
        "recommendation_limit": 5,
        "effective_filters": {},
        "original_filters": {},
        "exclusions": {},
        "relaxed_filters": [],
        "candidate_count_by_stage": {},
        "refinement_sort": None,
        "clarification_count": 0,
        "intents": [],
        "is_pet_override": False,
        "is_result_refinement": False,
        "pet_mismatch": False,
    }
    result = graph.invoke(initial_state, config=config)
    return {
        "response": result.get("response", ""),
        "product_cards": result.get("product_cards", []),
    }
