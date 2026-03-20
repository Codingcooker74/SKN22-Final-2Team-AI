from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send

from pipeline.state import ChatState
from pipeline.nodes.intent   import intent_node
from pipeline.nodes.clarify  import clarify_node
from pipeline.nodes.domain_qa import general_node, rag_node
from pipeline.nodes.recommend import profile_node, query_node, search_node, rerank_node
from pipeline.nodes.merge    import merge_node
from pipeline.nodes.respond  import respond_node


# ── 라우팅 함수 ────────────────────────────────────────────────────────────────

def route_intent(state: ChatState):
    """
    INTENT → 조건부 분기.
    - unclear → CLARIFY
    - domain_qa → GENERAL (또는 GENERAL + PROFILE 동시 via Send)
    - recommend → PROFILE
    - domain_qa + recommend → Send 병렬 fan-out
    """
    intents = state.get("intents") or ["unclear"]

    if "unclear" in intents:
        return "clarify"

    has_domain  = "domain_qa"  in intents
    has_recommend = "recommend" in intents

    if has_domain and has_recommend:
        # 병렬 fan-out: Send API
        return [Send("general", state), Send("profile", state)]
    elif has_domain:
        return "general"
    elif has_recommend:
        return "profile"
    else:
        return "clarify"


def route_rerank(state: ChatState) -> str:
    """
    RERANK → QUERY (필터 완화 재검색) or MERGE.
    결과 < 3개 + 완화 미사용 시 QUERY로 순환.
    """
    results    = state.get("reranked_results") or []
    relaxation = state.get("filter_relaxation_count", 0)

    if len(results) < 3 and relaxation < 2:
        print(f"[ROUTE_RERANK] 결과 {len(results)}개 → QUERY 재시도 (relaxation={relaxation})")
        return "query"
    return "merge"


# ── 그래프 빌드 ───────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    g = StateGraph(ChatState)

    # 노드 등록
    g.add_node("intent",  intent_node)
    g.add_node("clarify", clarify_node)
    g.add_node("general", general_node)
    g.add_node("rag",     rag_node)
    g.add_node("profile", profile_node)
    g.add_node("query",   query_node)
    g.add_node("search",  search_node)
    g.add_node("rerank",  rerank_node)
    g.add_node("merge",   merge_node)
    g.add_node("respond", respond_node)

    # 엣지
    g.add_edge(START, "intent")

    g.add_conditional_edges(
        "intent",
        route_intent,
        {
            "clarify": "clarify",
            "general": "general",
            "profile": "profile",
        },
    )

    g.add_edge("clarify", END)

    # domain_qa 서브플로우
    g.add_edge("general", "rag")
    g.add_edge("rag",     "merge")

    # recommend 서브플로우
    g.add_edge("profile", "query")
    g.add_edge("query",   "search")
    g.add_edge("search",  "rerank")

    g.add_conditional_edges(
        "rerank",
        route_rerank,
        {
            "query": "query",
            "merge": "merge",
        },
    )

    g.add_edge("merge",   "respond")
    g.add_edge("respond", END)

    cp = checkpointer or MemorySaver()
    return g.compile(checkpointer=cp)


# ── 싱글턴 인스턴스 ───────────────────────────────────────────────────────────
graph = build_graph()


# ── 간편 실행 헬퍼 ────────────────────────────────────────────────────────────

def chat(
    user_input: str,
    thread_id: str = "default",
    pet_profile: dict | None = None,
    health_concerns: list[str] | None = None,
    allergies: list[str] | None = None,
    food_preferences: list[str] | None = None,
    user_id: str | None = None,
) -> dict:
    """
    단일 턴 실행 헬퍼.
    같은 thread_id로 반복 호출하면 MemorySaver가 대화 히스토리를 유지한다.
    """
    config = {"configurable": {"thread_id": thread_id}}
    init_state = {
        "user_input":      user_input,
        "messages":        [],
        "pet_profile":     pet_profile,
        "health_concerns": health_concerns or [],
        "allergies":       allergies       or [],
        "food_preferences": food_preferences or [],
        "user_id":         user_id,
        # 초기화 필드
        "search_results":          [],
        "reranked_results":        [],
        "domain_contexts":         [],
        "product_cards":           [],
        "filter_relaxation_count": 0,
        "clarification_count":     0,
        "intents":                 [],
    }
    result = graph.invoke(init_state, config=config)
    return {
        "response":      result.get("response", ""),
        "product_cards": result.get("product_cards", []),
    }
