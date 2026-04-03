from final_ai.application.recommendation.service import build_search_query_state
from final_ai.infrastructure.observability import traceable
from final_ai.graph.state import ChatState


@traceable(name="query_node", run_type="chain")
def query_node(state: ChatState) -> dict:
    return build_search_query_state(state)
