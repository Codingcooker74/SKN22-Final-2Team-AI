from final_ai.application.recommendation.service import execute_search_state
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import traceable


@traceable(name="search_node", run_type="chain")
def search_node(state: ChatState) -> dict:
    return execute_search_state(state)
