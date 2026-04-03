from final_ai.domain.response.compose_service import build_response_state
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import traceable


@traceable(name="respond_node", run_type="chain")
def respond_node(state: ChatState) -> dict:
    return build_response_state(state)
