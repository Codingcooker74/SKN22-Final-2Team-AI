from final_ai.application.recommendation.service import build_profile_state
from final_ai.infrastructure.observability import traceable
from final_ai.graph.state import ChatState


@traceable(name="profile_node", run_type="chain")
def profile_node(state: ChatState) -> dict:
    return build_profile_state(state)
