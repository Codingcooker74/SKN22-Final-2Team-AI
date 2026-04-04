from final_ai.domain.intent.service import classify_intent
from final_ai.infrastructure.observability import traceable
from final_ai.graph.state import ChatState


@traceable(name="intent_node", run_type="chain")
def intent_node(state: ChatState) -> dict:
    return classify_intent(state)
