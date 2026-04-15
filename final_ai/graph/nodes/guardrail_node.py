from langchain_core.messages import AIMessage

from final_ai.domain.guardrails import check_input_guardrail
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import traceable


@traceable(name="guardrail_node", run_type="chain")
def guardrail_node(state: ChatState) -> dict:
    decision = check_input_guardrail(state.get("user_input"))
    if not decision.blocked:
        return {
            "guardrail_blocked": False,
            "guardrail_reason": "",
        }

    return {
        "guardrail_blocked": True,
        "guardrail_reason": decision.reason,
        "messages": [AIMessage(content=decision.response)],
        "response": decision.response,
        "product_cards": [],
        "intents": ["blocked"],
        "response_mode": "blocked",
    }
