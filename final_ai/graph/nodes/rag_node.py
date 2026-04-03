from final_ai.domain.domain_qa.retrieval_service import search_domain_contexts
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import traceable


@traceable(name="rag_node", run_type="chain")
def rag_node(state: ChatState) -> dict:
    query = state.get("search_query") or state["user_input"]
    domain_intent = state.get("domain_intent")
    species = (state.get("pet_profile") or {}).get("species")
    return {"domain_contexts": search_domain_contexts(query, domain_intent, species)}
