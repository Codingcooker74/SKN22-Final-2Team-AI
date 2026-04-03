from final_ai.domain.domain_qa.query_service import rewrite_domain_query
from final_ai.infrastructure.observability import traceable
from final_ai.graph.state import ChatState


@traceable(name="general_node", run_type="chain")
def general_node(state: ChatState) -> dict:
    return rewrite_domain_query(state)
