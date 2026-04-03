from final_ai.application.recommendation.service import rerank_search_results
from final_ai.infrastructure.observability import traceable
from final_ai.graph.state import ChatState


@traceable(name="rerank_node", run_type="chain")
def rerank_node(state: ChatState) -> dict:
    return rerank_search_results(state)
