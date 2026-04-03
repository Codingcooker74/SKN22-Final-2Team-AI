from final_ai.application.recommendation.service import serialize_product_card
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import get_logger, traceable

logger = get_logger(__name__)


@traceable(name="merge_node", run_type="chain")
def merge_node(state: ChatState) -> dict:
    domain_contexts = state.get("domain_contexts") or []
    reranked_results = state.get("reranked_results") or []

    if domain_contexts and reranked_results:
        mode = "combined"
    elif domain_contexts:
        mode = "domain_qa"
    elif reranked_results:
        mode = "recommend"
    else:
        mode = "empty"

    logger.info("merge mode=%s contexts=%s products=%s", mode, len(domain_contexts), len(reranked_results))

    product_cards = [serialize_product_card(product) for product in reranked_results]

    return {"product_cards": product_cards}
