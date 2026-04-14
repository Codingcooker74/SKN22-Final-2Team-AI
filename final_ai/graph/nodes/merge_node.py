from final_ai.application.recommendation.service import serialize_product_card
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import get_logger, traceable

logger = get_logger(__name__)


@traceable(name="merge_node", run_type="chain")
def merge_node(state: ChatState) -> dict:
    domain_contexts = state.get("domain_contexts") or []
    reranked_results = state.get("reranked_results") or []
    intents = state.get("intents") or []
    has_domain = "domain_qa" in intents
    has_recommend = "recommend" in intents or "popularity" in intents

    if has_domain and has_recommend:
        mode = "combined"
    elif has_domain:
        mode = "domain_qa"
    elif has_recommend:
        mode = "recommend"
    elif domain_contexts and reranked_results:
        mode = "combined"
    elif domain_contexts:
        mode = "domain_qa"
    elif reranked_results:
        mode = "recommend"
    else:
        mode = "empty"

    logger.info("merge mode=%s contexts=%s products=%s", mode, len(domain_contexts), len(reranked_results))

    product_cards = [serialize_product_card(product) for product in reranked_results]
    last_recommended_goods_ids = [
        str(card.get("goods_id"))
        for card in product_cards
        if card.get("goods_id") is not None
    ]

    return {
        "response_mode": mode,
        "product_cards": product_cards,
        "last_recommended_goods_ids": last_recommended_goods_ids or list(state.get("last_recommended_goods_ids") or []),
    }
