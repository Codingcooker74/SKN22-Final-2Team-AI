import ast

from final_ai.domain.recommendation.constants import HEALTH_TRAIT_KEYWORDS
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import get_logger

ALPHA = 0.50
BETA = 0.25
GAMMA = 0.15
DELTA = 0.10
EPSILON = 0.10
TOP_K = 5
logger = get_logger(__name__)


def _normalize(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(value - mn) / (mx - mn) for value in values]


def rerank_search_results(state: ChatState) -> dict:
    candidates = state.get("search_results") or []
    detected_aspect = state.get("detected_aspect")
    intents = state.get("intents") or []
    relaxation = state.get("filter_relaxation_count", 0)

    if not candidates:
        should_retry = relaxation < 1
        next_relaxation = relaxation + 1 if should_retry else relaxation
        logger.info("rerank empty candidates relaxation=%s retry=%s", relaxation, should_retry)
        return {
            "reranked_results": [],
            "filter_relaxation_count": next_relaxation,
            "recommend_retry_pending": should_retry,
        }

    is_popularity_mode = "popularity" in intents

    rrf_scores = [float(candidate.get("_score", 0.0)) for candidate in candidates]
    pop_scores = [float(candidate.get("popularity_score") or 0.0) for candidate in candidates]
    sent_scores = [float(candidate.get("sentiment_avg") or 0.0) for candidate in candidates]
    rep_scores = [float(candidate.get("repeat_rate") or 0.0) for candidate in candidates]

    norm_rrf = _normalize(rrf_scores)
    norm_pop = _normalize(pop_scores)
    norm_sent = _normalize(sent_scores)
    norm_rep = _normalize(rep_scores)

    scored = []
    for index, candidate in enumerate(candidates):
        if is_popularity_mode:
            score = (
                0.50 * norm_pop[index]
                + 0.25 * norm_sent[index]
                + 0.25 * norm_rep[index]
                + 0.01 * norm_rrf[index]
            )
        else:
            has_pop = candidate.get("popularity_score") is not None
            has_sentiment = candidate.get("sentiment_avg") is not None
            has_repeat = candidate.get("repeat_rate") is not None

            if not has_pop and not has_sentiment and not has_repeat:
                score = norm_rrf[index]
            elif not has_sentiment and not has_repeat:
                score = ALPHA * norm_rrf[index] + 0.35 * norm_pop[index]
            else:
                score = (
                    ALPHA * norm_rrf[index]
                    + BETA * norm_pop[index]
                    + GAMMA * norm_sent[index]
                    + DELTA * norm_rep[index]
                )

        if not is_popularity_mode and detected_aspect and candidate.get("sentiment_avg") is not None:
            score += EPSILON * float(candidate["sentiment_avg"])

        user_concerns = state.get("health_concerns") or []
        health_traits = state.get("health_traits") or ""
        product_tags = candidate.get("health_concern_tags") or []

        if isinstance(product_tags, str):
            try:
                product_tags = ast.literal_eval(product_tags)
            except Exception:
                product_tags = [product_tags]

        if any(concern in product_tags for concern in user_concerns):
            score += 0.20

        if health_traits:
            trait_keywords = [keyword for keyword in HEALTH_TRAIT_KEYWORDS if keyword in health_traits]
            if any(keyword in product_tags for keyword in trait_keywords):
                score += 0.10

        scored.append((score, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = [candidate for _, candidate in scored[:TOP_K]]

    should_retry = len(top) < 3 and relaxation < 1
    new_relaxation = relaxation + 1 if should_retry else relaxation
    mode_str = "POPULARITY" if is_popularity_mode else "NORMAL"
    logger.info("rerank mode=%s final=%s relaxation=%s", mode_str, len(top), relaxation)

    return {
        "reranked_results": top,
        "filter_relaxation_count": new_relaxation,
        "recommend_retry_pending": should_retry,
    }
