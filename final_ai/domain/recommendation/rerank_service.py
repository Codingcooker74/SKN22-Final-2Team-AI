import ast
import re

from final_ai.domain.recommendation.constants import HEALTH_TRAIT_KEYWORDS, RECOMMENDATION_TOP_K
from final_ai.domain.recommendation.filter_relaxation import (
    clamp_relaxation_count,
    next_relaxation_count,
    should_retry_recommendation,
    target_recommendation_count,
)
from final_ai.domain.recommendation.product_intent import candidate_matches_requested_terms
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import get_logger

ALPHA = 0.50
BETA = 0.25
GAMMA = 0.15
DELTA = 0.10
EPSILON = 0.10
REQUESTED_PRODUCT_MATCH_BONUS = 1.25
TOP_K = RECOMMENDATION_TOP_K
logger = get_logger(__name__)


def _choose_best_results(current: list[dict], previous: list[dict], *, target_count: int) -> list[dict]:
    if len(current) >= target_count:
        return current
    if len(previous) > len(current):
        return previous
    return current


def _normalize(values: list[float]) -> list[float]:
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(value - mn) / (mx - mn) for value in values]


def _get_base_product_name(full_name: str) -> str:
    """상품명에서 용량(kg, g, 그램 등) 정보를 제거하여 기본 상품명을 반환합니다."""
    # 용량 패턴: 숫자 + kg/g/키로/그램/팩/p/입 등 (공백 허용)
    weight_pattern = r"\b\d+(\.\d+)?\s*(kg|g|키로|그램|팩|p|입|개입|l|ml)\b"
    # 대괄호 안의 용량 정보도 포함 (예: [1.2kg])
    bracket_pattern = r"\[\d+(\.\d+)?\s*(kg|g|키로|그램|팩|p|입|개입|l|ml)\]"
    
    name = re.sub(weight_pattern, "", full_name, flags=re.IGNORECASE)
    name = re.sub(bracket_pattern, "", name, flags=re.IGNORECASE)
    
    # 연속된 공백 및 특수기호 정리 (예: "상품명   " -> "상품명")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _effective_price(candidate: dict) -> float:
    value = candidate.get("discount_price")
    if value is None:
        value = candidate.get("price")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _apply_refinement_sort(
    scored: list[tuple[float, dict]],
    *,
    refinement_sort: str | None,
) -> list[tuple[float, dict]]:
    if not refinement_sort:
        return scored

    if refinement_sort == "price_low":
        return sorted(
            scored,
            key=lambda item: (
                _effective_price(item[1]),
                -item[0],
                str(item[1].get("goods_id") or ""),
            ),
        )
    if refinement_sort == "price_high":
        return sorted(
            scored,
            key=lambda item: (
                -_effective_price(item[1]),
                -item[0],
                str(item[1].get("goods_id") or ""),
            ),
        )
    if refinement_sort == "popularity":
        return sorted(
            scored,
            key=lambda item: (
                -(float(item[1].get("popularity_score") or 0.0)),
                -(float(item[1].get("review_count") or 0.0)),
                -item[0],
                _effective_price(item[1]),
            ),
        )
    if refinement_sort == "rating":
        return sorted(
            scored,
            key=lambda item: (
                -(float(item[1].get("rating") or 0.0)),
                -(float(item[1].get("review_count") or 0.0)),
                -item[0],
                _effective_price(item[1]),
            ),
        )
    if refinement_sort == "review_count":
        return sorted(
            scored,
            key=lambda item: (
                -(float(item[1].get("review_count") or 0.0)),
                -(float(item[1].get("rating") or 0.0)),
                -item[0],
                _effective_price(item[1]),
            ),
        )
    return scored


def rerank_search_results(state: ChatState) -> dict:
    candidates = state.get("search_results") or []
    detected_aspect = state.get("detected_aspect")
    intents = state.get("intents") or []
    relaxation = clamp_relaxation_count(state.get("filter_relaxation_count", 0))
    refinement_sort = state.get("refinement_sort")
    target_count = target_recommendation_count(state)
    best_results = list(state.get("best_reranked_results") or [])
    requested_product_terms = list(state.get("requested_product_terms") or [])

    if not candidates:
        final_results = _choose_best_results([], best_results, target_count=target_count)
        should_retry = should_retry_recommendation(
            result_count=len(final_results),
            relaxation=relaxation,
            target_count=target_count,
        )
        new_relaxation = next_relaxation_count(relaxation) if should_retry else relaxation
        logger.info(
            "rerank empty candidates relaxation=%s target=%s retry=%s best=%s",
            relaxation,
            target_count,
            should_retry,
            len(final_results),
        )
        return {
            "reranked_results": final_results,
            "best_reranked_results": final_results,
            "filter_relaxation_count": new_relaxation,
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

        requested_product_match = bool(candidate.get("_requested_product_match")) or (
            candidate_matches_requested_terms(candidate, requested_product_terms)
        )
        if requested_product_match:
            score += REQUESTED_PRODUCT_MATCH_BONUS

        scored.append((score, candidate))

    # 점수 높은 순으로 정렬
    scored.sort(key=lambda item: item[0], reverse=True)
    if state.get("is_result_refinement"):
        scored = _apply_refinement_sort(scored, refinement_sort=refinement_sort)
    
    # 중복 상품군 필터링 (용량만 다른 상품 중 점수가 가장 높은 것 하나만 선택)
    unique_top = []
    seen_products = set()
    
    for score, candidate in scored:
        full_name = candidate.get("goods_name", "")
        base_name = _get_base_product_name(full_name)
        
        # 이미 선택된 상품군이면 건너뜀 (이미 점수가 높은 순으로 들어오고 있으므로)
        if base_name and base_name in seen_products:
            continue
            
        seen_products.add(base_name)
        
        logger.info(
            "PRODUCT SCORE: name=%r score=%.4f rrf=%.2f pop=%.2f sent=%.2f",
            full_name,
            score,
            candidate.get("_score", 0.0),
            candidate.get("popularity_score") or 0.0,
            candidate.get("sentiment_avg") or 0.0,
        )
        
        candidate_with_score = dict(candidate)
        candidate_with_score["rerank_score"] = float(score)
        unique_top.append(candidate_with_score)
        
        # 상위 K개만 수집
        if len(unique_top) >= TOP_K:
            break

    final_results = _choose_best_results(unique_top, best_results, target_count=target_count)
    should_retry = should_retry_recommendation(
        result_count=len(final_results),
        relaxation=relaxation,
        target_count=target_count,
    )
    new_relaxation = next_relaxation_count(relaxation) if should_retry else relaxation
    mode_str = "POPULARITY" if is_popularity_mode else "NORMAL"
    logger.info(
        "rerank mode=%s final=%s target=%s relaxation=%s retry=%s refinement=%s refinement_sort=%s",
        mode_str,
        len(final_results),
        target_count,
        relaxation,
        should_retry,
        bool(state.get("is_result_refinement")),
        refinement_sort,
    )

    return {
        "reranked_results": final_results,
        "best_reranked_results": final_results,
        "filter_relaxation_count": new_relaxation,
        "recommend_retry_pending": should_retry,
    }
