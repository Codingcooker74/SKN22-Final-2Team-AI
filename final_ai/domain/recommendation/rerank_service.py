import ast
import re

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

    # 점수 높은 순으로 정렬
    scored.sort(key=lambda item: item[0], reverse=True)
    
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

    should_retry = len(unique_top) < 3 and relaxation < 1
    new_relaxation = relaxation + 1 if should_retry else relaxation
    mode_str = "POPULARITY" if is_popularity_mode else "NORMAL"
    logger.info("rerank mode=%s final=%s relaxation=%s", mode_str, len(unique_top), relaxation)

    return {
        "reranked_results": unique_top,
        "filter_relaxation_count": new_relaxation,
        "recommend_retry_pending": should_retry,
    }
