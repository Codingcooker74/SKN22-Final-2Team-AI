from __future__ import annotations

import math

from final_ai.contracts.filters import normalize_search_filters

from .policy_checks import evaluate_recommendation_policy
from .schemas import RecommendationRunResult, RecommendationScore, StateRunResult, StateScore


def _rank_metrics(goods_ids: list[str], relevant_goods_ids: list[str]) -> dict[str, float]:
    if not relevant_goods_ids:
        return {}

    relevant = set(relevant_goods_ids)
    binary_hits = [1.0 if goods_id in relevant else 0.0 for goods_id in goods_ids]

    def _hit_at(k: int) -> float:
        return 1.0 if any(binary_hits[:k]) else 0.0

    def _mrr_at(k: int) -> float:
        for index, hit in enumerate(binary_hits[:k], start=1):
            if hit:
                return 1.0 / index
        return 0.0

    def _dcg(values: list[float]) -> float:
        return sum(value / math.log2(index + 2) for index, value in enumerate(values))

    def _ndcg_at(k: int) -> float:
        dcg = _dcg(binary_hits[:k])
        ideal_hits = [1.0] * min(len(relevant), k)
        idcg = _dcg(ideal_hits)
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def _recall_at(k: int) -> float:
        return sum(binary_hits[:k]) / len(relevant)

    return {
        "hit_at_3": _hit_at(3),
        "hit_at_5": _hit_at(5),
        "mrr_at_5": _mrr_at(5),
        "ndcg_at_5": _ndcg_at(5),
        "recall_at_5": _recall_at(5),
    }


def score_recommendation_result(result: RecommendationRunResult) -> RecommendationScore:
    policy_failures = evaluate_recommendation_policy(result)
    goods_ids = [
        str(product.get("goods_id"))
        for product in result.raw_top_products
        if product.get("goods_id") is not None
    ]
    metrics: dict[str, float | int | None] = {
        "result_count": len(result.raw_top_products),
        "search_results_count": result.search_results_count,
        "filter_relaxation_count": result.filter_relaxation_count,
        "recommend_retry_pending": int(result.recommend_retry_pending),
        "duration_ms": result.duration_ms,
    }
    metrics.update(_rank_metrics(goods_ids, result.case.expect.relevant_goods_ids))
    return RecommendationScore(
        passed=not policy_failures and result.error is None,
        metrics=metrics,
        policy_failures=policy_failures,
    )


def score_state_result(result: StateRunResult) -> StateScore:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    if result.error:
        return StateScore(
            passed=False,
            checks={},
            metrics={"duration_ms": result.duration_ms},
            failures=[f"runner_error: {result.error}"],
        )

    expectation = result.case.expect_state
    merged_state = result.merged_state or {}
    filters = normalize_search_filters(merged_state.get("filters"))

    if expectation.intents:
        checks["intents"] = set(merged_state.get("intents") or []) == set(expectation.intents)
    if expectation.pet_type:
        checks["pet_type"] = filters.get("pet_type") == expectation.pet_type
    if expectation.category:
        checks["category"] = filters.get("category") == expectation.category
    if expectation.subcategory:
        checks["subcategory"] = filters.get("subcategory") == expectation.subcategory
    if expectation.brand:
        checks["brand"] = filters.get("brand") == expectation.brand
    if expectation.health_concerns:
        actual = set(merged_state.get("health_concerns") or [])
        checks["health_concerns"] = set(expectation.health_concerns).issubset(actual)
    if expectation.target_pet_id:
        checks["target_pet_id"] = merged_state.get("target_pet_id") == expectation.target_pet_id
    if expectation.target_pet_name:
        checks["target_pet_name"] = result.resolved_pet_name == expectation.target_pet_name
    if expectation.route is not None:
        checks["route"] = result.route == expectation.route
    if expectation.min_decomposed_tasks is not None:
        checks["min_decomposed_tasks"] = (
            len(merged_state.get("decomposed_tasks") or []) >= expectation.min_decomposed_tasks
        )
    if expectation.should_clarify is not None:
        checks["should_clarify"] = result.should_clarify == expectation.should_clarify
    if expectation.is_result_refinement is not None:
        checks["is_result_refinement"] = bool(merged_state.get("is_result_refinement")) == expectation.is_result_refinement

    for key, passed in checks.items():
        if not passed:
            failures.append(f"{key}_failed")

    slot_check_keys = [
        key for key in ("pet_type", "category", "subcategory", "brand", "target_pet_id", "target_pet_name") if key in checks
    ]
    slot_accuracy = None
    if slot_check_keys:
        slot_accuracy = sum(1 for key in slot_check_keys if checks[key]) / len(slot_check_keys)

    intent_accuracy = None
    if "intents" in checks:
        intent_accuracy = 1.0 if checks["intents"] else 0.0

    metrics: dict[str, float | int | None] = {
        "duration_ms": result.duration_ms,
        "slot_accuracy": slot_accuracy,
        "intent_accuracy": intent_accuracy,
        "clarification_accuracy": (1.0 if checks.get("should_clarify") else 0.0)
        if "should_clarify" in checks
        else None,
        "refinement_accuracy": (1.0 if checks.get("is_result_refinement") else 0.0)
        if "is_result_refinement" in checks
        else None,
        "decomposed_task_count": len(merged_state.get("decomposed_tasks") or []),
        "pending_request_count": len(merged_state.get("pending_requests") or []),
    }

    return StateScore(
        passed=not failures,
        checks=checks,
        metrics=metrics,
        failures=failures,
    )
