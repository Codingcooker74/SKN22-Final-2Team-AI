from __future__ import annotations

import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from .schemas import RecommendationRunResult, RecommendationScore, StateRunResult, StateScore


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _build_recommendation_case_summary(
    result: RecommendationRunResult,
    score: RecommendationScore,
) -> dict[str, Any]:
    top_products = [
        {
            "goods_id": product.get("goods_id"),
            "product_name": product.get("product_name") or product.get("goods_name"),
            "effective_price": product.get("discount_price") or product.get("price"),
        }
        for product in result.products[:5]
    ]
    profile_pet = result.profile_state.get("pet_profile") or {}
    allowed_goods_ids = result.initial_state.get("allowed_goods_ids") or []
    if not allowed_goods_ids and result.initial_state.get("is_result_refinement"):
        allowed_goods_ids = result.initial_state.get("last_recommended_goods_ids") or []
    return {
        "case_id": result.case.case_id,
        "query": result.case.query,
        "passed": score.passed,
        "policy_failures": score.policy_failures,
        "search_query": result.search_query,
        "filters": result.query_state.get("filters") or result.initial_state.get("filters") or {},
        "allowed_goods_ids": allowed_goods_ids,
        "last_recommended_goods_ids": result.initial_state.get("last_recommended_goods_ids") or [],
        "is_result_refinement": bool(result.initial_state.get("is_result_refinement")),
        "refinement_sort": result.initial_state.get("refinement_sort"),
        "target_pet_id": result.case.target_pet_id,
        "profile_pet_name": profile_pet.get("name"),
        "profile_health_concerns": result.profile_state.get("health_concerns") or [],
        "allergies": result.profile_state.get("allergies") or result.initial_state.get("allergies") or [],
        "budget": result.profile_state.get("budget") or result.initial_state.get("budget"),
        "result_count": len(result.products),
        "top_products": top_products,
        "route_retry_pending": result.recommend_retry_pending,
        "filter_relaxation_count": result.filter_relaxation_count,
        "duration_ms": result.duration_ms,
        "error": result.error,
    }


def _build_state_case_summary(
    result: StateRunResult,
    score: StateScore,
) -> dict[str, Any]:
    merged_state = result.merged_state or {}
    decomposed_tasks = merged_state.get("decomposed_tasks") or []
    pending_requests = merged_state.get("pending_requests") or []
    return {
        "case_id": result.case.case_id,
        "message": result.case.message,
        "original_user_input": merged_state.get("original_user_input") or result.case.message,
        "normalized_user_input": merged_state.get("normalized_user_input")
        or merged_state.get("user_input")
        or result.case.message,
        "passed": score.passed,
        "failures": score.failures,
        "intents": merged_state.get("intents") or [],
        "filters": merged_state.get("filters") or {},
        "last_recommended_goods_ids": merged_state.get("last_recommended_goods_ids") or [],
        "allowed_goods_ids": merged_state.get("allowed_goods_ids") or [],
        "is_result_refinement": bool(merged_state.get("is_result_refinement")),
        "health_concerns": merged_state.get("health_concerns") or [],
        "target_pet_id": merged_state.get("target_pet_id"),
        "resolved_pet_name": result.resolved_pet_name,
        "route": result.route,
        "should_clarify": result.should_clarify,
        "decomposed_task_count": len(decomposed_tasks),
        "decomposed_tasks": decomposed_tasks[:3],
        "pending_request_count": len(pending_requests),
        "pending_requests": pending_requests[:3],
        "duration_ms": result.duration_ms,
        "error": result.error,
    }


def build_recommendation_report(
    results: list[RecommendationRunResult],
    scores: list[RecommendationScore],
) -> dict[str, Any]:
    paired = list(zip(results, scores))
    passed = sum(1 for _, score in paired if score.passed)
    failed = len(paired) - passed

    metric_names = sorted(
        {
            name
            for _, score in paired
            for name, value in score.metrics.items()
            if isinstance(value, (int, float))
        }
    )
    averages = {
        name: _mean(
            [
                float(score.metrics[name])
                for _, score in paired
                if isinstance(score.metrics.get(name), (int, float))
            ]
        )
        for name in metric_names
    }

    return {
        "summary": {
            "total_cases": len(paired),
            "passed_cases": passed,
            "failed_cases": failed,
            "average_metrics": averages,
        },
        "case_summaries": [
            _build_recommendation_case_summary(result, score)
            for result, score in paired
        ],
        "cases": [
            {
                "case_id": result.case.case_id,
                "variant": result.variant_name,
                "passed": score.passed,
                "policy_failures": score.policy_failures,
                "metrics": score.metrics,
                "result": asdict(result),
            }
            for result, score in paired
        ],
    }


def build_state_report(results: list[StateRunResult], scores: list[StateScore]) -> dict[str, Any]:
    paired = list(zip(results, scores))
    passed = sum(1 for _, score in paired if score.passed)
    failed = len(paired) - passed

    metric_names = sorted(
        {
            name
            for _, score in paired
            for name, value in score.metrics.items()
            if isinstance(value, (int, float))
        }
    )
    averages = {
        name: _mean(
            [
                float(score.metrics[name])
                for _, score in paired
                if isinstance(score.metrics.get(name), (int, float))
            ]
        )
        for name in metric_names
    }

    return {
        "summary": {
            "total_cases": len(paired),
            "passed_cases": passed,
            "failed_cases": failed,
            "average_metrics": averages,
        },
        "case_summaries": [
            _build_state_case_summary(result, score)
            for result, score in paired
        ],
        "cases": [
            {
                "case_id": result.case.case_id,
                "variant": result.variant_name,
                "passed": score.passed,
                "failures": score.failures,
                "checks": score.checks,
                "metrics": score.metrics,
                "result": asdict(result),
            }
            for result, score in paired
        ],
    }


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
