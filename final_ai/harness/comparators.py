from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HIGHER_IS_BETTER_METRICS = {
    "hit_at_3",
    "hit_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "recall_at_5",
    "slot_accuracy",
    "intent_accuracy",
    "clarification_accuracy",
}


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(case["case_id"]): case for case in report.get("cases", [])}


def compare_harness_reports(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, Any]:
    baseline_cases = _case_map(baseline_report)
    candidate_cases = _case_map(candidate_report)

    shared_case_ids = sorted(set(baseline_cases) & set(candidate_cases))
    baseline_only_case_ids = sorted(set(baseline_cases) - set(candidate_cases))
    candidate_only_case_ids = sorted(set(candidate_cases) - set(baseline_cases))

    status_regressions: list[str] = []
    status_improvements: list[str] = []
    case_metric_regressions: list[dict[str, Any]] = []

    for case_id in shared_case_ids:
        baseline_case = baseline_cases[case_id]
        candidate_case = candidate_cases[case_id]
        baseline_passed = bool(baseline_case.get("passed"))
        candidate_passed = bool(candidate_case.get("passed"))

        if baseline_passed and not candidate_passed:
            status_regressions.append(case_id)
        elif not baseline_passed and candidate_passed:
            status_improvements.append(case_id)

        baseline_metrics = baseline_case.get("metrics") or {}
        candidate_metrics = candidate_case.get("metrics") or {}
        for metric_name in sorted(HIGHER_IS_BETTER_METRICS):
            baseline_value = baseline_metrics.get(metric_name)
            candidate_value = candidate_metrics.get(metric_name)
            if not isinstance(baseline_value, (int, float)) or not isinstance(candidate_value, (int, float)):
                continue
            if candidate_value < baseline_value:
                case_metric_regressions.append(
                    {
                        "case_id": case_id,
                        "metric": metric_name,
                        "baseline": baseline_value,
                        "candidate": candidate_value,
                        "delta": candidate_value - baseline_value,
                    }
                )

    baseline_avgs = baseline_report.get("summary", {}).get("average_metrics") or {}
    candidate_avgs = candidate_report.get("summary", {}).get("average_metrics") or {}
    shared_metrics = sorted(set(baseline_avgs) & set(candidate_avgs))

    average_metric_deltas: dict[str, dict[str, float]] = {}
    average_metric_regressions: list[str] = []
    for metric_name in shared_metrics:
        baseline_value = baseline_avgs.get(metric_name)
        candidate_value = candidate_avgs.get(metric_name)
        if not isinstance(baseline_value, (int, float)) or not isinstance(candidate_value, (int, float)):
            continue
        delta = candidate_value - baseline_value
        average_metric_deltas[metric_name] = {
            "baseline": float(baseline_value),
            "candidate": float(candidate_value),
            "delta": float(delta),
        }
        if metric_name in HIGHER_IS_BETTER_METRICS and delta < 0:
            average_metric_regressions.append(metric_name)

    baseline_failed_cases = int(baseline_report.get("summary", {}).get("failed_cases", 0))
    candidate_failed_cases = int(candidate_report.get("summary", {}).get("failed_cases", 0))

    return {
        "summary": {
            "baseline_total_cases": int(baseline_report.get("summary", {}).get("total_cases", 0)),
            "candidate_total_cases": int(candidate_report.get("summary", {}).get("total_cases", 0)),
            "baseline_failed_cases": baseline_failed_cases,
            "candidate_failed_cases": candidate_failed_cases,
            "failed_case_delta": candidate_failed_cases - baseline_failed_cases,
        },
        "shared_case_ids": shared_case_ids,
        "baseline_only_case_ids": baseline_only_case_ids,
        "candidate_only_case_ids": candidate_only_case_ids,
        "status_regressions": status_regressions,
        "status_improvements": status_improvements,
        "case_metric_regressions": case_metric_regressions,
        "average_metric_deltas": average_metric_deltas,
        "average_metric_regressions": average_metric_regressions,
        "passed": not status_regressions and candidate_failed_cases <= baseline_failed_cases,
    }
