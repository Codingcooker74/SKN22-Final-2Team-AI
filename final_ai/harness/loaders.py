from __future__ import annotations

import json
from pathlib import Path

from .schemas import RecommendationCase, StateCase


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def load_recommendation_cases(*paths: str | Path) -> list[RecommendationCase]:
    cases: list[RecommendationCase] = []
    for path in paths:
        cases.extend(RecommendationCase.from_dict(row) for row in _read_jsonl(path))
    return cases


def load_state_cases(*paths: str | Path) -> list[StateCase]:
    cases: list[StateCase] = []
    for path in paths:
        cases.extend(StateCase.from_dict(row) for row in _read_jsonl(path))
    return cases
