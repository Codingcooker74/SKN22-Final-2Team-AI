import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable


HEALTH_CONCERN_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "health_concern_map.json"


@lru_cache(maxsize=1)
def load_health_concern_map() -> dict[str, str]:
    with HEALTH_CONCERN_MAP_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): str(value) for key, value in payload.items()}


def normalize_health_concern(concern: str | None) -> str:
    if concern is None:
        return ""

    text = str(concern).strip()
    if not text:
        return ""

    mapping = load_health_concern_map()
    lowered = text.lower()
    return mapping.get(text, mapping.get(lowered, text))


def normalize_health_concerns(concerns: Iterable[str] | None) -> list[str]:
    if not concerns:
        return []

    normalized = []
    seen = set()
    for concern in concerns:
        mapped = normalize_health_concern(concern)
        if not mapped or mapped in seen:
            continue
        seen.add(mapped)
        normalized.append(mapped)
    return normalized
