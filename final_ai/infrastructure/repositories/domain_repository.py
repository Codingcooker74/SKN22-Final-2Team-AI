from functools import lru_cache
from pathlib import Path

import pandas as pd

from final_ai.infrastructure.observability import get_logger

logger = get_logger(__name__)


def resolve_domain_csv_path() -> Path | None:
    base = Path(__file__).resolve().parents[3]
    candidates = [
        base / "test" / "data" / "merged_QnA_final.csv",
        Path(__file__).resolve().parents[2] / "pipeline" / "data" / "merged_QnA_final.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


@lru_cache(maxsize=1)
def _load_domain_dataframe() -> tuple[pd.DataFrame | None, str | None]:
    csv_path = resolve_domain_csv_path()
    if csv_path is None:
        logger.warning("domain csv file not found")
        return None, None

    try:
        return pd.read_csv(csv_path), csv_path.name
    except Exception:
        logger.exception("failed to load domain csv")
        return None, csv_path.name


def search_domain_rows(
    query: str,
    *,
    species: str | None = None,
    limit: int = 5,
) -> tuple[list[dict], str | None]:
    dataframe, csv_name = _load_domain_dataframe()
    if dataframe is None:
        return [], csv_name

    filtered = dataframe.copy()

    if species:
        species_kr = "강아지" if species == "dog" else "고양이"
        if "분류" in filtered.columns:
            filtered = filtered[filtered["분류"].astype(str).str.contains(species_kr, na=False)]

    keywords = [keyword for keyword in query.split() if len(keyword) > 1]
    if keywords and "질문" in filtered.columns:
        mask = filtered["질문"].astype(str).str.contains("|".join(keywords), na=False)
        candidates = filtered[mask].head(limit)
    else:
        candidates = filtered.head(limit)

    return candidates.to_dict(orient="records"), csv_name
