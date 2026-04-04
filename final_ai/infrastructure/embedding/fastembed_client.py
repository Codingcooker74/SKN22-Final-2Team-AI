import os

from fastembed import TextEmbedding

from final_ai.infrastructure.observability import get_logger
from final_ai.infrastructure.settings import EMBED_MODEL_NAME

_embed_model = None
_embed_model_unavailable_reason = None
_supported_embed_models = {
    model["model"]: model for model in TextEmbedding.list_supported_models()
}
logger = get_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_memory_limit_bytes() -> int | None:
    paths = (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        if 0 < limit < (1 << 60):
            return limit
    return None


def _read_host_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or total_pages <= 0:
        return None
    return page_size * total_pages


def _total_memory_gb() -> float | None:
    limits = [value for value in (_read_memory_limit_bytes(), _read_host_memory_bytes()) if value]
    if not limits:
        return None
    return min(limits) / (1024 ** 3)


def _embed_model_size_gb() -> float | None:
    model = _supported_embed_models.get(EMBED_MODEL_NAME)
    if not model:
        return None
    size = model.get("size_in_GB")
    try:
        return float(size)
    except (TypeError, ValueError):
        return None


def _dense_model_allowed() -> tuple[bool, str | None]:
    if not _env_flag("FASTEMBED_ENABLED", default=True):
        return False, "FASTEMBED_ENABLED=false"

    if not _env_flag("FASTEMBED_AUTO_DISABLE_LOW_MEMORY", default=True):
        return True, None

    total_memory_gb = _total_memory_gb()
    model_size_gb = _embed_model_size_gb()
    headroom_gb = float(os.getenv("FASTEMBED_MEMORY_HEADROOM_GB", "0.75"))

    if total_memory_gb is None or model_size_gb is None:
        return True, None

    required_memory_gb = model_size_gb + headroom_gb
    if total_memory_gb < required_memory_gb:
        return (
            False,
            "insufficient memory for dense embedding "
            f"(available={total_memory_gb:.2f}GB, required~={required_memory_gb:.2f}GB, model={EMBED_MODEL_NAME})",
        )

    return True, None


def get_embed_model() -> TextEmbedding | None:
    global _embed_model, _embed_model_unavailable_reason

    if _embed_model is not None:
        return _embed_model
    if _embed_model_unavailable_reason is not None:
        return None

    allowed, reason = _dense_model_allowed()
    if not allowed:
        _embed_model_unavailable_reason = reason
        logger.warning("dense embedding disabled: %s", reason)
        return None

    try:
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        local_files_only = _env_flag("FASTEMBED_LOCAL_FILES_ONLY", default=False)
        kwargs = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            kwargs["cache_dir"] = cache_dir
        if local_files_only:
            kwargs["local_files_only"] = True

        logger.info(
            "loading embedding model model=%s local_only=%s cache_dir=%s",
            EMBED_MODEL_NAME,
            local_files_only,
            cache_dir or "default",
        )
        _embed_model = TextEmbedding(EMBED_MODEL_NAME, **kwargs)
        logger.info("embedding model loaded")
    except Exception as exc:
        _embed_model_unavailable_reason = str(exc)
        raise

    return _embed_model


def embed_query(query: str) -> list[float] | None:
    try:
        model = get_embed_model()
        if model is None:
            if _embed_model_unavailable_reason:
                logger.warning("dense embedding unavailable: %s", _embed_model_unavailable_reason)
            return None
        return list(model.embed([f"query: {query}"]))[0].tolist()
    except Exception as exc:
        logger.warning("dense embedding unavailable: %s", exc)
        return None
