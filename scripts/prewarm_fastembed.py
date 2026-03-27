import os

from fastembed import TextEmbedding


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    model_name = os.getenv("FASTEMBED_MODEL", "intfloat/multilingual-e5-large")
    cache_dir = os.getenv("FASTEMBED_CACHE_PATH", "/opt/fastembed-cache")
    local_files_only = env_flag("FASTEMBED_LOCAL_FILES_ONLY", default=False)

    os.makedirs(cache_dir, exist_ok=True)
    print(
        f"[prewarm_fastembed] model={model_name} "
        f"cache_dir={cache_dir} local_only={local_files_only}"
    )

    model = TextEmbedding(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    vector = next(model.embed(["query: fastembed prewarm check"]))
    print(f"[prewarm_fastembed] ready dim={len(vector)}")


if __name__ == "__main__":
    main()
