from final_ai.infrastructure.settings import LLM_MODEL, LLM_TIMEOUT_SECONDS
from final_ai.infrastructure.observability import wrap_openai

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        from openai import OpenAI

        _llm = wrap_openai(OpenAI(timeout=LLM_TIMEOUT_SECONDS, max_retries=0))
    return _llm


class _LazyLLM:
    def __getattr__(self, name):
        return getattr(get_llm(), name)


llm = _LazyLLM()
