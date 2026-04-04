from final_ai.api.dependencies.request_context import ensure_request_active
from final_ai.domain.domain_qa.prompts import build_general_query_prompt
from final_ai.domain.profile.service import build_pet_context
from final_ai.infrastructure.llm.openai_client import LLM_MODEL, llm
from final_ai.infrastructure.observability import get_logger
from final_ai.graph.state import ChatState

logger = get_logger(__name__)


def rewrite_domain_query(state: ChatState) -> dict:
    pet_context = build_pet_context(state)
    prompt = build_general_query_prompt(pet_context, state["user_input"])

    ensure_request_active()
    refined = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ).choices[0].message.content.strip()

    logger.info("domain query rewritten=%s", refined)
    return {"search_query": refined}
