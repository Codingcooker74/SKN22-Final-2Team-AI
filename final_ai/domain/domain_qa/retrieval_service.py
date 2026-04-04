from final_ai.infrastructure.repositories.domain_repository import search_domain_rows
from final_ai.infrastructure.observability import get_logger

logger = get_logger(__name__)


def search_domain_contexts(query: str, domain_intent: str | None, species: str | None) -> list[str]:
    rows, csv_name = search_domain_rows(query, species=species, limit=5)
    if csv_name is None:
        logger.warning("rag csv file not found")
        return []

    contexts = []
    for row in rows:
        question = row.get("질문", "")
        answer = row.get("답변", "")
        if question or answer:
            contexts.append(f"질문: {question}\n답변: {answer}")

    logger.info("rag contexts=%s csv=%s domain=%s", len(contexts), csv_name, domain_intent or "all")
    return contexts
