import json

from final_ai.contracts.filters import normalize_search_filters
from final_ai.infrastructure.llm.openai_client import LLM_MODEL, llm
from final_ai.infrastructure.observability import get_logger

logger = get_logger(__name__)

MEMORY_DIALOG_STATE_KEYS = (
    "intents",
    "filters",
    "domain_intent",
    "clarification_count",
    "pending_pet_ids",
    "pending_categories",
    "target_pet_id",
    "pet_profile",
    "health_concerns",
    "allergies",
    "food_preferences",
    "is_pet_override",
    "pet_mismatch",
    "detected_aspect",
    "budget",
    "filter_relaxation_count",
    "recommend_retry_pending",
    "decomposed_tasks",
    "pending_requests",
)
SUMMARY_MAX_BULLETS = 6
SUMMARY_MAX_LINE_LENGTH = 120


def format_conversation_history(history: list[dict] | None, *, limit: int | None = None) -> str:
    entries = list(history or [])
    if limit is not None:
        entries = entries[-limit:]

    lines = []
    for entry in entries:
        role = (entry or {}).get("role")
        content = ((entry or {}).get("content") or "").strip()
        if not content:
            continue
        speaker = "사용자" if role == "user" else "어시스턴트" if role == "assistant" else "시스템"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "없음"


def _parse_summary_lines(summary: str | None) -> list[str]:
    lines = []
    for raw_line in (summary or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if not line or line in lines:
            continue
        lines.append(line[:SUMMARY_MAX_LINE_LENGTH])
    return lines


def _normalize_summary_text(summary: str | None) -> str:
    lines = _parse_summary_lines(summary)
    return "\n".join(f"- {line}" for line in lines[:SUMMARY_MAX_BULLETS])


def extract_dialog_state(state: dict) -> dict:
    dialog_state = {}
    for key in MEMORY_DIALOG_STATE_KEYS:
        if key == "filters":
            dialog_state["filters"] = normalize_search_filters(state.get("filters"))
            continue
        value = state.get(key)
        if key in {
            "intents",
            "pending_pet_ids",
            "pending_categories",
            "health_concerns",
            "allergies",
            "food_preferences",
            "decomposed_tasks",
            "pending_requests",
        }:
            dialog_state[key] = list(value or [])
            continue
        if key == "pet_profile":
            dialog_state[key] = dict(value or {})
            continue
        if key in {"clarification_count", "filter_relaxation_count"}:
            dialog_state[key] = int(value or 0)
            continue
        if key in {"is_pet_override", "pet_mismatch", "recommend_retry_pending"}:
            dialog_state[key] = bool(value)
            continue
        dialog_state[key] = value
    return dialog_state


def _fallback_memory_summary(existing_summary: str, summary_candidates: list[dict]) -> str:
    lines = _parse_summary_lines(existing_summary)
    for entry in summary_candidates[-SUMMARY_MAX_BULLETS:]:
        role = entry.get("role")
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자 요청" if role == "user" else "이전 응답"
        candidate = f"{prefix}: {content[:SUMMARY_MAX_LINE_LENGTH]}"
        if candidate not in lines:
            lines.append(candidate)
    return "\n".join(f"- {line}" for line in lines[-SUMMARY_MAX_BULLETS:])


def _summarize_with_llm(existing_summary: str, summary_candidates: list[dict], dialog_state: dict) -> str:
    candidate_text = format_conversation_history(summary_candidates)
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 반려동물 커머스 챗봇의 장기 메모리 압축기다. "
                    "기존 요약과 오래된 대화 조각을 받아 다음 턴에도 유지해야 할 정보만 한국어 bullet로 정리하라.\n"
                    "규칙:\n"
                    "- 최대 6개 bullet만 작성한다.\n"
                    "- 펫 정보, 건강/알레르기/선호, 예산, 반복된 구매 의도, 미해결 후속 요청만 남긴다.\n"
                    "- 인사, 중복 표현, 직전 최근 대화 기록과 겹치는 세부 표현은 버린다.\n"
                    "- 추정하지 말고 대화에 나온 사실만 적는다.\n"
                    "- 출력은 bullet 리스트만 작성한다."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"기존 요약:\n{existing_summary or '없음'}\n\n"
                    f"현재 구조화 상태:\n{json.dumps(dialog_state, ensure_ascii=False)}\n\n"
                    f"새로 요약할 오래된 대화:\n{candidate_text}"
                ),
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def build_memory_summary(state: dict) -> str:
    existing_summary = (state.get("memory_summary") or "").strip()
    summary_candidates = list(state.get("summary_candidates") or [])
    if not summary_candidates:
        return existing_summary

    dialog_state = extract_dialog_state(state)
    try:
        next_summary = _summarize_with_llm(existing_summary, summary_candidates, dialog_state)
    except Exception as exc:
        logger.warning("memory summary fallback used: %s", exc)
        next_summary = _fallback_memory_summary(existing_summary, summary_candidates)

    normalized_summary = _normalize_summary_text(next_summary)
    if normalized_summary:
        return normalized_summary
    return _normalize_summary_text(existing_summary)


def build_memory_payload(state: dict) -> dict:
    summary_candidates = list(state.get("summary_candidates") or [])
    last_compacted_message_id = state.get("last_compacted_message_id")
    if summary_candidates:
        last_compacted_message_id = summary_candidates[-1].get("message_id") or last_compacted_message_id
    return {
        "dialog_state": extract_dialog_state(state),
        "memory_summary": build_memory_summary(state),
        "last_compacted_message_id": last_compacted_message_id,
    }
