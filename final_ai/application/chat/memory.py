from final_ai.contracts.filters import normalize_search_filters

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
    "form_hint",
    "detected_aspect",
    "budget",
    "filter_relaxation_count",
    "recommend_retry_pending",
)


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


def extract_dialog_state(state: dict) -> dict:
    dialog_state = {}
    for key in MEMORY_DIALOG_STATE_KEYS:
        if key == "filters":
            dialog_state["filters"] = normalize_search_filters(state.get("filters"))
            continue
        value = state.get(key)
        if key in {"intents", "pending_pet_ids", "pending_categories", "health_concerns", "allergies", "food_preferences"}:
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


def build_memory_summary(state: dict) -> str:
    existing_lines = [
        line.lstrip("- ").strip()
        for line in ((state.get("memory_summary") or "").splitlines())
        if line.strip()
    ]
    pet_profile = state.get("pet_profile") or {}
    filters = normalize_search_filters(state.get("filters"))

    current_lines = []
    pet_bits = [pet_profile.get("name"), pet_profile.get("species"), pet_profile.get("breed"), pet_profile.get("age")]
    pet_line = " / ".join(str(bit) for bit in pet_bits if bit)
    if pet_line:
        current_lines.append(f"현재 펫 정보: {pet_line}")
    intents = ", ".join(state.get("intents") or [])
    if intents:
        current_lines.append(f"현재 의도: {intents}")
    if filters:
        filter_text = ", ".join(f"{key}={value}" for key, value in filters.items() if value)
        if filter_text:
            current_lines.append(f"현재 필터: {filter_text}")
    if state.get("health_concerns"):
        current_lines.append(f"건강 관심사: {', '.join(state['health_concerns'])}")
    if state.get("allergies"):
        current_lines.append(f"알레르기: {', '.join(state['allergies'])}")
    if state.get("food_preferences"):
        current_lines.append(f"선호 사항: {', '.join(state['food_preferences'])}")
    if state.get("pending_categories"):
        current_lines.append(f"대기 카테고리: {', '.join(state['pending_categories'])}")
    if state.get("user_input"):
        current_lines.append(f"최근 사용자 요청: {state['user_input']}")
    if state.get("response"):
        current_lines.append(f"최근 응답 요지: {state['response'][:160]}")

    merged_lines = []
    for line in existing_lines + current_lines:
        if line and line not in merged_lines:
            merged_lines.append(line)
    return "\n".join(f"- {line}" for line in merged_lines[-8:])


def build_memory_payload(state: dict) -> dict:
    return {
        "dialog_state": extract_dialog_state(state),
        "memory_summary": build_memory_summary(state),
        "last_compacted_message_id": None,
    }
