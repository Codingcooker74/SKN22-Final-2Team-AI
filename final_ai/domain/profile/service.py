from final_ai.graph.state import ChatState

from final_ai.infrastructure.observability import get_logger
from final_ai.infrastructure.repositories.pet_repository import (
    fetch_pet_full_profile,
    fetch_future_pet_profile,
    fetch_pet_name_for_user,
    fetch_user_pets,
)
from final_ai.infrastructure.search.hybrid_search import normalize_pet_species

logger = get_logger(__name__)

HEALTH_CONCERN_MAP = {
    "skin": "피부",
    "joint": "관절",
    "digestion": "소화",
    "weight": "체중",
    "urinary": "요로",
    "eye": "눈물",
    "hairball": "헤어볼",
    "dental": "치아",
    "immunity": "면역",
}


def translate_health_concerns(concerns: list[str] | None) -> list[str]:
    if not concerns:
        return []
    return [HEALTH_CONCERN_MAP.get(concern, concern) for concern in concerns]


def build_pet_context(state: ChatState) -> str:
    pet_profile = state.get("pet_profile") or {}
    parts = []

    if pet_profile.get("species"):
        species = normalize_pet_species(pet_profile["species"]) or pet_profile["species"]
        parts.append(f"종: {species}")
    if pet_profile.get("breed"):
        parts.append(f"품종: {pet_profile['breed']}")
    if pet_profile.get("age"):
        parts.append(f"나이: {pet_profile['age']}")

    concerns = translate_health_concerns(state.get("health_concerns"))
    if concerns:
        parts.append(f"건강관심사: {', '.join(concerns)}")

    if state.get("allergies"):
        parts.append(f"알레르기: {', '.join(state['allergies'])}")
    if state.get("food_preferences"):
        parts.append(f"선호사료타입: {', '.join(state['food_preferences'])}")

    return " / ".join(parts) if parts else "펫 프로필 없음"


def get_pet_name_for_user(user_id: str | None, target_pet_id: str | None = None) -> str:
    if not user_id:
        return "우리 아이"

    try:
        name = fetch_pet_name_for_user(user_id, target_pet_id=target_pet_id)
        if name:
            return name
    except Exception as exc:
        logger.warning("failed to fetch pet name: %s", exc)

    return "우리 아이"


def get_user_pets(user_id: str) -> list[dict]:
    if not user_id:
        return []

    try:
        return fetch_user_pets(user_id)
    except Exception as exc:
        logger.warning("failed to fetch user pets: %s", exc)
        return []


def get_pet_full_profile(pet_id: str) -> dict:
    if not pet_id:
        return {}

    try:
        return fetch_pet_full_profile(pet_id)
    except Exception as exc:
        logger.warning("failed to fetch pet full profile: %s", exc)
        return {}


def get_future_pet_profile(user_id: str) -> dict:
    if not user_id:
        return {}

    try:
        return fetch_future_pet_profile(user_id) or {}
    except Exception as exc:
        logger.warning("failed to fetch future pet profile: %s", exc)
        return {}
