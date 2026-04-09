import re

from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import get_logger
from final_ai.infrastructure.repositories.pet_repository import (
    fetch_pet_for_user,
    fetch_pet_preferences,
)
from final_ai.infrastructure.search.hybrid_search import normalize_pet_species

logger = get_logger(__name__)


def _parse_pet_age(age_str) -> float:
    if not age_str:
        return 1.0

    age_str = str(age_str).replace(" ", "")

    match_full = re.search(r"(\d+)(?:년|살)(\d+)개월", age_str)
    if match_full:
        years = int(match_full.group(1))
        months = int(match_full.group(2))
        return years + (months / 12.0)

    match_months = re.search(r"(\d+)개월", age_str)
    if match_months:
        return int(match_months.group(1)) / 12.0

    nums = re.findall(r"(\d+\.?\d*)", age_str)
    if nums:
        return float(nums[0])

    return 1.0


def _resolve_budget_limit(budget_range: str | None) -> int | None:
    if budget_range == "under_5":
        return 50000
    if budget_range == "5_10":
        return 100000
    if budget_range == "10_20":
        return 200000
    return None


def _determine_age_group(target_age: float, target_species: str | None) -> str:
    if target_age < 1:
        return "키튼" if target_species in ["cat", "고양이"] else "퍼피"
    if target_age >= 7:
        return "시니어"
    return "어덜트"


def build_profile_state(state: ChatState) -> dict:
    user_id = state.get("user_id")
    target_pet_id = state.get("target_pet_id")
    pet_profile = dict(state.get("pet_profile") or {})
    health_concerns = list(state.get("health_concerns") or [])
    allergies = list(state.get("allergies") or [])
    food_prefs = list(state.get("food_preferences") or [])
    target_breed = pet_profile.get("breed")
    target_species = pet_profile.get("species")
    target_age = 0.0
    is_pet_override = state.get("is_pet_override", False)
    budget_val = state.get("budget")
    pet_mismatch = False

    try:
        age_val = pet_profile.get("age") or ""
        if isinstance(age_val, (int, float)):
            target_age = float(age_val)
        else:
            target_age = _parse_pet_age(age_val)
    except Exception:
        pass

    if user_id:
        try:
            if target_pet_id:
                # 사용자가 특정 펫을 선택한 경우에만 미스매치 검증을 수행함
                pet_row = fetch_pet_for_user(user_id, target_pet_id=target_pet_id, auto_latest=True)
            elif is_pet_override:
                pet_row = None
                logger.info("explicit pet override detected; skipping db auto-load")
            else:
                # [수정] 펫을 선택하지 않은 경우(선택안함), DB에서 자동으로 가져오는 것을 중단함
                # 사용자가 "사료 추천" -> "강아지"라고 답했을 때, DB의 "고양이"가 덮어씌워지는 것을 방지
                pet_row = None
                logger.info("No pet selected (target_pet_id is None); skipping auto-load to respect user choice")

            if pet_row:
                db_species = normalize_pet_species(pet_row["species"])
                chat_species = normalize_pet_species(target_species)
                db_breed = pet_row["breed"]

                # [수정] target_pet_id가 있을 때만 엄격하게 미스매치를 체크함
                # target_pet_id가 없으면 사용자가 입력 기반 검색을 원한다고 가정함
                if target_pet_id and not state.get("is_pet_switched", False):
                    if chat_species and db_species != chat_species:
                        pet_mismatch = True
                    if target_breed and db_breed != target_breed:
                        pet_mismatch = True

                if not pet_mismatch:
                    pet_id = pet_row["pet_id"]
                    target_breed = pet_row["breed"] or target_breed
                    target_species = pet_row["species"] or target_species
                    target_age = pet_row["age_years"] or target_age

                    pet_profile.update(
                        {
                            "name": pet_row["name"],
                            "species": pet_row["species"],
                            "breed": pet_row["breed"],
                            "age": f"{pet_row['age_years']}세 {pet_row['age_months']}개월",
                            "weight": f"{pet_row['weight_kg']}kg",
                            "gender": pet_row["gender"],
                        }
                    )

                    db_budget_limit = _resolve_budget_limit(pet_row.get("budget_range"))
                    if budget_val is None:
                        budget_val = db_budget_limit

                    preferences = fetch_pet_preferences(str(pet_id))
                    health_concerns = preferences["health_concerns"] or health_concerns
                    allergies = preferences["allergies"] or allergies
                    food_prefs = preferences["food_preferences"] or food_prefs
        except Exception as exc:
            logger.warning("profile db lookup failed: %s", exc)

    age_group = _determine_age_group(target_age, target_species)
    
    # [수정] 펫을 선택하지 않았거나, popularity 인텐트가 있는 경우 미스매치 무시
    intents = state.get("intents") or []
    if not target_pet_id or "popularity" in intents:
        pet_mismatch = False

    logger.info("profile built user=%s pet=%s breed=%s", user_id, pet_profile.get("name"), pet_profile.get("breed"))
    return {
        "pet_profile": pet_profile,
        "health_concerns": health_concerns,
        "allergies": allergies,
        "food_preferences": food_prefs,
        "budget": budget_val,
        "pet_mismatch": pet_mismatch,
        "age_group": age_group,
    }
