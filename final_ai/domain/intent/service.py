import json

from final_ai.api.dependencies.request_context import ensure_request_active
from final_ai.application.chat.memory import format_conversation_history
from final_ai.contracts.filters import (
    SearchFilters,
    build_search_filters,
    normalize_filter_value,
    normalize_search_filters,
)
from final_ai.domain.intent.prompts import CATEGORIES, build_intent_prompt
from final_ai.domain.profile.service import get_pet_full_profile, get_user_pets
from final_ai.infrastructure.llm.openai_client import LLM_MODEL, llm
from final_ai.infrastructure.observability import get_logger
from final_ai.graph.state import ChatState

logger = get_logger(__name__)


def _build_context(
    *,
    state: ChatState,
    user_input: str,
    user_pets: list[dict],
    prev_intents: list[str],
    prev_filters: SearchFilters,
    prev_pet: dict,
    target_pet_id: str | None,
) -> str:
    context_parts = []
    memory_summary = (state.get("memory_summary") or "").strip()
    if memory_summary:
        context_parts.append(f"누적 대화 요약:\n{memory_summary}")

    history_text = format_conversation_history(state.get("conversation_history"), limit=10)
    if history_text != "없음":
        context_parts.append(f"최근 대화 기록:\n{history_text}")

    if not ((state.get("clarification_count", 0) > 0 or prev_intents) and user_input):
        return "\n\n".join(context_parts)

    prev_data = {
        "intents": prev_intents,
        "filters": prev_filters,
        "pet_profile": prev_pet,
        "current_pet_id": target_pet_id,
        "user_registered_pets": [pet["name"] for pet in user_pets],
    }
    context_parts.append(f"이전 대화 상태 및 등록된 펫 정보: {json.dumps(prev_data, ensure_ascii=False)}")
    return "\n\n".join(context_parts)


def _classify_user_input(user_input: str, context: str) -> dict:
    ensure_request_active()
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": build_intent_prompt(context)},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def classify_intent(state: ChatState) -> dict:
    user_input = state["user_input"]
    user_id = state.get("user_id")
    prev_intents = state.get("intents") or []
    prev_filters = normalize_search_filters(state.get("filters"))
    prev_pet = state.get("pet_profile") or {}
    target_pet_id = state.get("target_pet_id")
    pending_pet_ids = state.get("pending_pet_ids") or []
    pending_categories = state.get("pending_categories") or []

    user_pets = get_user_pets(user_id) if user_id else []
    context = _build_context(
        state=state,
        user_input=user_input,
        user_pets=user_pets,
        prev_intents=prev_intents,
        prev_filters=prev_filters,
        prev_pet=prev_pet,
        target_pet_id=target_pet_id,
    )
    result = _classify_user_input(user_input, context)

    new_intents = result.get("intents") or []
    mentioned_names = result.get("mentioned_pet_names") or []
    exclude_ingredients = result.get("exclude_ingredients") or []
    is_next_request = result.get("is_next_request", False)
    target_categories = result.get("target_categories") or []
    is_explicit_pet_info = bool(result.get("pet_type") or result.get("breed"))

    is_pet_switched = False
    switched_pet_name = None
    overridden_metadata = {}

    if is_next_request:
        if pending_categories:
            target_categories = [pending_categories.pop(0)]
            new_intents = ["recommend"]
        elif pending_pet_ids:
            next_id = pending_pet_ids.pop(0)
            full_profile = get_pet_full_profile(next_id)
            if full_profile:
                target_pet_id = next_id
                prev_pet = full_profile["pet_profile"]
                overridden_metadata = {
                    "health_concerns": full_profile["health_concerns"],
                    "allergies": full_profile["allergies"],
                    "food_preferences": full_profile["food_preferences"],
                }
                is_pet_switched = True
                switched_pet_name = prev_pet.get("name")
                new_intents = ["recommend"]
    elif mentioned_names:
        matched_pets = [pet for pet in user_pets if pet["name"] in mentioned_names]
        if matched_pets:
            first_pet = matched_pets[0]
            if str(first_pet["pet_id"]) != str(target_pet_id):
                full_profile = get_pet_full_profile(str(first_pet["pet_id"]))
                if full_profile:
                    target_pet_id = str(first_pet["pet_id"])
                    prev_pet = full_profile["pet_profile"]
                    overridden_metadata = {
                        "health_concerns": full_profile["health_concerns"],
                        "allergies": full_profile["allergies"],
                        "food_preferences": full_profile["food_preferences"],
                        "breed_context": "",
                        "health_traits": "",
                    }
                    is_pet_switched = True
                    switched_pet_name = prev_pet.get("name")
                    if "recommend" not in new_intents:
                        new_intents.append("recommend")
            for pet in matched_pets[1:]:
                pet_id = str(pet["pet_id"])
                if pet_id != target_pet_id and pet_id not in pending_pet_ids:
                    pending_pet_ids.append(pet_id)
    elif is_explicit_pet_info:
        new_species = None
        if result.get("pet_type") == "강아지":
            new_species = "dog"
        elif result.get("pet_type") == "고양이":
            new_species = "cat"

        new_breed = result.get("breed")
        current_species = prev_pet.get("species")
        current_breed = prev_pet.get("breed")

        is_contradictory = False
        if new_species and current_species and new_species != current_species:
            is_contradictory = True
        if new_breed and current_breed and new_breed != current_breed:
            is_contradictory = True

        if is_contradictory or (new_breed and not target_pet_id):
            target_pet_id = None
            prev_pet = {}
            overridden_metadata = {
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "breed_context": "",
                "health_traits": "",
            }
            logger.info("switching to general breed context=%s", new_breed or new_species)
            is_pet_switched = True
        else:
            logger.info("pet context matched current profile=%s", prev_pet.get("name"))

    if not new_intents:
        if "recommend" in prev_intents:
            new_intents = ["recommend"]
        else:
            new_intents = ["unclear"]

    new_pet = dict(prev_pet)
    if is_explicit_pet_info and not target_pet_id:
        new_pet = {}
        if result.get("pet_type"):
            new_pet["species"] = "dog" if result["pet_type"] == "강아지" else "cat"
        if result.get("breed"):
            new_pet["breed"] = result["breed"]
        if result.get("age"):
            new_pet["age"] = result["age"]
    elif result.get("age"):
        new_pet["age"] = result["age"]

    new_filters: SearchFilters = {}
    is_major_switch = is_pet_switched or target_categories

    species = new_pet.get("species")
    if species:
        new_filters["pet_type"] = "강아지" if species == "dog" else "고양이"
    elif result.get("pet_type"):
        new_filters["pet_type"] = result["pet_type"]
    elif prev_filters.get("pet_type"):
        new_filters["pet_type"] = prev_filters["pet_type"]

    current_pet_kr = new_filters.get("pet_type") or "강아지"
    pet_category_map = CATEGORIES.get(current_pet_kr, {})

    detected_category = None
    if target_categories:
        detected_category = target_categories[0]
        if len(target_categories) > 1:
            for category in target_categories[1:]:
                if category not in pending_categories:
                    pending_categories.append(category)
    elif "recommend" in new_intents and not is_major_switch:
        detected_category = prev_filters.get("category")

    detected_subcategory = normalize_filter_value(result.get("subcategory"))
    if not detected_subcategory and "recommend" in new_intents and not is_major_switch:
        if detected_category == prev_filters.get("category"):
            detected_subcategory = prev_filters.get("subcategory")

    if not detected_subcategory and "recommend" in new_intents:
        found_subcategory = None
        found_category = None
        targets = [detected_category] if detected_category else pet_category_map.keys()
        for category_name in targets:
            subcategories = pet_category_map.get(category_name, {}).get("subcategories", [])
            for subcategory in subcategories:
                keywords = subcategory.split("/") if "/" in subcategory else [subcategory]
                if any(keyword in user_input and len(keyword) > 1 for keyword in keywords):
                    found_subcategory = subcategory
                    found_category = category_name
                    break
            if found_subcategory:
                break
        if found_subcategory:
            detected_subcategory = found_subcategory
            detected_category = found_category
            if detected_category != prev_filters.get("category"):
                detected_subcategory = found_subcategory

    if detected_category:
        new_filters["category"] = detected_category
    if detected_subcategory:
        new_filters["subcategory"] = detected_subcategory

    form_to_subcategory = {
        ("고양이", "캔", "사료"): "주식캔",
        ("고양이", "캔", "간식"): "간식캔",
        ("고양이", "파우치", "사료"): "주식파우치",
        ("고양이", "파우치", "간식"): "간식파우치",
        ("강아지", "캔", "간식"): "캔/파우치",
    }
    previous_form_hint = None if is_major_switch else state.get("form_hint")
    llm_form_hint = result.get("form_hint")
    detected_form = next((keyword for keyword in ("캔", "파우치") if keyword in user_input), None)
    final_form_hint = detected_form or llm_form_hint or previous_form_hint

    form_hint = None
    if final_form_hint and current_pet_kr:
        category = new_filters.get("category")
        subcategory = form_to_subcategory.get((current_pet_kr, final_form_hint, category))
        if subcategory:
            new_filters["subcategory"] = subcategory
            form_hint = None
        else:
            form_hint = final_form_hint

    if "popularity" in new_intents and "subcategory" in new_filters:
        del new_filters["subcategory"]

    logger.info(
        "intent classified input=%s intents=%s pet=%s filters=%s",
        user_input,
        new_intents,
        new_pet.get("name", new_pet.get("breed", "Unknown")),
        build_search_filters(
            pet_type=new_filters.get("pet_type"),
            category=new_filters.get("category"),
            subcategory=new_filters.get("subcategory"),
        ),
    )

    combined_allergies = list(
        set((overridden_metadata.get("allergies") or state.get("allergies") or []) + exclude_ingredients)
    )

    return {
        "intents": new_intents,
        "target_pet_id": target_pet_id,
        "pending_pet_ids": pending_pet_ids,
        "pending_categories": pending_categories,
        "is_pet_switched": is_pet_switched,
        "switched_pet_name": switched_pet_name,
        "domain_intent": result.get("domain_intent") or state.get("domain_intent"),
        "detected_aspect": result.get("detected_aspect") or state.get("detected_aspect"),
        "budget": int(result["budget"]) if result.get("budget") else state.get("budget"),
        "filters": build_search_filters(
            pet_type=new_filters.get("pet_type"),
            category=new_filters.get("category"),
            subcategory=new_filters.get("subcategory"),
        ),
        "pet_profile": new_pet,
        "is_pet_override": is_explicit_pet_info or is_pet_switched,
        "pet_mismatch": False if is_pet_switched else state.get("pet_mismatch", False),
        "form_hint": form_hint,
        "filter_relaxation_count": 0 if target_categories else state.get("filter_relaxation_count", 0),
        "allergies": combined_allergies,
        **overridden_metadata,
    }
