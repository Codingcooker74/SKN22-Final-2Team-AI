import json
from pathlib import Path

from langchain_core.messages import AIMessage

from final_ai.contracts.filters import normalize_search_filters
from final_ai.graph.state import ChatState
from final_ai.infrastructure.observability import get_logger, traceable

logger = get_logger(__name__)

CATEGORY_FILE = Path(__file__).resolve().parents[2] / "pipeline" / "data" / "category.json"
with open(CATEGORY_FILE, encoding="utf-8") as file:
    CATEGORIES = json.load(file)


@traceable(name="clarify_node", run_type="chain")
def clarify_node(state: ChatState) -> dict:
    intents = state.get("intents") or []
    filters = normalize_search_filters(state.get("filters"))
    pet_profile = state.get("pet_profile") or {}

    pet_type_detected = filters.get("pet_type")
    pet_species_profile = pet_profile.get("species")
    if pet_species_profile == "dog":
        pet_species_kr = "강아지"
    elif pet_species_profile == "cat":
        pet_species_kr = "고양이"
    else:
        pet_species_kr = None

    current_pet_type = pet_type_detected or pet_species_kr
    category = filters.get("category")

    if "recommend" in intents and not current_pet_type:
        if "popularity" in intents:
            question = "인기 있는 상품을 찾으시는군요! 어떤 반려동물(강아지/고양이)을 위한 상품인가요?"
        else:
            question = "어떤 반려동물을 키우고 계세요? 강아지인가요, 고양이인가요?"
    elif "recommend" in intents and state.get("form_hint"):
        form = state["form_hint"]
        pet_text = f"{current_pet_type}용" if current_pet_type else "반려동물용"
        question = f"{form} 상품을 찾으시는군요! {pet_text} 주식(사료)을 찾으시나요, 아니면 간식을 찾으시나요?"
    elif "recommend" in intents and not category:
        pop_str = " 인기 상품" if "popularity" in intents else ""
        pet_text = f"{current_pet_type}를 위한" if current_pet_type else "반려동물을 위한"
        question = f"{pet_text} 어떤{pop_str}을 찾으시나요? (사료, 간식)"
    else:
        question = "반려동물 상품 추천이나 건강 정보 상담을 도와드릴 수 있어요. 궁금한 점이 있으시면 말씀해 주세요!"

    count = state.get("clarification_count", 0) + 1
    logger.info("clarify count=%s question=%s", count, question)
    return {
        "messages": [AIMessage(content=question)],
        "response": question,
        "clarification_count": count,
    }
