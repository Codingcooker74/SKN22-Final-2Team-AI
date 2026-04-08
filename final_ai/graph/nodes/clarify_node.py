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
    
    # 펫 프로필 정보 한글화
    if pet_species_profile == "dog":
        pet_species_kr = "강아지"
    elif pet_species_profile == "cat":
        pet_species_kr = "고양이"
    else:
        pet_species_kr = None

    current_pet_type = pet_type_detected or pet_species_kr
    category = filters.get("category")
    
    # 추천 관련 의도인지 확인
    is_recommend_flow = "recommend" in intents or "popularity" in intents
    pop_str = " 인기 상품" if "popularity" in intents else ""

    # 1. 반려동물 종류(강아지/고양이) 정보가 없는 경우
    if is_recommend_flow and not current_pet_type:
        question = f"반려동물을 위한{pop_str}을 찾으시는군요! 어떤 반려동물(강아지/고양이)을 위한 상품인가요?"
        
    # 2. 카테고리(사료, 간식 등) 정보가 없는 경우
    elif is_recommend_flow and not category:
        pet_text = f"{current_pet_type}를 위한" if current_pet_type else "반려동물을 위한"
        if current_pet_type == "강아지":
            question = f"{pet_text} 어떤{pop_str}을 찾으시나요? (사료, 간식, 용품, 배변용품, 덴탈관)"
        else:    
            question = f"{pet_text} 어떤{pop_str}을 찾으시나요? (사료, 간식, 용품, 모래, 습식관)"

    # 3. 그 외 기본 안내
    else:
        question = "반려동물 상품 추천이나 건강 정보 상담을 도와드릴 수 있어요. 궁금한 점이 있으시면 말씀해 주세요!"

    count = state.get("clarification_count", 0) + 1
    logger.info("clarify count=%s question=%s", count, question)
    return {
        "messages": [AIMessage(content=question)],
        "response": question,
        "clarification_count": count,
    }
