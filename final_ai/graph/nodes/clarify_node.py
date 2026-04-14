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


def _topic_object_phrase(noun: str, suffix: str) -> str:
    text = str(noun or "").strip()
    if not text:
        return suffix
    last_char = text[-1]
    if not ("가" <= last_char <= "힣"):
        return f"{text}{suffix}"
    has_batchim = (ord(last_char) - ord("가")) % 28 != 0
    particle = "을" if has_batchim else "를"
    return f"{text}{particle}"


def _fallback_category_from_text(text: str | None) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    best_match = None
    best_len = 0
    for pet_category_map in CATEGORIES.values():
        for category_name, category_info in pet_category_map.items():
            targets = [category_name, *(category_info.get("aliases") or [])]
            for target in targets:
                if target and target in raw and len(target) > best_len:
                    best_match = category_name
                    best_len = len(target)

    return best_match


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
    fallback_category = _fallback_category_from_text(state.get("user_input"))
    resolved_category = category or fallback_category
    
    # 추천 관련 의도인지 확인
    is_recommend_flow = "recommend" in intents or "popularity" in intents
    pop_str = " 인기 상품" if "popularity" in intents else ""

    # 1. 반려동물 종류(강아지/고양이) 정보가 없는 경우
    if is_recommend_flow and not current_pet_type:
        if resolved_category:
            category_text = _topic_object_phrase(f"{resolved_category}{pop_str}", "을")
            question = f"{category_text} 찾으시는군요! 어떤 반려동물(강아지/고양이)을 위한 상품인가요?"
        else:
            question = "어떤 제품을 찾으시나요? 어떤 반려동물(강아지/고양이)을 위한 상품인가요?"
        
    # 2. 카테고리(사료, 간식 등) 정보가 없는 경우
    elif is_recommend_flow and not category:
        if fallback_category and current_pet_type:
            category_text = _topic_object_phrase(f"{fallback_category}{pop_str}", "을")
            question = f"{current_pet_type}를 위한 {category_text} 찾으시는군요. 예산이나 원하는 조건이 있을까요?"
        elif current_pet_type:
            question = "어떤 제품을 찾으시나요?"
        else:
            question = "어떤 제품을 찾으시나요?"

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
