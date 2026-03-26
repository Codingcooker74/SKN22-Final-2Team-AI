import json
from pathlib import Path

from langchain_core.messages import AIMessage
from final_ai.pipeline.state import ChatState

CATEGORY_FILE = Path(__file__).resolve().parents[1] / "data" / "category.json"
with open(CATEGORY_FILE, encoding="utf-8") as f:
    _categories = json.load(f)


def clarify_node(state: ChatState) -> dict:
    """
    재질문 생성 후 END.
    다음 턴에 같은 thread_id로 재호출 → INTENT 재시도 (MemorySaver).
    """
    intents  = state.get("intents") or []
    filters  = state.get("filters") or {}
    pet_profile = state.get("pet_profile") or {}
    
    # 펫 종류 (AI 추출값 또는 프로필 정보)
    pet_type_detected = filters.get("pet_type")
    pet_species_profile = pet_profile.get("species")
    
    if pet_species_profile == "dog":
        pet_species_kr = "강아지"
    elif pet_species_profile == "cat":
        pet_species_kr = "고양이"
    else:
        pet_species_kr = pet_species_profile

    current_pet_type = pet_type_detected or pet_species_kr
    category = filters.get("category")

    if "recommend" in intents and not current_pet_type:
        q = "어떤 반려동물을 키우고 계세요? 강아지인가요, 고양이인가요?"
    elif "recommend" in intents and not category:
        avail_cats = list(_categories.get(current_pet_type, {}).keys())
        if not avail_cats: # fallback
            avail_cats = ["사료", "간식", "용품"]
        q = f"{current_pet_type}를 위한 어떤 상품을 찾으시나요? ({', '.join(avail_cats)})"
    else:
        q = "반려동물 상품 추천이나 건강 정보 상담을 도와드릴 수 있어요. 궁금한 점이 있으시면 말씀해 주세요!"

    count = state.get("clarification_count", 0) + 1
    print(f"[CLARIFY] count={count}: {q}")
    return {
        "messages":            [AIMessage(content=q)],
        "response":            q,
        "clarification_count": count,
    }
