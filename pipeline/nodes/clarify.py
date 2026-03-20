import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage
from pipeline.state import ChatState

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
    pet_type = filters.get("pet_type")
    category = filters.get("category")

    if "recommend" in intents and not pet_type:
        q = "어떤 반려동물을 키우고 계세요? 강아지인가요, 고양이인가요?"
    elif "recommend" in intents and not category:
        avail = list(_categories.get(pet_type, {}).keys())
        q = f"{pet_type}를 위한 어떤 상품을 찾으시나요? ({', '.join(avail)})"
    else:
        q = "반려동물 상품 추천이나 건강 정보 상담을 도와드릴 수 있어요. 궁금한 점이 있으시면 말씀해 주세요!"

    count = state.get("clarification_count", 0) + 1
    print(f"[CLARIFY] count={count}: {q}")
    return {
        "messages":            [AIMessage(content=q)],
        "response":            q,
        "clarification_count": count,
    }
