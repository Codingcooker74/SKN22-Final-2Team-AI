from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    # 대화
    messages:   Annotated[list, add_messages]  # HumanMessage / AIMessage
    user_input: str                            # 현재 턴 원문

    # 사용자 / 펫 (API 요청 페이로드에서 주입, DB 조회 없음)
    user_id:          str | None
    pet_profile:      dict | None    # species(dog/cat), breed, age, weight, gender
    health_concerns:  list[str]      # PET_HEALTH_CONCERN
    allergies:        list[str]      # PET_ALLERGY
    food_preferences: list[str]      # PET_FOOD_PREFERENCE

    # 의도 분류
    intents:             list[str]   # ["recommend"] / ["domain_qa"] / ["domain_qa","recommend"] / ["unclear"]
    domain_intent:       str | None  # health_disease / care_management / nutrition_diet / behavior_psychology / travel
    clarification_count: int
    detected_aspect:     str | None  # ABSA 속성
    budget:              int | None

    # recommend 플로우
    search_query:            str | None
    filters:                 dict | None
    search_results:          list[dict]
    reranked_results:        list[dict]
    filter_relaxation_count: int        # RERANK→QUERY 루프 횟수 (최대 1)

    # domain_qa 플로우
    domain_contexts: list[str]   # RAG 검색 결과

    # 최종 출력
    response:      str
    product_cards: list[dict]
