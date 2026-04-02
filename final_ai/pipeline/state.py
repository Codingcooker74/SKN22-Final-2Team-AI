from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    # 대화
    messages:   Annotated[list, add_messages]  # HumanMessage / AIMessage
    user_input: str                            # 현재 턴 원문

    # 사용자 / 펫 (API 요청 페이로드에서 주입, DB 조회 없음)
    user_id:          str | None
    target_pet_id:    str | None
    pending_pet_ids:  list[str]      # 추천 대기 중인 펫 ID 목록
    pending_categories: list[str]    # 추천 대기 중인 카테고리 목록
    is_pet_switched:  bool            # 현재 턴에서 펫 전환 발생 여부

    switched_pet_name: str | None    # 전환된 펫의 이름 (응답 출력용)
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
    is_pet_override:     bool        # 채팅 기반 펫 정보 우선 반영 여부
    pet_mismatch:        bool        # 등록된 펫 정보와 채팅 정보 불일치 여부

    # recommend 플로우
    search_query:            str | None
    filters:                 dict | None
    breed_context:           str | None         # 품종별 지식 전체
    health_traits:           str | None         # 품종별 건강 특징 (추천 이유 생성용)
    search_results:            list[dict]
    reranked_results:          list[dict]
    filter_relaxation_count:   int       # 추천 필터 완화 횟수
    recommend_retry_pending:   bool      # RERANK 이후 완화 재검색 필요 여부
    form_hint:                 str | None  # 캔/파우치 등 제형 힌트 (사료/간식 구분 전까지 보관)

    # domain_qa 플로우
    domain_contexts: list[str]   # RAG 검색 결과

    # 최종 출력
    response:      str
    product_cards: list[dict]
