from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from final_ai.contracts.filters import SearchExclusions, SearchFilters


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    conversation_history: list[dict]
    summary_candidates: list[dict]
    memory_summary: str
    last_compacted_message_id: str | None

    user_id: str | None
    target_pet_id: str | None
    last_recommended_goods_ids: list[str]
    last_search_goods_ids: list[str]
    allowed_goods_ids: list[str]
    # 분해된 하위 작업 리스트 (Query Decomposition 결과물)
    # 형식: [{"pet_name": "초코", "category": "사료", "health_concern": "체중", "age": "노령"}, ...]
    decomposed_tasks: list[dict]
    # 순차 추천 통합 큐: 펫+카테고리를 쌍으로 관리하여 순서 보장
    # 형식: [{"pet_id": "...", "category": "모래"}, ...]
    # pet_id가 None이면 현재 펫 유지
    pending_requests: list[dict]
    is_pet_switched: bool
    is_result_refinement: bool

    switched_pet_name: str | None
    pet_profile: dict | None
    health_concerns: list[str]
    allergies: list[str]
    food_preferences: list[str]

    intents: list[str]
    domain_intent: str | None
    clarification_count: int
    detected_aspect: str | None
    budget: int | None
    min_budget: int | None
    is_pet_override: bool
    pet_mismatch: bool

    age_group: str | None
    refinement_sort: str | None
    search_query: str | None
    filters: SearchFilters
    exclusions: SearchExclusions
    original_filters: SearchFilters
    breed_context: str | None
    health_traits: str | None
    search_results: list[dict]
    reranked_results: list[dict]
    best_reranked_results: list[dict]
    filter_relaxation_count: int
    recommend_retry_pending: bool
    recommendation_limit: int
    effective_filters: SearchFilters
    relaxed_filters: list[str]
    candidate_count_by_stage: dict[str, int]

    domain_contexts: list[str]

    response_mode: str
    response: str
    product_cards: list[dict]
