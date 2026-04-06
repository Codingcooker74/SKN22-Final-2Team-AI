from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from final_ai.contracts.filters import SearchFilters


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    conversation_history: list[dict]
    memory_summary: str

    user_id: str | None
    target_pet_id: str | None
    pending_pet_ids: list[str]
    pending_categories: list[str]
    is_pet_switched: bool

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
    is_pet_override: bool
    pet_mismatch: bool

    age_group: str | None
    search_query: str | None
    filters: SearchFilters
    breed_context: str | None
    health_traits: str | None
    search_results: list[dict]
    reranked_results: list[dict]
    filter_relaxation_count: int
    recommend_retry_pending: bool
    form_hint: str | None

    domain_contexts: list[str]

    response: str
    product_cards: list[dict]
