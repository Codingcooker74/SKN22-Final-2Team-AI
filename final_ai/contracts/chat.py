from typing import Any

from pydantic import BaseModel, Field


class ConversationHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    request_id: str | None = None
    user_id: str | None = None
    target_pet_id: str | None = None
    pet_profile: dict | None = None
    health_concerns: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    food_preferences: list[str] = Field(default_factory=list)
    conversation_history: list[ConversationHistoryItem] = Field(default_factory=list)
    memory_summary: str = ""
    dialog_state: dict[str, Any] = Field(default_factory=dict)


class SessionCreateRequest(BaseModel):
    title: str | None = None
    target_pet_id: str | None = None
