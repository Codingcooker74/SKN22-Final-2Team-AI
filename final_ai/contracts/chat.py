from pydantic import BaseModel, Field


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


class SessionCreateRequest(BaseModel):
    title: str | None = None
    target_pet_id: str | None = None
