from typing import Optional, List
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    user_id: Optional[str] = None           # 로그인 시 전달됨
    pet_profile: Optional[dict] = None       # {species, breed, age, gender, weight}
    health_concerns: List[str] = []
    allergies: List[str] = []
    food_preferences: List[str] = []
