from typing import Optional, List
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    pet_profile: Optional[dict] = None       # {species, breed, age, gender, weight}
    health_concerns: List[str] = []
    allergies: List[str] = []
    food_preferences: List[str] = []
