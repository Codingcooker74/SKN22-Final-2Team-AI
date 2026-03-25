import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "TailTalk AI API"
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # .env 파일 로드 설정
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
