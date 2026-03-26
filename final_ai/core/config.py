import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TailTalk AI API"
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # PostgreSQL
    postgres_db:       str = os.getenv("POSTGRES_DB",       "tailtalk_db")
    postgres_user:     str = os.getenv("POSTGRES_USER",     "mungnyang")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "finalprojectljs1908")
    postgres_host:     str = os.getenv("POSTGRES_HOST",     "localhost")
    postgres_port:     str = os.getenv("POSTGRES_PORT",     "5432")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # .env 파일 로드 (상위 디렉토리 순서대로 탐색)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
