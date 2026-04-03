import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

LLM_MODEL = "gpt-4o-mini"
LLM_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
POSTGRES_CONNECT_TIMEOUT_SECONDS = int(os.getenv("POSTGRES_CONNECT_TIMEOUT_SECONDS", "5"))
POSTGRES_STATEMENT_TIMEOUT_MS = int(os.getenv("POSTGRES_STATEMENT_TIMEOUT_MS", "20000"))
EMBED_MODEL_NAME = os.getenv("FASTEMBED_MODEL", "intfloat/multilingual-e5-large")
