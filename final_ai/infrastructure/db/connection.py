import os

import psycopg2

from final_ai.infrastructure.settings import (
    POSTGRES_CONNECT_TIMEOUT_SECONDS,
    POSTGRES_STATEMENT_TIMEOUT_MS,
)


def get_db_connection():
    host = (os.getenv("POSTGRES_HOST") or "localhost").strip() or "localhost"
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "tailtalk_db"),
        user=os.getenv("POSTGRES_USER", "mungnyang"),
        password=os.getenv("POSTGRES_PASSWORD", "finalprojectljs1908"),
        host=host,
        port=os.getenv("POSTGRES_PORT", "5432"),
        connect_timeout=POSTGRES_CONNECT_TIMEOUT_SECONDS,
        options=f"-c statement_timeout={POSTGRES_STATEMENT_TIMEOUT_MS}",
        application_name="tailtalk-fastapi",
    )
