import psycopg2.extras

from final_ai.infrastructure.db.connection import get_db_connection

RECENT_HISTORY_LIMIT = 12


def _serialize_message_row(row: dict) -> dict:
    return {
        "message_id": str(row["message_id"]),
        "role": row["role"],
        "content": row["content"],
    }


def fetch_chat_context(
    user_id: str,
    session_id: str,
    current_user_message_id: str,
    *,
    recent_history_limit: int = RECENT_HISTORY_LIMIT,
) -> dict | None:
    if not (user_id and session_id and current_user_message_id):
        return None

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT s.session_id, s.user_id, s.target_pet_id, s.profile_context_type,
                   m.summary_text, m.dialog_state, m.last_compacted_message_id
            FROM chat_session AS s
            LEFT JOIN chat_session_memory AS m
              ON m.session_id = s.session_id
            WHERE s.session_id = %s
              AND s.user_id = %s
            LIMIT 1
            """,
            (session_id, user_id),
        )
        session_row = cur.fetchone()
        if not session_row:
            return None

        cur.execute(
            """
            SELECT message_id, role, content, created_at
            FROM chat_message
            WHERE session_id = %s
              AND message_id = %s
              AND role = 'user'
            LIMIT 1
            """,
            (session_id, current_user_message_id),
        )
        current_message = cur.fetchone()
        if not current_message:
            return None

        cur.execute(
            """
            SELECT message_id, role, content, created_at
            FROM chat_message
            WHERE session_id = %s
              AND (created_at < %s OR (created_at = %s AND message_id < %s))
            ORDER BY created_at ASC, message_id ASC
            """,
            (
                session_id,
                current_message["created_at"],
                current_message["created_at"],
                current_message["message_id"],
            ),
        )
        previous_messages = [dict(row) for row in cur.fetchall()]
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    conversation_rows = previous_messages[-recent_history_limit:]
    summary_rows = previous_messages[:-recent_history_limit] if len(previous_messages) > recent_history_limit else []

    last_compacted_message_id = (
        str(session_row["last_compacted_message_id"]) if session_row.get("last_compacted_message_id") else None
    )
    if last_compacted_message_id:
        for index, row in enumerate(summary_rows):
            if str(row["message_id"]) == last_compacted_message_id:
                summary_rows = summary_rows[index + 1 :]
                break

    return {
        "message": current_message["content"],
        "target_pet_id": str(session_row["target_pet_id"]) if session_row.get("target_pet_id") else None,
        "profile_context_type": session_row.get("profile_context_type"),
        "conversation_history": [
            {"role": row["role"], "content": row["content"]}
            for row in conversation_rows
        ],
        "summary_candidates": [_serialize_message_row(row) for row in summary_rows],
        "memory_summary": session_row.get("summary_text") or "",
        "dialog_state": dict(session_row.get("dialog_state") or {}),
        "last_compacted_message_id": last_compacted_message_id,
    }
