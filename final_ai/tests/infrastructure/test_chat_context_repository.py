from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

from final_ai.infrastructure.repositories import chat_context_repository


class _FakeCursor:
    def __init__(self, *, fetchone_results=None, fetchall_results=None):
        self._fetchone_results = list(fetchone_results or [])
        self._fetchall_results = list(fetchall_results or [])
        self.executed = []

    def execute(self, query, params):
        self.executed.append((query, params))

    def fetchone(self):
        return self._fetchone_results.pop(0)

    def fetchall(self):
        return self._fetchall_results.pop(0)

    def close(self):
        return None


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self, **kwargs):
        return self._cursor

    def close(self):
        return None


class ChatContextRepositoryTests(unittest.TestCase):
    def test_fetch_chat_context_splits_recent_history_and_summary_candidates(self):
        now = datetime(2026, 4, 6, 12, 0, 0)
        previous_messages = [
            {
                "message_id": f"m-{index + 1}",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"이전 대화 {index + 1}",
                "created_at": now - timedelta(minutes=20 - index),
            }
            for index in range(14)
        ]
        cursor = _FakeCursor(
            fetchone_results=[
                {
                    "session_id": "session-1",
                    "user_id": "user-1",
                    "target_pet_id": "pet-1",
                    "profile_context_type": "pet",
                    "summary_text": "기존 요약",
                    "dialog_state": {"intents": ["recommend"]},
                    "last_compacted_message_id": "m-1",
                },
                {
                    "message_id": "current-message",
                    "role": "user",
                    "content": "현재 질문",
                    "created_at": now,
                },
            ],
            fetchall_results=[previous_messages],
        )
        connection = _FakeConnection(cursor)

        with patch.object(chat_context_repository, "get_db_connection", return_value=connection):
            context = chat_context_repository.fetch_chat_context("user-1", "session-1", "current-message")

        self.assertEqual(context["message"], "현재 질문")
        self.assertEqual(context["target_pet_id"], "pet-1")
        self.assertEqual(context["memory_summary"], "기존 요약")
        self.assertEqual(context["dialog_state"], {"intents": ["recommend"]})
        self.assertEqual(len(context["conversation_history"]), 12)
        self.assertEqual(context["conversation_history"][0]["content"], "이전 대화 3")
        self.assertEqual(
            context["summary_candidates"],
            [{"message_id": "m-2", "role": "assistant", "content": "이전 대화 2"}],
        )
        self.assertEqual(context["last_compacted_message_id"], "m-1")
