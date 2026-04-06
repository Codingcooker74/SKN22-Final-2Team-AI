import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from final_ai.application.chat.context import ChatContextLoadError, hydrate_chat_request
from final_ai.application.chat.dto import build_chat_execution_request
from final_ai.application.chat.memory import build_memory_payload
from final_ai.application.chat.service import stream_chat_events
from final_ai.contracts.chat import ChatRequest


class _FakeAuthContext:
    def __init__(self, request_id: str, user_id: str | None, session_id: str | None):
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id

    def log_extra(self, **extra):
        payload = {
            "request_id": self.request_id,
            "session_id": self.session_id or "",
            "user_id": self.user_id or "",
        }
        payload.update({key: "" if value is None else str(value) for key, value in extra.items()})
        return payload


class _FakeRequest:
    def __init__(self, auth_context=None, disconnected: bool = False):
        self.state = SimpleNamespace(auth_context=auth_context)
        self._disconnected = disconnected

    async def is_disconnected(self):
        return self._disconnected


class ChatExecutionRequestTests(unittest.TestCase):
    def test_build_chat_execution_request_includes_metadata(self):
        request = ChatRequest(
            message="사료 추천",
            thread_id="thread-1",
            request_id="req-1",
            user_id="user-1",
            target_pet_id="pet-1",
            allergies=["닭"],
            conversation_history=[{"role": "user", "content": "이전 질문"}],
            summary_candidates=[{"message_id": "m-1", "role": "assistant", "content": "이전 요약 대상"}],
            memory_summary="기존 요약",
            dialog_state={
                "intents": ["recommend"],
                "filters": {"pet_type": "강아지", "category": "사료"},
                "clarification_count": 2,
            },
            last_compacted_message_id="m-0",
        )

        execution = build_chat_execution_request(request)

        self.assertEqual(execution.initial_state["user_input"], "사료 추천")
        self.assertEqual(execution.initial_state["allergies"], ["닭"])
        self.assertEqual(execution.initial_state["conversation_history"], [{"role": "user", "content": "이전 질문"}])
        self.assertEqual(
            execution.initial_state["summary_candidates"],
            [{"message_id": "m-1", "role": "assistant", "content": "이전 요약 대상"}],
        )
        self.assertEqual(execution.initial_state["memory_summary"], "기존 요약")
        self.assertEqual(execution.initial_state["last_compacted_message_id"], "m-0")
        self.assertEqual(execution.initial_state["intents"], ["recommend"])
        self.assertEqual(execution.initial_state["filters"], {"pet_type": "강아지", "category": "사료"})
        self.assertEqual(execution.initial_state["clarification_count"], 2)
        self.assertEqual(execution.config["configurable"]["thread_id"], "thread-1")
        self.assertEqual(
            execution.config["metadata"],
            {
                "request_id": "req-1",
                "session_id": "thread-1",
                "user_id": "user-1",
                "target_pet_id": "pet-1",
            },
        )

    def test_build_chat_execution_request_prefers_dialog_state_target_pet_id(self):
        request = ChatRequest(
            message="추천",
            thread_id="thread-1",
            user_id="user-1",
            target_pet_id="pet-session",
            dialog_state={"target_pet_id": "pet-memory"},
        )

        execution = build_chat_execution_request(request)

        self.assertEqual(execution.initial_state["target_pet_id"], "pet-memory")
        self.assertEqual(execution.config["metadata"]["target_pet_id"], "pet-memory")


class ChatContextHydrationTests(unittest.TestCase):
    def test_hydrate_chat_request_loads_context_from_db(self):
        request = ChatRequest(
            thread_id="session-1",
            current_user_message_id="message-1",
            user_id="user-1",
            target_pet_id="pet-session",
        )

        with (
            patch(
                "final_ai.application.chat.context.fetch_chat_context",
                return_value={
                    "message": "DB에서 읽은 질문",
                    "target_pet_id": "pet-db",
                    "profile_context_type": "pet",
                    "conversation_history": [{"role": "user", "content": "이전 질문"}],
                    "summary_candidates": [{"message_id": "old-1", "role": "assistant", "content": "이전 응답"}],
                    "memory_summary": "- 기존 요약",
                    "dialog_state": {
                        "intents": ["recommend"],
                        "filters": {"pet_type": "고양이"},
                        "target_pet_id": "pet-memory",
                    },
                    "last_compacted_message_id": "old-1",
                },
            ),
            patch(
                "final_ai.application.chat.context.get_pet_full_profile",
                return_value={
                    "pet_profile": {"species": "cat"},
                    "health_concerns": ["skin"],
                    "allergies": ["chicken"],
                    "food_preferences": ["dry"],
                },
            ),
        ):
            hydrated = hydrate_chat_request(request)

        self.assertEqual(hydrated.message, "DB에서 읽은 질문")
        self.assertEqual(hydrated.target_pet_id, "pet-memory")
        self.assertEqual(hydrated.conversation_history, [{"role": "user", "content": "이전 질문"}])
        self.assertEqual(
            hydrated.summary_candidates,
            [{"message_id": "old-1", "role": "assistant", "content": "이전 응답"}],
        )
        self.assertEqual(hydrated.memory_summary, "- 기존 요약")
        self.assertEqual(hydrated.dialog_state["intents"], ["recommend"])
        self.assertEqual(hydrated.pet_profile, {"species": "cat"})
        self.assertEqual(hydrated.health_concerns, ["skin"])
        self.assertEqual(hydrated.allergies, ["chicken"])
        self.assertEqual(hydrated.food_preferences, ["dry"])
        self.assertEqual(hydrated.last_compacted_message_id, "old-1")


class StreamChatEventsTests(unittest.TestCase):
    def test_stream_chat_events_emits_info_tokens_products_final_and_done(self):
        request = ChatRequest(
            message="사료 추천",
            thread_id="thread-1",
            request_id="req-1",
            user_id="user-1",
        )
        fake_request = _FakeRequest(auth_context=_FakeAuthContext("req-1", "user-1", "thread-1"))
        
        async def fake_to_thread(*args, **kwargs):
            return {
                "response": "좋은 사료입니다",
                "product_cards": [{"goods_id": "A1"}],
            }

        with (
            patch("final_ai.application.chat.service.get_pet_name_for_user", return_value="초코"),
            patch("final_ai.application.chat.service.asyncio.to_thread", side_effect=fake_to_thread),
        ):
            events = asyncio.run(_collect_events(stream_chat_events(request, fake_request)))

        self.assertEqual(events[0], ("info", {"content": "초코에 어울리는 사료를 찾는 중입니다..."}))
        self.assertEqual("".join(payload["content"] for event_type, payload in events if event_type == "token"), "좋은 사료입니다")
        self.assertEqual(events[-3], ("products", {"cards": [{"goods_id": "A1"}]}))
        self.assertEqual(events[-2][0], "final")
        self.assertEqual(events[-2][1]["message"], "좋은 사료입니다")
        self.assertEqual(events[-2][1]["cards"], [{"goods_id": "A1"}])
        self.assertEqual(events[-2][1]["meta"], {"request_id": "req-1", "session_id": "thread-1"})
        self.assertIn("memory", events[-2][1])
        self.assertEqual(events[-2][1]["memory"]["dialog_state"]["intents"], [])
        self.assertEqual(events[-1], ("done", {}))

    def test_stream_chat_events_yields_error_when_context_load_fails(self):
        request = ChatRequest(thread_id="thread-1", current_user_message_id="message-1", request_id="req-1", user_id="user-1")
        fake_request = _FakeRequest(auth_context=_FakeAuthContext("req-1", "user-1", "thread-1"))

        with patch(
            "final_ai.application.chat.service.hydrate_chat_request",
            side_effect=ChatContextLoadError("missing context"),
        ):
            events = asyncio.run(_collect_events(stream_chat_events(request, fake_request)))

        self.assertEqual(events, [("error", {"message": "대화 문맥을 불러오지 못했습니다. 다시 시도해 주세요."})])

    def test_stream_chat_events_yields_error_event_when_graph_fails(self):
        request = ChatRequest(message="사료 추천", thread_id="thread-1", request_id="req-1")
        fake_request = _FakeRequest()
        
        async def fake_to_thread(*args, **kwargs):
            raise RuntimeError("graph failed")

        with (
            patch("final_ai.application.chat.service.get_pet_name_for_user", return_value="우리 아이"),
            patch("final_ai.application.chat.service.asyncio.to_thread", side_effect=fake_to_thread),
        ):
            events = asyncio.run(_collect_events(stream_chat_events(request, fake_request)))

        self.assertEqual(events[0][0], "info")
        self.assertEqual(events[-1], ("error", {"message": "응답 생성 중 오류가 발생했습니다."}))


class ChatMemoryPayloadTests(unittest.TestCase):
    def test_build_memory_payload_compacts_summary_candidates_and_updates_cursor(self):
        state = {
            "memory_summary": "- 기존 요약",
            "summary_candidates": [
                {"message_id": "m-1", "role": "user", "content": "우리 고양이 알레르기 있는 사료는 피하고 싶어요."},
                {"message_id": "m-2", "role": "assistant", "content": "닭고기 알레르기를 고려한 사료를 추천해드릴게요."},
            ],
            "intents": ["recommend"],
            "filters": {"pet_type": "고양이", "category": "사료"},
            "clarification_count": 1,
        }

        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="- 닭고기 알레르기를 고려한 고양이 사료를 반복적으로 찾고 있음"))]
        )

        with patch("final_ai.application.chat.memory.llm.chat.completions.create", return_value=fake_response):
            payload = build_memory_payload(state)

        self.assertEqual(
            payload["memory_summary"],
            "- 닭고기 알레르기를 고려한 고양이 사료를 반복적으로 찾고 있음",
        )
        self.assertEqual(payload["last_compacted_message_id"], "m-2")


async def _collect_events(event_stream):
    return [event async for event in event_stream]
