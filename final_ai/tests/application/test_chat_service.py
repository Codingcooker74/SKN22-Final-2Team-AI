import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from final_ai.application.chat.dto import build_chat_execution_request
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
        )

        execution = build_chat_execution_request(request)

        self.assertEqual(execution.initial_state["user_input"], "사료 추천")
        self.assertEqual(execution.initial_state["allergies"], ["닭"])
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
        self.assertEqual(
            events[-2],
            (
                "final",
                {
                    "message": "좋은 사료입니다",
                    "cards": [{"goods_id": "A1"}],
                    "meta": {"request_id": "req-1", "session_id": "thread-1"},
                },
            ),
        )
        self.assertEqual(events[-1], ("done", {}))

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


async def _collect_events(event_stream):
    return [event async for event in event_stream]
