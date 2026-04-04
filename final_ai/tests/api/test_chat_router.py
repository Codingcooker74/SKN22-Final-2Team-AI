import asyncio
import unittest
from fastapi import Response
from unittest.mock import patch

from final_ai.api.routers import chat as chat_router
from final_ai.api.dependencies.auth import RequestAuthContext
from final_ai.contracts.chat import ChatRequest, SessionCreateRequest


async def _collect_streaming_response(response):
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        chunks.append(chunk)
    return "".join(chunks)


class ChatRouterTests(unittest.TestCase):
    def test_chat_streams_events_and_applies_request_headers(self):
        captured = {}

        async def fake_stream_chat_events(req, request):
            captured["request_id"] = req.request_id
            captured["user_id"] = req.user_id
            captured["thread_id"] = req.thread_id
            yield "info", {"content": "start"}
            yield "final", {"message": "done", "cards": [], "meta": {"request_id": "req-123", "session_id": "session-99"}}
            yield "done", {}

        with patch.object(chat_router, "stream_chat_events", fake_stream_chat_events):
            response = asyncio.run(
                chat_router.chat(
                    ChatRequest(message="사료 추천"),
                    request=object(),
                    auth_context=RequestAuthContext(
                        request_id="req-123",
                        user_id="user-7",
                        session_id="session-99",
                        authorization=None,
                        access_token=None,
                        client_ip=None,
                        user_agent=None,
                    ),
                )
            )
            payload = asyncio.run(_collect_streaming_response(response))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-request-id"], "req-123")
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn('"type": "info"', payload)
        self.assertIn('"type": "final"', payload)
        self.assertIn('"type": "done"', payload)
        self.assertEqual(
            captured,
            {
                "request_id": "req-123",
                "user_id": "user-7",
                "thread_id": "session-99",
            },
        )

    def test_create_session_returns_request_id_header(self):
        response = Response()
        payload = asyncio.run(
            chat_router.create_session(
                SessionCreateRequest(title="테스트 대화"),
                response=response,
                auth_context=RequestAuthContext(
                    request_id="req-session",
                    user_id=None,
                    session_id=None,
                    authorization=None,
                    access_token=None,
                    client_ip=None,
                    user_agent=None,
                ),
            )
        )

        self.assertEqual(response.headers["x-request-id"], "req-session")
        self.assertEqual(payload["title"], "테스트 대화")
