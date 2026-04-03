import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from final_ai.api.dependencies import RequestAuthContext, get_request_auth_context
from final_ai.api.presenters import sse_event
from final_ai.application.chat.service import stream_chat_events
from final_ai.contracts.chat import ChatRequest, SessionCreateRequest
from final_ai.infrastructure.observability import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _apply_request_context(req: ChatRequest, auth_context: RequestAuthContext) -> ChatRequest:
    updates = {
        "request_id": auth_context.request_id,
        "user_id": req.user_id or auth_context.user_id,
    }
    if (not req.thread_id or req.thread_id == "default") and auth_context.session_id:
        updates["thread_id"] = auth_context.session_id

    if hasattr(req, "model_copy"):
        return req.model_copy(update=updates)
    return req.copy(update=updates)


async def _render_event_stream(req: ChatRequest, request: Request) -> AsyncIterator[str]:
    async for event_type, payload in stream_chat_events(req, request):
        yield sse_event(event_type, payload)


@router.post("/")
async def chat(
    req: ChatRequest,
    request: Request,
    auth_context: RequestAuthContext = Depends(get_request_auth_context),
):
    req = _apply_request_context(req, auth_context)
    logger.info("chat request accepted", extra=auth_context.log_extra(thread_id=req.thread_id))
    return StreamingResponse(
        _render_event_stream(req, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": auth_context.request_id,
        },
    )


@router.post("/sessions/")
async def create_session(
    req: SessionCreateRequest,
    response: Response,
    auth_context: RequestAuthContext = Depends(get_request_auth_context),
):
    session_id = str(uuid.uuid4())
    response.headers["X-Request-Id"] = auth_context.request_id
    logger.info("chat session created", extra=auth_context.log_extra(session_id=session_id))
    return {
        "session_id": session_id,
        "title": req.title or "새 대화",
        "display_date": "오늘",
    }


@router.post("/sessions/{session_id}/messages/")
async def session_chat(
    session_id: str,
    req: ChatRequest,
    request: Request,
    auth_context: RequestAuthContext = Depends(get_request_auth_context),
):
    req.thread_id = session_id
    return await chat(req, request, auth_context)


@router.get("/sessions/{session_id}/messages/")
async def get_messages(
    session_id: str,
    response: Response,
    auth_context: RequestAuthContext = Depends(get_request_auth_context),
):
    response.headers["X-Request-Id"] = auth_context.request_id
    logger.info("chat session messages requested", extra=auth_context.log_extra(session_id=session_id))
    return {"messages": []}


@router.delete("/sessions/{session_id}/")
async def delete_session(
    session_id: str,
    response: Response,
    auth_context: RequestAuthContext = Depends(get_request_auth_context),
):
    response.headers["X-Request-Id"] = auth_context.request_id
    logger.info("chat session deleted", extra=auth_context.log_extra(session_id=session_id))
    return {"status": "success"}
