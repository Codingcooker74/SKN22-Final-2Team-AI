import json
import asyncio
import uuid
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from final_ai.observability import traceable
from final_ai.pipeline.chatbot_graph import build_graph
from final_ai.schemas.chat import ChatRequest

router = APIRouter()

# 앱 기동 시 한 번만 빌드 (MemorySaver 포함)
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


@traceable(name="tailtalk_fastapi_chat_graph", run_type="chain")
def _invoke_graph(initial_state: dict, config: dict) -> dict:
    return get_graph().invoke(initial_state, config=config)

def _json_default(value):
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return str(value)


def _sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False, default=_json_default)}\n\n"

async def _stream(req: ChatRequest):
    initial_state = {
        "user_input": req.message,
        "pet_profile": req.pet_profile,
        "health_concerns": req.health_concerns,
        "allergies": req.allergies,
        "food_preferences": req.food_preferences,
        "user_id": req.user_id,
        "target_pet_id": req.target_pet_id,
    }

    config = {"configurable": {"thread_id": req.thread_id}}

    # [추가] LLM이 응답을 생성하는 동안 보여줄 실시간 멘트 (DB 실제 이름 조회)
    pet_name = "우리 아이"
    if req.user_id:
        try:
            from final_ai.pipeline.utils import get_db_connection
            import psycopg2.extras
            conn = get_db_connection()
            # 펫 테이블에서 user_id에 해당하는 진짜 이름을 가져옵니다.
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                if req.target_pet_id:
                    cur.execute(
                        "SELECT name FROM pet WHERE user_id = %s AND pet_id = %s LIMIT 1",
                        (req.user_id, req.target_pet_id),
                    )
                else:
                    cur.execute(
                        "SELECT name FROM pet WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                        (req.user_id,),
                    )
                row = cur.fetchone()
                if row and row["name"]:
                    pet_name = row["name"]  # DB에 저장된 실제 이름 (예: 초코)
            conn.close()
        except Exception as e:
            print(f"[CHAT_ROUTER] 펫 이름 조회 중 오류: {e}")
    
    # 카테고리 추출
    category = "상품"
    if req.message:
        if "사료" in req.message:   category = "사료"
        elif "간식" in req.message: category = "간식"
        elif "영양제" in req.message: category = "영양제"
        elif "용품" in req.message: category = "용품"

    # 제일 먼저 실시간 멘트 전송 (name에 DB의 실제 이름이 들어감)
    yield _sse("info", {"content": f"{pet_name}에 어울리는 {category}를 찾는 중입니다..."})

    loop = asyncio.get_event_loop()
    try:
        final_state = await loop.run_in_executor(
            None,
            lambda: _invoke_graph(initial_state, config),
        )
    except Exception as e:
        yield _sse("error", {"message": str(e)})
        return

    response_text = final_state.get("response", "")
    product_cards = final_state.get("product_cards", [])

    # 응답 스트리밍
    words = response_text.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == 0 else " " + word
        yield _sse("token", {"content": chunk})
        await asyncio.sleep(0.01)

    if product_cards:
        yield _sse("products", {"cards": product_cards})

    yield _sse("done", {})

# 1. POST / (기본 채팅)
@router.post("/")
async def chat(req: ChatRequest):
    return StreamingResponse(
        _stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# 2. POST /sessions/ (Django의 세션 생성 대응)
class SessionCreateRequest(BaseModel):
    title: Optional[str] = None
    target_pet_id: Optional[str] = None

@router.post("/sessions/")
async def create_session(req: SessionCreateRequest):
    # 실제 DB 세션 생성은 Django가 담당하므로, 여기서는 호환성을 위해 ID만 반환
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "title": req.title or "새 대화",
        "display_date": "오늘"
    }

# 3. POST /sessions/{session_id}/messages/ (Django의 메시지 전송 대응)
@router.post("/sessions/{session_id}/messages/")
async def session_chat(session_id: str, req: ChatRequest):
    # thread_id를 장고의 session_id로 고정하여 상태 유지
    req.thread_id = session_id
    return await chat(req)

# 4. GET /sessions/{session_id}/messages/ (Django의 메시지 조회 대응)
@router.get("/sessions/{session_id}/messages/")
async def get_messages(session_id: str):
    # LangGraph의 checkpoint에서 내역을 가져올 수도 있으나, 
    # 현재는 프론트엔드 UI를 위해 빈 배열 또는 기본 인사를 반환
    return {"messages": []}

# 5. DELETE /sessions/{session_id}/ (Django의 세션 삭제 대응)
@router.delete("/sessions/{session_id}/")
async def delete_session(session_id: str):
    return {"status": "success"}
