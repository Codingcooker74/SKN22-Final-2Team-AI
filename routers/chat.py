import json
import asyncio
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pipeline.chatbot_graph import build_graph

router = APIRouter()

# 앱 기동 시 한 번만 빌드 (MemorySaver 포함)
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    pet_profile: Optional[dict] = None       # {species, breed, age, gender, weight}
    health_concerns: list[str] = []
    allergies: list[str] = []
    food_preferences: list[str] = []


def _sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data}, ensure_ascii=False)}\n\n"


async def _stream(req: ChatRequest):
    graph = get_graph()

    initial_state = {
        "messages": [],
        "user_input": req.message,
        "user_id": None,
        "pet_profile": req.pet_profile,
        "health_concerns": req.health_concerns,
        "allergies": req.allergies,
        "food_preferences": req.food_preferences,
        "intents": [],
        "domain_intent": None,
        "clarification_count": 0,
        "detected_aspect": None,
        "budget": None,
        "search_query": None,
        "filters": None,
        "search_results": [],
        "reranked_results": [],
        "filter_relaxation_count": 0,
        "domain_contexts": [],
        "response": "",
        "product_cards": [],
    }

    config = {"configurable": {"thread_id": req.thread_id}}

    # 그래프 실행 (동기 함수를 스레드풀에서 실행)
    loop = asyncio.get_event_loop()
    try:
        final_state = await loop.run_in_executor(
            None,
            lambda: graph.invoke(initial_state, config=config),
        )
    except Exception as e:
        yield _sse("error", {"message": str(e)})
        return

    response_text = final_state.get("response", "")
    product_cards = final_state.get("product_cards", [])

    # 응답 텍스트를 단어 단위로 스트리밍
    words = response_text.split(" ")
    for i, word in enumerate(words):
        chunk = word if i == 0 else " " + word
        yield _sse("token", {"content": chunk})
        await asyncio.sleep(0.03)

    # 상품 카드 전송
    if product_cards:
        yield _sse("products", {"cards": product_cards})

    yield _sse("done", {})


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
