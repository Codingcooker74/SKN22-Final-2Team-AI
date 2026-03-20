from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def recommend():
    # TODO: Qdrant Hybrid Search 연결
    return {"message": "TODO"}
