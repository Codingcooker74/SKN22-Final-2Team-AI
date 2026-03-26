from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def recommend():
    # TODO: PostgreSQL hybrid search 결과 연결
    return {"message": "TODO"}
