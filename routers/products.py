from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_products():
    # TODO: PostgreSQL 상품 조회
    return {"message": "TODO"}
