from fastapi import APIRouter, Depends, Query, Response

from final_ai.api.dependencies import RequestAuthContext, get_request_auth_context
from final_ai.api.presenters import bad_request_response, internal_error_response
from final_ai.application.recommendation.service import list_products
from final_ai.infrastructure.observability import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/")
async def list_products_route(
    response: Response,
    query: str | None = None,
    pet_type: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    brand: str | None = None,
    budget: int | None = Query(default=None, ge=0),
    include_soldout: bool = False,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    auth_context: RequestAuthContext = Depends(get_request_auth_context),
):
    response.headers["X-Request-Id"] = auth_context.request_id
    if query is not None and not query.strip():
        return bad_request_response(
            "상품 검색어가 비어 있습니다.",
            code="empty_query",
            headers={"X-Request-Id": auth_context.request_id},
        )

    try:
        logger.info(
            "product list requested",
            extra=auth_context.log_extra(query=query, pet_type=pet_type, category=category, subcategory=subcategory),
        )
        return list_products(
            query=query,
            pet_type=pet_type,
            category=category,
            subcategory=subcategory,
            brand=brand,
            budget=budget,
            include_soldout=include_soldout,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.exception("product list failed", extra=auth_context.log_extra(query=query, pet_type=pet_type))
        return internal_error_response(
            "상품 목록을 조회하지 못했습니다.",
            code="product_list_failed",
            details={"reason": str(exc)},
            headers={"X-Request-Id": auth_context.request_id},
        )
