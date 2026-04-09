from fastapi import APIRouter, Depends, Query, Response

from final_ai.api.dependencies import RequestAuthContext, get_request_auth_context
from final_ai.api.presenters import bad_request_response, internal_error_response
from final_ai.application.recommendation.service import recommend_products
from final_ai.infrastructure.observability import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/")
async def recommend(
    response: Response,
    query: str = Query(..., min_length=1),
    user_id: str | None = None,
    target_pet_id: str | None = None,
    pet_type: str | None = None,
    breed: str | None = None,
    age: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    brand: str | None = None,
    health_concerns: list[str] | None = Query(default=None),
    allergies: list[str] | None = Query(default=None),
    food_preferences: list[str] | None = Query(default=None),
    budget: int | None = Query(default=None, ge=0),
    limit: int = Query(default=5, ge=1, le=20),
    auth_context: RequestAuthContext = Depends(get_request_auth_context),
):
    response.headers["X-Request-Id"] = auth_context.request_id
    effective_user_id = user_id or auth_context.user_id
    if not query.strip():
        return bad_request_response(
            "추천 질의어가 비어 있습니다.",
            code="missing_query",
            headers={"X-Request-Id": auth_context.request_id},
        )

    try:
        logger.info(
            "recommendation requested",
            extra=auth_context.log_extra(
                user_id=effective_user_id,
                target_pet_id=target_pet_id,
                pet_type=pet_type,
                category=category,
                subcategory=subcategory,
                brand=brand,
            ),
        )
        return recommend_products(
            query=query,
            user_id=effective_user_id,
            target_pet_id=target_pet_id,
            pet_type=pet_type,
            breed=breed,
            age=age,
            category=category,
            subcategory=subcategory,
            brand=brand,
            health_concerns=health_concerns,
            allergies=allergies,
            food_preferences=food_preferences,
            budget=budget,
            limit=limit,
        )
    except Exception as exc:
        logger.exception(
            "recommendation failed",
            extra=auth_context.log_extra(user_id=effective_user_id, target_pet_id=target_pet_id),
        )
        return internal_error_response(
            "추천 상품을 조회하지 못했습니다.",
            code="recommendation_failed",
            details={"reason": str(exc)},
            headers={"X-Request-Id": auth_context.request_id},
        )
