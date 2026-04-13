import unittest
from unittest.mock import patch

from final_ai.application.recommendation.service import recommend_products


class RecommendationServiceTests(unittest.TestCase):
    def test_recommend_products_retries_until_five_products_then_stops(self):
        products = [
            {
                "goods_id": f"GI{index}",
                "goods_name": f"강아지 기본 사료 {index}",
                "brand_name": "테스트",
                "price": 10000,
                "discount_price": 9000,
                "rating": 9.5,
                "review_count": 12,
                "thumbnail_url": f"https://example.com/{index}.jpg",
                "product_url": f"https://example.com/{index}",
            }
            for index in range(1, 6)
        ]
        seen_relaxations = []

        def fake_execute_search_state(state):
            seen_relaxations.append(state.get("filter_relaxation_count", 0))
            relaxation = state.get("filter_relaxation_count", 0)
            if relaxation == 0:
                return {"search_results": []}
            if relaxation == 1:
                return {"search_results": products[:3]}
            return {"search_results": products}

        def fake_rerank_search_results(state):
            results = state.get("search_results") or []
            relaxation = state.get("filter_relaxation_count", 0)
            if not results:
                return {
                    "reranked_results": [],
                    "filter_relaxation_count": relaxation + 1,
                    "recommend_retry_pending": relaxation < 4,
                }
            return {
                "reranked_results": results,
                "filter_relaxation_count": relaxation + 1 if len(results) < 5 else relaxation,
                "recommend_retry_pending": len(results) < 5 and relaxation < 4,
            }

        with (
            patch(
                "final_ai.application.recommendation.service.build_profile_state",
                return_value={
                    "pet_profile": {"species": "dog"},
                    "health_concerns": ["요로"],
                    "allergies": [],
                    "food_preferences": [],
                    "budget": None,
                    "pet_mismatch": False,
                    "age_group": "어덜트",
                },
            ),
            patch(
                "final_ai.application.recommendation.service.build_search_query_state",
                return_value={
                    "search_query": "강아지 사료 요로",
                    "filters": {"pet_type": "강아지", "category": "사료"},
                },
            ),
            patch(
                "final_ai.application.recommendation.service.execute_search_state",
                side_effect=fake_execute_search_state,
            ),
            patch(
                "final_ai.application.recommendation.service.rerank_search_results",
                side_effect=fake_rerank_search_results,
            ),
        ):
            result = recommend_products(
                query="사료추천해줘",
                pet_type="강아지",
                category="사료",
                health_concerns=["요로"],
                limit=5,
            )

        self.assertEqual(seen_relaxations, [0, 1, 2])
        self.assertEqual([product["goods_id"] for product in result["products"]], ["GI1", "GI2", "GI3", "GI4", "GI5"])
        self.assertEqual(result["meta"]["filter_relaxation_count"], 2)
        self.assertFalse(result["meta"]["recommend_retry_pending"])
