import unittest
from unittest.mock import patch

from final_ai.application.recommendation.service import recommend_products


class RecommendationServiceTests(unittest.TestCase):
    def test_recommend_products_retries_when_initial_search_has_no_candidates(self):
        product = {
            "goods_id": "GI1",
            "goods_name": "강아지 기본 사료",
            "brand_name": "테스트",
            "price": 10000,
            "discount_price": 9000,
            "rating": 9.5,
            "review_count": 12,
            "thumbnail_url": "https://example.com/a.jpg",
            "product_url": "https://example.com/a",
        }
        seen_relaxations = []

        def fake_execute_search_state(state):
            seen_relaxations.append(state.get("filter_relaxation_count", 0))
            if state.get("filter_relaxation_count", 0) == 0:
                return {"search_results": []}
            return {"search_results": [product]}

        def fake_rerank_search_results(state):
            if not state.get("search_results"):
                return {
                    "reranked_results": [],
                    "filter_relaxation_count": 1,
                    "recommend_retry_pending": True,
                }
            return {
                "reranked_results": [product],
                "filter_relaxation_count": state.get("filter_relaxation_count", 0),
                "recommend_retry_pending": False,
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

        self.assertEqual(seen_relaxations, [0, 1])
        self.assertEqual(result["products"][0]["goods_id"], "GI1")
        self.assertEqual(result["meta"]["filter_relaxation_count"], 1)
        self.assertFalse(result["meta"]["recommend_retry_pending"])
