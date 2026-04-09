import unittest

from final_ai.domain.recommendation.rerank_service import rerank_search_results


class RerankServiceTests(unittest.TestCase):
    def test_rerank_search_results_prioritizes_scored_candidates(self):
        state = {
            "search_results": [
                {
                    "goods_id": "A",
                    "_score": 0.1,
                    "popularity_score": 0.1,
                    "sentiment_avg": 0.1,
                    "repeat_rate": 0.1,
                    "health_concern_tags": [],
                },
                {
                    "goods_id": "B",
                    "_score": 0.2,
                    "popularity_score": 0.9,
                    "sentiment_avg": 0.9,
                    "repeat_rate": 0.9,
                    "health_concern_tags": ["눈물"],
                },
                {
                    "goods_id": "C",
                    "_score": 0.05,
                    "popularity_score": 0.5,
                    "sentiment_avg": 0.4,
                    "repeat_rate": 0.4,
                    "health_concern_tags": [],
                },
            ],
            "detected_aspect": None,
            "intents": ["recommend"],
            "filter_relaxation_count": 0,
            "health_concerns": ["눈물"],
            "health_traits": "",
        }

        result = rerank_search_results(state)

        self.assertEqual(result["reranked_results"][0]["goods_id"], "B")
        self.assertFalse(result["recommend_retry_pending"])
        self.assertEqual(result["filter_relaxation_count"], 0)

    def test_rerank_search_results_requests_retry_when_empty(self):
        result = rerank_search_results(
            {
                "search_results": [],
                "filter_relaxation_count": 0,
                "intents": ["recommend"],
            }
        )

        self.assertEqual(result["reranked_results"], [])
        self.assertTrue(result["recommend_retry_pending"])
        self.assertEqual(result["filter_relaxation_count"], 1)

    def test_rerank_search_results_applies_price_low_sort_for_refinement(self):
        result = rerank_search_results(
            {
                "search_results": [
                    {
                        "goods_id": "A",
                        "goods_name": "상품 A 1kg",
                        "_score": 0.9,
                        "price": 12000,
                        "discount_price": 11000,
                        "popularity_score": 10,
                        "sentiment_avg": 0.9,
                        "repeat_rate": 0.6,
                        "health_concern_tags": [],
                    },
                    {
                        "goods_id": "B",
                        "goods_name": "상품 B 1kg",
                        "_score": 0.4,
                        "price": 9000,
                        "discount_price": 8000,
                        "popularity_score": 5,
                        "sentiment_avg": 0.3,
                        "repeat_rate": 0.2,
                        "health_concern_tags": [],
                    },
                ],
                "detected_aspect": None,
                "intents": ["recommend"],
                "filter_relaxation_count": 0,
                "health_concerns": [],
                "health_traits": "",
                "is_result_refinement": True,
                "refinement_sort": "price_low",
            }
        )

        self.assertEqual(result["reranked_results"][0]["goods_id"], "B")

    def test_rerank_search_results_applies_popularity_sort_for_refinement(self):
        result = rerank_search_results(
            {
                "search_results": [
                    {
                        "goods_id": "A",
                        "goods_name": "상품 A 1kg",
                        "_score": 0.9,
                        "price": 10000,
                        "discount_price": 10000,
                        "popularity_score": 10,
                        "review_count": 30,
                        "sentiment_avg": 0.9,
                        "repeat_rate": 0.6,
                        "health_concern_tags": [],
                    },
                    {
                        "goods_id": "B",
                        "goods_name": "상품 B 1kg",
                        "_score": 0.4,
                        "price": 10000,
                        "discount_price": 10000,
                        "popularity_score": 25,
                        "review_count": 10,
                        "sentiment_avg": 0.3,
                        "repeat_rate": 0.2,
                        "health_concern_tags": [],
                    },
                ],
                "detected_aspect": None,
                "intents": ["recommend"],
                "filter_relaxation_count": 0,
                "health_concerns": [],
                "health_traits": "",
                "is_result_refinement": True,
                "refinement_sort": "popularity",
            }
        )

        self.assertEqual(result["reranked_results"][0]["goods_id"], "B")
