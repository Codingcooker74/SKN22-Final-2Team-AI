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
