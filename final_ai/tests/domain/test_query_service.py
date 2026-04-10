import unittest

from final_ai.domain.recommendation.query_service import build_search_query_state


class QueryServiceTests(unittest.TestCase):
    def test_build_search_query_state_normalizes_health_concerns_to_korean(self):
        result = build_search_query_state(
            {
                "user_input": "사료 추천",
                "filters": {"pet_type": "dog", "category": "사료"},
                "pet_profile": {"species": "dog", "breed": "진돗개"},
                "health_concerns": ["digestion"],
                "age_group": "퍼피",
                "filter_relaxation_count": 0,
                "is_result_refinement": False,
            }
        )

        self.assertIn("소화", result["search_query"])
        self.assertNotIn("digestion", result["search_query"])
