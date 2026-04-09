import unittest
from unittest.mock import patch

from final_ai.domain.recommendation.search_service import execute_search_state


class SearchServiceTests(unittest.TestCase):
    @patch("final_ai.domain.recommendation.search_service.hybrid_search_pg", return_value=[])
    def test_execute_search_state_uses_previous_recommendations_for_refinement(
        self,
        mock_hybrid_search_pg,
    ):
        execute_search_state(
            {
                "user_input": "이 중에서 더 싼 거로 보여줘",
                "search_query": "이 중에서 더 싼 거 고양이 사료",
                "filters": {"pet_type": "고양이", "category": "사료"},
                "pet_profile": {"species": "cat"},
                "health_concerns": [],
                "allergies": [],
                "budget": None,
                "filter_relaxation_count": 0,
                "is_result_refinement": True,
                "last_recommended_goods_ids": ["GI1", "GI2"],
                "allowed_goods_ids": [],
            }
        )

        self.assertEqual(mock_hybrid_search_pg.call_count, 1)
        self.assertEqual(mock_hybrid_search_pg.call_args.kwargs["allowed_goods_ids"], ["GI1", "GI2"])

    @patch("final_ai.domain.recommendation.search_service.hybrid_search_pg", return_value=[])
    def test_execute_search_state_prefers_explicit_allowed_goods_ids_over_memory_scope(
        self,
        mock_hybrid_search_pg,
    ):
        execute_search_state(
            {
                "user_input": "이 중에서 더 싼 거로 보여줘",
                "search_query": "이 중에서 더 싼 거 고양이 사료",
                "filters": {"pet_type": "고양이", "category": "사료"},
                "pet_profile": {"species": "cat"},
                "health_concerns": [],
                "allergies": [],
                "budget": None,
                "filter_relaxation_count": 0,
                "is_result_refinement": True,
                "last_recommended_goods_ids": ["GI1", "GI2"],
                "allowed_goods_ids": ["GI9"],
            }
        )

        self.assertEqual(mock_hybrid_search_pg.call_count, 1)
        self.assertEqual(mock_hybrid_search_pg.call_args.kwargs["allowed_goods_ids"], ["GI9"])
