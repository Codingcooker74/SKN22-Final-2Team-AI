import unittest
from unittest.mock import patch

from final_ai.domain.recommendation.search_service import execute_search_state


class SearchServiceTests(unittest.TestCase):
    @patch("final_ai.domain.recommendation.search_service.hybrid_search_pg", return_value=[])
    def test_execute_search_state_normalizes_health_concerns_for_search(
        self,
        mock_hybrid_search_pg,
    ):
        execute_search_state(
            {
                "user_input": "사료 추천",
                "search_query": "강아지 진돗개 사료 digestion 퍼피",
                "filters": {"pet_type": "강아지", "category": "사료"},
                "pet_profile": {"species": "dog"},
                "health_concerns": ["digestion"],
                "allergies": [],
                "budget": None,
                "filter_relaxation_count": 0,
                "is_result_refinement": False,
                "allowed_goods_ids": [],
            }
        )

        self.assertEqual(mock_hybrid_search_pg.call_count, 1)
        self.assertEqual(mock_hybrid_search_pg.call_args.kwargs["health_concerns"], ["소화"])

    @patch("final_ai.domain.recommendation.search_service.hybrid_search_pg", return_value=[])
    def test_execute_search_state_relaxes_health_concern_filter_after_retry(
        self,
        mock_hybrid_search_pg,
    ):
        execute_search_state(
            {
                "user_input": "사료 추천",
                "search_query": "강아지 시츄 사료 요로",
                "filters": {"pet_type": "강아지", "category": "사료"},
                "pet_profile": {"species": "dog"},
                "health_concerns": ["urinary"],
                "allergies": [],
                "budget": None,
                "filter_relaxation_count": 1,
                "is_result_refinement": False,
                "allowed_goods_ids": [],
            }
        )

        self.assertEqual(mock_hybrid_search_pg.call_count, 1)
        self.assertEqual(mock_hybrid_search_pg.call_args.kwargs["health_concerns"], [])

    @patch("final_ai.domain.recommendation.search_service.hybrid_search_pg", return_value=[])
    def test_execute_search_state_relaxes_subcategory_from_second_retry(
        self,
        mock_hybrid_search_pg,
    ):
        result = execute_search_state(
            {
                "user_input": "습식사료 추천",
                "search_query": "강아지 사료",
                "filters": {"pet_type": "강아지", "category": "사료", "subcategory": "습식사료"},
                "pet_profile": {"species": "dog"},
                "health_concerns": ["피부"],
                "allergies": [],
                "budget": None,
                "filter_relaxation_count": 2,
                "is_result_refinement": False,
                "allowed_goods_ids": [],
            }
        )

        self.assertIsNone(mock_hybrid_search_pg.call_args.kwargs["subcategory"])
        self.assertEqual(mock_hybrid_search_pg.call_args.kwargs["health_concerns"], [])
        self.assertEqual(result["relaxed_filters"], ["health_concern", "subcategory"])

    @patch(
        "final_ai.domain.recommendation.search_service.hybrid_search_pg",
        return_value=[
            {
                "goods_id": "GI1",
                "goods_name": "강아지 퍼피 사료",
                "category": ["사료"],
                "subcategory": ["퍼피"],
                "health_concern_tags": [],
            }
        ],
    )
    def test_execute_search_state_skips_age_filter_from_third_retry(
        self,
        mock_hybrid_search_pg,
    ):
        strict_result = execute_search_state(
            {
                "user_input": "사료 추천",
                "search_query": "강아지 사료",
                "filters": {"pet_type": "강아지", "category": "사료"},
                "pet_profile": {"species": "dog", "breed": "말티즈"},
                "health_concerns": [],
                "allergies": [],
                "budget": None,
                "age_group": "어덜트",
                "filter_relaxation_count": 0,
                "is_result_refinement": False,
                "allowed_goods_ids": [],
            }
        )
        relaxed_result = execute_search_state(
            {
                "user_input": "사료 추천",
                "search_query": "강아지 사료",
                "filters": {"pet_type": "강아지", "category": "사료"},
                "pet_profile": {"species": "dog", "breed": "말티즈"},
                "health_concerns": [],
                "allergies": [],
                "budget": None,
                "age_group": "어덜트",
                "filter_relaxation_count": 3,
                "is_result_refinement": False,
                "allowed_goods_ids": [],
            }
        )

        self.assertEqual(strict_result["search_results"], [])
        self.assertEqual(len(relaxed_result["search_results"]), 1)
        self.assertEqual(relaxed_result["relaxed_filters"], ["age_group", "breed"])

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

    @patch("final_ai.domain.recommendation.search_service.hybrid_search_pg", return_value=[])
    def test_execute_search_state_passes_structured_exclusions_to_hybrid_search(
        self,
        mock_hybrid_search_pg,
    ):
        execute_search_state(
            {
                "user_input": "브랜드 제외 추천",
                "search_query": "고양이 사료",
                "filters": {"pet_type": "고양이", "category": "사료"},
                "exclusions": {"brands": ["로얄캐닌"], "health_concerns": ["요로"], "goods_ids": ["GI1"]},
                "pet_profile": {"species": "cat"},
                "health_concerns": [],
                "allergies": [],
                "budget": None,
                "filter_relaxation_count": 0,
                "is_result_refinement": False,
                "allowed_goods_ids": [],
            }
        )

        self.assertEqual(mock_hybrid_search_pg.call_args.kwargs["exclude_brands"], ["로얄캐닌"])
        self.assertEqual(mock_hybrid_search_pg.call_args.kwargs["exclude_health_concerns"], ["요로"])
        self.assertEqual(mock_hybrid_search_pg.call_args.kwargs["exclude_goods_ids"], ["GI1"])

    @patch(
        "final_ai.domain.recommendation.search_service.hybrid_search_pg",
        return_value=[
            {
                "goods_id": "GI1",
                "goods_name": "로얄캐닌 고양이 사료",
                "brand_name": "로얄캐닌",
                "category": ["사료"],
                "subcategory": ["건식사료"],
                "health_concern_tags": ["요로"],
                "main_ingredients": ["닭"],
            },
            {
                "goods_id": "GI2",
                "goods_name": "기타브랜드 고양이 사료",
                "brand_name": "기타브랜드",
                "category": ["사료"],
                "subcategory": ["건식사료"],
                "health_concern_tags": [],
                "main_ingredients": ["닭"],
            },
        ],
    )
    def test_execute_search_state_filters_excluded_candidates_after_search(
        self,
        _mock_hybrid_search_pg,
    ):
        result = execute_search_state(
            {
                "user_input": "브랜드 제외 추천",
                "search_query": "고양이 사료",
                "filters": {"pet_type": "고양이", "category": "사료"},
                "exclusions": {"brands": ["로얄캐닌"]},
                "pet_profile": {"species": "cat"},
                "health_concerns": [],
                "allergies": [],
                "budget": None,
                "filter_relaxation_count": 0,
                "is_result_refinement": False,
                "allowed_goods_ids": [],
            }
        )

        self.assertEqual([item["goods_id"] for item in result["search_results"]], ["GI2"])
