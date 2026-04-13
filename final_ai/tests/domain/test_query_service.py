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

    def test_build_search_query_state_relaxes_query_hints_by_stage(self):
        base_state = {
            "user_input": "6개월 말티즈 요로 습식사료 추천",
            "filters": {"pet_type": "dog", "category": "사료", "subcategory": "습식사료"},
            "pet_profile": {"species": "dog", "breed": "말티즈"},
            "health_concerns": ["urinary"],
            "age_group": "퍼피",
            "is_result_refinement": False,
        }

        strict = build_search_query_state({**base_state, "filter_relaxation_count": 0})
        no_health = build_search_query_state({**base_state, "filter_relaxation_count": 1})
        no_subcategory = build_search_query_state({**base_state, "filter_relaxation_count": 2})
        core_query = build_search_query_state({**base_state, "filter_relaxation_count": 3})

        self.assertIn("요로", strict["search_query"])
        self.assertIn("습식사료", strict["search_query"])
        self.assertIn("말티즈", strict["search_query"])
        self.assertIn("퍼피", strict["search_query"])

        self.assertNotIn("요로", no_health["search_query"])
        self.assertIn("습식사료", no_health["search_query"])

        self.assertNotIn("습식사료", no_subcategory["search_query"])
        self.assertEqual(no_subcategory["filters"], {"pet_type": "강아지", "category": "사료"})

        self.assertEqual(core_query["search_query"], "강아지 사료")
        self.assertEqual(core_query["relaxed_filters"], ["health_concern", "subcategory", "age_group", "breed"])

    def test_build_search_query_state_removes_exclusion_terms_from_refinement_query(self):
        result = build_search_query_state(
            {
                "user_input": "로얄캐닌 제외하고 더 싼 거 보여줘",
                "filters": {"pet_type": "고양이", "category": "사료"},
                "exclusions": {"brands": ["로얄캐닌"]},
                "pet_profile": {"species": "cat"},
                "health_concerns": [],
                "filter_relaxation_count": 0,
                "is_result_refinement": True,
            }
        )

        self.assertNotIn("로얄캐닌", result["search_query"])
        self.assertIn("고양이 사료", result["search_query"])
