import unittest

from final_ai.contracts.filters import (
    build_search_exclusions,
    build_search_filters,
    normalize_filter_value,
    normalize_search_exclusions,
    normalize_search_filters,
)


class SearchFiltersContractTests(unittest.TestCase):
    def test_normalize_filter_value_returns_first_valid_scalar(self):
        self.assertEqual(normalize_filter_value(" 고양이 "), "고양이")
        self.assertEqual(normalize_filter_value(["", "  ", "이동장/캐리어", "스크래쳐"]), "이동장/캐리어")
        self.assertIsNone(normalize_filter_value({"bad": "shape"}))

    def test_build_search_filters_drops_empty_values(self):
        self.assertEqual(
            build_search_filters(
                pet_type="강아지",
                category="  ",
                subcategory=["", "주식캔"],
            ),
            {
                "pet_type": "강아지",
                "subcategory": "주식캔",
            },
        )

    def test_normalize_search_filters_coerces_legacy_list_payloads(self):
        self.assertEqual(
            normalize_search_filters(
                {
                    "pet_type": ["고양이"],
                    "category": ["용품", "장난감"],
                    "subcategory": ["이동장/캐리어", "스크래쳐/캣타워"],
                }
            ),
            {
                "pet_type": "고양이",
                "category": "용품",
                "subcategory": "이동장/캐리어",
            },
        )

    def test_build_search_exclusions_drops_empty_values_and_normalizes_lists(self):
        self.assertEqual(
            build_search_exclusions(
                brands=["", "로얄캐닌", "로얄캐닌"],
                ingredients="연어",
                keywords=["", "그레인프리"],
            ),
            {
                "brands": ["로얄캐닌"],
                "ingredients": ["연어"],
                "keywords": ["그레인프리"],
            },
        )

    def test_normalize_search_exclusions_coerces_legacy_scalar_payloads(self):
        self.assertEqual(
            normalize_search_exclusions(
                {
                    "brand": "로얄캐닌",
                    "subcategory": ["습식사료", "건식사료"],
                    "keyword": "연어",
                }
            ),
            {
                "brands": ["로얄캐닌"],
                "subcategories": ["습식사료", "건식사료"],
                "keywords": ["연어"],
            },
        )
