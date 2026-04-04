import unittest

from final_ai.infrastructure.repositories.product_filters import (
    build_product_filter_clauses,
    normalize_filter_values,
)


class ProductFilterTests(unittest.TestCase):
    def test_normalize_filter_values_supports_scalar_and_list(self):
        self.assertEqual(normalize_filter_values("이동장/캐리어"), ["이동장/캐리어"])
        self.assertEqual(
            normalize_filter_values(["이동장/캐리어", "이동장/캐리어", "  ", "급수기"]),
            ["이동장/캐리어", "급수기"],
        )
        self.assertEqual(normalize_filter_values(None), [])

    def test_build_product_filter_clauses_uses_overlap_for_multi_value_category_filters(self):
        filters, params = build_product_filter_clauses(
            pet_type="고양이",
            category=["용품", "이동"],
            subcategory=["이동장/캐리어"],
            budget=50000,
        )

        self.assertEqual(
            filters,
            [
                "%s = ANY(pet_type)",
                "(category && %s::text[] OR subcategory && %s::text[])",
                "%s = ANY(subcategory)",
                "price <= %s",
            ],
        )
        self.assertEqual(params, ["고양이", ["용품", "이동"], ["용품", "이동"], "이동장/캐리어", 50000])

    def test_build_product_filter_clauses_uses_overlap_for_multi_value_subcategory(self):
        filters, params = build_product_filter_clauses(
            pet_type=["고양이", "강아지"],
            subcategory=["이동장/캐리어", "스크래쳐/캣타워"],
        )

        self.assertEqual(
            filters,
            [
                "pet_type && %s::text[]",
                "subcategory && %s::text[]",
            ],
        )
        self.assertEqual(params, [["고양이", "강아지"], ["이동장/캐리어", "스크래쳐/캣타워"]])
