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
                "(category && %s::varchar[] OR subcategory && %s::varchar[])",
                "EXISTS (SELECT 1 FROM unnest(subcategory) s WHERE s ILIKE %s)",
                "price <= %s",
            ],
        )
        self.assertEqual(params, ["고양이", ["용품", "이동"], ["용품", "이동"], "%이동장/캐리어%", 50000])

    def test_build_product_filter_clauses_uses_overlap_for_multi_value_subcategory(self):
        filters, params = build_product_filter_clauses(
            pet_type=["고양이", "강아지"],
            subcategory=["이동장/캐리어", "스크래쳐/캣타워"],
        )

        self.assertEqual(
            filters,
            [
                "pet_type && %s::varchar[]",
                "subcategory && %s::varchar[]",
            ],
        )
        self.assertEqual(params, [["고양이", "강아지"], ["이동장/캐리어", "스크래쳐/캣타워"]])

    def test_build_product_filter_clauses_uses_varchar_overlap_for_multi_health_concerns(self):
        filters, params = build_product_filter_clauses(
            pet_type="강아지",
            category="사료",
            health_concerns=["관절", "피부"],
        )

        self.assertEqual(
            filters,
            [
                "%s = ANY(pet_type)",
                (
                    "("
                    "EXISTS (SELECT 1 FROM unnest(category) c WHERE c ILIKE %s) OR "
                    "EXISTS (SELECT 1 FROM unnest(subcategory) s WHERE s ILIKE %s)"
                    ")"
                ),
                "health_concern_tags && %s::varchar[]",
            ],
        )
        self.assertEqual(params, ["강아지", "%사료%", "%사료%", ["관절", "피부"]])

    def test_build_product_filter_clauses_supports_exclusion_filters(self):
        filters, params = build_product_filter_clauses(
            pet_type="고양이",
            category="사료",
            exclude_brands=["로얄캐닌"],
            exclude_subcategories=["습식사료"],
            exclude_health_concerns=["요로"],
            exclude_goods_ids=["GI1", "GI2"],
        )

        self.assertEqual(
            filters,
            [
                "%s = ANY(pet_type)",
                (
                    "("
                    "EXISTS (SELECT 1 FROM unnest(category) c WHERE c ILIKE %s) OR "
                    "EXISTS (SELECT 1 FROM unnest(subcategory) s WHERE s ILIKE %s)"
                    ")"
                ),
                "brand_name NOT ILIKE %s",
                "NOT EXISTS (SELECT 1 FROM unnest(subcategory) s WHERE s ILIKE %s)",
                "NOT (%s = ANY(health_concern_tags))",
                "NOT (goods_id = ANY(%s::text[]))",
            ],
        )
        self.assertEqual(
            params,
            ["고양이", "%사료%", "%사료%", "%로얄캐닌%", "%습식사료%", "요로", ["GI1", "GI2"]],
        )
