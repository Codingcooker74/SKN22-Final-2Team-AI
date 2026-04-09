import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from final_ai.domain.intent.service import classify_intent
from final_ai.harness import (
    RecommendationCase,
    RecommendationVariant,
    StateCase,
    StateVariant,
    compare_harness_reports,
    load_recommendation_cases,
    load_state_cases,
    run_recommendation_case,
    run_state_case,
    score_recommendation_result,
    score_state_result,
)


class HarnessLoaderTests(unittest.TestCase):
    def test_loaders_parse_jsonl_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            recommend_path = Path(tmp_dir) / "recommend.jsonl"
            state_path = Path(tmp_dir) / "state.jsonl"

            recommend_path.write_text(
                json.dumps(
                    {
                        "case_id": "recommend-1",
                        "query": "강아지 간식 추천",
                        "pet_type": "강아지",
                        "category": "간식",
                        "expect": {"min_results": 1},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            state_path.write_text(
                json.dumps(
                    {
                        "case_id": "state-1",
                        "message": "고양이 사료 추천해줘",
                        "expect_state": {
                            "pet_type": "고양이",
                            "category": "사료",
                            "route": ["general", "query"],
                            "min_decomposed_tasks": 1,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            recommend_cases = load_recommendation_cases(recommend_path)
            state_cases = load_state_cases(state_path)

        self.assertEqual(len(recommend_cases), 1)
        self.assertEqual(recommend_cases[0].case_id, "recommend-1")
        self.assertEqual(recommend_cases[0].expect.min_results, 1)

        self.assertEqual(len(state_cases), 1)
        self.assertEqual(state_cases[0].case_id, "state-1")
        self.assertEqual(state_cases[0].expect_state.category, "사료")
        self.assertEqual(state_cases[0].expect_state.route, ["general", "query"])
        self.assertEqual(state_cases[0].expect_state.min_decomposed_tasks, 1)


class RecommendationHarnessTests(unittest.TestCase):
    def test_recommendation_runner_collects_intermediate_state_and_scores(self):
        case = RecommendationCase.from_dict(
            {
                "case_id": "recommend-dental",
                "query": "강아지 덴탈껌 추천",
                "pet_type": "강아지",
                "category": "간식",
                "subcategory": "덴탈껌",
                "expect": {
                    "min_results": 1,
                    "allowed_pet_types": ["강아지"],
                    "allowed_categories": ["간식"],
                    "max_filter_relaxation_count": 0,
                    "relevant_goods_ids": ["A1"],
                },
            }
        )

        def build_profile_state_fn(state):
            return {
                "pet_profile": {"species": "dog", "name": "초코"},
                "age_group": "어덜트",
                "health_concerns": state.get("health_concerns") or [],
                "allergies": state.get("allergies") or [],
                "food_preferences": state.get("food_preferences") or [],
                "budget": state.get("budget"),
                "pet_mismatch": False,
            }

        def build_query_state_fn(state):
            return {
                "search_query": "강아지 간식 덴탈껌",
                "filters": {"pet_type": "강아지", "category": "간식", "subcategory": "덴탈껌"},
            }

        search_results = [
            {
                "goods_id": "A1",
                "goods_name": "덴탈껌 100g",
                "pet_type": ["강아지"],
                "category": ["간식"],
                "subcategory": ["덴탈껌"],
                "_score": 0.9,
                "popularity_score": 10,
                "sentiment_avg": 4.9,
                "repeat_rate": 0.4,
                "thumbnail_url": "https://example.com/a1.jpg",
                "product_url": "https://example.com/a1",
                "price": 1000,
                "discount_price": 900,
                "review_count": 20,
                "rating": 4.8,
            }
        ]

        def execute_search_state_fn(_state):
            return {"search_results": search_results}

        def rerank_search_results_fn(_state):
            return {
                "reranked_results": [{**search_results[0], "rerank_score": 0.95}],
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
            }

        variant = RecommendationVariant(
            name="test-variant",
            build_profile_state_fn=build_profile_state_fn,
            build_search_query_state_fn=build_query_state_fn,
            execute_search_state_fn=execute_search_state_fn,
            rerank_search_results_fn=rerank_search_results_fn,
        )

        result = run_recommendation_case(case, variant)
        score = score_recommendation_result(result)

        self.assertIsNone(result.error)
        self.assertEqual(result.search_query, "강아지 간식 덴탈껌")
        self.assertEqual(result.search_results_count, 1)
        self.assertEqual(result.products[0]["goods_id"], "A1")
        self.assertTrue(score.passed)
        self.assertEqual(score.metrics["hit_at_3"], 1.0)
        self.assertEqual(score.metrics["mrr_at_5"], 1.0)

    def test_recommendation_score_fails_on_policy_violation(self):
        case = RecommendationCase.from_dict(
            {
                "case_id": "recommend-policy-fail",
                "query": "강아지 사료 추천",
                "pet_type": "강아지",
                "category": "사료",
                "expect": {
                    "must_exclude_goods_ids": ["B2"],
                    "max_duplicate_base_products": 1,
                },
            }
        )
        result = run_recommendation_case(
            case,
            RecommendationVariant(
                name="policy-fail",
                build_profile_state_fn=lambda state: {},
                build_search_query_state_fn=lambda state: {"search_query": state["user_input"], "filters": state["filters"]},
                execute_search_state_fn=lambda state: {
                    "search_results": [
                        {
                            "goods_id": "B1",
                            "goods_name": "사료 1kg",
                            "pet_type": ["강아지"],
                            "category": ["사료"],
                        },
                        {
                            "goods_id": "B2",
                            "goods_name": "사료 2kg",
                            "pet_type": ["강아지"],
                            "category": ["사료"],
                        },
                    ]
                },
                rerank_search_results_fn=lambda state: {
                    "reranked_results": state["search_results"],
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                },
            ),
        )

        score = score_recommendation_result(result)

        self.assertFalse(score.passed)
        self.assertTrue(any("must_exclude_goods_ids_failed" in failure for failure in score.policy_failures))
        self.assertTrue(any("duplicate_base_product_failed" in failure for failure in score.policy_failures))

    def test_recommendation_score_fails_on_budget_and_allergy_policy(self):
        case = RecommendationCase.from_dict(
            {
                "case_id": "recommend-budget-allergy-fail",
                "query": "고양이 간식 추천",
                "pet_type": "고양이",
                "category": "간식",
                "expect": {
                    "max_effective_price": 20000,
                    "forbidden_ingredient_keywords": ["닭", "chicken"],
                },
            }
        )
        result = run_recommendation_case(
            case,
            RecommendationVariant(
                name="budget-allergy-fail",
                build_profile_state_fn=lambda state: {},
                build_search_query_state_fn=lambda state: {"search_query": state["user_input"], "filters": state["filters"]},
                execute_search_state_fn=lambda state: {
                    "search_results": [
                        {
                            "goods_id": "C1",
                            "goods_name": "치킨 트릿",
                            "pet_type": ["고양이"],
                            "category": ["간식"],
                            "price": 25000,
                            "discount_price": 23000,
                            "main_ingredients": ["Chicken Meat", "Salmon"],
                        }
                    ]
                },
                rerank_search_results_fn=lambda state: {
                    "reranked_results": state["search_results"],
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                },
            ),
        )

        score = score_recommendation_result(result)

        self.assertFalse(score.passed)
        self.assertTrue(any("max_effective_price_failed" in failure for failure in score.policy_failures))
        self.assertTrue(any("forbidden_ingredient_keywords_failed" in failure for failure in score.policy_failures))

    def test_recommendation_score_fails_on_brand_filter_violation(self):
        case = RecommendationCase.from_dict(
            {
                "case_id": "recommend-brand-fail",
                "query": "로얄캐닌 고양이 사료 추천",
                "pet_type": "고양이",
                "category": "사료",
                "brand": "로얄캐닌",
                "expect": {"min_results": 1},
            }
        )
        result = run_recommendation_case(
            case,
            RecommendationVariant(
                name="brand-fail",
                build_profile_state_fn=lambda state: {},
                build_search_query_state_fn=lambda state: {"search_query": state["user_input"], "filters": state["filters"]},
                execute_search_state_fn=lambda state: {
                    "search_results": [
                        {
                            "goods_id": "R1",
                            "goods_name": "고양이 사료",
                            "brand_name": "오리젠",
                            "pet_type": ["고양이"],
                            "category": ["사료"],
                            "price": 10000,
                            "discount_price": 9000,
                        }
                    ]
                },
                rerank_search_results_fn=lambda state: {
                    "reranked_results": state["search_results"],
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                },
            ),
        )

        score = score_recommendation_result(result)

        self.assertFalse(score.passed)
        self.assertTrue(any("brand_filter_failed" in failure for failure in score.policy_failures))

    def test_recommendation_score_fails_on_allowed_goods_scope_violation(self):
        case = RecommendationCase.from_dict(
            {
                "case_id": "recommend-scope-fail",
                "query": "이 중에서 더 싼 거",
                "pet_type": "고양이",
                "category": "사료",
                "is_result_refinement": True,
                "last_recommended_goods_ids": ["GI1", "GI2"],
                "expect": {"allowed_goods_ids": ["GI1", "GI2"]},
            }
        )
        result = run_recommendation_case(
            case,
            RecommendationVariant(
                name="scope-fail",
                build_profile_state_fn=lambda state: {},
                build_search_query_state_fn=lambda state: {"search_query": state["user_input"], "filters": state["filters"]},
                execute_search_state_fn=lambda state: {
                    "search_results": [
                        {
                            "goods_id": "GI3",
                            "goods_name": "범위 밖 상품",
                            "pet_type": ["고양이"],
                            "category": ["사료"],
                        }
                    ]
                },
                rerank_search_results_fn=lambda state: {
                    "reranked_results": state["search_results"],
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                },
            ),
        )

        score = score_recommendation_result(result)

        self.assertFalse(score.passed)
        self.assertTrue(any("allowed_goods_ids_failed" in failure for failure in score.policy_failures))

    def test_recommendation_score_fails_on_top_goods_prefix_violation(self):
        case = RecommendationCase.from_dict(
            {
                "case_id": "recommend-top-prefix-fail",
                "query": "이 중에서 더 싼 거",
                "pet_type": "고양이",
                "category": "사료",
                "is_result_refinement": True,
                "expect": {"top_goods_ids_prefix": ["GI1"]},
            }
        )
        result = run_recommendation_case(
            case,
            RecommendationVariant(
                name="top-prefix-fail",
                build_profile_state_fn=lambda state: {},
                build_search_query_state_fn=lambda state: {"search_query": state["user_input"], "filters": state["filters"]},
                execute_search_state_fn=lambda state: {
                    "search_results": [
                        {"goods_id": "GI2", "goods_name": "상품 2", "pet_type": ["고양이"], "category": ["사료"]}
                    ]
                },
                rerank_search_results_fn=lambda state: {
                    "reranked_results": state["search_results"],
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                },
            ),
        )

        score = score_recommendation_result(result)

        self.assertFalse(score.passed)
        self.assertTrue(any("top_goods_ids_prefix_failed" in failure for failure in score.policy_failures))

    def test_recommendation_runner_supports_synthetic_pet_profiles(self):
        case = RecommendationCase.from_dict(
            {
                "case_id": "recommend-profiled-cat-food",
                "query": "사료 추천",
                "user_id": "user-1",
                "target_pet_id": "11",
                "category": "사료",
                "pet_rows": {
                    "11": {
                        "pet_id": "11",
                        "user_id": "user-1",
                        "name": "바나나",
                        "species": "cat",
                        "breed": "코숏",
                        "age_years": 3,
                        "age_months": 0,
                        "weight_kg": 4.2,
                        "gender": "F",
                        "budget_range": "under_5",
                    }
                },
                "pet_preferences_by_pet_id": {
                    "11": {
                        "health_concerns": ["요로"],
                        "allergies": ["닭"],
                        "food_preferences": ["습식"],
                    }
                },
                "expect": {
                    "min_results": 1,
                    "search_query_contains": ["고양이", "사료", "요로"],
                },
            }
        )

        search_results = [
            {
                "goods_id": "P1",
                "goods_name": "요로 케어 사료",
                "pet_type": ["고양이"],
                "category": ["사료"],
                "price": 12000,
                "discount_price": 9900,
                "review_count": 10,
                "rating": 4.8,
            }
        ]

        result = run_recommendation_case(
            case,
            RecommendationVariant(
                name="profile-stub",
                execute_search_state_fn=lambda state: {"search_results": search_results},
                rerank_search_results_fn=lambda state: {
                    "reranked_results": state["search_results"],
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                },
            ),
        )
        score = score_recommendation_result(result)

        self.assertIsNone(result.error)
        self.assertEqual(result.profile_state["pet_profile"]["name"], "바나나")
        self.assertEqual(result.profile_state["budget"], 50000)
        self.assertEqual(result.profile_state["health_concerns"], ["요로"])
        self.assertEqual(result.profile_state["allergies"], ["닭"])
        self.assertEqual(result.query_state["filters"]["pet_type"], "고양이")
        self.assertTrue(score.passed)

    def test_recommendation_runner_keeps_explicit_budget_over_pet_profile_budget(self):
        case = RecommendationCase.from_dict(
            {
                "case_id": "recommend-explicit-budget",
                "query": "사료 추천",
                "user_id": "user-1",
                "target_pet_id": "21",
                "category": "사료",
                "budget": 20000,
                "pet_rows": {
                    "21": {
                        "pet_id": "21",
                        "user_id": "user-1",
                        "name": "초코",
                        "species": "dog",
                        "breed": "말티즈",
                        "age_years": 3,
                        "age_months": 0,
                        "weight_kg": 3.6,
                        "gender": "M",
                        "budget_range": "under_5",
                    }
                },
                "pet_preferences_by_pet_id": {
                    "21": {
                        "health_concerns": ["관절"],
                        "allergies": [],
                        "food_preferences": [],
                    }
                },
                "expect": {"min_results": 1},
            }
        )

        result = run_recommendation_case(
            case,
            RecommendationVariant(
                name="explicit-budget",
                execute_search_state_fn=lambda state: {
                    "search_results": [
                        {
                            "goods_id": "P2",
                            "goods_name": "관절 사료",
                            "pet_type": ["강아지"],
                            "category": ["사료"],
                            "price": 19000,
                            "discount_price": 18000,
                            "review_count": 10,
                            "rating": 4.8,
                        }
                    ]
                },
                rerank_search_results_fn=lambda state: {
                    "reranked_results": state["search_results"],
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                },
            ),
        )

        self.assertIsNone(result.error)
        self.assertEqual(result.profile_state["budget"], 20000)


class StateHarnessTests(unittest.TestCase):
    def test_state_runner_extracts_and_scores_expected_state(self):
        case = StateCase.from_dict(
            {
                "case_id": "state-dental",
                "message": "강아지 덴탈껌 추천해줘",
                "expect_state": {
                    "intents": ["recommend"],
                    "pet_type": "강아지",
                    "category": "간식",
                    "subcategory": "덴탈껌",
                    "target_pet_name": "초코",
                    "route": "profile",
                    "should_clarify": False,
                },
            }
        )

        def build_execution_request_fn(req):
            return SimpleNamespace(
                initial_state={
                    "user_input": req.message,
                    "target_pet_id": req.target_pet_id,
                    "pet_profile": req.pet_profile,
                    "health_concerns": req.health_concerns,
                    "allergies": req.allergies,
                    "food_preferences": req.food_preferences,
                    "intents": [],
                    "filters": {},
                    "decomposed_tasks": [],
                    "pending_requests": [],
                    "clarification_count": 0,
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                    "pet_mismatch": False,
                }
            )

        variant = StateVariant(
            name="state-test",
            hydrate_request_fn=lambda req: req,
            build_execution_request_fn=build_execution_request_fn,
            intent_classifier_fn=lambda state: {
                "intents": ["recommend"],
                "filters": {"pet_type": "강아지", "category": "간식", "subcategory": "덴탈껌"},
                "pet_profile": {"name": "초코"},
            },
            route_intent_fn=lambda merged: "profile",
        )

        result = run_state_case(case, variant)
        score = score_state_result(result)

        self.assertIsNone(result.error)
        self.assertEqual(result.route, "profile")
        self.assertEqual(result.resolved_pet_name, "초코")
        self.assertTrue(score.passed)
        self.assertEqual(score.metrics["slot_accuracy"], 1.0)

    def test_state_runner_supports_route_and_decomposition_expectations(self):
        case = StateCase.from_dict(
            {
                "case_id": "state-multi-task",
                "message": "강아지 사료랑 간식 둘 다 추천해줘",
                "expect_state": {
                    "intents": ["recommend"],
                    "pet_type": "강아지",
                    "route": "query",
                    "min_decomposed_tasks": 1,
                    "should_clarify": False,
                },
            }
        )

        def build_execution_request_fn(req):
            return SimpleNamespace(
                initial_state={
                    "user_input": req.message,
                    "target_pet_id": req.target_pet_id,
                    "pet_profile": req.pet_profile,
                    "health_concerns": req.health_concerns,
                    "allergies": req.allergies,
                    "food_preferences": req.food_preferences,
                    "intents": [],
                    "filters": {},
                    "decomposed_tasks": [],
                    "pending_requests": [],
                    "clarification_count": 0,
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                    "pet_mismatch": False,
                }
            )

        variant = StateVariant(
            name="state-decompose-test",
            hydrate_request_fn=lambda req: req,
            build_execution_request_fn=build_execution_request_fn,
            intent_classifier_fn=lambda state: {
                "intents": ["recommend"],
                "filters": {"pet_type": "강아지", "category": "사료"},
                "decomposed_tasks": [{"pet_name": None, "category": "간식", "subcategory": None}],
            },
            route_intent_fn=lambda merged: "query",
        )

        result = run_state_case(case, variant)
        score = score_state_result(result)

        self.assertIsNone(result.error)
        self.assertEqual(result.route, "query")
        self.assertEqual(len(result.merged_state["decomposed_tasks"]), 1)
        self.assertTrue(score.passed)
        self.assertEqual(score.metrics["decomposed_task_count"], 1)

    def test_state_runner_supports_result_refinement_expectation(self):
        case = StateCase.from_dict(
            {
                "case_id": "state-result-refinement",
                "message": "이 중에서 더 싼 거로 보여줘",
                "expect_state": {
                    "intents": ["recommend"],
                    "pet_type": "고양이",
                    "category": "사료",
                    "route": "query",
                    "should_clarify": False,
                    "is_result_refinement": True,
                },
            }
        )

        def build_execution_request_fn(req):
            return SimpleNamespace(
                initial_state={
                    "user_input": req.message,
                    "target_pet_id": req.target_pet_id,
                    "last_recommended_goods_ids": ["GI1", "GI2"],
                    "allowed_goods_ids": [],
                    "pet_profile": {"species": "cat"},
                    "health_concerns": req.health_concerns,
                    "allergies": req.allergies,
                    "food_preferences": req.food_preferences,
                    "intents": ["recommend"],
                    "filters": {"pet_type": "고양이", "category": "사료"},
                    "decomposed_tasks": [],
                    "pending_requests": [],
                    "clarification_count": 0,
                    "filter_relaxation_count": 0,
                    "recommend_retry_pending": False,
                    "is_result_refinement": False,
                    "pet_mismatch": False,
                }
            )

        variant = StateVariant(
            name="state-refinement",
            hydrate_request_fn=lambda req: req,
            build_execution_request_fn=build_execution_request_fn,
            intent_classifier_fn=lambda state: {
                "intents": ["recommend"],
                "filters": {"pet_type": "고양이", "category": "사료"},
                "allowed_goods_ids": ["GI1", "GI2"],
                "last_recommended_goods_ids": ["GI1", "GI2"],
                "is_result_refinement": True,
            },
            route_intent_fn=lambda merged: "query",
        )

        result = run_state_case(case, variant)
        score = score_state_result(result)

        self.assertIsNone(result.error)
        self.assertTrue(score.passed)
        self.assertEqual(result.merged_state["allowed_goods_ids"], ["GI1", "GI2"])
        self.assertEqual(score.metrics["refinement_accuracy"], 1.0)

    @patch(
        "final_ai.domain.intent.service._classify_user_input",
        return_value={"intents": ["recommend"], "mentioned_pet_names": ["바나나"]},
    )
    def test_state_runner_injects_registered_pets_and_pet_profiles(self, _mock_classify_user_input):
        case = StateCase.from_dict(
            {
                "case_id": "state-named-pet-switch",
                "message": "바나나로 바꿔줘",
                "user_id": "user-1",
                "thread_context": {
                    "dialog_state": {
                        "intents": ["recommend"],
                        "target_pet_id": "10",
                        "filters": {"pet_type": "강아지", "category": "사료"},
                        "pet_profile": {"name": "초코", "species": "dog", "breed": "말티즈"},
                    }
                },
                "registered_pets": [
                    {"pet_id": "10", "name": "초코", "species": "dog", "breed": "말티즈", "age": "3살"},
                    {"pet_id": "11", "name": "바나나", "species": "cat", "breed": "코숏", "age": "2살"},
                ],
                "pet_profiles": {
                    "11": {
                        "pet_id": "11",
                        "pet_profile": {"name": "바나나", "species": "cat", "breed": "코숏", "age": "2살"},
                        "health_concerns": [],
                        "allergies": [],
                        "food_preferences": [],
                    }
                },
                "expect_state": {
                    "intents": ["recommend"],
                    "pet_type": "고양이",
                    "category": "사료",
                    "target_pet_id": "11",
                    "target_pet_name": "바나나",
                    "route": "profile",
                    "should_clarify": False,
                },
            }
        )

        result = run_state_case(
            case,
            StateVariant(
                name="state-named-pet-test",
                hydrate_request_fn=lambda req: req,
                build_execution_request_fn=lambda req: SimpleNamespace(
                    initial_state={
                        "user_input": req.message,
                        "user_id": req.user_id,
                        "target_pet_id": req.target_pet_id,
                        "pet_profile": req.pet_profile or req.dialog_state.get("pet_profile"),
                        "health_concerns": req.health_concerns,
                        "allergies": req.allergies,
                        "food_preferences": req.food_preferences,
                        "intents": list((req.dialog_state or {}).get("intents") or []),
                        "filters": dict((req.dialog_state or {}).get("filters") or {}),
                        "decomposed_tasks": [],
                        "pending_requests": [],
                        "clarification_count": 0,
                        "filter_relaxation_count": 0,
                        "recommend_retry_pending": False,
                        "pet_mismatch": False,
                        "conversation_history": [],
                        "summary_candidates": [],
                        "memory_summary": "",
                    }
                ),
                intent_classifier_fn=classify_intent,
            ),
        )
        score = score_state_result(result)

        self.assertIsNone(result.error)
        self.assertEqual(result.merged_state["target_pet_id"], "11")
        self.assertEqual(result.resolved_pet_name, "바나나")
        self.assertEqual(result.merged_state["filters"]["category"], "사료")
        self.assertTrue(score.passed)


class HarnessComparatorTests(unittest.TestCase):
    def test_compare_reports_detects_regression(self):
        baseline_report = {
            "summary": {
                "total_cases": 2,
                "failed_cases": 0,
                "average_metrics": {"hit_at_5": 1.0, "mrr_at_5": 0.8},
            },
            "cases": [
                {"case_id": "case-1", "passed": True, "metrics": {"hit_at_5": 1.0, "mrr_at_5": 1.0}},
                {"case_id": "case-2", "passed": True, "metrics": {"hit_at_5": 1.0, "mrr_at_5": 0.6}},
            ],
        }
        candidate_report = {
            "summary": {
                "total_cases": 2,
                "failed_cases": 1,
                "average_metrics": {"hit_at_5": 0.5, "mrr_at_5": 0.4},
            },
            "cases": [
                {"case_id": "case-1", "passed": True, "metrics": {"hit_at_5": 1.0, "mrr_at_5": 0.8}},
                {"case_id": "case-2", "passed": False, "metrics": {"hit_at_5": 0.0, "mrr_at_5": 0.0}},
            ],
        }

        comparison = compare_harness_reports(baseline_report, candidate_report)

        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["summary"]["failed_case_delta"], 1)
        self.assertEqual(comparison["status_regressions"], ["case-2"])
        self.assertIn("hit_at_5", comparison["average_metric_regressions"])
        self.assertTrue(any(item["case_id"] == "case-2" for item in comparison["case_metric_regressions"]))

    def test_compare_reports_passes_when_candidate_improves(self):
        baseline_report = {
            "summary": {
                "total_cases": 1,
                "failed_cases": 1,
                "average_metrics": {"slot_accuracy": 0.5},
            },
            "cases": [
                {"case_id": "state-1", "passed": False, "metrics": {"slot_accuracy": 0.5}},
            ],
        }
        candidate_report = {
            "summary": {
                "total_cases": 1,
                "failed_cases": 0,
                "average_metrics": {"slot_accuracy": 1.0},
            },
            "cases": [
                {"case_id": "state-1", "passed": True, "metrics": {"slot_accuracy": 1.0}},
            ],
        }

        comparison = compare_harness_reports(baseline_report, candidate_report)

        self.assertTrue(comparison["passed"])
        self.assertEqual(comparison["status_improvements"], ["state-1"])
        self.assertFalse(comparison["average_metric_regressions"])
