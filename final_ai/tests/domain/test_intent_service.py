import unittest
from unittest.mock import patch

from final_ai.domain.intent.service import classify_intent


class IntentServiceTests(unittest.TestCase):
    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_keeps_pet_type_for_multi_task_query(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["recommend"],
            "decomposed_tasks": [
                {"pet_name": None, "category": "사료", "subcategory": None},
                {"pet_name": None, "category": "간식", "subcategory": None},
            ],
        }

        result = classify_intent(
            {
                "user_input": "강아지 사료랑 간식 둘 다 추천해줘",
                "user_id": None,
                "intents": [],
                "filters": {},
                "pet_profile": {},
                "target_pet_id": None,
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["intents"], ["recommend"])
        self.assertEqual(result["filters"]["pet_type"], "강아지")
        self.assertEqual(result["filters"]["category"], "사료")
        self.assertEqual(result["pet_profile"]["species"], "dog")
        self.assertEqual(len(result["decomposed_tasks"]), 1)
        self.assertEqual(result["decomposed_tasks"][0]["category"], "간식")

    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_treats_pet_switch_followup_as_recommend(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["unclear"],
            "pet_type": "강아지",
        }

        result = classify_intent(
            {
                "user_input": "강아지로 바꿔줘",
                "user_id": None,
                "intents": ["recommend"],
                "filters": {"pet_type": "고양이", "category": "사료"},
                "pet_profile": {"species": "cat"},
                "target_pet_id": None,
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["intents"], ["recommend"])
        self.assertEqual(result["filters"]["pet_type"], "강아지")
        self.assertEqual(result["filters"]["category"], "사료")

    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_marks_result_refinement_followup_as_recommend(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["unclear"],
            "is_result_refinement": True,
        }

        result = classify_intent(
            {
                "user_input": "이 중에서 더 싼 거로 보여줘",
                "user_id": None,
                "intents": ["recommend"],
                "filters": {"pet_type": "고양이", "category": "사료"},
                "pet_profile": {"species": "cat"},
                "target_pet_id": None,
                "last_recommended_goods_ids": ["GI1", "GI2"],
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["intents"], ["recommend"])
        self.assertEqual(result["filters"]["pet_type"], "고양이")
        self.assertEqual(result["filters"]["category"], "사료")
        self.assertTrue(result["is_result_refinement"])
        self.assertEqual(result["allowed_goods_ids"], ["GI1", "GI2"])

    @patch(
        "final_ai.domain.intent.service.get_pet_full_profile",
        return_value={
            "pet_id": "11",
            "pet_profile": {"name": "바나나", "species": "cat", "breed": "코숏", "age": "2살"},
            "health_concerns": [],
            "allergies": [],
            "food_preferences": [],
        },
    )
    @patch(
        "final_ai.domain.intent.service.get_user_pets",
        return_value=[
            {"pet_id": "10", "name": "초코", "species": "dog", "breed": "말티즈", "age": "3살"},
            {"pet_id": "11", "name": "바나나", "species": "cat", "breed": "코숏", "age": "2살"},
        ],
    )
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_treats_named_pet_switch_followup_as_recommend(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
        _mock_get_pet_full_profile,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["unclear"],
            "mentioned_pet_names": ["바나나"],
        }

        result = classify_intent(
            {
                "user_input": "바나나로 바꿔줘",
                "user_id": "user-1",
                "intents": ["recommend"],
                "filters": {"pet_type": "강아지", "category": "사료"},
                "pet_profile": {"name": "초코", "species": "dog", "breed": "말티즈"},
                "target_pet_id": "10",
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["intents"], ["recommend"])
        self.assertEqual(result["target_pet_id"], "11")
        self.assertEqual(result["switched_pet_name"], "바나나")
        self.assertEqual(result["filters"]["pet_type"], "고양이")
        self.assertEqual(result["filters"]["category"], "사료")

    @patch(
        "final_ai.domain.intent.service.get_pet_full_profile",
        return_value={
            "pet_id": "10",
            "pet_profile": {"name": "초코", "species": "dog", "breed": "말티즈", "age": "3살"},
            "health_concerns": [],
            "allergies": [],
            "food_preferences": [],
        },
    )
    @patch(
        "final_ai.domain.intent.service.get_user_pets",
        return_value=[
            {"pet_id": "10", "name": "초코", "species": "dog", "breed": "말티즈", "age": "3살"},
            {"pet_id": "11", "name": "바나나", "species": "cat", "breed": "코숏", "age": "2살"},
        ],
    )
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_keeps_health_concern_on_named_pet_switch(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
        _mock_get_pet_full_profile,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["recommend"],
            "decomposed_tasks": [
                {"pet_name": "초코", "category": "간식", "subcategory": None, "health_concern": "관절"},
            ],
        }

        result = classify_intent(
            {
                "user_input": "이번엔 초코 관절 관리용 간식으로 바꿔줘",
                "user_id": "user-1",
                "intents": ["recommend"],
                "filters": {"pet_type": "고양이", "category": "사료"},
                "pet_profile": {"name": "바나나", "species": "cat", "breed": "코숏"},
                "target_pet_id": "11",
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["target_pet_id"], "10")
        self.assertEqual(result["filters"]["pet_type"], "강아지")
        self.assertEqual(result["filters"]["category"], "간식")
        self.assertIn("관절", result["health_concerns"])

    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_adds_domain_qa_for_explanation_request(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["recommend"],
            "pet_type": "강아지",
            "target_categories": ["사료"],
            "health_concerns": ["피부"],
        }

        result = classify_intent(
            {
                "user_input": "강아지 피부에 좋은 사료 추천해주고 왜 좋은지도 설명해줘",
                "user_id": None,
                "intents": [],
                "filters": {},
                "pet_profile": {},
                "target_pet_id": None,
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["filters"]["pet_type"], "강아지")
        self.assertEqual(result["filters"]["category"], "사료")
        self.assertIn("recommend", result["intents"])
        self.assertIn("domain_qa", result["intents"])

    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_keeps_brand_for_brand_specific_request(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["recommend"],
            "pet_type": "고양이",
            "target_categories": ["사료"],
            "brand": "로얄캐닌",
        }

        result = classify_intent(
            {
                "user_input": "로얄캐닌 고양이 사료 추천해줘",
                "user_id": None,
                "intents": [],
                "filters": {},
                "pet_profile": {},
                "target_pet_id": None,
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["filters"]["pet_type"], "고양이")
        self.assertEqual(result["filters"]["category"], "사료")
        self.assertEqual(result["filters"]["brand"], "로얄캐닌")

    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_recovers_compact_recommend_request_from_category_keyword(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["unclear"],
            "target_categories": [],
        }

        result = classify_intent(
            {
                "user_input": "사료추천해줘",
                "user_id": None,
                "intents": [],
                "filters": {},
                "exclusions": {},
                "pet_profile": {},
                "target_pet_id": None,
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["intents"], ["recommend"])
        self.assertEqual(result["filters"]["category"], "사료")
        self.assertNotIn("pet_type", result["filters"])

    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_moves_excluded_brand_to_exclusions(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["recommend"],
            "pet_type": "고양이",
            "target_categories": ["사료"],
            "brand": "로얄캐닌",
        }

        result = classify_intent(
            {
                "user_input": "로얄캐닌 제외하고 고양이 사료 추천해줘",
                "user_id": None,
                "intents": [],
                "filters": {},
                "exclusions": {},
                "pet_profile": {},
                "target_pet_id": None,
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["filters"]["pet_type"], "고양이")
        self.assertEqual(result["filters"]["category"], "사료")
        self.assertNotIn("brand", result["filters"])
        self.assertEqual(result["exclusions"]["brands"], ["로얄캐닌"])

    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_excludes_previous_results_for_alternative_request(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["unclear"],
        }

        result = classify_intent(
            {
                "user_input": "방금 추천한 거 말고 다른 거 보여줘",
                "user_id": None,
                "intents": ["recommend"],
                "filters": {"pet_type": "고양이", "category": "사료"},
                "exclusions": {"brands": ["로얄캐닌"]},
                "pet_profile": {"species": "cat"},
                "target_pet_id": None,
                "last_recommended_goods_ids": ["GI1", "GI2"],
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["intents"], ["recommend"])
        self.assertFalse(result["is_result_refinement"])
        self.assertEqual(result["allowed_goods_ids"], [])
        self.assertEqual(result["exclusions"]["goods_ids"], ["GI1", "GI2"])
        self.assertEqual(result["exclusions"]["brands"], ["로얄캐닌"])

    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_extracts_generic_exclusion_keyword_from_text_fallback(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["recommend"],
            "pet_type": "강아지",
            "target_categories": ["사료"],
        }

        result = classify_intent(
            {
                "user_input": "'그레인프리' 제외하고 강아지 사료 추천해줘",
                "user_id": None,
                "intents": [],
                "filters": {},
                "exclusions": {},
                "pet_profile": {},
                "target_pet_id": None,
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertEqual(result["filters"]["pet_type"], "강아지")
        self.assertEqual(result["filters"]["category"], "사료")
        self.assertEqual(result["exclusions"]["keywords"], ["그레인프리"])

    @patch("final_ai.domain.intent.service.get_user_pets", return_value=[])
    @patch("final_ai.domain.intent.service._classify_user_input")
    def test_classify_intent_keeps_domain_qa_after_decomposition(
        self,
        mock_classify_user_input,
        _mock_get_user_pets,
    ):
        mock_classify_user_input.return_value = {
            "intents": ["recommend"],
            "decomposed_tasks": [
                {"pet_name": None, "category": "사료", "subcategory": None, "health_concern": "피부/모질"},
            ],
        }

        result = classify_intent(
            {
                "user_input": "강아지 피부에 좋은 사료 추천해주고 왜 좋은지도 설명해줘",
                "user_id": None,
                "intents": [],
                "filters": {},
                "pet_profile": {},
                "target_pet_id": None,
                "pending_requests": [],
                "decomposed_tasks": [],
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "clarification_count": 0,
                "filter_relaxation_count": 0,
                "recommend_retry_pending": False,
                "pet_mismatch": False,
                "conversation_history": [],
                "summary_candidates": [],
                "memory_summary": "",
            }
        )

        self.assertIn("recommend", result["intents"])
        self.assertIn("domain_qa", result["intents"])
