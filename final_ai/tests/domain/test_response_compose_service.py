import unittest
from types import SimpleNamespace
from unittest.mock import patch

from final_ai.domain.response import compose_service


class ResponseComposeServiceTests(unittest.TestCase):
    def test_build_response_state_uses_fallback_when_llm_fails(self):
        failing_llm = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("llm down")))
            )
        )
        state = {
            "user_input": "사료 추천해줘",
            "pet_profile": {"name": "초코", "species": "dog", "breed": "말티즈", "age": "3살"},
            "filters": {"category": "사료"},
            "reranked_results": [{"brand_name": "브랜드", "goods_name": "테스트 사료"}],
            "domain_contexts": [],
            "health_concerns": [],
            "pending_pet_ids": [],
            "pending_categories": [],
            "health_traits": "",
            "is_pet_switched": False,
        }

        with patch.object(compose_service, "llm", failing_llm):
            result = compose_service.build_response_state(state)

        self.assertIn("추천 상품을 확인해 주세요", result["response"])
        self.assertEqual(result["messages"][0].content, result["response"])
