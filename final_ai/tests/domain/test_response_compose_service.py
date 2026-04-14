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

    def test_build_response_state_uses_domain_qa_prompt_for_domain_mode(self):
        captured = {}

        def fake_create(**kwargs):
            captured["messages"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="초콜릿 섭취는 위험할 수 있어 바로 동물병원에 연락해 주세요.")
                    )
                ]
            )

        fake_llm = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
        state = {
            "user_input": "강아지가 초콜릿을 먹었어요",
            "response_mode": "domain_qa",
            "intents": ["domain_qa"],
            "pet_profile": {"name": "초코", "species": "dog", "breed": "말티즈", "age": "3살"},
            "filters": {},
            "reranked_results": [],
            "domain_contexts": ["초콜릿은 강아지에게 독성을 일으킬 수 있습니다."],
            "health_concerns": [],
            "pending_requests": [],
            "decomposed_tasks": [],
            "health_traits": "",
            "is_pet_switched": False,
        }

        with patch.object(compose_service, "llm", fake_llm):
            result = compose_service.build_response_state(state)

        system_message = captured["messages"][0]["content"]
        user_message = captured["messages"][1]["content"]
        self.assertIn("건강/행동 지식 상담", system_message)
        self.assertIn("상품을 추천하지 마세요", system_message)
        self.assertIn("응답 모드: domain_qa", user_message)
        self.assertIn("추천 상품 후보 있음: NO", user_message)
        self.assertNotIn("추천 상품을 확인해 주세요", result["response"])

    def test_build_response_state_domain_qa_fallback_does_not_recommend(self):
        failing_llm = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("llm down")))
            )
        )
        state = {
            "user_input": "강아지가 초콜릿을 먹었어요",
            "response_mode": "domain_qa",
            "intents": ["domain_qa"],
            "pet_profile": {"name": "초코", "species": "dog", "breed": "말티즈", "age": "3살"},
            "filters": {},
            "reranked_results": [],
            "domain_contexts": ["초콜릿은 강아지에게 독성을 일으킬 수 있습니다."],
            "health_concerns": [],
            "pending_requests": [],
            "decomposed_tasks": [],
            "health_traits": "",
            "is_pet_switched": False,
        }

        with patch.object(compose_service, "llm", failing_llm):
            result = compose_service.build_response_state(state)

        self.assertIn("동물병원", result["response"])
        self.assertNotIn("추천 상품", result["response"])
