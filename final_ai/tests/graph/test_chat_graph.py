import unittest

from langgraph.types import Send

from final_ai.graph.builder import build_graph, route_intent, route_profile_node, route_rerank


class ChatGraphTests(unittest.TestCase):
    def test_route_intent_requests_clarification_without_pet_type(self):
        state = {
            "intents": ["recommend"],
            "filters": {},
            "pet_profile": {},
            "filter_relaxation_count": 0,
        }

        self.assertEqual(route_intent(state), "clarify")

    def test_route_intent_fans_out_for_domain_and_recommend(self):
        state = {
            "intents": ["domain_qa", "recommend"],
            "filters": {"pet_type": "강아지", "category": "사료"},
            "pet_profile": {"species": "dog"},
            "filter_relaxation_count": 0,
        }

        result = route_intent(state)

        self.assertTrue(all(isinstance(item, Send) for item in result))
        self.assertEqual([item.node for item in result], ["general", "query"])

    def test_route_intent_fans_out_through_profile_when_pet_is_selected(self):
        state = {
            "intents": ["domain_qa", "recommend"],
            "filters": {"pet_type": "강아지", "category": "사료"},
            "pet_profile": {"species": "dog"},
            "target_pet_id": "pet-1",
            "filter_relaxation_count": 0,
        }

        result = route_intent(state)

        self.assertTrue(all(isinstance(item, Send) for item in result))
        self.assertEqual([item.node for item in result], ["general", "profile"])

    def test_route_rerank_returns_query_when_retry_pending(self):
        self.assertEqual(
            route_rerank({"reranked_results": [], "filter_relaxation_count": 0, "recommend_retry_pending": True}),
            "query",
        )

    def test_route_profile_node_merges_domain_only_intent(self):
        state = {
            "intents": ["domain_qa"],
            "pet_mismatch": False,
        }

        self.assertEqual(route_profile_node(state), "merge")

    def test_route_profile_node_continues_for_recommend_intent(self):
        state = {
            "intents": ["recommend"],
            "pet_mismatch": False,
        }

        self.assertEqual(route_profile_node(state), "query")

    def test_route_profile_node_merges_pet_mismatch(self):
        state = {
            "intents": ["recommend"],
            "pet_mismatch": True,
        }

        self.assertEqual(route_profile_node(state), "merge")

    def test_build_graph_returns_compiled_graph(self):
        self.assertEqual(type(build_graph()).__name__, "CompiledStateGraph")
