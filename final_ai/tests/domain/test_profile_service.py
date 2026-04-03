import unittest

from final_ai.domain.profile.service import build_pet_context, translate_health_concerns


class ProfileServiceTests(unittest.TestCase):
    def test_translate_health_concerns_maps_known_values(self):
        self.assertEqual(translate_health_concerns(["skin", "joint", "unknown"]), ["피부", "관절", "unknown"])

    def test_build_pet_context_includes_profile_and_preferences(self):
        state = {
            "pet_profile": {"species": "dog", "breed": "말티즈", "age": "3살"},
            "health_concerns": ["eye"],
            "allergies": ["닭"],
            "food_preferences": ["건식"],
        }

        context = build_pet_context(state)

        self.assertIn("종: 강아지", context)
        self.assertIn("품종: 말티즈", context)
        self.assertIn("나이: 3살", context)
        self.assertIn("건강관심사: 눈물", context)
        self.assertIn("알레르기: 닭", context)
        self.assertIn("선호사료타입: 건식", context)
