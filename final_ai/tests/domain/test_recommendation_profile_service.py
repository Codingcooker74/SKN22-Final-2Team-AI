import unittest
from unittest.mock import patch

from final_ai.domain.recommendation.profile_service import build_profile_state


class RecommendationProfileServiceTests(unittest.TestCase):
    @patch("final_ai.domain.recommendation.profile_service.fetch_pet_preferences")
    @patch("final_ai.domain.recommendation.profile_service.fetch_pet_for_user")
    def test_build_profile_state_normalizes_db_health_concerns(self, mock_fetch_pet_for_user, mock_fetch_pet_preferences):
        mock_fetch_pet_for_user.return_value = {
            "pet_id": "pet-1",
            "name": "초코",
            "species": "dog",
            "breed": "진돗개",
            "age_years": 0,
            "age_months": 8,
            "weight_kg": 12.4,
            "gender": "M",
            "budget_range": None,
        }
        mock_fetch_pet_preferences.return_value = {
            "health_concerns": ["digestion"],
            "allergies": [],
            "food_preferences": [],
        }

        result = build_profile_state(
            {
                "user_id": "user-1",
                "target_pet_id": "pet-1",
                "pet_profile": {"species": "dog"},
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
            }
        )

        self.assertEqual(result["health_concerns"], ["소화"])
