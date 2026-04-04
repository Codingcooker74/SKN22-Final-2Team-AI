import unittest

from final_ai.infrastructure.search.hybrid_search import normalize_pet_species


class HybridSearchTests(unittest.TestCase):
    def test_normalize_pet_species_maps_known_values(self):
        self.assertEqual(normalize_pet_species("dog"), "강아지")
        self.assertEqual(normalize_pet_species("cat"), "고양이")
        self.assertEqual(normalize_pet_species("강아지"), "강아지")
        self.assertIsNone(normalize_pet_species("hamster"))
