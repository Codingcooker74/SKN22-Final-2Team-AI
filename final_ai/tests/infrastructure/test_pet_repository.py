import unittest
from unittest.mock import patch

from final_ai.infrastructure.repositories import pet_repository


class _FakeCursor:
    def __init__(self, *, fetchone_result=None, fetchall_result=None):
        self._fetchone_result = fetchone_result
        self._fetchall_result = fetchall_result or []
        self.executed = []

    def execute(self, query, params):
        self.executed.append((query, params))

    def fetchone(self):
        return self._fetchone_result

    def fetchall(self):
        return self._fetchall_result

    def close(self):
        return None


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self, **kwargs):
        return self._cursor

    def close(self):
        return None


class PetRepositoryTests(unittest.TestCase):
    def test_fetch_user_pets_formats_rows(self):
        cursor = _FakeCursor(
            fetchall_result=[
                {"pet_id": 10, "name": "초코", "species": "dog", "breed": "말티즈", "age_years": 3},
                {"pet_id": 11, "name": "바나나", "species": "cat", "breed": "코숏", "age_years": 2},
            ]
        )
        connection = _FakeConnection(cursor)

        with patch.object(pet_repository, "get_db_connection", return_value=connection):
            pets = pet_repository.fetch_user_pets("user-1")

        self.assertEqual(
            pets,
            [
                {"pet_id": "10", "name": "초코", "species": "dog", "breed": "말티즈", "age": "3살"},
                {"pet_id": "11", "name": "바나나", "species": "cat", "breed": "코숏", "age": "2살"},
            ],
        )

    def test_fetch_pet_name_for_user_returns_name(self):
        with patch.object(pet_repository, "fetch_pet_for_user", return_value={"name": "초코"}):
            self.assertEqual(pet_repository.fetch_pet_name_for_user("user-1"), "초코")
