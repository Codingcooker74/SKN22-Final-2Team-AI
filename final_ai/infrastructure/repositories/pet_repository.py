import psycopg2.extras

from final_ai.infrastructure.db.connection import get_db_connection


def fetch_pet_for_user(
    user_id: str,
    *,
    target_pet_id: str | None = None,
    auto_latest: bool = True,
) -> dict | None:
    if not user_id:
        return None

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if target_pet_id:
            cur.execute(
                """
                SELECT pet_id, user_id, name, species, breed, age_years, age_months, weight_kg, gender,
                       budget_range, created_at
                FROM pet
                WHERE user_id = %s AND pet_id = %s
                LIMIT 1
                """,
                (user_id, target_pet_id),
            )
        elif auto_latest:
            cur.execute(
                """
                SELECT pet_id, user_id, name, species, breed, age_years, age_months, weight_kg, gender,
                       budget_range, created_at
                FROM pet
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id,),
            )
        else:
            return None
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def fetch_pet_name_for_user(user_id: str, *, target_pet_id: str | None = None) -> str | None:
    row = fetch_pet_for_user(user_id, target_pet_id=target_pet_id, auto_latest=True)
    return row.get("name") if row else None


def fetch_user_pets(user_id: str) -> list[dict]:
    if not user_id:
        return []

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT pet_id, name, species, breed, age_years
            FROM pet
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "pet_id": str(row["pet_id"]),
                "name": row["name"],
                "species": row["species"],
                "breed": row["breed"],
                "age": f"{row['age_years']}살",
            }
            for row in rows
        ]
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def fetch_pet_preferences(pet_id: str) -> dict[str, list[str]]:
    if not pet_id:
        return {
            "health_concerns": [],
            "allergies": [],
            "food_preferences": [],
        }

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT concern FROM pet_health_concern WHERE pet_id = %s", (pet_id,))
        health_concerns = [row[0] for row in cur.fetchall()]

        cur.execute("SELECT ingredient FROM pet_allergy WHERE pet_id = %s", (pet_id,))
        allergies = [row[0] for row in cur.fetchall()]

        cur.execute("SELECT food_type FROM pet_food_preference WHERE pet_id = %s", (pet_id,))
        food_preferences = [row[0] for row in cur.fetchall()]

        return {
            "health_concerns": health_concerns,
            "allergies": allergies,
            "food_preferences": food_preferences,
        }
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def fetch_pet_full_profile(pet_id: str) -> dict:
    if not pet_id:
        return {}

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT pet_id, name, species, breed, age_years, gender, weight_kg
            FROM pet
            WHERE pet_id = %s
            LIMIT 1
            """,
            (pet_id,),
        )
        base = cur.fetchone()
        if not base:
            return {}
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    preferences = fetch_pet_preferences(pet_id)
    return {
        "pet_id": pet_id,
        "pet_profile": {
            "name": base["name"],
            "species": base["species"],
            "breed": base["breed"],
            "age": f"{base['age_years']}살",
            "gender": base["gender"],
            "weight": float(base["weight_kg"]) if base["weight_kg"] else None,
        },
        "health_concerns": preferences["health_concerns"],
        "allergies": preferences["allergies"],
        "food_preferences": preferences["food_preferences"],
    }


def fetch_breed_meta(breed_name: str, age_group: str) -> dict | None:
    if not breed_name:
        return None

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT preferred_food, health_products, chunk_text
            FROM breed_meta
            WHERE (breed_name = %s OR breed_name_en ILIKE %s)
            ORDER BY CASE WHEN age_group = %s THEN 0 ELSE 1 END, id ASC
            LIMIT 1
            """,
            (breed_name, f"%{breed_name}%", age_group),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def fetch_future_pet_profile(user_id: str) -> dict | None:
    if not user_id:
        return None

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT preferred_species, housing_type, experience_level, interests
            FROM future_pet_profile
            WHERE user_id = %s
            LIMIT 1
            """,
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "species": row.get("preferred_species") or "",
            "lifecycle": "future_guardian",
            "preferred_species": row.get("preferred_species") or "",
            "housing_type": row.get("housing_type") or "",
            "experience_level": row.get("experience_level") or "",
            "interests": row.get("interests") or [],
        }
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
