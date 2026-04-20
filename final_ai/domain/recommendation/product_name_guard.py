import re
import unicodedata

_PET_SPECIES_KR = {
    "dog": "강아지",
    "cat": "고양이",
    "강아지": "강아지",
    "고양이": "고양이",
}


def _normalize_pet_species(value: str | list[str] | None) -> str | None:
    if not value:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    text = str(value).strip()
    return _PET_SPECIES_KR.get(text.lower()) or _PET_SPECIES_KR.get(text)


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text)).lower().replace(" ", "")


def has_pet_type_name_conflict(candidate: dict, *, target_pet_type: str | None) -> bool:
    target = _normalize_pet_species(target_pet_type)
    if target not in {"강아지", "고양이"}:
        return False

    goods_name = str(candidate.get("goods_name") or "").lower()
    compact_name = _normalize_text(goods_name)
    if not compact_name:
        return False

    if target == "강아지":
        return any(term in compact_name for term in ("고양이", "캣")) or bool(re.search(r"\bcat(s)?\b", goods_name))

    has_dog_term = any(term in compact_name for term in ("강아지", "반려견"))
    has_dog_word = bool(re.search(r"\bdogs?\b", goods_name))
    # "개"는 "8개" 같은 수량 표현과 충돌하므로 독립 토큰 또는 용도 표현만 제외합니다.
    has_dog_korean_token = bool(re.search(r"(?<![0-9A-Za-z가-힣])개(?![0-9A-Za-z가-힣])", goods_name))
    has_dog_usage = "개용" in compact_name
    return has_dog_term or has_dog_word or has_dog_korean_token or has_dog_usage


def filter_pet_type_name_conflicts(candidates: list[dict], *, target_pet_type: str | None) -> list[dict]:
    if not candidates:
        return candidates

    return [
        candidate
        for candidate in candidates
        if not has_pet_type_name_conflict(candidate, target_pet_type=target_pet_type)
    ]
