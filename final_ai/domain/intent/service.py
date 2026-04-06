import json

from final_ai.api.dependencies.request_context import ensure_request_active
from final_ai.application.chat.memory import format_conversation_history
from final_ai.contracts.filters import (
    SearchFilters,
    build_search_filters,
    normalize_filter_value,
    normalize_search_filters,
)
from final_ai.domain.intent.prompts import CATEGORIES, build_intent_prompt
from final_ai.domain.profile.service import get_pet_full_profile, get_user_pets
from final_ai.infrastructure.llm.openai_client import LLM_MODEL, llm
from final_ai.infrastructure.observability import get_logger
from final_ai.graph.state import ChatState

logger = get_logger(__name__)

# 건강 고민 표준 태그 매핑 사전
HEALTH_CONCERN_MAP = {
    "체중": ["다이어트", "살", "비만", "체중조절", "저칼로리", "슬림"],
    "눈물": ["눈물자국", "눈건강", "눈세정", "아이케어"],
    "피부": ["아토피", "가려움", "알러지", "피부염", "피부건강", "피부/모질"],
    "관절": ["슬개골", "뼈", "관절건강", "다리", "튼튼"],
    "소화": ["장", "변비", "설사", "소화불량", "위건강", "소화/장"],
    "치아": ["양치", "치석", "구강", "입냄새", "덴탈"],
    "요로": ["신장", "방광", "결석", "신장건강"],
    "헤어볼": ["그루밍", "헤어볼제거"],
    "면역": ["체력", "활력", "항산화", "면역력"],
}


def _build_context(
    *,
    state: ChatState,
    user_input: str,
    user_pets: list[dict],
    prev_intents: list[str],
    prev_filters: SearchFilters,
    prev_pet: dict,
    target_pet_id: str | None,
) -> str:
    context_parts = []
    memory_summary = (state.get("memory_summary") or "").strip()
    if memory_summary:
        context_parts.append(f"누적 대화 요약:\n{memory_summary}")

    summary_candidates_text = format_conversation_history(state.get("summary_candidates"), limit=8)
    if summary_candidates_text != "없음":
        context_parts.append(f"이번 턴에 메모리로 편입할 이전 대화:\n{summary_candidates_text}")

    history_text = format_conversation_history(state.get("conversation_history"), limit=10)
    if history_text != "없음":
        context_parts.append(f"최근 대화 기록:\n{history_text}")

    if not ((state.get("clarification_count", 0) > 0 or prev_intents) and user_input):
        return "\n\n".join(context_parts)

    prev_data = {
        "intents": prev_intents,
        "filters": prev_filters,
        "pet_profile": prev_pet,
        "current_pet_id": target_pet_id,
        "user_registered_pets": [pet["name"] for pet in user_pets],
    }
    context_parts.append(f"이전 대화 상태 및 등록된 펫 정보: {json.dumps(prev_data, ensure_ascii=False)}")
    return "\n\n".join(context_parts)


def _classify_user_input(user_input: str, context: str) -> dict:
    ensure_request_active()
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": build_intent_prompt(context)},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


def classify_intent(state: ChatState) -> dict:
    user_input = state["user_input"]
    user_id = state.get("user_id")
    prev_intents = state.get("intents") or []
    prev_filters = normalize_search_filters(state.get("filters"))
    prev_pet = state.get("pet_profile") or {}
    target_pet_id = state.get("target_pet_id")
    pending_requests = list(state.get("pending_requests") or [])

    user_pets = get_user_pets(user_id) if user_id else []
    context = _build_context(
        state=state,
        user_input=user_input,
        user_pets=user_pets,
        prev_intents=prev_intents,
        prev_filters=prev_filters,
        prev_pet=prev_pet,
        target_pet_id=target_pet_id,
    )
    result = _classify_user_input(user_input, context)

    new_intents = result.get("intents") or []
    mentioned_names = result.get("mentioned_pet_names") or []
    exclude_ingredients = result.get("exclude_ingredients") or []
    raw_health_concerns = result.get("health_concerns") or []
    is_next_request = result.get("is_next_request", False)
    target_categories = result.get("target_categories") or []
    is_explicit_pet_info = bool(result.get("pet_type") or result.get("breed"))

    # 건강 고민 표준 태그로 변환 로직
    detected_health_concerns = []
    for raw in raw_health_concerns:
        mapped_tag = raw
        for tag, keywords in HEALTH_CONCERN_MAP.items():
            if any(keyword in raw for keyword in keywords) or raw == tag:
                mapped_tag = tag
                break
        detected_health_concerns.append(mapped_tag)

    is_pet_switched = False
    switched_pet_name = None
    overridden_metadata = {}
    _pending_form_hint = None  # pending entry에 저장된 form_hint 복원용 (다중 펫 시나리오)

    if is_next_request:
        if pending_requests:
            next_req = pending_requests.pop(0)
            next_pet_id = next_req.get("pet_id")
            next_category = next_req.get("category")

            # 펫 전환이 필요한 경우
            if next_pet_id:
                full_profile = get_pet_full_profile(next_pet_id)
                if full_profile:
                    target_pet_id = next_pet_id
                    prev_pet = full_profile["pet_profile"]
                    overridden_metadata = {
                        "health_concerns": full_profile["health_concerns"],
                        "allergies": full_profile["allergies"],
                        "food_preferences": full_profile["food_preferences"],
                    }
                    is_pet_switched = True
                    switched_pet_name = prev_pet.get("name")

            # 카테고리 설정 (펫 전환과 무관하게 적용)
            if next_category:
                target_categories = [next_category]

            new_intents = ["recommend"]

            # pending entry에 저장된 form_hint를 현재 처리에 복원
            # (이전 턴에 다중 펫 처리 시 pending에 저장해뒀던 form_hint)
            _pending_form_hint = next_req.get("form_hint")
        else:
            _pending_form_hint = None
    elif mentioned_names:
        # ★ 핵심 수정: DB 등록 순서가 아닌 사용자 입력 순서(mentioned_names)로 정렬
        # 예: "초코 사료랑 바나나 모래" → mentioned_names=["초코","바나나"] 순서 보장
        raw_matched = [pet for pet in user_pets if pet["name"] in mentioned_names]
        matched_pets = sorted(
            raw_matched,
            key=lambda p: mentioned_names.index(p["name"]) if p["name"] in mentioned_names else 999,
        )
        if matched_pets:
            first_pet = matched_pets[0]
            if str(first_pet["pet_id"]) != str(target_pet_id):
                # 현재 펫과 다를 경우 → 프로필 전환
                full_profile = get_pet_full_profile(str(first_pet["pet_id"]))
                if full_profile:
                    target_pet_id = str(first_pet["pet_id"])
                    prev_pet = full_profile["pet_profile"]
                    overridden_metadata = {
                        "health_concerns": full_profile["health_concerns"],
                        "allergies": full_profile["allergies"],
                        "food_preferences": full_profile["food_preferences"],
                        "breed_context": "",
                        "health_traits": "",
                    }
                    is_pet_switched = True
                    switched_pet_name = prev_pet.get("name")
                    if "recommend" not in new_intents:
                        new_intents.append("recommend")
            else:
                # 현재 펫과 동일(초코가 이미 선택된 상태) → 프로필 전환 없이 카테고리만 처리
                logger.info("first mentioned pet is already the current pet: %s", first_pet.get("name"))

            # 두 번째 이후 펫+카테고리를 통합 큐에 쌍으로 등록
            # target_categories[0] = 첫 번째 펫 카테고리, target_categories[1:] = 나머지 펫 카테고리
            remaining_categories = list(target_categories[1:]) if len(target_categories) > 1 else []
            for i, pet in enumerate(matched_pets[1:]):
                pet_id = str(pet["pet_id"])
                pending_cat = remaining_categories[i] if i < len(remaining_categories) else None
                pending_entry = {"pet_id": pet_id, "category": pending_cat}
                # 이미 동일한 pet_id가 큐에 없을 때만 추가
                if not any(r.get("pet_id") == pet_id for r in pending_requests):
                    pending_requests.append(pending_entry)
            # 다중 펫+카테고리가 이미 쌍으로 등록됐음을 표시 → 아래 단일 펫 카테고리 로직 중복 방지
            multi_pet_categories_registered = len(matched_pets) > 1 and bool(remaining_categories)
    elif is_explicit_pet_info:
        multi_pet_categories_registered = False
        new_species = None
        if result.get("pet_type") == "강아지":
            new_species = "dog"
        elif result.get("pet_type") == "고양이":
            new_species = "cat"

        new_breed = result.get("breed")
        current_species = prev_pet.get("species")
        current_breed = prev_pet.get("breed")

        is_contradictory = False
        if new_species and current_species and new_species != current_species:
            is_contradictory = True
        if new_breed and current_breed and new_breed != current_breed:
            is_contradictory = True

        if is_contradictory or (new_breed and not target_pet_id):
            target_pet_id = None
            prev_pet = {}
            overridden_metadata = {
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "breed_context": "",
                "health_traits": "",
            }
            logger.info("switching to general breed context=%s", new_breed or new_species)
            is_pet_switched = True
        else:
            logger.info("pet context matched current profile=%s", prev_pet.get("name"))

    if not new_intents:
        if "recommend" in prev_intents:
            new_intents = ["recommend"]
        else:
            new_intents = ["unclear"]

    else:
        multi_pet_categories_registered = False

    new_pet = dict(prev_pet)
    if is_explicit_pet_info and not target_pet_id:
        new_pet = {}
        if result.get("pet_type"):
            new_pet["species"] = "dog" if result["pet_type"] == "강아지" else "cat"
        if result.get("breed"):
            new_pet["breed"] = result["breed"]
        if result.get("age"):
            new_pet["age"] = result["age"]
    elif result.get("age"):
        new_pet["age"] = result["age"]

    new_filters: SearchFilters = {}
    is_major_switch = is_pet_switched or target_categories

    species = new_pet.get("species")
    if species:
        new_filters["pet_type"] = "강아지" if species == "dog" else "고양이"
    elif result.get("pet_type"):
        new_filters["pet_type"] = result["pet_type"]
    elif prev_filters.get("pet_type"):
        new_filters["pet_type"] = prev_filters["pet_type"]

    current_pet_kr = new_filters.get("pet_type") or "강아지"
    pet_category_map = CATEGORIES.get(current_pet_kr, {})

    detected_category = None
    if target_categories:
        detected_category = target_categories[0]
        # 남은 카테고리(2번째~)는 다중 펫 처리가 안 된 경우에만 현재 펫 유지 상태로 큐에 등록
        # 다중 펫 처리(초코+사료, 바나나+모래)를 이미 한 경우엔 건너뜀 → 중복 방지
        if len(target_categories) > 1 and not multi_pet_categories_registered:
            for category in target_categories[1:]:
                pending_requests.append({"pet_id": None, "category": category})
    elif "recommend" in new_intents and not is_major_switch:
        detected_category = prev_filters.get("category")

    detected_subcategory = normalize_filter_value(result.get("subcategory"))
    if not detected_subcategory and "recommend" in new_intents and not is_major_switch:
        if detected_category == prev_filters.get("category"):
            detected_subcategory = prev_filters.get("subcategory")

    if not detected_subcategory and "recommend" in new_intents:
        found_subcategory = None
        found_category = None
        targets = [detected_category] if detected_category else pet_category_map.keys()
        for category_name in targets:
            subcategories = pet_category_map.get(category_name, {}).get("subcategories", [])
            for subcategory in subcategories:
                keywords = subcategory.split("/") if "/" in subcategory else [subcategory]
                if any(keyword in user_input and len(keyword) > 1 for keyword in keywords):
                    found_subcategory = subcategory
                    found_category = category_name
                    break
            if found_subcategory:
                break
        if found_subcategory:
            detected_subcategory = found_subcategory
            detected_category = found_category
            if detected_category != prev_filters.get("category"):
                detected_subcategory = found_subcategory

    if detected_category:
        new_filters["category"] = detected_category
    if detected_subcategory:
        new_filters["subcategory"] = detected_subcategory

    form_to_subcategory = {
        ("고양이", "캔", "사료"): "주식캔",
        ("고양이", "캔", "간식"): "간식캔",
        ("고양이", "파우치", "사료"): "주식파우치",
        ("고양이", "파우치", "간식"): "간식파우치",
        ("강아지", "캔", "간식"): "캔/파우치",
    }
    previous_form_hint = None if is_major_switch else state.get("form_hint")
    llm_form_hint = result.get("form_hint")
    # 다중 펫 시나리오: user_input 전체에서 form 탐지하되, is_next_request면 pending entry에서 복원
    if is_next_request:
        detected_form = _pending_form_hint  # pending entry에 저장된 form_hint 사용
    else:
        detected_form = next((keyword for keyword in ("캔", "파우치") if keyword in user_input), None)
    final_form_hint = detected_form or llm_form_hint or previous_form_hint

    form_hint = None
    if final_form_hint and current_pet_kr:
        category = new_filters.get("category")
        subcategory = form_to_subcategory.get((current_pet_kr, final_form_hint, category))
        if subcategory:
            new_filters["subcategory"] = subcategory
            form_hint = None
        else:
            form_hint = final_form_hint

    # ★ 다중 펫 시나리오: form_hint가 현재 펫이 아닌 pending 펫에 해당할 경우
    # → pending entry에 form_hint를 저장하고 현재 state에서 제거 (clarify 방지)
    if form_hint and pending_requests and not is_next_request:
        for entry in pending_requests:
            if entry.get("form_hint") is None:
                entry["form_hint"] = form_hint
                logger.info(
                    "form_hint '%s' moved to pending entry pet_id=%s",
                    form_hint, entry.get("pet_id"),
                )
                break
        form_hint = None  # 현재 state에서 제거 → route_intent가 clarify로 빠지지 않음

    if "popularity" in new_intents and "subcategory" in new_filters:
        del new_filters["subcategory"]

    logger.info(
        "intent classified input=%s intents=%s pet=%s filters=%s",
        user_input,
        new_intents,
        new_pet.get("name", new_pet.get("breed", "Unknown")),
        build_search_filters(
            pet_type=new_filters.get("pet_type"),
            category=new_filters.get("category"),
            subcategory=new_filters.get("subcategory"),
        ),
    )

    combined_allergies = list(
        set((overridden_metadata.get("allergies") or state.get("allergies") or []) + exclude_ingredients)
    )

    current_health_concerns = overridden_metadata.get("health_concerns") or state.get("health_concerns") or []
    combined_health_concerns = list(set(current_health_concerns + detected_health_concerns))

    return {
        "intents": new_intents,
        "target_pet_id": target_pet_id,
        "pending_requests": pending_requests,
        "is_pet_switched": is_pet_switched,
        "switched_pet_name": switched_pet_name,
        "domain_intent": result.get("domain_intent") or state.get("domain_intent"),
        "detected_aspect": result.get("detected_aspect") or state.get("detected_aspect"),
        "budget": int(result["budget"]) if result.get("budget") else state.get("budget"),
        "filters": build_search_filters(
            pet_type=new_filters.get("pet_type"),
            category=new_filters.get("category"),
            subcategory=new_filters.get("subcategory"),
        ),
        "pet_profile": new_pet,
        "is_pet_override": is_explicit_pet_info or is_pet_switched,
        "pet_mismatch": False if is_pet_switched else state.get("pet_mismatch", False),
        "form_hint": form_hint,
        "filter_relaxation_count": 0 if target_categories else state.get("filter_relaxation_count", 0),
        "allergies": combined_allergies,
        "health_concerns": combined_health_concerns,
        **overridden_metadata,
    }
