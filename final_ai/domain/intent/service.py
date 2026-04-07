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

# ── 서브카테고리 동의어 매핑 사전 ──────────────────────────────────────────
SUBCATEGORY_SYNONYMS = {
    "껌": "덴탈껌",
    "정수기": "급식/급수기",
    "물그릇": "급식/급수기",
    "식기": "급식/급수기",
    "밥그릇": "급식/급수기",
    "츄르": "져키/스틱",
    "스크래쳐": "스크래쳐/캣타워",
    "캣타워": "스크래쳐/캣타워",
    "이동장": "이동장/캐리어",
    "캐리어": "이동장/캐리어",
    "하우스": "하우스/방석",
    "방석": "하우스/방석",
    "패드": "배변패드",
    "모래": "벤토나이트",
}


def _resolve_category_subcategory(
    *,
    text_to_search: str,
    category: str | None,
    subcategory: str | None,
    pet_type_kr: str,
) -> tuple[str | None, str | None]:
    """
    동의어 매핑 및 계층 구조를 바탕으로 카테고리와 서브카테고리를 보정합니다.
    """
    pet_category_map = CATEGORIES.get(pet_type_kr or "강아지", {})
    combined_text = f"{category or ''} {subcategory or ''} {text_to_search}".strip()
    
    # 1. 동의어 보정
    for alias, canonical in SUBCATEGORY_SYNONYMS.items():
        if alias in combined_text:
            subcategory = canonical
            break

    # 2. 키워드 매칭 (표준 명칭이 아닐 때만)
    standard_names = []
    for cat_info in pet_category_map.values():
        standard_names.extend(cat_info.get("subcategories", []))
        
    if subcategory not in standard_names:
        found_sub, found_cat = None, None
        for cat_name, cat_info in pet_category_map.items():
            for sub in cat_info.get("subcategories", []):
                keywords = [k.strip() for k in sub.replace("(", "/").replace(")", "/").split("/") if k.strip()]
                if any(k in combined_text and len(k) > 1 for k in keywords):
                    found_sub, found_cat = sub, cat_name
                    break
            if found_sub: break
        if found_sub:
            subcategory, category = found_sub, found_cat

    # 3. 부모 카테고리 역추론
    if subcategory:
        for cat_name, cat_info in pet_category_map.items():
            if subcategory in cat_info.get("subcategories", []):
                category = cat_name
                break
                
    return category, subcategory


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
    original_user_input = state["user_input"]
    current_user_input = original_user_input
    user_id = state.get("user_id")
    prev_intents = state.get("intents") or []
    prev_filters = normalize_search_filters(state.get("filters"))
    prev_pet = state.get("pet_profile") or {}
    target_pet_id = state.get("target_pet_id")
    pending_requests = list(state.get("pending_requests") or [])
    decomposed_tasks = list(state.get("decomposed_tasks") or [])

    user_pets = get_user_pets(user_id) if user_id else []
    
    # 1. 초기 분류
    context = _build_context(
        state=state,
        user_input=current_user_input,
        user_pets=user_pets,
        prev_intents=prev_intents,
        prev_filters=prev_filters,
        prev_pet=prev_pet,
        target_pet_id=target_pet_id,
    )
    result = _classify_user_input(current_user_input, context)
    
    is_next_request = result.get("is_next_request", False)

    # 2. 후속 요청 처리 (Sequential Processing)
    if is_next_request and decomposed_tasks:
        next_task = decomposed_tasks.pop(0)
        pet_name = next_task.get("pet_name") or ""
        category = next_task.get("category") or ""
        subcategory = next_task.get("subcategory") or ""
        health = next_task.get("health_concern") or ""
        age = next_task.get("age") or ""
        
        # [개선] LLM 재호출 대신 큐의 데이터를 직접 결과에 매핑하여 정확도 보장
        logger.info("Processing queued task: %s", next_task)
        result = {
            "intents": ["recommend"],
            "mentioned_pet_names": [pet_name] if pet_name else [],
            "target_categories": [category] if category else [],
            "subcategory": subcategory,
            "health_concerns": [health] if health else [],
            "age": age,
            "is_next_request": False,
            "is_synthetic": True # 합성된 데이터임을 표시
        }
        current_user_input = f"{pet_name} {health} {subcategory} {category} 추천".strip()

    new_intents = result.get("intents") or []
    mentioned_names = result.get("mentioned_pet_names") or []
    exclude_ingredients = result.get("exclude_ingredients") or []
    raw_health_concerns = result.get("health_concerns") or []
    target_categories = result.get("target_categories") or []
    is_explicit_pet_info = bool(result.get("pet_type") or result.get("breed"))
    new_decomposed_tasks = result.get("decomposed_tasks") or []

    if new_decomposed_tasks:
        logger.info("─── Query Decomposition Detected ───")
        for i, task in enumerate(new_decomposed_tasks):
            logger.info(
                "Task [%d]: pet_name=%s, category=%s, subcategory=%s, health_concern=%s",
                i + 1,
                task.get("pet_name", "N/A"),
                task.get("category", "N/A"),
                task.get("subcategory", "N/A"),
                task.get("health_concern", "N/A"),
            )
        logger.info("───────────────────────────────────")

    # ── [중요] Decomposition 맥락 방어 로직 ──
    # 질문 원문(original_user_input)에 펫 이름이 2개 이상 직접 언급되지 않았다면, 
    # 과거 이력 때문에 질문을 쪼개는 것을 방지합니다.
    # 단, 합성된 입력(is_synthetic)에 대해서는 이 로직을 건너뜁니다.
    if not result.get("is_synthetic"):
        actual_mentions_in_input = [pet["name"] for pet in user_pets if pet["name"] in original_user_input]
        if len(new_decomposed_tasks) > 1 and len(actual_mentions_in_input) < 2:
            logger.info("Preventing excessive decomposition triggered by history context.")
            new_decomposed_tasks = [] # 쪼개지 않고 현재 입력 전체를 하나로 처리

    def map_health_concerns(concerns):
        detected = []
        for raw in concerns:
            mapped_tag = raw
            for tag, keywords in HEALTH_CONCERN_MAP.items():
                if any(keyword in raw for keyword in keywords) or raw == tag:
                    mapped_tag = tag
                    break
            detected.append(mapped_tag)
        return detected

    detected_health_concerns = map_health_concerns(raw_health_concerns)

    is_pet_switched = False
    switched_pet_name = None
    overridden_metadata = {}
    
    # 펫 타입 결정
    temp_pet = dict(prev_pet)
    if is_explicit_pet_info and not target_pet_id:
        if result.get("pet_type"):
            temp_pet["species"] = "dog" if result["pet_type"] == "강아지" else "cat"
    pet_type_kr = "고양이" if (temp_pet.get("species") == "cat" or result.get("pet_type") == "고양이") else "강아지"

    # ── [중요] decomposed_tasks 개별 보정 로직 ──
    if new_decomposed_tasks and "recommend" in new_intents:
        for task in new_decomposed_tasks:
            task_text = f"{task.get('category', '')} {task.get('subcategory', '')}".strip()
            t_cat, t_sub = _resolve_category_subcategory(
                text_to_search=task_text, 
                category=task.get("category"),
                subcategory=task.get("subcategory"),
                pet_type_kr=pet_type_kr
            )
            task["category"] = t_cat
            task["subcategory"] = t_sub

    # 4. Query Decomposition 처리
    if new_decomposed_tasks and not is_next_request:
        # [수정] 여러 작업이 감지되면, 현재 턴에서는 첫 번째 작업만 수행하고
        # 나머지는 모두 decomposed_tasks 큐에 쌓습니다.
        # 이렇게 함으로써 응답 노드에서 "다음 상품도 보여드릴까요?" 질문이 나가게 됩니다.
        first_task = new_decomposed_tasks[0]
        # 첫 번째 작업을 제외한 나머지를 큐에 저장 (기존에는 1:부터였으므로 동일하지만 의미 명확화)
        decomposed_tasks.extend(new_decomposed_tasks[1:]) 
        
        logger.info("Multi-task detected. Processing first task and queuing %d tasks.", len(new_decomposed_tasks) - 1)

        if first_task.get("pet_name"):
            mentioned_names = [first_task["pet_name"]]
        if first_task.get("category"):
            target_categories = [first_task["category"]]
        if first_task.get("subcategory"):
            result["subcategory"] = first_task["subcategory"]
        if first_task.get("health_concern"):
            detected_health_concerns = map_health_concerns([first_task["health_concern"]])
        if first_task.get("age"):
            result["age"] = first_task["age"]
        
        new_intents = ["recommend"]

    # 5. 펫 매칭 및 전환 로직
    if mentioned_names:
        raw_matched = [pet for pet in user_pets if pet["name"] in mentioned_names]
        matched_pets = sorted(
            raw_matched,
            key=lambda p: mentioned_names.index(p["name"]) if p["name"] in mentioned_names else 999,
        )
        if matched_pets:
            first_pet = matched_pets[0]
            if str(first_pet["pet_id"]) != str(target_pet_id):
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

            if not new_decomposed_tasks:
                remaining_categories = list(target_categories[1:]) if len(target_categories) > 1 else []
                for i, pet in enumerate(matched_pets[1:]):
                    pet_id = str(pet["pet_id"])
                    pending_cat = remaining_categories[i] if i < len(remaining_categories) else None
                    pending_entry = {"pet_id": pet_id, "category": pending_cat}
                    if not any(r.get("pet_id") == pet_id for r in pending_requests):
                        pending_requests.append(pending_entry)
                multi_pet_categories_registered = len(matched_pets) > 1 and bool(remaining_categories)
            else:
                multi_pet_categories_registered = True
    elif is_explicit_pet_info:
        # 이름 언급은 없지만 종/품종 정보가 명시된 경우
        multi_pet_categories_registered = False
        new_breed = result.get("breed")
        current_breed = prev_pet.get("breed")
        if (new_breed and current_breed and new_breed != current_breed) or (new_breed and not target_pet_id):
            target_pet_id = None
            prev_pet = {}
            overridden_metadata = {
                "health_concerns": [],
                "allergies": [],
                "food_preferences": [],
                "breed_context": "",
                "health_traits": "",
            }
            is_pet_switched = True

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

    if new_pet.get("species"):
        new_filters["pet_type"] = "강아지" if new_pet["species"] == "dog" else "고양이"
    elif result.get("pet_type"):
        new_filters["pet_type"] = result["pet_type"]
    elif prev_filters.get("pet_type"):
        new_filters["pet_type"] = prev_filters["pet_type"]

    # 6. 메인 분석 결과 보정
    detected_cat = target_categories[0] if target_categories else None
    detected_sub = normalize_filter_value(result.get("subcategory"))

    if "recommend" in new_intents:
        detected_cat, detected_sub = _resolve_category_subcategory(
            text_to_search=current_user_input,
            category=detected_cat,
            subcategory=detected_sub,
            pet_type_kr=pet_type_kr
        )

    # 필터 적용
    if detected_cat:
        new_filters["category"] = detected_cat
        if target_categories and len(target_categories) > 1 and not multi_pet_categories_registered:
            for category in target_categories[1:]:
                pending_requests.append({"pet_id": None, "category": category})
    elif "recommend" in new_intents and not is_major_switch:
        new_filters["category"] = prev_filters.get("category")

    if detected_sub:
        new_filters["subcategory"] = detected_sub
    elif "recommend" in new_intents and not is_major_switch:
        if new_filters.get("category") == prev_filters.get("category"):
            new_filters["subcategory"] = prev_filters.get("subcategory")

    if "popularity" in new_intents and "subcategory" in new_filters:
        del new_filters["subcategory"]

    logger.info(
        "intent classified input=%s intents=%s pet=%s filters=%s decomposed_count=%s",
        current_user_input,
        new_intents,
        new_pet.get("name", new_pet.get("breed", "Unknown")),
        build_search_filters(
            pet_type=new_filters.get("pet_type"),
            category=new_filters.get("category"),
            subcategory=new_filters.get("subcategory"),
        ),
        len(decomposed_tasks)
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
        "decomposed_tasks": decomposed_tasks,
        "new_decomposed_tasks": new_decomposed_tasks,
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
        "filter_relaxation_count": 0 if target_categories else state.get("filter_relaxation_count", 0),
        "allergies": combined_allergies,
        "health_concerns": combined_health_concerns,
        **overridden_metadata,
    }
