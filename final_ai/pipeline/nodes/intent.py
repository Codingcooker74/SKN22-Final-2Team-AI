import json
from pathlib import Path

from final_ai.observability import traceable
from final_ai.pipeline.state import ChatState
from final_ai.pipeline.utils import (
    LLM_MODEL, llm, get_user_pets, get_pet_full_profile
)

CATEGORY_FILE = Path(__file__).resolve().parents[1] / "data" / "category.json"
with open(CATEGORY_FILE, encoding="utf-8") as f:
    _categories = json.load(f)

INTENT_SYSTEM = f"""
당신은 반려동물 쇼핑 서비스의 의도 분류기입니다. 사용자 입력을 분석해 JSON으로만 반환하세요.

### intents 규칙
- recommend : 상품 추천/검색. "A 중에서 B" 형태 포함.
- popularity : "인기 있는", "잘나가는", "베스트셀러", "많이 팔린" 등의 요청이 있을 때 반드시 포함. (일반적으로 recommend와 함께 사용)
- domain_qa : 반려동물 건강·사료·행동 전문 지식 질문
- unclear   : 잡담·인사·무관·의도불명 (small_talk 없음, 모두 unclear)
두 의도 동시 감지 시 복수 반환: ["recommend", "popularity"]

### 펫 정보 추출 (pet_profile)
질문에서 다음 정보를 찾아내세요 (JSON의 루트 레벨에 포함):
- pet_type: 강아지 / 고양이 / null (사용자가 "개", "댕댕이"라고 하면 "강아지"로, "냥이", "고냥이"라고 하면 "고양이"로 정규화하세요.)
- breed: 품종명 (예: 말티즈, 포메라니안, 리트리버 등) / null
- age: 나이 (예: 7살, 3개월, 시니어 등) / null
- mentioned_pet_names: 질문에 언급된 반려동물의 이름 리스트 (예: ["초코", "바나나"]) / []
- is_next_pet_request: 사용자가 "다음 펫도 보여줘", "응 보여줘", "다른 애는?" 등 대기 중인 다른 펫의 추천을 요청하는 긍정 답변인 경우 true / false

### domain_intent (domain_qa 포함 시)
health_disease / care_management / nutrition_diet / behavior_psychology / travel

### 카테고리 추출 규칙
- 제공된 카테고리 목록에서 가장 적합한 것을 선택하세요.
- **소분류(subcategory)를 추출할 때는 반드시 그에 부합하는 대분류(category)도 함께 응답하세요.**
- **사료(주식)와 간식(보상용)을 엄격히 구분하세요.**
- **캔(Can)과 파우치(Pouch)는 서로 다른 제형입니다. 반려동물 종류와 맥락에 따라 아래 매핑을 따르세요:**
  - 고양이 + 캔 + 사료 맥락 → subcategory: "주식캔"
  - 고양이 + 캔 + 간식 맥락 → subcategory: "간식캔"
  - 고양이 + 파우치 + 사료 맥락 → subcategory: "주식파우치"
  - 고양이 + 파우치 + 간식 맥락 → subcategory: "간식파우치"
- **사료인지 간식인지 맥락이 불분명할 때는 subcategory를 설정하지 말고, form_hint 필드에 "캔" 또는 "파우치"를 넣어 반환하세요.**

### Few-shot (문맥 활용 예시)
1. 신규: "7살 말티즈 사료 추천해줘"
   -> {{"intents":["recommend"],"pet_type":"강아지","breed":"말티즈","age":"7살","category":"사료"}}
2. 인기 상품: "개 사료 뭐가 제일 잘나가?"
   -> {{"intents":["recommend", "popularity"],"pet_type":"강아지","category":"사료"}}
3. 고양이 캔 사료 명확: "고양이 주식캔 추천해줘"
   -> {{"intents":["recommend"],"pet_type":"고양이","category":"사료","subcategory":"주식캔"}}
4. 고양이 캔 간식 명확: "고양이 간식캔 보여줘"
   -> {{"intents":["recommend"],"pet_type":"고양이","category":"간식","subcategory":"간식캔"}}
5. 고양이 캔 ambiguous: "고양이 캔 추천해줘" (사료/간식 불명확)
   -> {{"intents":["recommend"],"pet_type":"고양이","form_hint":"캔"}}
6. 강아지 캔 사료: "우리 애 캔 사료 있어?"
   -> {{"intents":["recommend"],"pet_type":"강아지","category":"사료","subcategory":"습식사료"}}
7. 후속(카테고리 변경): "간식은?" (이전: 사료)
   -> {{"intents":["recommend"],"category":"간식"}}
8. 복합 질문: "눈물 왜 생겨? 좋은 사료도 알려줘"
   -> {{"intents":["domain_qa","recommend"],"domain_intent":"health_disease","category":"사료"}}


### 카테고리
{json.dumps(_categories, ensure_ascii=False)}

### 추가 규칙
- 만약 `mentioned_pet_names`가 비어있지 않다면, 해당 이름의 주인공을 위한 추천(recommend) 의도가 포함된 것으로 간주하세요.

출력: JSON only (**반드시 subcategory가 추출되면 category도 함께 포함하세요.**)
"""


@traceable(name="intent_node", run_type="chain")
def intent_node(state: ChatState) -> dict:
    user_input = state["user_input"]
    user_id = state.get("user_id")
    prev_intents = state.get("intents") or []
    prev_filters = state.get("filters") or {}
    prev_pet = state.get("pet_profile") or {}
    pending_pet_ids = state.get("pending_pet_ids") or []
    target_pet_id = state.get("target_pet_id")

    # ── 1. 사용자 펫 목록 조회 ────────────────────────────────────────────────
    user_pets = []
    if user_id:
        user_pets = get_user_pets(user_id)

    context = ""
    if (state.get("clarification_count", 0) > 0 or prev_intents) and user_input:
        prev_data = {
            "intents": prev_intents,
            "filters": prev_filters,
            "pet_profile": prev_pet,
            "pending_pets": [p["name"] for p in user_pets if p["pet_id"] in pending_pet_ids]
        }
        context = f"\n이전 추출 정보: {json.dumps(prev_data, ensure_ascii=False)}"

    res = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM + context},
            {"role": "user",   "content": user_input},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    r = json.loads(res.choices[0].message.content)
    
    new_intents = r.get("intents") or []
    mentioned_names = r.get("mentioned_pet_names") or []
    is_next_request = r.get("is_next_pet_request", False)

    # ── 2. 펫 전환 및 큐잉 로직 ───────────────────────────────────────────────
    is_pet_switched = False
    switched_pet_name = None
    overridden_metadata = {}

    # (A) "다음 펫 보여줘" 긍정 응답 처리
    if is_next_request and pending_pet_ids:
        next_id = pending_pet_ids.pop(0)
        full_p = get_pet_full_profile(next_id)
        if full_p:
            target_pet_id = next_id
            prev_pet = full_p["pet_profile"]
            overridden_metadata = {
                "health_concerns": full_p["health_concerns"],
                "allergies": full_p["allergies"],
                "food_preferences": full_p["food_preferences"],
                "health_traits": "", "breed_context": ""
            }
            is_pet_switched = True
            switched_pet_name = prev_pet.get("name")
            new_intents = ["recommend"] # 추천 메시지 생성 유도
            print(f"[PET_SWITCH] Next pet from queue: {switched_pet_name} ({target_pet_id})")

    # (B) 이름 언급에 의한 전환 및 큐잉
    elif mentioned_names:
        matched_pets = [p for p in user_pets if p["name"] in mentioned_names]
        if matched_pets:
            # 첫 번째 언급된 펫으로 즉시 전환 (현재 타겟과 다를 경우만)
            first_pet = matched_pets[0]
            if str(first_pet["pet_id"]) != str(target_pet_id):
                full_p = get_pet_full_profile(str(first_pet["pet_id"]))
                if full_p:
                    target_pet_id = str(first_pet["pet_id"])
                    prev_pet = full_p["pet_profile"]
                    overridden_metadata = {
                        "health_concerns": full_p["health_concerns"],
                        "allergies": full_p["allergies"],
                        "food_preferences": full_p["food_preferences"],
                        "health_traits": "", "breed_context": ""
                    }
                    is_pet_switched = True
                    switched_pet_name = prev_pet.get("name")
                    if "recommend" not in new_intents: new_intents.append("recommend")
                    print(f"[PET_SWITCH] Name matched switch: {switched_pet_name} ({target_pet_id})")
            
            # 나머지 펫들은 대기열에 추가 (중복 제거)
            for p in matched_pets[1:]:
                pid = str(p["pet_id"])
                if pid != target_pet_id and pid not in pending_pet_ids:
                    pending_pet_ids.append(pid)
            if matched_pets[1:]:
                print(f"[PET_QUEUE] Pending pets updated: {pending_pet_ids}")

    # ── 3. 도메인 분류 및 필터 병합 ─────────────────────────────────────────────
    if not any(i in ["recommend", "domain_qa"] for i in new_intents):
        if "recommend" in prev_intents:
            new_intents = ["recommend"]
        elif "domain_qa" in prev_intents:
            new_intents = ["domain_qa"]

    new_pet = dict(prev_pet)
    is_explicit_pet_info = bool(r.get("pet_type") or r.get("breed"))
    if is_explicit_pet_info:
        new_pet = {}
        if r.get("pet_type"):
            new_pet["species"] = "dog" if r["pet_type"] == "강아지" else "cat"
        if r.get("breed"):
            new_pet["breed"] = r["breed"]
        if r.get("age"):
            new_pet["age"] = r["age"]
    else:
        if r.get("age"):
            new_pet["age"] = r["age"]

    new_filters = prev_filters.copy()
    if is_explicit_pet_info or is_pet_switched:
        new_filters["pet_type"] = r.get("pet_type") or ("강아지" if new_pet.get("species")=="dog" else "고양이")

    pet_species = new_pet.get("species") or prev_pet.get("species")
    current_pet_kr = (
        r.get("pet_type")
        or new_filters.get("pet_type")
        or ("고양이" if pet_species == "cat" else "강아지" if pet_species == "dog" else "강아지")
    )
    pet_cat_map = _categories.get(current_pet_kr, {})

    detected_cat = r.get("category")
    detected_sub = r.get("subcategory")
    
    if detected_sub and not detected_cat:
        for cat_name, info in pet_cat_map.items():
            if detected_sub in info.get("subcategories", []):
                detected_cat = cat_name
                break

    if not detected_sub:
        found_sub, found_cat = None, None
        target_cats = [detected_cat] if detected_cat else pet_cat_map.keys()
        for cat_name in target_cats:
            subs = pet_cat_map.get(cat_name, {}).get("subcategories", [])
            for s in subs:
                keywords = s.split("/") if "/" in s else [s]
                if any(kw in user_input and len(kw) > 1 for kw in keywords):
                    found_sub, found_cat = s, cat_name
                    break
            if found_sub: break
        
        if found_sub:
            detected_sub, detected_cat = found_sub, found_cat

    if detected_cat: new_filters["category"] = detected_cat
    if detected_sub: new_filters["subcategory"] = detected_sub
    
    if detected_cat and not detected_sub and "subcategory" in new_filters:
        safe_subs = pet_cat_map.get(detected_cat, {}).get("subcategories", [])
        if new_filters["subcategory"] not in safe_subs:
            del new_filters["subcategory"]

    FORM_TO_SUB = {
        ("고양이", "캔",    "사료"): "주식캔",
        ("고양이", "캔",    "간식"): "간식캔",
        ("고양이", "파우치", "사료"): "주식파우치",
        ("고양이", "파우치", "간식"): "간식파우치",
        ("강아지", "캔",    "간식"): "캔/파우치",
    }
    prev_form_hint = state.get("form_hint")
    form_hint = None
    detected_form = next((kw for kw in ("캔", "파우치") if kw in user_input), None)

    if detected_form and current_pet_kr:
        new_cat = new_filters.get("category")
        possible = {c: s for (p, f, c), s in FORM_TO_SUB.items() if p == current_pet_kr and f == detected_form}
        if len(possible) == 1:
            only_cat, only_sub = list(possible.items())[0]
            new_filters.update({"category": only_cat, "subcategory": only_sub})
            if new_cat and new_cat in possible:
                new_filters["subcategory"] = possible[new_cat]
                form_hint = None
            elif not new_cat:
                form_hint = detected_form
                new_filters.pop("category", None)
                new_filters.pop("subcategory", None)

    elif prev_form_hint and not detected_form:
        new_cat = new_filters.get("category")
        if new_cat and current_pet_kr:
            resolved_sub = FORM_TO_SUB.get((current_pet_kr, prev_form_hint, new_cat))
            if resolved_sub:
                new_filters["subcategory"] = resolved_sub
            form_hint = None
        else:
            form_hint = prev_form_hint

    print(f"[INTENT] input='{user_input}' -> intents={new_intents}, filters={new_filters}, pet={new_pet.get('name') or new_pet.get('breed')}")

    return {
        "intents":         new_intents or ["unclear"],
        "target_pet_id":   target_pet_id,
        "pending_pet_ids": pending_pet_ids,
        "is_pet_switched": is_pet_switched,
        "switched_pet_name": switched_pet_name,
        "domain_intent":   r.get("domain_intent") or state.get("domain_intent"),
        "detected_aspect": r.get("detected_aspect") or state.get("detected_aspect"),
        "budget":          int(r["budget"]) if r.get("budget") else state.get("budget"),
        "filters":         new_filters,
        "pet_profile":     new_pet,
        "is_pet_override": is_explicit_pet_info or is_pet_switched,
        "pet_mismatch":    False if is_pet_switched else state.get("pet_mismatch", False),
        "form_hint":       form_hint,
        "filter_relaxation_count": state.get("filter_relaxation_count", 0) if not r.get("category") else 0,
        **overridden_metadata
    }

