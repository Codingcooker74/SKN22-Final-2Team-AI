import json
from pathlib import Path

from final_ai.observability import traceable
from final_ai.pipeline.state import ChatState
from final_ai.pipeline.utils import LLM_MODEL, llm

CATEGORY_FILE = Path(__file__).resolve().parents[1] / "data" / "category.json"
with open(CATEGORY_FILE, encoding="utf-8") as f:
    _categories = json.load(f)

INTENT_SYSTEM = f"""
당신은 반려동물 쇼핑 서비스의 의도 분류기입니다. 사용자 입력을 분석해 JSON으로만 반환하세요.

### intents 규칙
- recommend : 상품 추천/검색. "A 중에서 B" 형태 포함.
- domain_qa : 반려동물 건강·사료·행동 전문 지식 질문
- unclear   : 잡담·인사·무관·의도불명 (small_talk 없음, 모두 unclear)
두 의도 동시 감지 시 복수 반환: ["domain_qa", "recommend"]

### 펫 정보 추출 (pet_profile)
질문에서 다음 정보를 찾아내세요 (JSON의 루트 레벨에 포함):
- pet_type: 강아지 / 고양이 / null
- breed: 품종명 (예: 말티즈, 포메라니안, 리트리버 등) / null
- age: 나이 (예: 7살, 3개월, 시니어 등) / null

### domain_intent (domain_qa 포함 시)
health_disease / care_management / nutrition_diet / behavior_psychology / travel

### 카테고리 추출 규칙
- 제공된 카테고리 목록에서 가장 적합한 것을 선택하세요.
- **소분류(subcategory)를 추출할 때는 반드시 그에 부합하는 대분류(category)도 함께 응답하세요.**
- **사료(주식)와 간식(보상용)을 엄격히 구분하세요.**

### Few-shot (문맥 활용 예시)
1. 신규: "7살 말티즈 사료 추천해줘"
   -> {{"intents":["recommend"],"pet_type":"강아지","breed":"말티즈","age":"7살","category":"사료"}}
2. 후속(카테고리 변경): "간식은?" (이전: 사료)
   -> {{"intents":["recommend"],"category":"간식"}}
3. 후속(용품 변경): "급수기 추천해줘" (이전: 간식)
   -> {{"intents":["recommend"],"category":"용품","subcategory":"급식/급수기"}}
4. 복합 질문: "눈물 왜 생겨? 좋은 사료도 알려줘"
   -> {{"intents":["domain_qa","recommend"],"domain_intent":"health_disease","category":"사료"}}

### 카테고리
{json.dumps(_categories, ensure_ascii=False)}

출력: JSON only
"""


@traceable(name="intent_node", run_type="chain")
def intent_node(state: ChatState) -> dict:
    user_input = state["user_input"]
    prev_intents = state.get("intents") or []
    prev_filters = state.get("filters") or {}
    prev_pet = state.get("pet_profile") or {}

    context = ""
    # 재질문 중이거나 이전 정보가 있는 경우 컨텍스트 제공
    if (state.get("clarification_count", 0) > 0 or prev_intents) and user_input:
        prev_data = {
            "intents": prev_intents,
            "filters": prev_filters,
            "pet_profile": prev_pet
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
    
    # [Merge & Override Logic]
    new_intents = r.get("intents") or []
    
    # 1. 의도 유지 (슬롯 채우기성 일 때)
    if not any(i in ["recommend", "domain_qa"] for i in new_intents):
        if "recommend" in prev_intents:
            new_intents = ["recommend"]
        elif "domain_qa" in prev_intents:
            new_intents = ["domain_qa"]

    # 2. 펫 프로필 병합/오버라이드 판단
    new_pet = dict(prev_pet)
    
    # 펫 핵심 정보(종, 품종)가 새롭게 감지되면 기존 정보를 '오버라이드' 한다고 간주
    is_new_pet = bool(r.get("pet_type") or r.get("breed"))
    
    if is_new_pet:
        # 완전히 새로운 펫 정보로 교체
        new_pet = {}
        if r.get("pet_type"):
            new_pet["species"] = "dog" if r["pet_type"] == "강아지" else "cat"
        if r.get("breed"):
            new_pet["breed"] = r["breed"]
        if r.get("age"):
            new_pet["age"] = r["age"]
    else:
        # 펫 정보는 유지하고 개별 필드만 업데이트 (예: 나이만 추가되는 경우 등)
        if r.get("age"):
            new_pet["age"] = r["age"]

    # 3. 검색 필터 병합
    new_filters = prev_filters.copy()
    
    # 펫 정보가 바뀌었다면 필터 상의 pet_type도 강제 업데이트
    if is_new_pet:
        new_filters["pet_type"] = r.get("pet_type")
    
    # 카테고리 및 기타 필드 업데이트
    # [정합성 보정] subcategory는 있는데 category가 없거나 불일치할 경우 보정
    detected_cat = r.get("category")
    detected_sub = r.get("subcategory")
    
    if detected_sub and not detected_cat:
        # subcategory가 속한 category를 _categories에서 찾기
        current_pet_type = r.get("pet_type") or prev_filters.get("pet_type") or "강아지"
        pet_cat_map = _categories.get(current_pet_type, {})
        for cat_name, info in pet_cat_map.items():
            if detected_sub in info.get("subcategories", []):
                detected_cat = cat_name
                break
    
    if detected_cat:
        new_filters["category"] = detected_cat
    if detected_sub:
        new_filters["subcategory"] = detected_sub
    
    # 만약 새로운 카테고리가 들어왔는데 기존 소분류가 남아있다면 초기화 (정합성)
    if detected_cat and not detected_sub and "subcategory" in new_filters:
        # 기존 소분류가 새 카테고리에 속하지 않으면 삭제
        current_pet_type = r.get("pet_type") or prev_filters.get("pet_type") or "강아지"
        safe_subs = _categories.get(current_pet_type, {}).get(detected_cat, {}).get("subcategories", [])
        if new_filters["subcategory"] not in safe_subs:
            del new_filters["subcategory"]

    print(f"[INTENT] input='{user_input}' -> merged_intents={new_intents}, merged_filters={new_filters}, pet={new_pet}")

    # 4. 펫 정보가 바뀌었다면 기존의 특정 펫 기반 메타데이터(관심사, 알러지 등)는 초기화
    # (새로운 펫에게 이전 펫의 건강 고민이나 알러지가 적용되는 것을 방지)
    overridden_metadata = {}
    if is_new_pet:
        overridden_metadata = {
            "health_concerns": [],
            "allergies": [],
            "food_preferences": [],
            "health_traits": "",
            "breed_context": ""
        }

    return {
        "intents":        new_intents or ["unclear"],
        "domain_intent":  r.get("domain_intent") or state.get("domain_intent"),
        "detected_aspect": r.get("detected_aspect") or state.get("detected_aspect"),
        "budget":         int(r["budget"]) if r.get("budget") else state.get("budget"),
        "filters":        new_filters,
        "pet_profile":    new_pet,
        "is_pet_override": is_new_pet,
        "filter_relaxation_count": state.get("filter_relaxation_count", 0) if not r.get("category") else 0,
        **overridden_metadata
    }
