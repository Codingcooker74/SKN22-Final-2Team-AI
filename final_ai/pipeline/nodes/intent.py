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

### Few-shot (문맥 활용 예시)
1. 신규: "7살 말티즈 사료 추천해줘"
   -> {{"intents":["recommend"],"pet_type":"강아지","breed":"말티즈","age":"7살","category":"사료"}}
2. 후속(카테고리 변경): "간식은?" (이전: 7살 말티즈 사료)
   -> {{"intents":["recommend"],"pet_type":null,"breed":null,"age":null,"category":"간식"}}
3. 오버라이드(펫 변경): "4살 페르시안 고양이 장난감은?"
   -> {{"intents":["recommend"],"pet_type":"고양이","breed":"페르시안","age":"4살","category":"장난감"}}
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
    for k in ["category", "subcategory"]:
        if r.get(k):
            new_filters[k] = r[k]

    print(f"[INTENT] input='{user_input}' -> merged_intents={new_intents}, merged_filters={new_filters}, pet={new_pet}")

    return {
        "intents":        new_intents or ["unclear"],
        "domain_intent":  r.get("domain_intent") or state.get("domain_intent"),
        "detected_aspect": r.get("detected_aspect") or state.get("detected_aspect"),
        "budget":         int(r["budget"]) if r.get("budget") else state.get("budget"),
        "filters":        new_filters,
        "pet_profile":    new_pet,
        "filter_relaxation_count": state.get("filter_relaxation_count", 0) if not r.get("category") else 0,
    }
