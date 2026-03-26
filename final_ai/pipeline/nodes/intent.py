import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.utils import llm, LLM_MODEL
from pipeline.state import ChatState

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

### domain_intent (domain_qa 포함 시)
health_disease / care_management / nutrition_diet / behavior_psychology / travel

### Few-shot
"눈물 자국 심한 포메 사료 추천해줘" → {{"intents":["recommend"],"domain_intent":null,"pet_type":"강아지","category":"사료","subcategory":"눈/눈물","detected_aspect":null,"budget":null}}
"눈물 자국 왜 생겨? 좋은 사료도 알려줘" → {{"intents":["domain_qa","recommend"],"domain_intent":"health_disease","pet_type":null,"category":"사료","subcategory":null,"detected_aspect":null,"budget":null}}
"안녕 반가워" → {{"intents":["unclear"],"domain_intent":null,"pet_type":null,"category":null,"subcategory":null,"detected_aspect":null,"budget":null}}

### 카테고리
{json.dumps(_categories, ensure_ascii=False)}

출력: JSON only
"""


def intent_node(state: ChatState) -> dict:
    user_input = state["user_input"]
    prev_intents = state.get("intents") or []
    prev_filters = state.get("filters") or {}

    context = ""
    # 재질문 중이거나 이전 의도가 있는 경우 컨텍스트 제공
    if (state.get("clarification_count", 0) > 0 or prev_intents) and user_input:
        prev_data = {"intents": prev_intents, "filters": prev_filters}
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
    
    # [Merge Logic] 기존 정보와 새 추출 정보 병합
    new_intents = r.get("intents") or []
    
    # 만약 현재 답변이 "강아지", "사료" 같이 짧은 슬롯 채우기성 답변이라면 
    # 이전 의도(recommend/domain_qa)를 유지할 확률이 높음
    if not any(i in ["recommend", "domain_qa"] for i in new_intents):
        if "recommend" in prev_intents:
            new_intents = ["recommend"]
        elif "domain_qa" in prev_intents:
            new_intents = ["domain_qa"]

    new_filters = prev_filters.copy()
    for k in ["pet_type", "category", "subcategory"]:
        if r.get(k):
            new_filters[k] = r[k]

    print(f"[INTENT] original={r}, merged_intents={new_intents}, merged_filters={new_filters}")

    pet_profile = dict(state.get("pet_profile") or {})
    if new_filters.get("pet_type") and not pet_profile.get("species"):
        pet_profile["species"] = "dog" if new_filters["pet_type"] == "강아지" else "cat"

    return {
        "intents":        new_intents or ["unclear"],
        "domain_intent":  r.get("domain_intent") or state.get("domain_intent"),
        "detected_aspect": r.get("detected_aspect") or state.get("detected_aspect"),
        "budget":         int(r["budget"]) if r.get("budget") else state.get("budget"),
        "filters":        new_filters,
        "pet_profile":             pet_profile,
        "filter_relaxation_count": state.get("filter_relaxation_count", 0),
    }
