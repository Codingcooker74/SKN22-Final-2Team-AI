import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. 환경 변수 로드 및 클라이언트 설정
from dotenv import find_dotenv
load_dotenv(find_dotenv())
client = OpenAI()
LLM_MODEL = "gpt-4o-mini"

# 2. 데이터 경로 설정 및 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORY_FILE = os.path.join(BASE_DIR, "data", "category.json")

def load_categories():
    if not os.path.exists(CATEGORY_FILE):
        return {}
    with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

categories_data = load_categories()

# 3. 시스템 프롬프트 정의
SYSTEM_PROMPT = f"""
당신은 반려동물 쇼핑 및 지식 상담 서비스의 지능형 분류기입니다.
사용자의 입력을 분석하여 지정된 형식의 JSON으로 반환하세요.

### 고도화 규칙 ###
- "A 중에서 B인 거 있어?" 형태의 질문은 100% [recommend]입니다. 
  예: "사료 중에서 연어 없는 거 있어?", "모래 중에서 먼지 안 나는 거 있어?"
- 특정 상품의 '존재 여부'를 묻는 것은 구매 의사가 있는 것으로 간주합니다.

### 분류 규칙:
    ## 분류 가이드라인
    1. 반려동물의 특정 증상(관절, 소화불량, 눈물 등)이나 건강 상태에 대한 언급이 있는 경우, 원인 파악(`domain_qa`)과 해결을 위한 상품 추천(`recommend`) 의도가 모두 포함된 것으로 간주하여 `[domain_qa, recommend]`를 모두 반환합니다.
    2. 단, "A는 왜 생기는 거야?"와 같이 순수하게 지식적인 원인만 묻는 경우에는 `[domain_qa]`만, "A에 좋은 사료 있어?"와 같이 상품 추천만 묻는 경우에는 `[recommend]`만 반환합니다.


1. **intents**: 다음 중 하나 이상을 선택 (리스트 형식)
    - `recommend`: 상품 추천 요청
    - `domain_qa`: 전문적인 지식 질문 (건강, 사료 성분 등)
    - `unclear`: 인사, 잡담, 의도가 불분명하거나 서비스와 무관한 질문 (small_talk 없음, 모두 unclear)

    ### Few-shot Examples ###
    Input: "안녕 반가워" -> Output: ["unclear"]
    Input: "캣잎이 뭐야?" -> Output: ["domain_qa"]
    Input: "강아지가 사료를 안 먹어" -> Output: ["domain_qa"]
    Input: "오늘 날씨 어때?" -> Output: ["unclear"]
    Input: "간식 좀 골라줄래?" -> Output: ["recommend"]

    ### 의도 분류 예시 ###
    입력: "요즘 우리 애 관절이 안 좋아.." -> 출력: [domain_qa, recommend] (상태 언급 -> 원인 상담 + 상품 추천)
    입력: "강아지가 소화불량인 것 같아 왜 그럴까?" -> 출력: [domain_qa, recommend] (원인 파악 + 추천 필요)
    입력: "소화불량일 때 먹이기 좋은 사료나 간식 있어?" -> 출력: [recommend] (구매 목적 명확)
    입력: "우리 고양이가 갑자기 밥을 안 먹어요. 어디 아픈가?" -> 출력: [domain_qa, recommend]
    입력: "눈물 자국은 왜 생기는 거야?" -> 출력: [domain_qa] (지식 탐색)
    입력: "눈물 자국이 심한데 어떤 사료가 좋아?" -> 출력: [recommend] (해결책 검색)
    입력: "소화가 잘되는 사료는 뭐야?" -> 출력: [recommend] (구매 목적)
    입력: "우리 고양이가 갑자기 밥을 안 먹어요. 사료가 맛이 없나?" -> 출력: [recommend, domain_qa]


2. **domain_qa 서브 분류** (intents에 `domain_qa`가 포함된 경우에만 해당, 아니면 null):
   - `health_disease`: 질병 및 건강 관련
   - `care_management`: 일상 관리 및 케어
   - `nutrition_diet`: 영양 및 식단
   - `behavior_psychology`: 행동 및 심리
   - `travel`: 여행 및 외출

3. **category & subcategory**:
   - 아래 제공되는 카테고리 데이터를 기반으로 분류하세요.
   - `pet_type`이 `강아지`인 경우: {list(categories_data.get("강아지", {}).keys())}
   - `pet_type`이 `고양이`인 경우: {list(categories_data.get("고양이", {}).keys())}
   - 각 카테고리에 속하는 `subcategories` 목록에서 가장 적절한 것을 선택하세요.
   - **중요**: 사용자가 "사료", "간식" 등 대분류 이름만 말한 경우에도 이를 `category`에 정확히 매핑하세요.

4. **pet_type**:
   - `강아지`, `고양이` 중 하나. 
   - 사용자가 "냥이", "댕댕이", "개", "고양이" 등 종을 나타내는 단어를 사용하면 즉시 해당 종으로 분류하세요.
   - 만약 "subcategory"의 값이 특정 종(예: 고양이 모래 -> 고양이)에만 해당한다면 이를 바탕으로 종을 추론할 수 있습니다.
   - **중요**: 입력 및 기존 맥락에서 종을 도저히 특정할 수 없는 경우에만 `null`을 반환하세요.

### Few-shot Examples (Multi-turn) ###
- Context: (비어있음) / Input: "사료 추천해줘" 
  -> {{"intents": ["recommend"], "pet_type": null, "category": "사료"}}
- Context: (pet_type 보완 중) / Input: "냥이용" 
  -> {{"intents": ["recommend"], "pet_type": "고양이"}}
- Context: (category 보완 중) / Input: "간식" 
  -> {{"intents": ["recommend"], "category": "간식"}}

5. **detected_aspect**: 다음 중 해당되는 것들을 리스트로 추출 (해당 없으면 null)
   - `기호성`, `생체반응`, `소화/배변`, `제품 성상`, `성분/원료`, `냄새`, `가격/구매`, `배송/포장`

6. **budget**:
   - 사용자가 명시한 금액이 있다면 추출 (예: "2만원대" -> "20000", "5000원 이하" -> "5000").
   - 금액 언급이 없으면 null.

### 카테고리 데이터:
{json.dumps(categories_data, ensure_ascii=False, indent=2)}

### 출력 형식 (JSON):
{{
  "intents": ["recommend"],
  "domain_qa_sub": null,
  "category": "사료",
  "subcategory": "습식사료",
  "pet_type": "강아지",
  "detected_aspect": ["기호성"],
  "budget": "20000"
}}

반드시 JSON 형식으로만 답변하세요.
"""

# --- [NEW] 세션 상태 관리 (LangGraph State 역할) ---
session_state = {
    "current_extraction": None,
    "waiting_for": None,  # 'pet_type', 'category' 등
    "handled_intents": set()  # 이미 처리된 의도 (예: domain_qa)
}

def analyze_intent(user_input):
    # 만약 특정 정보를 기다리는 중이라면, 현재 대화의 맥락(기존 추출 결과)을 프롬프트에 추가
    context_prompt = ""
    if session_state["current_extraction"]:
        context_prompt = f"\n기존 추출 정보: {json.dumps(session_state['current_extraction'], ensure_ascii=False)}"
        if session_state["waiting_for"]:
            context_prompt += f"\n현재 '{session_state['waiting_for']}' 정보를 보완하는 중입니다."

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + context_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def process_node(result):
    """LangGraph의 노드 로직: 추출 결과를 검증하고 다음 행동을 결정"""
    intents = result.get("intents") or []
    
    # [1] recommend 의도 처리 (multi-intent인 경우에도 recommend가 포함되어 있으면 추천 로직 수행)
    if "recommend" in intents:
        # pet_type 확인
        if not result.get("pet_type"):
            session_state["waiting_for"] = "pet_type"
            session_state["current_extraction"] = result
            return "챗봇: 어떤 반려동물 제품을 찾으시나요? (강아지용인가요? 고양이용인가요?)"
        
        # category 확인
        if not result.get("category"):
            session_state["waiting_for"] = "category"
            session_state["current_extraction"] = result
            pet_type = result.get("pet_type")
            avail_cats = ", ".join(categories_data.get(pet_type, {}).keys())
            return f"챗봇: {pet_type}를 위한 어떤 카테고리의 제품을 추천해 드릴까요? ({avail_cats} 등)"
        
        # 정보가 모두 있으면 완료
        session_state["waiting_for"] = None
        session_state["current_extraction"] = None
        return f"챗봇: {result.get('pet_type')}용 {result.get('category')} 카테고리에서 좋은 상품을 찾아드릴게요!"

    # [2] domain_qa 단독 의도 처리
    elif "domain_qa" in intents:
        session_state["waiting_for"] = None
        session_state["current_extraction"] = None
        return f"챗봇: 분석 결과에 따라 지식 답변을 준비 중입니다."

    # [3] unclear 의도 처리 (인사, 잡담 포함)
    elif "unclear" in intents:
        session_state["waiting_for"] = None
        session_state["current_extraction"] = None
        return "챗봇: 죄송합니다, 말씀하신 내용을 잘 이해하지 못했어요. 상품 추천이나 반려동물 건강 관련 질문이 있으신가요? 어떤 도움이 필요하신지 구체적으로 말씀해 주세요."

    return "챗봇: 어떤 도움이 필요하신가요?"

def get_refined_intent(user_input):
    """
    사용자의 입력을 분석하고, 세션 상태(session_state)와 병합하여 
    최종 정제된 의도와 파라미터를 반환합니다.
    """
    # 1. LLM 분석 (context 포함)
    raw_result = analyze_intent(user_input)
    
    # 2. 지능형 병합 로직
    if session_state["current_extraction"]:
        result = session_state["current_extraction"].copy()
        
        # 새로운 결과에서 의미 있는 값들을 기존 정보에 업데이트
        for key, value in raw_result.items():
            if key == "intents":
                new_intents = value or []
                # 신규 의도가 명확하다면(recommend, domain_qa) 업데이트
                if any(i in ["recommend", "domain_qa"] for i in new_intents):
                    result["intents"] = new_intents
                # 만약 신규 의도가 없거나 unclear인 경우, 기존 의도가 recommend라면 유지
                # (추천 프로세스 진행 중에는 intent가 recommend로 계속 유지되어야 함)
                elif "recommend" in (result.get("intents") or []):
                    # 하지만 domain_qa는 이미 처리했을 수 있으므로, 
                    # 이미 처리된(handled) 녀석들은 신류 입력에서 새로 나오지 않는 이상 제거 고려
                    # 여기서는 일단 recommend만 살리고, domain_qa는 신규 발화에서 안 보이면 제외하는 식으로 유도 가능
                    pass
            elif value and value != "None" and value is not None:
                result[key] = value
    else:
        result = raw_result

    # 3. 데이터 정제
    formatted_result = {
        "intents": result.get("intents") or [],
        "category": result.get("category"),
        "subcategory": result.get("subcategory"),
        "pet_type": result.get("pet_type"),
        "detected_aspect": result.get("detected_aspect"),
        "budget": result.get("budget"),
        "domain_qa_sub": result.get("domain_qa_sub")
    }

    # 4. 이미 처리된 의도 처리
    current_intents = list(formatted_result["intents"])
    
    # 만약 recommend가 이미 있었고, 지금 슬롯을 채우고 있다면 recommend를 명시적으로 유지
    if session_state.get("waiting_for") and any(formatted_result.get(k) for k in ["pet_type", "category"]):
         if "recommend" not in current_intents:
             current_intents.append("recommend")

    # domain_qa 처리 여부 확인
    if "domain_qa" in session_state["handled_intents"]:
        # 신규 분석 결과에서 도메인 질문이 명확하게 다시 나오지 않는 한 제거
        # (단순 단어나 슬롯 채우기 답변인 경우 LLM이 이전 컨텍스트 때문에 domain_qa를 또 줄 수 있음)
        if len(user_input.split()) <= 2: # 짧은 답변(예: "강아지", "사료")은 새로운 질문일 확률이 낮음
            current_intents = [i for i in current_intents if i != "domain_qa"]
        elif "domain_qa" not in (raw_result.get("intents") or []):
            current_intents = [i for i in current_intents if i != "domain_qa"]
    
    formatted_result["intents"] = list(set(current_intents)) # 중복 제거

    # 의도 정제: recommend나 domain_qa가 있으면 unclear 제거
    if any(i in ["recommend", "domain_qa"] for i in formatted_result["intents"]):
        formatted_result["intents"] = [i for i in formatted_result["intents"] if i != "unclear"]
    
    # 상태 업데이트
    session_state["current_extraction"] = formatted_result
    
    return formatted_result

def main():
    print(f"[{LLM_MODEL}] LangGraph 스타일 상태 관리 테스트 (종료: q)")
    while True:
        user_input = input("\n사용자: ").strip()
        if user_input.lower() == 'q':
            break
        if not user_input:
            continue
            
        # 1. 정제된 의도 추출
        formatted_result = get_refined_intent(user_input)

        print("\n[현재 분석 상태]")
        print(json.dumps(formatted_result, ensure_ascii=False, indent=2))

        # 2. 로직 처리 노드 (다음 대화 결정)
        response_msg = process_node(formatted_result)
        
        # 만약 domain_qa만 있는 경우라면 즉시 처리된 것으로 간주 (단순 테스트용)
        if "domain_qa" in formatted_result["intents"] and "recommend" not in formatted_result["intents"]:
             session_state["handled_intents"].add("domain_qa")
        
        print(f"\n{response_msg}")

if __name__ == "__main__":
    main()
