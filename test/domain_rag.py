import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CSV 데이터 경로 (절대 경로 보장)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "test", "data", "merged_QnA_final.csv")

def get_domain_answer(user_input, pet_type=None):
    """
    사용자 입력과 pet_type을 받아 CSV 기반 RAG 또는 LLM 직접 답변 수행
    """
    print(f"\n[DEBUG] 사용자 입력: {user_input} (분류: {pet_type})")
    
    # [1] CSV 데이터 로드
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"[ERROR] CSV 로드 실패: {e}")
        df = pd.DataFrame()

    # [2] pet_type 필터링 ('분류' 컬럼 기준)
    if not df.empty and pet_type:
        df = df[df['분류'].astype(str).str.contains(pet_type, na=False)]
        print(f"[DEBUG] 필터링된 데이터 개수: {len(df)}")

    # [3] 간단한 키워드 기반 유사 질문 매칭
    # 여기서는 입력 문장의 키워드가 포함된 질문들을 후보로 수집합니다.
    candidate_context = ""
    if not df.empty:
        keywords = [k for k in user_input.split() if len(k) > 1] # 2글자 이상 키워드
        if keywords:
            # 질문 컬럼에서 키워드가 하나라도 포함된 행 검색
            mask = df['질문'].str.contains('|'.join(keywords), na=False)
            candidates = df[mask].head(5)
            
            if not candidates.empty:
                print(f"[DEBUG] 유사 질문 {len(candidates)}개 발견")
                candidate_list = []
                for _, row in candidates.iterrows():
                    candidate_list.append(f"질문: {row['질문']}\n답변: {row['답변']}")
                candidate_context = "\n\n".join(candidate_list)
            else:
                print("[DEBUG] 유사 질문을 찾지 못함. 직접 답변으로 전환합니다.")
        else:
            print("[DEBUG] 유효한 키워드가 없어 직접 답변으로 전환합니다.")

    # [4] LLM(gpt-4o-mini)을 통한 최종 답변 생성
    # CSV에서 추출한 데이터가 있으면 이를 바탕으로, 없으면 LLM의 기본 지식으로 답변
    
    system_prompt = f"""
당신은 반려동물 전문가입니다. 아래 제공된 [참조 데이터]를 바탕으로 사용자의 질문에 친절하게 답변하세요.

### 규칙:
1. [참조 데이터]에 질문과 직접적인 연관이 있는 답변이 있다면 해당 내용을 적극 활용하세요.
2. 만약 [참조 데이터]에 적절한 내용이 없거나 부족하다면, 당신의 전문 지식을 동원해 최상의 전문가적인 답변을 제공하세요.
3. 반려동물 주인에게 도움이 될 수 있도록 따뜻하고 명확한 정보를 제공해야 합니다.

[참조 데이터]
{candidate_context if candidate_context else '관련된 CSV 데이터가 없습니다. 본인의 지식으로 답변하세요.'}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {e}"

if __name__ == "__main__":
    # 대화형 테스트 코드
    print("=" * 50)
    print("반려동물 도메인 QA RAG 테스트 (종료하려면 'exit' 입력)")
    print("=" * 50)
    
    pet_type = input("반려동물 종류를 입력하세요 (강아지/고양이/없음): ").strip()
    if pet_type in ["없음", ""]:
        pet_type = None
        
    while True:
        query = input("\n질문을 입력하세요: ").strip()
        
        if query.lower() in ["exit", "quit", "종료", "q"]:
            print("테스트를 종료합니다.")
            break
            
        if not query:
            continue
            
        result = get_domain_answer(query, pet_type)
        print("-" * 50)
        print(f"답변:\n{result}")
        print("-" * 50)
