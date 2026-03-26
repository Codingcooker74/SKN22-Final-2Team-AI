import os
import json
import psycopg2
from dotenv import load_dotenv, find_dotenv
from fastembed import TextEmbedding

# 1. 환경 변수 로드 및 DB 연결 설정
load_dotenv(find_dotenv())

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "tailtalk_db"),
        user=os.getenv("POSTGRES_USER", "mungnyang"),
        password=os.getenv("POSTGRES_PASSWORD", "finalprojectljs1908"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )

# 2. 임베딩 모델 초기화 (최초 실행 시 모델 다운로드 가능)
print("임베딩 모델 로딩 중...")
model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

def get_recommendations(result):
    """
    Intents 분류 결과(dict)를 받아 하이브리드 검색(Vector + Keyword) 후 
    RRF로 정렬하여 상위 5개 상품명을 반환합니다.
    """
    pet_type = result.get("pet_type") or ""
    category = result.get("category") or ""
    subcategory = result.get("subcategory") or ""
    
    # 쿼리 문자열 생성 (예: "강아지 사료 습식사료")
    query_str = f"{pet_type} {category} {subcategory}".strip()
    if not query_str:
        return []
    
    print(f"\n검색 쿼리: {query_str}")
    
    # [A] 벡터 임베딩 생성
    # query_str에 "query: " 프리픽스를 붙이는 것이 E5 모델의 권장 방식입니다.
    query_embedding = list(model.embed([f"query: {query_str}"]))[0].tolist()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # [B] 벡터 검색 수행 (Cosine Similarity)
        # 1 - (embedding <=> %s) 가 유사도 점수 (1에 가까울수록 유사)
        cur.execute("""
            SELECT goods_name
            FROM product
            ORDER BY embedding <=> %s::vector
            LIMIT 100
        """, (query_embedding,))
        vector_results = [row[0] for row in cur.fetchall()]
        
        # [C] 키워드 검색 수행 (Full Text Search)
        # plainto_tsquery를 사용하여 자연어 처리를 수행합니다.
        cur.execute("""
            SELECT goods_name
            FROM product
            WHERE search_vector @@ plainto_tsquery('simple', %s)
            ORDER BY ts_rank(search_vector, plainto_tsquery('simple', %s)) DESC
            LIMIT 100
        """, (query_str, query_str))
        keyword_results = [row[0] for row in cur.fetchall()]
        
        # [D] RRF (Reciprocal Rank Fusion) 적용
        k = 60
        scores = {}
        
        # 벡터 검색 순위 반영
        for rank, name in enumerate(vector_results):
            scores[name] = scores.get(name, 0) + (1 / (k + rank + 1))
            
        # 키워드 검색 순위 반영
        for rank, name in enumerate(keyword_results):
            scores[name] = scores.get(name, 0) + (1 / (k + rank + 1))
            
        # [E] 최종 정렬 및 상위 5개 추출
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_5 = [item[0] for item in sorted_results[:5]]
        
        return top_5

    except Exception as e:
        print(f"Error during search: {e}")
        return []
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # 테스트용 데이터
    test_result = {
        "intents": ["recommend"],
        "category": "사료",
        "subcategory": "습식사료",
        "pet_type": "강아지",
        "detected_aspect": None,
        "budget": None,
        "domain_qa_sub": None
    }
    
    recommendations = get_recommendations(test_result)
    
    print("\n=== 추천 결과 (Top 5) ===")
    for i, name in enumerate(recommendations, 1):
        print(f"{i}. {name}")
