# ai/chatbot/rag/airline_info.py

from sentence_transformers import SentenceTransformer
from ai.chatbot.rag.client import query_vector_store
import os
from dotenv import load_dotenv

# .env 파일 로드 (루트 폴더에 .env 파일이 있는지 확인하세요)
load_dotenv()

# 모델 로딩은 애플리케이션 시작 시 한 번만 이루어지도록 전역 변수 또는 싱글톤 패턴 활용
# 여기서는 간단하게 모듈 레벨에서 초기화
embedding_model = SentenceTransformer("dragonkue/snowflake-arctic-embed-l-v2.0-ko")

# get_airline_info 함수가 검색된 원본 결과를 반환하도록 수정
# LLM 연동 전 디버깅을 위함
def get_airline_info(user_input: str) -> list: # str 대신 list[dict]를 반환하도록 타입 힌트 변경
    # 1) 쿼리 임베딩 생성
    query_embedding = embedding_model.encode(user_input).tolist()

    # 2) 벡터 검색 실행
    # 🚨 filter={"doc_type": "airline_info"} 부분을 제거합니다.
    results = query_vector_store(query_embedding, top_k=3) 
    
    # 3) 검색된 모든 결과를 그대로 반환
    return results

# --- 테스트 코드 (이 파일을 직접 실행할 때만 동작) ---
if __name__ == "__main__":
    print("--- get_airline_info 함수 테스트 시작 ---")
    
    test_queries = [
        "대한항공 연락처 뭐야?",
        "아시아나항공 고객센터 번호 알려줘",
        "제주항공 전화번호",
        "인천공항 시설 정보", # 항공사 정보가 아닌 쿼리 (이제 필터링되지 않고 검색될 수 있음)
        "없는 항공사 정보"
    ]

    for query in test_queries:
        print(f"\n--- 사용자 쿼리: '{query}' ---")
        
        # get_airline_info 함수 호출
        retrieved_docs = get_airline_info(query)
        
        if retrieved_docs:
            print(f"✔️ 검색 결과 ({len(retrieved_docs)}개):")
            for idx, doc in enumerate(retrieved_docs, 1):
                # 검색된 문서의 주요 필드들을 출력하여 확인
                print(f"  📦 결과 {idx}:")
                print(f"    - 항공사: {doc.get('airline_name_kor', 'N/A')}")
                print(f"    - 연락처: {doc.get('airline_contact', 'N/A')}")
                print(f"    - 원문 텍스트: {doc.get('text', 'N/A')}") # 임베딩에 사용된 원본 텍스트
                # 필요한 경우 다른 필드들도 출력
        else:
            print("❌ 검색 결과가 없습니다.")
    
    print("\n--- get_airline_info 함수 테스트 종료 ---")