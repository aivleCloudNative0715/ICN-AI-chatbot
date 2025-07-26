# ai/chatbot/rag/facility_guide.py

from sentence_transformers import SentenceTransformer
from .client import get_model, query_vector_store

def get_facility_guide_info(user_input: str) -> list:
    """
    사용자 쿼리를 기반으로 공항 시설 및 입점업체 정보를 검색합니다.
    AirportFacility 컬렉션과 AirportEnterprise 컬렉션을 활용합니다.

    Args:
        user_input (str): 사용자의 질문 문자열.

    Returns:
        list: 검색된 관련 문서들의 리스트 (Dictionary 형태).
    """
    model = get_model() # 임베딩 모델 로드
    query_embedding = model.encode(user_input).tolist() # 사용자 쿼리 임베딩

    print(f"\n--- [Facility Guide] 처리 중 쿼리: '{user_input}' ---")

    combined_results = []

    # 1. 'AirportFacility' 컬렉션에서 정보 검색
    print(f"  > 'AirportFacility' 컬렉션 검색 시작...")
    # top_k는 검색할 문서의 개수입니다. 필요에 따라 조절할 수 있습니다.
    facility_results = query_vector_store("AirportFacility", query_embedding, top_k=3) 
    print(f"  > 'AirportFacility' 컬렉션 검색 결과 ({len(facility_results)}개):")
    for i, res in enumerate(facility_results):
        print(f"    - Facility Result {i+1}: Name: {res.get('facility_name', 'N/A')}, Location: {res.get('location', 'N/A')}, Desc: {res.get('description', 'N/A')[:50]}...")
    
    combined_results.extend(facility_results)

    # 2. 'AirportEnterprise' 컬렉션에서 정보 검색
    print(f"  > 'AirportEnterprise' 컬렉션 검색 시작...")
    # top_k는 검색할 문서의 개수입니다. 필요에 따라 조절할 수 있습니다.
    # AirportEnterprise 컬렉션 이름이 "aiportEnterprise_vector_index" 였으니,
    # 정확한 컬렉션 이름을 "AirportEnterprise"로 가정하고 진행합니다.
    enterprise_results = query_vector_store("AirportEnterprise", query_embedding, top_k=3) 
    print(f"  > 'AirportEnterprise' 컬렉션 검색 결과 ({len(enterprise_results)}개):")
    for i, res in enumerate(enterprise_results):
        print(f"    - Enterprise Result {i+1}: Name: {res.get('enterprise_name', 'N/A')}, Location: {res.get('location', 'N/A')}, Type: {res.get('type', 'N/A')}, Open: {res.get('operating_hours', 'N/A')}")
    
    combined_results.extend(enterprise_results)
    
    # 중복 제거 (선택 사항): 필요하다면 여기에서 문서 ID 등을 기준으로 중복을 제거할 수 있습니다.
    # 지금은 간단하게 두 리스트를 합칩니다.

    print(f"--- [Facility Guide] 총 결합된 문서 수: {len(combined_results)} ---")
    return combined_results

# --- 테스트 코드 (이 파일을 직접 실행할 때만 동작) ---
if __name__ == "__main__":
    print("--- get_facility_guide_info 함수 테스트 시작 ---")
    
    import os
    from dotenv import load_dotenv
    load_dotenv()

    test_queries = [
        "면세점 어디에 있어?",
        "약국이 어디에 있나요?",
        "스타벅스 운영 시간 알려줘",
        "환전소 위치는?",
        "제1터미널에 식당 추천해줘",
        "인천공항 흡연실 위치 알려줘"
    ]

    for query in test_queries:
        print(f"\n--- 사용자 쿼리: '{query}' ---")
        
        retrieved_docs = get_facility_guide_info(query)
        
        if retrieved_docs:
            print(f"✔️ 검색 결과 ({len(retrieved_docs)}개):")
            for idx, doc in enumerate(retrieved_docs, 1):
                print(f"  📦 결과 {idx}:")
                if doc.get('facility_name'):
                    print(f"    - 유형: 시설, 이름: {doc.get('facility_name', 'N/A')}, 위치: {doc.get('location', 'N/A')}, 설명: {doc.get('description', 'N/A')[:100]}...")
                elif doc.get('enterprise_name'):
                    print(f"    - 유형: 입점업체, 이름: {doc.get('enterprise_name', 'N/A')}, 위치: {doc.get('location', 'N/A')}, 운영시간: {doc.get('operating_hours', 'N/A')}, 전화: {doc.get('contact', 'N/A')}")
                else:
                    print(f"    - 알 수 없는 유형: {doc}")
        else:
            print("❌ 검색 결과가 없습니다.")
    
    print("\n--- get_facility_guide_info 함수 테스트 종료 ---")