from sentence_transformers import SentenceTransformer
from .client import get_model, query_vector_store

def get_transfer_policy_info(user_input: str) -> list:
    """
    사용자 쿼리를 기반으로 환승 절차 관련 정보를 검색합니다.
    AirportProcedure 컬렉션을 활용하며, '환승' 유형 문서만 필터링합니다.

    Args:
        user_input (str): 사용자의 질문 문자열.

    Returns:
        list: 검색된 관련 문서들의 리스트 (Dictionary 형태).
    """
    model = get_model() # 임베딩 모델 로드
    query_embedding = model.encode(user_input).tolist() # 사용자 쿼리 임베딩

    print(f"\n--- [Transfer Policy] 처리 중 쿼리: '{user_input}' ---")

    # 'AirportProcedure' 컬렉션에서 환승 절차 정보 검색
    # top_k는 검색할 문서의 개수입니다. 필요에 따라 조절할 수 있습니다.
    print(f"  > 'AirportProcedure' 컬렉션에서 환승 정보 검색 시작...")
    raw_procedure_results = query_vector_store("AirportProcedure", query_embedding, top_k=5) 
    print(f"  > 'AirportProcedure' 컬렉션 원본 검색 결과 ({len(raw_procedure_results)}개):")
    for i, res in enumerate(raw_procedure_results):
        print(f"    - Procedure Raw Result {i+1}: Type: {res.get('procedure_type')}, Desc: {res.get('description')[:50]}...")
    
    # airportProcedure 결과 중 'procedure_type'이 '환승'인 문서만 필터링
    print(f"  > 'AirportProcedure' 결과 '환승' 유형으로 필터링 중...")
    filtered_procedure_results = []
    for doc in raw_procedure_results:
        if doc.get('procedure_type') == '환승':
            filtered_procedure_results.append(doc)
            print(f"    - FILTERED_IN: Type: {doc.get('procedure_type')}, Desc: {doc.get('description')[:50]}...")
        else:
            print(f"    - FILTERED_OUT: Type: {doc.get('procedure_type')}, Desc: {doc.get('description')[:50]}...")
    
    # 필터링된 환승 절차 문서를 반환 (최대 3개 정도로 제한)
    final_results = filtered_procedure_results[:3]
    
    print(f"--- [Transfer Policy] 총 결합된 문서 수: {len(final_results)} ---")
    return final_results

# --- 테스트 코드 (이 파일을 직접 실행할 때만 동작) ---
if __name__ == "__main__":
    print("--- get_transfer_policy_info 함수 테스트 시작 ---")
    
    import os
    from dotenv import load_dotenv
    load_dotenv()

    test_queries = [
        "환승 절차는 어떻게 돼?",
        "환승 보안 심사 받아야 해?",
        "공항에서 환승하려면 어디로 가야 해?"
    ]

    for query in test_queries:
        print(f"\n--- 사용자 쿼리: '{query}' ---")
        
        retrieved_docs = get_transfer_policy_info(query)
        
        if retrieved_docs:
            print(f"✔️ 검색 결과 ({len(retrieved_docs)}개):")
            for idx, doc in enumerate(retrieved_docs, 1):
                print(f"  📦 결과 {idx}:")
                if doc.get('procedure_type'):
                    print(f"    - 유형: 공항 절차, 절차 유형: {doc.get('procedure_type')}, 내용: {doc.get('description', '')[:100]}...")
                else:
                    print(f"    - 알 수 없는 유형: {doc}")
        else:
            print("❌ 검색 결과가 없습니다.")
    
    print("\n--- get_transfer_policy_info 함수 테스트 종료 ---")