# C:\Users\User\Desktop\ICN-AI-chatbot\ai\chatbot\graph\handlers\facility.py
from ai.chatbot.graph.state import ChatState
from ai.chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from ai.chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller
from ai.chatbot.rag.llm_tools import extract_location_with_llm

def facility_guide_handler(state: ChatState) -> ChatState:
    """
    'facility_guide' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 공항 시설 정보를 검색하고 답변을 생성합니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "facility_guide")
    slots = state.get("slots", [])

    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}
    
    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # 1. 슬롯에서 'B-facility_name'을 추출
    facility_names = [word for word, slot in slots if slot == 'B-facility_name']
    if not facility_names:
        facility_names = [user_query] 
        
    print(f"디버그: 검색할 시설 목록 - {facility_names}")

    # 2. llm_tools.py의 함수를 사용해 위치 정보 추출
    location_keyword = extract_location_with_llm(user_query)
    print(f"디버그: LLM으로 추출된 위치 정보 - {location_keyword}")

    # RAG_SEARCH_CONFIG에서 현재 의도에 맞는 설정 가져오기
    rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
    intent_description = rag_config.get("description", intent_name)
    query_filter = rag_config.get("query_filter")

    # 3. 각 시설 이름별로 RAG 검색을 수행하여 모든 결과를 모읍니다.
    all_retrieved_docs_text = []
    try:
        for facility_name in facility_names:
            # LLM이 추출한 위치 정보가 있다면, 이를 검색 쿼리에 추가
            search_query = f"{location_keyword} {facility_name}" if location_keyword else facility_name
            print(f"디버그: '{search_query}'에 대해 검색 시작...")
            
            query_embedding = get_query_embedding(search_query)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=rag_config.get("collection_name"),
                vector_index_name=rag_config.get("vector_index_name"),
                query_filter=query_filter,
                top_k=3
            )
            all_retrieved_docs_text.extend(retrieved_docs_text)
            
        print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs_text)}개 문서 검색 완료.")

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}
    
    if not all_retrieved_docs_text:
        return {**state, "response": "죄송합니다. 요청하신 시설 정보를 찾을 수 없습니다."}

    context_for_llm = "\n\n".join(all_retrieved_docs_text)
    final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)
    
    return {**state, "response": final_response}