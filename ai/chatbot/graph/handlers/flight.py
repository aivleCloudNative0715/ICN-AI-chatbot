from ai.chatbot.graph.state import ChatState

from ai.chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client # utils에서 필요한 함수 임포트
from ai.chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller # config에서 설정 및 공통 LLM 호출 함수 임포트

def flight_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "항공편 정보입니다."}

def regular_schedule_query_handler(state: ChatState) -> ChatState:
    return {**state, "response": "정기 운항 스케줄입니다."}

def airline_info_query_handler(state: ChatState) -> ChatState:
    """
    'airline_info_query' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 항공사 정보를 검색하고 임시 답변을 생성합니다.
    """
    user_query = state.get("user_input", "") # ChatState에서 'user_input' 필드를 사용합니다.
    intent_name = state.get("intent", "airline_info_query") # 의도 이름을 가져옵니다.

    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # RAG_SEARCH_CONFIG에서 현재 의도에 맞는 설정 가져오기
    rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
    collection_name = rag_config.get("collection_name")
    vector_index_name = rag_config.get("vector_index_name")
    intent_description = rag_config.get("description", intent_name)
    query_filter = rag_config.get("query_filter") # 추가 필터링이 있다면

    if not collection_name:
        # 설정이 없는 경우 (오류 처리)
        error_msg = f"죄송합니다. '{intent_name}' 의도에 대한 정보 검색 설정을 찾을 수 없습니다."
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    try:
        # 1. 사용자 쿼리 임베딩 (utils에서 호출)
        query_embedding = get_query_embedding(user_query)
        print("디버그: 쿼리 임베딩 완료.")

        # 2. MongoDB 벡터 검색 (utils에서 호출)
        # `perform_vector_search`에 컬렉션 이름과 필터 인자 전달
        retrieved_docs_text = perform_vector_search(
            query_embedding,
            collection_name=collection_name, # config.py에서 가져온 컬렉션 이름 사용
            query_filter=query_filter,       # config.py에서 가져온 필터 사용 (없으면 None)
            vector_index_name=vector_index_name,
            top_k=5                          # 검색할 문서 개수
        )
        print(f"디버그: MongoDB에서 {len(retrieved_docs_text)}개 문서 검색 완료.")

        # 3. 검색된 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(retrieved_docs_text)
        
        # 4. 공통 LLM 호출 함수를 사용하여 최종 답변 생성 (현재는 임시 답변)
        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)
        
        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

def airport_info_query_handler(state: ChatState) -> ChatState:
    """
    'airport_info_query' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 공항 코드, 이름, 위치 정보를 검색하고 답변을 생성합니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "airport_info_query")

    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # RAG_SEARCH_CONFIG에서 현재 의도에 맞는 설정 가져오기
    rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
    collection_name = rag_config.get("collection_name")
    vector_index_name = rag_config.get("vector_index_name")
    intent_description = rag_config.get("description", intent_name)
    query_filter = rag_config.get("query_filter") # 추가 필터링이 있다면 config에서 가져옴

    if not (collection_name and vector_index_name):
        error_msg = f"죄송합니다. '{intent_name}' 의도에 대한 정보 검색 설정을 찾을 수 없거나 인덱스 이름이 누락되었습니다."
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    try:
        # 1. 사용자 쿼리 임베딩
        query_embedding = get_query_embedding(user_query)
        print("디버그: 쿼리 임베딩 완료.")

        # 2. MongoDB 벡터 검색
        retrieved_docs_text = perform_vector_search(
            query_embedding,
            collection_name=collection_name,
            vector_index_name=vector_index_name,
            query_filter=query_filter, # config에서 가져온 필터 사용 (없으면 None)
            top_k=5 # 검색할 문서 개수
        )
        print(f"디버그: MongoDB에서 {len(retrieved_docs_text)}개 문서 검색 완료.")

        # 3. 검색된 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(retrieved_docs_text)
        
        # 4. 공통 LLM 호출 함수를 사용하여 최종 답변 생성
        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)
        
        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}
    

if __name__ == "__main__":
    print("--- handlers.py 단독 테스트 모드 실행 ---")

    # airport_info_query_handler 테스트
    # 가상의 ChatState 객체 생성
    # 실제 의도 분류기를 거쳤다는 가정 하에 intent를 직접 지정합니다.
    test_airport_state = ChatState(
        user_input="LGA는 어떤 공항이야?",
        intent="airport_info_query",
        response="" # 초기 응답은 비워둠
    )

    print(f"\n[테스트 시나리오] 사용자 쿼리: '{test_airport_state['user_input']}'")
    
    # airport_info_query_handler 호출
    updated_state = airport_info_query_handler(test_airport_state)

    print("\n--- airport_info_query_handler 결과 ---")
    print(f"최종 응답: {updated_state.get('response', '응답 없음')}")
    print("--------------------------------------")

    # 테스트 후 MongoDB 클라이언트 연결 종료 (필요하다면)
    # 이 부분은 테스트의 종류에 따라 선택적입니다.
    # 만약 연결이 계속 유지되어야 한다면 주석 처리하거나 필요에 따라 호출 위치를 조절하세요.
    try:
        close_mongo_client()
        print("디버그: MongoDB 클라이언트 연결 종료.")
    except Exception as e:
        print(f"디버그: MongoDB 클라이언트 종료 중 오류 발생: {e}")

    print("\n--- 단독 테스트 완료 ---")
