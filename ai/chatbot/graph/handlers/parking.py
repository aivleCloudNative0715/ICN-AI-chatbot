from ai.chatbot.graph.state import ChatState

from ai.chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from ai.chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller

def parking_fee_info_handler(state: ChatState) -> ChatState:
    """
    'parking_fee_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 주차 요금 및 할인 정책 정보를 검색하고 답변을 생성합니다.
    ParkingLotPolicyVector 컬렉션을 사용합니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "parking_fee_info")

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
    query_filter = rag_config.get("query_filter") # config에서 가져온 필터 사용 (없으면 None)

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
            query_filter=query_filter, # config에서 가져온 필터가 있다면 전달
            top_k=5 # 검색할 문서 개수
        )
        print(f"디버그: MongoDB에서 {len(retrieved_docs_text)}개 문서 검색 완료.")

        if not retrieved_docs_text:
            print("디버그: 벡터 검색 결과, 관련 문서가 없습니다.")
            return {**state, "response": "죄송합니다. 요청하신 주차 요금 정보를 찾을 수 없습니다."}

        # 3. 검색된 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")
        
        # 4. 공통 LLM 호출 함수를 사용하여 최종 답변 생성
        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)
        
        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

def parking_congestion_prediction_handler(state: ChatState) -> ChatState:
    return {**state, "response": "주차 혼잡도 예측입니다."}

def parking_location_recommendation_handler(state: ChatState) -> ChatState:
    """
    'parking_location_recommendation' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 주차장 위치 정보를 검색하고 답변을 생성합니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "parking_location_recommendation")

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

def parking_availability_query_handler(state: ChatState) -> ChatState:
    return {**state, "response": "주차장 이용 가능 여부입니다."}

def parking_walk_time_info_handler(state: ChatState) -> ChatState:
    """
    'parking_walk_time_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 주차장 도보 시간 정보를 검색하고 답변을 생성합니다.
    ConnectionTimeVector 컬렉션을 사용하며, '주차장' 관련 문서만 필터링합니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "parking_walk_time_info")

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


    if not (collection_name and vector_index_name):
        error_msg = f"죄송합니다. '{intent_name}' 의도에 대한 정보 검색 설정을 찾을 수 없거나 인덱스 이름이 누락되었습니다."
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    try:
        # 1. 사용자 쿼리 임베딩
        query_embedding = get_query_embedding(user_query)
        print("디버그: 쿼리 임베딩 완료.")

        # 2. MongoDB 벡터 검색
        # 필터링된 결과 중에서도 '환승시간' 관련 내용이 있다면 LLM이 이를 무시하도록 프롬프트에 지시합니다.
        retrieved_docs_text = perform_vector_search(
            query_embedding,
            collection_name=collection_name,
            vector_index_name=vector_index_name,
            top_k=5 # 검색할 문서 개수
        )
        print(f"디버그: MongoDB에서 {len(retrieved_docs_text)}개 문서 검색 완료.")

        if not retrieved_docs_text:
            print("디버그: 필터링 및 벡터 검색 결과, 관련 문서가 없습니다.")
            return {**state, "response": "죄송합니다. 요청하신 주차장 도보 시간 정보를 찾을 수 없습니다."}

        # 3. 검색된 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")

        # 4. 공통 LLM 호출 함수를 사용하여 최종 답변 생성
        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}


if __name__ == "__main__":
    print("--- handlers.py 단독 테스트 모드 실행 ---")

    # parking_fee_info_handler 테스트
    # 가상의 ChatState 객체 생성
    # 실제 의도 분류기를 거쳤다는 가정 하에 intent를 직접 지정합니다.
    test_parking_state = ChatState(
        user_input="2터미널 장기 주차 P2에 주차할 건데, 주차 요금 사전 정산 시 모바일로도 가능한지 궁금합니다.",
        intent="parking_fee_info",
        response="" # 초기 응답은 비워둠
    )

    print(f"\n[테스트 시나리오] 사용자 쿼리: '{test_parking_state['user_input']}'")
    
    # parking_fee_info_handler 호출
    updated_state = parking_fee_info_handler(test_parking_state)

    print("\n--- parking_fee_info_handler 결과 ---")
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
