from chatbot.graph.state import ChatState

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller

def parking_fee_info_handler(state: ChatState) -> ChatState:
    """
    'parking_fee_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 주차 요금 및 할인 정책 정보를 검색하고 답변을 생성합니다.
    여러 주차 요금 토픽에 대한 복합 질문도 처리할 수 있도록 개선되었습니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "parking_fee_info")
    # slots 정보를 가져와서 사용합니다.
    slots = state.get("slots", [])

    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # 슬롯에서 'B-fee_topic' 태그가 붙은 키워드를 모두 추출합니다.
    search_keywords = [word for word, slot in slots if slot == 'B-fee_topic']
    
    # 만약 슬롯에서 키워드를 찾지 못했다면, 전체 쿼리를 사용합니다.
    if not search_keywords:
        search_keywords = [user_query]
        print("디버그: 슬롯에서 키워드를 찾지 못했습니다. 전체 쿼리로 검색을 시도합니다.")

    # RAG_SEARCH_CONFIG에서 현재 의도에 맞는 설정 가져오기
    rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
    collection_name = rag_config.get("collection_name")
    vector_index_name = rag_config.get("vector_index_name")
    intent_description = rag_config.get("description", intent_name)
    query_filter = rag_config.get("query_filter")

    if not (collection_name and vector_index_name):
        error_msg = f"죄송합니다. '{intent_name}' 의도에 대한 정보 검색 설정을 찾을 수 없거나 인덱스 이름이 누락되었습니다."
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    all_retrieved_docs_text = []
    try:
        # 추출된 각 키워드에 대해 RAG 검색을 개별적으로 수행합니다.
        for keyword in search_keywords:
            print(f"디버그: '{keyword}'에 대해 검색 시작...")
            
            # 키워드 임베딩 및 벡터 검색
            query_embedding = get_query_embedding(keyword)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter=query_filter,
                top_k=5 # 검색할 문서 개수
            )
            all_retrieved_docs_text.extend(retrieved_docs_text)
            
        print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs_text)}개 문서 검색 완료.")

        if not all_retrieved_docs_text:
            print("디버그: 벡터 검색 결과, 관련 문서가 없습니다.")
            return {**state, "response": "죄송합니다. 요청하신 주차 요금 정보를 찾을 수 없습니다."}

        # 검색된 모든 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")
        
        # 공통 LLM 호출 함수를 사용하여 최종 답변 생성
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
    여러 주차장 위치에 대한 복합 질문도 처리할 수 있도록 개선되었습니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "parking_location_recommendation")
    # slots 정보를 가져와서 사용합니다.
    slots = state.get("slots", [])

    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # 슬롯에서 'B-parking_lot' 태그가 붙은 주차장 이름을 모두 추출합니다.
    search_keywords = [word for word, slot in slots if slot == 'B-parking_lot']

    # 만약 슬롯에서 키워드를 찾지 못했다면, 전체 쿼리를 사용합니다.
    if not search_keywords:
        search_keywords = [user_query]
        print("디버그: 슬롯에서 주차장 이름을 찾지 못했습니다. 전체 쿼리로 검색을 시도합니다.")

    # RAG_SEARCH_CONFIG에서 현재 의도에 맞는 설정 가져오기
    rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
    collection_name = rag_config.get("collection_name")
    vector_index_name = rag_config.get("vector_index_name")
    intent_description = rag_config.get("description", intent_name)
    query_filter = rag_config.get("query_filter")

    if not (collection_name and vector_index_name):
        error_msg = f"죄송합니다. '{intent_name}' 의도에 대한 정보 검색 설정을 찾을 수 없거나 인덱스 이름이 누락되었습니다."
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    all_retrieved_docs_text = []
    try:
        # 추출된 각 키워드에 대해 RAG 검색을 개별적으로 수행합니다.
        for keyword in search_keywords:
            print(f"디버그: '{keyword}'에 대해 검색 시작...")

            # 키워드 임베딩 및 벡터 검색
            query_embedding = get_query_embedding(keyword)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter=query_filter,
                top_k=5 # 검색할 문서 개수
            )
            all_retrieved_docs_text.extend(retrieved_docs_text)

        print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs_text)}개 문서 검색 완료.")

        # 3. 검색된 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")
        
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