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
    사용자 쿼리를 기반으로 MongoDB에서 항공사 정보를 검색하고 답변을 생성합니다.
    여러 항공사에 대한 복합 질문도 처리할 수 있도록 개선되었습니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "airline_info_query")
    slots = state.get("slots", [])

    if not user_query:
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # 슬롯에서 여러 항공사 이름을 추출합니다.
    airline_names = [slot[0] for slot in slots if slot[1] == 'B-airline_name']
    
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

    try:
        # 1. 사용자 쿼리 임베딩
        query_embedding = get_query_embedding(user_query)
        print("디버그: 쿼리 임베딩 완료.")

        # 2. MongoDB 벡터 검색
        retrieved_docs_text = perform_vector_search(
            query_embedding,
            collection_name=collection_name,
            vector_index_name=vector_index_name,
            query_filter=query_filter,
            top_k=3
        )
        print(f"디버그: MongoDB에서 {len(retrieved_docs_text)}개 문서 검색 완료.")

        # 3. 검색된 문서를 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(retrieved_docs_text)

        # 4. 공통 LLM 호출 함수를 사용하여 최종 답변 생성
        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)

        # 복합 질문에 대한 추가 처리: 여러 항공사 이름이 슬롯에 포함된 경우
        if len(airline_names) > 1:
            # LLM이 생성한 답변에 각 항공사 정보가 모두 포함되었는지 확인
            # 간단한 검증 로직으로, 각 항공사 이름이 답변에 포함되어 있는지 확인
            found_all_airlines = all(name in final_response for name in airline_names)

            if not found_all_airlines:
                # LLM이 모든 항공사 정보를 포함하지 못했을 경우를 대비하여,
                # 각 항공사에 대해 별도로 RAG를 수행하여 답변을 결합하는 로직을 추가할 수 있습니다.
                # 여기서는 간단하게 경고 메시지를 추가합니다.
                print("디버그: LLM이 모든 항공사 정보를 포함하지 못했을 수 있습니다. 별도 처리가 필요합니다.")

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

def airport_info_handler(state: ChatState) -> ChatState:
    """
    'airport_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 공항 정보를 검색하고 답변을 생성합니다.
    여러 공항에 대한 복합 질문도 처리할 수 있도록 개선되었습니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "airport_info")
    # slots 정보를 가져와서 사용합니다.
    slots = state.get("slots", [])

    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # 슬롯에서 'B-airport_name' 태그가 붙은 공항 이름을 모두 추출합니다.
    airport_names = [word for word, slot in slots if slot == 'B-airport_name']
    
    # 만약 슬롯에서 공항 이름을 찾지 못했다면, 전체 쿼리를 사용합니다.
    if not airport_names:
        airport_names = [user_query]
        print("디버그: 슬롯에서 공항 이름을 찾지 못했습니다. 전체 쿼리로 검색을 시도합니다.")

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
        # 추출된 각 공항 이름에 대해 RAG 검색을 개별적으로 수행합니다.
        for airport_name in airport_names:
            print(f"디버그: '{airport_name}'에 대해 검색 시작...")
            
            query_embedding = get_query_embedding(airport_name)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter=query_filter,
                top_k=3
            )
            all_retrieved_docs_text.extend(retrieved_docs_text)
            
        print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs_text)}개 문서 검색 완료.")

        if not all_retrieved_docs_text:
            return {**state, "response": "죄송합니다. 요청하신 공항 정보를 찾을 수 없습니다."}

        # 검색된 모든 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        
        # 공통 LLM 호출 함수를 사용하여 최종 답변 생성
        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)
        
        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}