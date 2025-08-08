from chatbot.graph.state import ChatState

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller
from chatbot.rag.baggage_claim_info_helper import call_arrival_flight_api, _parse_flight_baggage_query_with_llm, _parse_airport_code_with_llm, _generate_final_answer_with_llm
from datetime import datetime, timedelta

def immigration_policy_info_handler(state: ChatState) -> ChatState:
    """
    'immigration_policy_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 입출국 정책 정보를 검색하고 답변을 생성합니다.
    AirportPolicieVector 컬렉션을 사용합니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "immigration_policy_info")

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




def baggage_claim_info_handler(state: ChatState) -> ChatState:
    """
    여객기 운항 현황 상세 조회 서비스 API를 호출하여 수하물 수취대 정보를 제공하는 핸들러.
    """
    print(f"\n--- 수하물 수취 정보 핸들러 실행 ---")
    user_query = state.get("user_input", "")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    parsed_queries = _parse_flight_baggage_query_with_llm(user_query)

    if not parsed_queries or not isinstance(parsed_queries, list):
        response_text = "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다. 다시 시도해 주세요."
        return {**state, "response": response_text}

    final_responses = []
    
    for query in parsed_queries:
        date_offset = query.get("date_offset", 0)
        flight_id = (query.get("flight_id") or "")
        searchday = query.get("searchday", "")
        from_time = query.get("from_time", 0000)
        to_time = query.get("to_time", 2359)
        airport_code = query.get("airport_code", "")
        
        print(f"디버그: 쿼리 정보 - {query}")
        # --- 이 부분을 수정하세요. ---

        if date_offset == "unsupported" or not isinstance(date_offset, (int, float)) or not (-3 <= date_offset <= 6):
            final_responses.append(f"죄송합니다. 조회일 기준 -3일부터 +6일까지만 조회가 가능합니다.")
            continue
        
        if not flight_id:
            if not searchday and not airport_code:
                final_responses.append(f"죄송합니다. 어느 시각에 도착한 항공편인지 더 자세히 알 수 있을까요? 출발지 공항 이름이나 편명을 알려주시면 더 정확한 정보를 제공할 수 있습니다.")
                text_response = "\n".join(final_responses)
                return {**state, "response": text_response}
        
        searchday = datetime.now() + timedelta(days=date_offset)
        searchday = searchday.strftime("%Y%m%d")
        
        if not from_time and not to_time:
            now = datetime.now()
            
            from_dt = now - timedelta(hours=1)
            to_dt = now + timedelta(hours=1)

            # HHMM 형식의 문자열로 변환
            from_time = str(from_dt.strftime("%H%M"))
            to_time = str(to_dt.strftime("%H%M"))
        
        print(f"디버그: 검색일 - {searchday}, 편명 - {flight_id}, 시각 범위 - {from_time} ~ {to_time}, 공항 이름 - {airport_code}")
        
        print(user_query)
        if not airport_code:
            query_embedding = get_query_embedding(user_query)
            print("디버그: 쿼리 임베딩 완료.")

            # MongoDB 벡터 검색
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name="AirportVector",
                vector_index_name="airport_vector_index",
                top_k=1 # 검색할 문서 개수
            )
            
            airport_code = _parse_airport_code_with_llm(retrieved_docs_text[0]) if retrieved_docs_text else None
            
        print(f"디버그: 공항 코드 - {airport_code}")

        # API 호출
        params = {
            "searchday": searchday,
            "flight_id": flight_id,
            "from_time": from_time,
            "to_time": to_time,
            "airport_code": airport_code,
        }

        # None 값 제거
        clean_params = {k: v for k, v in params.items() if v is not None}

        arrival_info = call_arrival_flight_api(clean_params)
        
        print(f"디버그: API 호출 결과 - {arrival_info}")
        
        llm_reponse = _generate_final_answer_with_llm(arrival_info, user_query)
        final_responses.append(llm_reponse)

    
    if not final_responses:
        response_text = f"죄송합니다. 해당하는 항공편들의 운항 정보를 찾을 수 없습니다."
    else:
        response_text = final_responses
        
    disclaimer = (
        "\n\n"  # 시각적 구분을 위한 줄바꿈
        "⚠️ 주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다.\n"
        "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트 또는 해당 항공사/기관/시설에 직접 확인하시기 바랍니다."
    )
    if isinstance(response_text, list):
        response_text = "\n".join(response_text)  # 리스트 → 문자열 변환

    response_text += disclaimer
    
    return {**state, "response": response_text}





def baggage_rule_query_handler(state: ChatState) -> ChatState:
    """
    'baggage_rule_query' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 수하물 규정 정보를 검색하고 답변을 생성합니다.
    AirportPolicyVector 컬렉션을 사용합니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "baggage_rule_query")

    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

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
        query_embedding = get_query_embedding(user_query)
        print("디버그: 쿼리 임베딩 완료.")

        retrieved_docs_text = perform_vector_search(
            query_embedding,
            collection_name=collection_name,
            vector_index_name=vector_index_name,
            query_filter=query_filter,
            top_k=5
        )
        print(f"디버그: MongoDB에서 {len(retrieved_docs_text)}개 문서 검색 완료.")

        context_for_llm = "\n\n".join(retrieved_docs_text)

        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}