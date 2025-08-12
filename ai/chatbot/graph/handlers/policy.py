from chatbot.graph.state import ChatState
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller
from chatbot.rag.baggage_helper import _parse_baggage_rule_query_with_llm
from chatbot.rag.baggage_claim_info_helper import call_arrival_flight_api, _parse_flight_baggage_query_with_llm, _parse_airport_code_with_llm, _generate_final_answer_with_llm
from chatbot.rag.immigration_helper import _parse_immigration_policy_query_with_llm
from chatbot.rag.flight_info_helper import _call_flight_api, _extract_flight_info_from_response

def immigration_policy_handler(state: ChatState) -> ChatState:
    """
    'immigration_policy_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 입출국 정책 정보를 검색하고 답변을 생성합니다.
    복합 질문(여러 정책 항목)도 처리할 수 있도록 개선되었습니다.
    """
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "immigration_policy_info")

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # ⭐ LLM으로 복합 질문을 분해합니다.
    # 📌 수정된 부분: _parse_immigration_policy_query_with_llm 함수에 재구성된 쿼리를 전달합니다.
    parsed_queries = _parse_immigration_policy_query_with_llm(query_to_process)

    search_queries = []
    if parsed_queries and parsed_queries.get("requests"):
        search_queries = [req.get("query") for req in parsed_queries["requests"]]
    
    if not search_queries:
        # 📌 수정된 부분: 복합 질문으로 파악되지 않으면, 재구성된 쿼리를 사용합니다.
        search_queries = [query_to_process]
        print("디버그: 복합 질문으로 파악되지 않아 최종 쿼리로 검색을 시도합니다.")

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
        for query in search_queries:
            print(f"디버그: '{query}'에 대해 검색 시작...")
            
            # 📌 수정된 부분: 검색을 위해 query_embedding에 query를 전달합니다.
            query_embedding = get_query_embedding(query)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter=query_filter,
                top_k=5
            )
            all_retrieved_docs_text.extend(retrieved_docs_text)
            
        print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs_text)}개 문서 검색 완료.")

        if not all_retrieved_docs_text:
            print("디버그: 필터링 및 벡터 검색 결과, 관련 문서가 없습니다.")
            return {**state, "response": "죄송합니다. 요청하신 입출국 정책 정보를 찾을 수 없습니다."}

        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")

        # 📌 수정된 부분: common_llm_rag_caller에 query_to_process를 전달합니다.
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)

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
    
    kst = ZoneInfo("Asia/Seoul")
    
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    parsed_queries = _parse_flight_baggage_query_with_llm(query_to_process)

    if not parsed_queries or not isinstance(parsed_queries, list):
        response_text = "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다. 다시 시도해 주세요."
        return {**state, "response": response_text}

    final_responses = []
    
    for query in parsed_queries:
        date_offset = query.get("date_offset")
        flight_id = (query.get("flight_id") or "")
        from_time = query.get("from_time")
        to_time = query.get("to_time")
        airport_code = query.get("airport_code", "")
        
        print(f"디버그: 쿼리 정보 - {query}")

        # 📌 수정된 로직: date_offset이 정수이면서 범위를 벗어났을 때만 continue
        if isinstance(date_offset, (int, float)) and not (-3 <= date_offset <= 6):
            final_responses.append(f"죄송합니다. 조회일 기준 -3일부터 +6일까지만 조회가 가능합니다.")
            continue
        
        # 📌 수정된 로직: 필수 정보(편명 또는 공항 코드)가 없을 때 에러 반환
        if not flight_id and not airport_code:
            final_responses.append(f"죄송합니다. 어느 시각에 도착한 항공편인지 더 자세히 알 수 있을까요? 출발지 공항 이름이나 편명을 알려주시면 더 정확한 정보를 제공할 수 있습니다.")
            text_response = "\n".join(final_responses)
            return {**state, "response": text_response}
        
        # 📌 수정된 로직: 날짜와 시간 파라미터 설정
        search_date = datetime.now(tz=kst) + timedelta(days=date_offset or 0)
        search_date_str = search_date.strftime("%Y%m%d")

        if not from_time and not to_time:
            now = datetime.now()
            from_dt = now - timedelta(hours=1)
            to_dt = now + timedelta(hours=1)
            from_time_str = str(from_dt.strftime("%H%M"))
            to_time_str = str(to_dt.strftime("%H%M"))
        else:
            from_time_str = from_time
            to_time_str = to_time
        
        print(f"디버그: 검색일 - {search_date_str}, 편명 - {flight_id}, 시각 범위 - {from_time_str} ~ {to_time_str}, 공항 코드 - {airport_code}")
        
        params = {
            "search_date": search_date_str, # 📌 수정된 매개변수 이름
            "flight_id": flight_id,
            "from_time": from_time_str,
            "to_time": to_time_str,
            "airport_code": airport_code, # 도착편의 경우, 이 코드는 출발 공항 코드를 의미
        }

        clean_params = {k: v for k, v in params.items() if v is not None}

        arrival_info = _call_flight_api(direction="arrival", **clean_params)
        
        print(f"디버그: API 호출 결과 - {arrival_info}")
        
        llm_reponse = _generate_final_answer_with_llm(arrival_info, query_to_process)
        final_responses.append(llm_reponse)

    if not final_responses:
        response_text = f"죄송합니다. 해당하는 항공편들의 운항 정보를 찾을 수 없습니다."
    else:
        response_text = final_responses
        
    disclaimer = (
        "\n\n"
        "주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다.\n"
        "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트 또는 해당 항공사/기관/시설에 직접 확인하시기 바랍니다."
    )
    if isinstance(response_text, list):
        response_text = "\n".join(response_text)

    response_text += disclaimer
    
    return {**state, "response": response_text}

def baggage_rule_query_handler(state: ChatState) -> ChatState:
    """
    'baggage_rule_query' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 수하물 규정 정보를 검색하고 답변을 생성합니다.
    복합 질문(여러 수하물 항목)도 처리할 수 있도록 개선되었습니다.
    """
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "baggage_rule_query")

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # ⭐ LLM으로 복합 질문을 분해합니다.
    # 📌 수정된 부분: _parse_baggage_rule_query_with_llm 함수에 재구성된 쿼리를 전달합니다.
    parsed_queries = _parse_baggage_rule_query_with_llm(query_to_process)

    search_queries = []
    if parsed_queries and parsed_queries.get("requests"):
        search_queries = [req.get("query") for req in parsed_queries["requests"]]
    
    if not search_queries:
        # 📌 수정된 부분: 복합 질문으로 파악되지 않으면, 재구성된 쿼리를 사용합니다.
        search_queries = [query_to_process]
        print("디버그: 복합 질문으로 파악되지 않아 최종 쿼리로 검색을 시도합니다.")

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
        for query in search_queries:
            print(f"디버그: '{query}'에 대해 검색 시작...")
            
            # 📌 수정된 부분: 검색을 위해 query_embedding에 query를 전달합니다.
            query_embedding = get_query_embedding(query)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter=query_filter,
                top_k=5
            )
            all_retrieved_docs_text.extend(retrieved_docs_text)
            
        print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs_text)}개 문서 검색 완료.")

        if not all_retrieved_docs_text:
            print("디버그: 필터링 및 벡터 검색 결과, 관련 문서가 없습니다.")
            return {**state, "response": "죄송합니다. 요청하신 수하물 규정 정보를 찾을 수 없습니다. 혹시 이용하시는 항공사를 알려주시면 더 정확한 규정을 찾아드릴 수 있습니다."}

        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")

        # 📌 수정된 부분: common_llm_rag_caller에 query_to_process를 전달합니다.
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}