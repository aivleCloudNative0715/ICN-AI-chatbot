from chatbot.graph.state import ChatState

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller

import os
from chatbot.graph.state import ChatState
from dotenv import load_dotenv

# 새로운 LLM 파싱 함수를 임포트합니다.
from chatbot.rag.parking_fee_helper import _parse_parking_fee_query_with_llm
from chatbot.rag.parking_walk_time_helper import _parse_parking_walk_time_query_with_llm

load_dotenv()

def parking_fee_info_handler(state: ChatState) -> ChatState:
    """
    'parking_fee_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 주차 요금 및 할인 정책 정보를 검색하고 답변을 생성합니다.
    여러 주차 요금 토픽에 대한 복합 질문도 처리할 수 있도록 개선되었습니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "parking_fee_info")
    slots = state.get("slots", [])
    
    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # ⭐ B-fee_topic 또는 I-fee_topic 슬롯이 여러 개 추출되었는지 확인합니다.
    fee_topic_slots = [word for word, slot in slots if slot in ['B-fee_topic', 'I-fee_topic']]
    
    search_queries = []
    if len(fee_topic_slots) > 1:
        # ⭐ 슬롯이 여러 개일 경우, LLM을 사용해 질문을 분해합니다.
        parsed_queries = _parse_parking_fee_query_with_llm(user_query)
        if parsed_queries and parsed_queries.get("requests"):
            search_queries = [req.get("query") for req in parsed_queries["requests"]]
        
    if not search_queries:
        # ⭐ 분해된 질문이 없거나 슬롯이 하나인 경우, 전체 쿼리를 검색 키워드로 사용합니다.
        search_queries = [user_query]
        print("디버그: 복합 질문으로 파악되지 않아 전체 쿼리로 검색을 시도합니다.")

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
        for query in search_queries:
            print(f"디버그: '{query}'에 대해 검색 시작...")
            
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
            print("디버그: 벡터 검색 결과, 관련 문서가 없습니다.")
            return {**state, "response": "죄송합니다. 요청하신 주차 요금 정보를 찾을 수 없습니다."}

        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")
        
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
    복합 질문(여러 출발지-도착지 쌍)도 처리할 수 있도록 개선되었습니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "parking_walk_time_info")
    
    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    parsed_queries = _parse_parking_walk_time_query_with_llm(user_query)
    
    search_queries = []
    if parsed_queries and parsed_queries.get("requests"):
        search_queries = [req.get("query") for req in parsed_queries["requests"]]
    
    if not search_queries:
        search_queries = [user_query]
        print("디버그: 복합 질문으로 파악되지 않아 전체 쿼리로 검색을 시도합니다.")

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
            # ⭐ 이 부분이 수정되었습니다.
            # RAG 검색 결과가 없을 때, 사용자에게 필요한 정보를 요청하는 응답 생성
            return {**state, "response": "죄송합니다. 해당 주차장 도보 시간 정보를 찾을 수 없습니다. 혹시 이용하시는 항공사나 카운터 번호를 알고 계시면 더 정확한 정보를 찾아드릴 수 있습니다."}

        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")

        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}