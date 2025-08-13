from chatbot.graph.state import ChatState

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller
from chatbot.rag.config import client

import os
import requests
from datetime import datetime
import re
from dotenv import load_dotenv
import json

# 새로운 LLM 파싱 함수를 임포트합니다.
from chatbot.rag.parking_fee_helper import _parse_parking_fee_query_with_llm
from chatbot.rag.parking_walk_time_helper import _parse_parking_walk_time_query_with_llm

load_dotenv()

SERVICE_KEY = os.getenv("SERVICE_KEY")
if not SERVICE_KEY:
    raise ValueError("SERVICE_KEY 환경 변수가 설정되지 않았습니다.")

# 주차장 현황 API URL
API_URL = "http://apis.data.go.kr/B551177/StatusOfParking/getTrackingParking"

def parking_fee_info_handler(state: ChatState) -> ChatState:
    """
    'parking_fee_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 주차 요금 및 할인 정책 정보를 검색하고 답변을 생성합니다.
    여러 주차 요금 토픽에 대한 복합 질문도 처리할 수 있도록 개선되었습니다.
    """
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "parking_fee_info")
    slots = state.get("slots", [])
    
    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    fee_topic_slots = [word for word, slot in slots if slot in ['B-fee_topic', 'I-fee_topic']]
    
    search_queries = []
    if len(fee_topic_slots) > 1:
        # 📌 수정된 부분: _parse_parking_fee_query_with_llm 함수에 재구성된 쿼리를 전달합니다.
        parsed_queries = _parse_parking_fee_query_with_llm(query_to_process)
        if parsed_queries and parsed_queries.get("requests"):
            search_queries = [req.get("query") for req in parsed_queries["requests"]]
            
    if not search_queries:
        # ⭐ 분해된 질문이 없거나 슬롯이 하나인 경우, 재구성된 쿼리를 검색 키워드로 사용합니다.
        search_queries = [query_to_process]
        print("디버그: 복합 질문으로 파악되지 않아 최종 쿼리로 검색을 시도합니다.")

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
            print("디버그: 벡터 검색 결과, 관련 문서가 없습니다.")
            return {**state, "response": "죄송합니다. 요청하신 주차 요금 정보를 찾을 수 없습니다."}

        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")
        
        # 📌 수정된 부분: common_llm_rag_caller에 query_to_process를 전달합니다.
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)
        
        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

def parking_congestion_prediction_handler(state: ChatState) -> ChatState:
    return {**state, "response": "추후 제공할 기능입니다! 현재는 실시간 주차장 현황에 대해서만 제공하고 있습니다."}

def parking_location_recommendation_handler(state: ChatState) -> ChatState:
    """
    'parking_location_recommendation' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 주차장 위치 정보를 검색하고 답변을 생성합니다.
    여러 주차장 위치에 대한 복합 질문도 처리할 수 있도록 개선되었습니다.
    """
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "parking_location_recommendation")
    slots = state.get("slots", [])

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # 슬롯에서 'B-parking_lot' 태그가 붙은 주차장 이름을 모두 추출합니다.
    search_keywords = [word for word, slot in slots if slot == ['B-parking_lot', 'I-parking_lot']]

    if not search_keywords:
        # 📌 수정된 부분: 슬롯에 키워드가 없으면, 재구성된 쿼리를 사용해 검색을 시도합니다.
        search_keywords = [query_to_process]
        print("디버그: 슬롯에서 주차장 이름을 찾지 못했습니다. 재구성된 쿼리로 검색을 시도합니다.")

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
        for keyword in search_keywords:
            print(f"디버그: '{keyword}'에 대해 검색 시작...")

            # 📌 수정된 부분: 검색을 위해 query_embedding에 keyword를 전달합니다.
            query_embedding = get_query_embedding(keyword)
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
            return {**state, "response": "죄송합니다. 요청하신 주차장 위치 정보를 찾을 수 없습니다."}

        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")
        
        # 📌 수정된 부분: common_llm_rag_caller에 query_to_process를 전달합니다.
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

def parking_availability_query_handler(state: ChatState) -> ChatState:
    """
    'parking_availability_query' 의도에 대한 RAG 기반 핸들러.
    API를 호출하여 주차장 이용 가능 여부를 확인하고 답변을 생성합니다.
    """
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "parking_availability_query")
    
    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")
    
    params = {
        "serviceKey": SERVICE_KEY,
        "type": "json",
        "numOfRows": 1000,
        "pageNo": 1,
    }
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        
        print(f"디버그: API 응답 텍스트: {response.text[:200]}")  # 처음 200자만 출력
        print(f"디버그: API 응답 상태 코드: {response.status_code}")
        
        response_data = response.json()
        print(response_data)
        items_container = response_data.get("response", {}).get("body", {}).get("items", {})
        if not items_container:
            response_text = "혼잡도 예측 정보를 찾을 수 없습니다. API 응답이 비어있거나 형식이 다릅니다."
            return {**state, "response": response_text}
        
        items = items_container.get("item", []) if isinstance(items_container, dict) else items_container
        if not items:
            response_text = "혼잡도 예측 정보를 찾을 수 없습니다. API 응답 데이터가 비어있습니다."
            return {**state, "response": response_text}
        if isinstance(items, dict): items = [items]
        
        # 주차 가능 대수 계산 및 마이너스 값 처리
        for item in items:
            available_spots = int(item['parkingarea']) - int(item['parking'])
            item['parking'] = str(max(0, available_spots))  # 마이너스면 0으로 설정
        
        print(items)
        # 📌 수정된 부분: 프롬프트에 query_to_process를 추가
        prompt_template = (
            "당신은 인천국제공항의 정보를 제공하는 친절하고 유용한 챗봇입니다. "
            "사용자 질문에 다음 정보를 바탕으로 답변해주세요.\n"
            "사용자 질문: {user_query}\n"
            "검색된 정보: {items}\n"
            "T1은 인천국제공항 제1여객터미널, T2는 제2여객터미널입니다. "
            "datetmp은 YYYY-MM-DD HH:MM:SS 형식입니다. 주차장 상태를 마지막으로 확인한 시간입니다. 이 시간을 가장 먼저 언급하세요. "
            "parking은 주차 가능 대수입니다. parking이 0이면 '만차'라고 표시해주세요.\n"
            "\n"
            "**답변 형식:**\n"
            "1. 먼저 확인 시간을 언급\n"
            "2. ## T1 (제1여객터미널) 섹션으로 T1 주차장들을 모두 나열\n"
            "3. ## T2 (제2여객터미널) 섹션으로 T2 주차장들을 모두 나열\n"
            "4. 각 주차장은 '- **주차장명**: 주차 가능 대수 **N**대 (또는 **만차**)' 형식으로 출력\n"
            "\n"
            "**지침: 답변에서 중요한 정보나 키워드는 Markdown의 볼드체(`**키워드**`)를 사용하여 강조해줘.**"
        )
        
        # 📌 수정된 부분: formatted_prompt에 query_to_process를 전달
        formatted_prompt = prompt_template.format(user_query=query_to_process, items=json.dumps(items, ensure_ascii=False, indent=2))
        
        llm_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.5,
            max_tokens=600
        )
        final_response_text = llm_response.choices[0].message.content
        print(f"\n--- [GPT-4o-mini 응답] ---")
        print(final_response_text)
        
    except requests.RequestException as e:
        print(f"디버그: API 호출 중 오류 발생 - {e}")
        final_response_text = "주차장 이용 가능 여부를 가져오는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 - {e}")
        final_response_text = "주차장 현황 정보를 처리하는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."

    return {**state, "response": final_response_text}

def parking_walk_time_info_handler(state: ChatState) -> ChatState:
    """
    'parking_walk_time_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 주차장 도보 시간 정보를 검색하고 답변을 생성합니다.
    복합 질문(여러 출발지-도착지 쌍)도 처리할 수 있도록 개선되었습니다.
    """
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "parking_walk_time_info")

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # 📌 수정된 부분: _parse_parking_walk_time_query_with_llm 함수에 재구성된 쿼리를 전달합니다.
    parsed_queries = _parse_parking_walk_time_query_with_llm(query_to_process)

    search_queries = []
    if parsed_queries and parsed_queries.get("requests"):
        search_queries = [req.get("query") for req in parsed_queries["requests"]]

    if not search_queries:
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
            return {**state, "response": "죄송합니다. 해당 주차장 도보 시간 정보를 찾을 수 없습니다. 혹시 이용하시는 항공사나 카운터 번호를 알고 계시면 더 정확한 정보를 찾아드릴 수 있습니다."}

        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")

        # 📌 수정된 부분: common_llm_rag_caller에 query_to_process를 전달합니다.
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}