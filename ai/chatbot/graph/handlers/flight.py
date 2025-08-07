from chatbot.graph.state import ChatState

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client # utils에서 필요한 함수 임포트
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller # config에서 설정 및 공통 LLM 호출 함수 임포트

from chatbot.rag.flight_info_helper import call_flight_api, _format_flight_info, _parse_flight_query_with_llm
from chatbot.rag.regular_schedule_helper import _parse_schedule_query_with_llm, _get_day_of_week_field
from chatbot.rag.utils import get_mongo_collection
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

SERVICE_KEY = os.getenv("SERVICE_KEY")

def flight_info_handler(state: ChatState) -> ChatState:
    """
    여객기 운항 현황 상세 조회 서비스 API를 호출하여 답변을 생성하는 핸들러.
    """
    print(f"\n--- 항공편 정보 핸들러 실행 ---")
    user_query = state.get("user_input", "")

    parsed_queries = _parse_flight_query_with_llm(user_query)

    if not parsed_queries or not isinstance(parsed_queries, list):
        response_text = "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다. 다시 시도해 주세요."
        return {**state, "response": response_text}

    final_responses = []

    # 도착 관련 키워드 목록
    arrival_keywords = ["출구"]
    # 출발 관련 키워드 목록
    departure_keywords = ["체크인", "카운터", "게이트"]

    for query in parsed_queries:
        date_offset = query.get("date_offset", 0)
        direction = query.get("direction")
        flight_id = query.get("flight_id", "").upper()
        
        # --- 이 부분을 수정하세요. ---
        # query.get() 메서드를 사용하여 키가 없을 때 기본값으로 빈 리스트를 반환합니다.
        requested_info_keywords = query.get("requested_info_keywords", [])

        if date_offset == "unsupported" or not isinstance(date_offset, (int, float)) or not (-3 <= date_offset <= 6):
            final_responses.append(f"죄송합니다. 항공편 정보는 조회일 기준 -3일부터 +6일까지만 조회가 가능합니다.")
            continue
        
        if not flight_id:
            final_responses.append(f"죄송합니다. 운항 정보를 확인하려면 항공편명을 알려주세요.")
            continue
        
        search_date = datetime.now() + timedelta(days=date_offset)
        search_day = search_date.strftime("%Y%m%d")
        date_label = f"{(abs(date_offset))}일 전" if date_offset < 0 else (f"{date_offset}일 뒤" if date_offset > 0 else "오늘")
        
        is_arrival_info_requested = any(kw in requested_info_keywords for kw in arrival_keywords)
        is_departure_info_requested = any(kw in requested_info_keywords for kw in departure_keywords)
        
        # 방향과 요청 정보의 불일치 확인 및 안내
        if is_arrival_info_requested and is_departure_info_requested:
            pass # 둘 다 요청한 경우, 일단 API를 호출하여 정보를 모두 제공
        elif direction == "departure" and is_arrival_info_requested:
            final_responses.append(f"{flight_id}편은 출발편입니다. 출구 정보는 도착편에만 제공됩니다.")
            continue
        elif direction == "arrival" and is_departure_info_requested:
            final_responses.append(f"{flight_id}편은 도착편입니다. 체크인 카운터나 게이트 정보는 출발편에만 제공됩니다.")
            continue

        # API 호출
        if direction is None:
            # 방향이 명확하지 않은 경우 도착과 출발 모두 조회
            arrival_info = call_flight_api({"searchday": search_day, "flight_id": flight_id}, "arrival")
            departure_info = call_flight_api({"searchday": search_day, "flight_id": flight_id}, "departure")

            response_parts = []
            if arrival_info and arrival_info != "api_error":
                response_parts.extend(_format_flight_info(arrival_info, date_label, "arrival", requested_info_keywords))
            if departure_info and departure_info != "api_error":
                if response_parts:
                    response_parts.append("\n- - -")
                response_parts.extend(_format_flight_info(departure_info, date_label, "departure", requested_info_keywords))
            
            if response_parts:
                final_responses.append("\n".join(response_parts))
            else:
                final_responses.append(f"죄송합니다. {date_label}에 해당하는 항공편 ({flight_id}) 운항 정보를 찾을 수 없습니다.")

        else:
            # 방향이 지정된 경우 해당 방향만 조회
            flight_info = call_flight_api({"searchday": search_day, "flight_id": flight_id}, direction)

            if flight_info and flight_info != "api_error":
                response_parts = _format_flight_info(flight_info, date_label, direction, requested_info_keywords)
                final_responses.append("\n".join(response_parts))
            else:
                final_responses.append(f"죄송합니다. {date_label}에 해당하는 {direction}편 ({flight_id}) 운항 정보를 찾을 수 없습니다.")
    
    if not final_responses:
        response_text = f"죄송합니다. {date_label}에 해당하는 항공편들의 운항 정보를 찾을 수 없습니다."
    else:
        response_text = "\n\n" + "\n\n".join(final_responses) + "\n"
        
    disclaimer = (
        "\n\n"  # 시각적 구분을 위한 줄바꿈
        "⚠️ 주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다.\n"
        "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트 또는 해당 항공사/기관/시설에 직접 확인하시기 바랍니다."
    )
    response_text += disclaimer
    
    return {**state, "response": response_text}

def regular_schedule_query_handler(state: ChatState) -> ChatState:
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "regular_schedule_query")

    if not user_query:
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    parsed_queries = _parse_schedule_query_with_llm(user_query)
    if not parsed_queries:
        return {**state, "response": "죄송합니다. 스케줄 정보를 파악하는 중 문제가 발생했습니다. 다시 시도해 주세요."}
    
    all_retrieved_docs = []
    
    for parsed_query in parsed_queries:
        airline_name = parsed_query.get("airline_name")
        airport_name = parsed_query.get("airport_name")
        day_name = parsed_query.get("day_of_week")
        direction = parsed_query.get("direction")
        time_period = parsed_query.get("time_period")
        
        query_filter = {}
        day_field = _get_day_of_week_field(day_name)
        query_filter[day_field] = True
        
        if airline_name:
            query_filter["airline_name_kor"] = airline_name
        
        if airport_name:
            query_filter["airport_name_kor"] = airport_name
        
        if direction:
            if direction == 'arrival':
                query_filter["direction"] = "도착"
            elif direction == 'departure':
                query_filter["direction"] = "출발"

        # 시즌 정보(운항 기간)는 first_date와 last_date 필터로만 처리
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        
        day_of_week_number = {'월요일': 0, '화요일': 1, '수요일': 2, '목요일': 3, '금요일': 4, '토요일': 5, '일요일': 6, '오늘': today.weekday()}
        target_day_number = day_of_week_number.get(day_name, today.weekday())
        
        target_date = today
        days_to_add = (target_day_number - today.weekday() + 7) % 7
        target_date += timedelta(days=days_to_add)
        
        query_filter["first_date"] = {"$lte": target_date}
        query_filter["last_date"] = {"$gte": target_date}
        
        # 시간대별 검색 로직
        if time_period:
            time_filter = {}
            if time_period == '오전':
                time_filter["$gte"] = "06:00"
                time_filter["$lt"] = "12:00"
            elif time_period == '오후':
                time_filter["$gte"] = "12:00"
                time_filter["$lt"] = "18:00"
            elif time_period == '저녁':
                time_filter["$gte"] = "18:00"
                time_filter["$lt"] = "24:00"
            
            if time_filter:
                query_filter["scheduled_time"] = time_filter
        
        try:
            collection = get_mongo_collection(collection_name="FlightSchedule")
            retrieved_docs = list(collection.find(query_filter).sort("scheduled_time"))
            all_retrieved_docs.extend(retrieved_docs)
            print(f"디버그: {parsed_query}에 대해 총 {len(retrieved_docs)}개 문서 검색 완료.")
        except Exception as e:
            error_msg = f"죄송합니다. DB 연결 또는 조회 중 오류가 발생했습니다: {e}"
            print(f"디버그: {error_msg}")
            return {**state, "response": error_msg}

    print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs)}개 문서 검색 완료.")

    if not all_retrieved_docs:
        response_text = "죄송합니다. 요청하신 조건에 맞는 정기 운항 스케줄 정보를 찾을 수 없습니다."
        return {**state, "response": response_text}
    
    context_for_llm = "\n".join([str(doc) for doc in all_retrieved_docs])
    intent_description = "사용자가 요청한 정기 운항 스케줄 정보를 요약하여 친절하게 답변해줘. 여러 항공편 정보를 구조화된 목록 형태로 보기 좋게 정리해줘."
    
    final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)
    
    return {**state, "response": final_response}

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
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # 슬롯에서 여러 항공사 이름을 추출합니다.
    airline_names = [word for word, slot in slots if slot == 'B-airline_name']
    
    # 만약 슬롯에서 항공사 이름을 찾지 못했다면, 전체 쿼리를 사용합니다.
    if not airline_names:
        airline_names = [user_query]
        print("디버그: 슬롯에서 항공사 이름을 찾지 못했습니다. 전체 쿼리로 검색을 시도합니다.")

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
        # 추출된 각 항공사 이름에 대해 RAG 검색을 개별적으로 수행합니다.
        for airline_name in airline_names:
            print(f"디버그: '{airline_name}'에 대해 검색 시작...")
            
            query_embedding = get_query_embedding(airline_name)
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
            return {**state, "response": "죄송합니다. 요청하신 항공사 정보를 찾을 수 없습니다."}

        # 검색된 모든 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        
        # 공통 LLM 호출 함수를 사용하여 최종 답변 생성
        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)
        
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