from chatbot.graph.state import ChatState

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client # utils에서 필요한 함수 임포트
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller # config에서 설정 및 공통 LLM 호출 함수 임포트

from chatbot.rag.regular_schedule_helper import (
    _parse_schedule_query_with_llm,
    _get_schedule_from_db
)
from chatbot.rag.flight_info_helper import (
    _normalize_time,
    _parse_flight_query_with_llm,
    _call_flight_api,
    _extract_flight_info_from_response,
    _get_airport_code_with_llm
)
from chatbot.rag.utils import get_mongo_collection
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

load_dotenv()

SERVICE_KEY = os.getenv("SERVICE_KEY")

def flight_info_handler(state: ChatState) -> ChatState:
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "flight_info_query")

    if not query_to_process:
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # 📌 수정된 부분: 이제 _parse_flight_query_with_llm 함수에 재구성된 쿼리를 전달합니다.
    parsed_queries = _parse_flight_query_with_llm(query_to_process)

    if not parsed_queries or not any(q.get("flight_id") or q.get("airport_name") or q.get("departure_airport_name") for q in parsed_queries):
        return {**state, "response": "죄송합니다. 요청하신 항공편 정보를 찾을 수 없습니다."}

    all_flight_results = []
    
    for query in parsed_queries:
        flight_id = query.get("flight_id")
        airport_name = query.get("airport_name")
        airline_name = query.get("airline_name")
        departure_airport_name = query.get("departure_airport_name")
        direction = query.get("direction", "departure")
        schedule_time = query.get("scheduleDateTime")

        time_to_check = []
        if schedule_time:
            try:
                normalized_time = _normalize_time(schedule_time)
                time_obj = datetime.strptime(normalized_time, "%H%M")
                
                if 1 <= time_obj.hour < 12:
                    from_time_am = (time_obj - timedelta(hours=1)).strftime("%H%M")
                    to_time_am = (time_obj + timedelta(hours=1)).strftime("%H%M")
                    time_to_check.append({"from": from_time_am, "to": to_time_am})
                    
                    time_obj_pm = time_obj + timedelta(hours=12)
                    from_time_pm = (time_obj_pm - timedelta(hours=1)).strftime("%H%M")
                    to_time_pm = (time_obj_pm + timedelta(hours=1)).strftime("%H%M")
                    time_to_check.append({"from": from_time_pm, "to": to_time_pm})
                else:
                    from_time = (time_obj - timedelta(hours=1)).strftime("%H%M")
                    to_time = (time_obj + timedelta(hours=1)).strftime("%H%M")
                    time_to_check.append({"from": from_time, "to": to_time})

            except (ValueError, TypeError) as e:
                print(f"디버그: 시간 파싱 오류 - {e}, 원본 시간: {schedule_time}")
                time_to_check = [{"from": None, "to": None}]
        else:
            time_to_check = [{"from": None, "to": None}]

        api_result = {"data": [], "total_count": 0}
        
        for time_params in time_to_check:
            current_from_time = time_params["from"]
            current_to_time = time_params["to"]
            
            current_api_result = {"data": [], "total_count": 0}

            if departure_airport_name:
                print(f"디버그: 출발지 '{departure_airport_name}'에 대한 API 호출 준비 (도착)")
                airport_code = _get_airport_code_with_llm(departure_airport_name)
                
                if airport_code:
                    current_api_result = _call_flight_api(
                        direction, 
                        airport_code=airport_code,
                        from_time=current_from_time,
                        to_time=current_to_time
                    )
            
            elif airport_name:
                print(f"디버그: 도착지 '{airport_name}'에 대한 API 호출 준비 ({'도착' if direction == 'arrival' else '출발'})")
                airport_code = _get_airport_code_with_llm(airport_name)
                
                if airport_code:
                    current_api_result = _call_flight_api(
                        direction, 
                        airport_code=airport_code,
                        from_time=current_from_time,
                        to_time=current_to_time
                    )

            elif flight_id:
                api_result_dep = _call_flight_api(
                    "departure", 
                    flight_id=flight_id,
                    from_time=current_from_time,
                    to_time=current_to_time
                )
                api_result_arr = _call_flight_api(
                    "arrival",
                    flight_id=flight_id,
                    from_time=current_from_time,
                    to_time=current_to_time
                )
                current_api_result["data"].extend(api_result_dep.get("data", []))
                current_api_result["data"].extend(api_result_arr.get("data", []))
            
            if current_api_result.get("data"):
                api_result["data"].extend(current_api_result.get("data"))
                api_result["total_count"] += current_api_result.get("total_count")
                if not api_result.get("found_date"):
                    api_result["found_date"] = current_api_result.get("found_date")


        retrieved_info = []
        if api_result.get("data"):
            retrieved_info = _extract_flight_info_from_response(
                api_result, 
                info_type=query.get("info_type"), 
                found_date=api_result.get("found_date"),
                airline_name=airline_name,
                departure_airport_name=departure_airport_name,
                departure_airport_code=airport_code if departure_airport_name else None
            )

        if not retrieved_info:
            continue

        for info in retrieved_info:
            info["운항날짜"] = api_result.get("found_date") if api_result.get("found_date") else "알 수 없음"

        all_flight_results.extend(retrieved_info)
    
    if not all_flight_results:
        final_response = "죄송합니다. 요청하신 항공편 정보를 찾을 수 없습니다."
        return {**state, "response": final_response}
    
    cleaned_results = []
    for result in all_flight_results:
        cleaned_item = {k: v for k, v in result.items() if v and v != "정보 없음"}
        if cleaned_item:
            cleaned_results.append(cleaned_item)

    if not cleaned_results:
        final_response = "죄송합니다. 요청하신 항공편 정보를 찾았으나, 세부 정보가 부족합니다."
    else:
        truncated_flight_results = cleaned_results[:2]
        context_for_llm = json.dumps(truncated_flight_results, ensure_ascii=False, indent=2)

        intent_description = (
            "사용자가 요청한 항공편에 대한 운항 현황입니다. 다음 검색 결과를 종합하여 "
            "친절하고 명확하게 답변해주세요. "
            "응답에는 찾은 정보만 포함하고, 정보가 없는 항목은 언급하지 마세요. "
        )
        
        # 📌 수정된 부분: common_llm_rag_caller에도 재구성된 쿼리를 전달합니다.
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)

    return {**state, "response": final_response}

def regular_schedule_query_handler(state: ChatState) -> ChatState:
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "regular_schedule_query")

    if not user_query:
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    parsed_queries_data = _parse_schedule_query_with_llm(user_query)
    if not parsed_queries_data or not parsed_queries_data.get('requests'):
        return {**state, "response": "죄송합니다. 스케줄 정보를 파악하는 중 문제가 발생했습니다. 다시 시도해 주세요."}
    
    parsed_queries = parsed_queries_data['requests']
    
    all_retrieved_docs = []
    not_found_messages = []

    for parsed_query in parsed_queries:
        airline_name = parsed_query.get("airline_name")
        airport_name = parsed_query.get("airport_name")
        
        # ⭐ LLM이 파싱한 airport_codes를 그대로 사용
        airport_codes = parsed_query.get("airport_codes", [])
        
        day_name = parsed_query.get("day_of_week")
        time_period = parsed_query.get("time_period")
        direction = parsed_query.get('direction', '출발')
        
        retrieved_db_docs = _get_schedule_from_db(
            direction=direction,
            airport_codes=airport_codes, 
            day_name=day_name,
            time_period=time_period,
            airline_name=airline_name
        )

        if isinstance(retrieved_db_docs, str):
            not_found_messages.append(f"데이터 조회 중 오류가 발생했습니다: {retrieved_db_docs}")
            continue

        retrieved_db_docs.sort(key=lambda x: x.get("scheduled_time", "99:99"))
        top_5_docs = retrieved_db_docs[:5]
        
        if not top_5_docs:
            not_found_messages.append(f"죄송합니다. '{airport_name}'에서 오는 {day_name} {time_period} {direction} 스케줄 정보를 찾을 수 없습니다.")
        else:
            sanitized_schedules = []
            for doc in top_5_docs:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
                if 'first_date' in doc and isinstance(doc['first_date'], datetime):
                    doc['first_date'] = doc['first_date'].isoformat()
                if 'last_date' in doc and isinstance(doc['last_date'], datetime):
                    doc['last_date'] = doc['last_date'].isoformat()
                if 'scheduled_datetime' in doc and isinstance(doc['scheduled_datetime'], datetime):
                    doc['scheduled_datetime'] = doc['scheduled_datetime'].isoformat()
                sanitized_schedules.append(doc)

            query_meta = {
                "query_info": {
                    "day": day_name,
                    "airport": airport_name,
                    "direction": direction,
                    "airline": airline_name
                },
                "schedules": sanitized_schedules
            }
            all_retrieved_docs.append(query_meta)

    if not all_retrieved_docs:
        final_response_text = "\n".join(not_found_messages)
        if not final_response_text:
            final_response_text = "죄송합니다. 요청하신 조건에 맞는 정보를 찾을 수 없습니다."
        return {**state, "response": final_response_text}
    
    context_for_llm = json.dumps(all_retrieved_docs, ensure_ascii=False, indent=2)
    
    intent_description = (
        "사용자가 여러 조건에 대한 정기 운항 스케줄 정보를 요청했습니다. "
        "다음 검색 결과를 종합하여, 각 조건별로 구분하여 친절하고 명확하게 답변해주세요. "
        "각 조건에 해당하는 항공편이 없을 경우, '찾을 수 없습니다'와 같은 명확한 메시지를 포함해 주세요."
    )

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