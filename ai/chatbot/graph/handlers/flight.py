from chatbot.graph.state import ChatState

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client # utils에서 필요한 함수 임포트
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller # config에서 설정 및 공통 LLM 호출 함수 임포트

from chatbot.rag.regular_schedule_helper import (
    _parse_schedule_query_with_llm,
    _get_schedule_from_db
)
from chatbot.rag.flight_info_helper import (
    _convert_slots_to_query_format,
    _parse_flight_query_with_llm,
    _call_flight_api,
    _extract_flight_info_from_response
)
from chatbot.rag.llm_tools import _extract_airline_name_with_llm 
from chatbot.rag.utils import get_mongo_collection
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

load_dotenv()

SERVICE_KEY = os.getenv("SERVICE_KEY")

def flight_info_handler(state: ChatState) -> ChatState:
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "flight_info")
    slots = state.get("slots", [])

    if not query_to_process:
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # 🚀 최적화: slot 정보 우선 활용, 없으면 LLM fallback
    parsed_queries = _convert_slots_to_query_format(slots, query_to_process)
    
    if not parsed_queries:
        print("디버그: slot 정보 부족, LLM으로 fallback")
        parsed_queries = _parse_flight_query_with_llm(query_to_process)
    else:
        print("디버그: ⚡ slot 정보로 빠른 처리 완료 (LLM 호출 생략)")

    if not parsed_queries:
        return {**state, "response": "죄송합니다. 요청하신 항공편 정보를 찾을 수 없습니다. 출발지 또는 도착지를 명확히 알려주시겠어요?"}

    all_flight_results = []
    
    for query in parsed_queries:
        flight_id = query.get("flight_id")
        airport_name = query.get("airport_name")
        airline_name = query.get("airline_name")
        departure_airport_name = query.get("departure_airport_name")
        direction = query.get("direction", "departure")
        terminal = query.get("terminal")
        
        from_time = query.get("from_time")
        to_time = query.get("to_time")
        
        if from_time and to_time and from_time == to_time:
            time_obj = datetime.strptime(from_time, "%H%M")
            from_time = (time_obj - timedelta(hours=0)).strftime("%H%M")
            to_time = (time_obj + timedelta(hours=3)).strftime("%H%M")
            
        if not from_time and not to_time:
            # 🕒 개선: 현재 시간에서 2시간 전부터 검색하여 최근 항공편도 포함
            current_time = datetime.now()
            from_time_obj = current_time
            from_time = from_time_obj.strftime("%H%M")
            to_time = "2359"
            print(f"디버그: 특정 시간 언급이 없어 현재 시각({current_time.strftime('%H%M')})부터 검색합니다.")
        
        date_offset = query.get("date_offset", 0)
        search_date = datetime.now() + timedelta(days=date_offset)
        search_date_str = search_date.strftime("%Y%m%d")
        
        api_result = {"data": [], "total_count": 0}
        
        # 📌 수정: 상대 공항 코드만 가져옵니다. 인천에 대한 쿼리는 빈 리스트가 됩니다.
        other_airport_codes = query.get("airport_codes", [])
        airport_code_for_api = other_airport_codes[0] if other_airport_codes else None

        # 📌 핵심 수정: 방향과 상대 공항 코드 유무에 따라 API 호출 로직을 분기합니다.
        # airport_code_for_api가 None일 경우, 해당 파라미터는 전달되지 않아 전체 도착/출발 항공편을 검색합니다.
        if flight_id and not airport_name and not departure_airport_name and not other_airport_codes:
            # 편명만 있고 출발지/도착지 정보가 없으면 양쪽 모두 검색
            print(f"디버그: 편명 '{flight_id}' 전용 검색 - departure/arrival 모두 호출")
            api_result_dep = _call_flight_api("departure", search_date=search_date_str, from_time=from_time, to_time=to_time, flight_id=flight_id)
            api_result_arr = _call_flight_api("arrival", search_date=search_date_str, from_time=from_time, to_time=to_time, flight_id=flight_id)
            api_result["data"].extend(api_result_dep.get("data", []))
            api_result["data"].extend(api_result_arr.get("data", []))
        elif direction == "departure":
            print(f"디버그: 인천 -> '{airport_code_for_api or '모든 도착지'}'에 대한 API 호출 준비 (출발 방향)")
            current_api_result = _call_flight_api(
                "departure",
                search_date=search_date_str,
                from_time=from_time,
                to_time=to_time,
                airport_code=airport_code_for_api,
                flight_id=flight_id
            )
            api_result = current_api_result
            
        elif direction == "arrival":
            print(f"디버그: '{airport_code_for_api or '모든 출발지'}' -> 인천에 대한 API 호출 준비 (도착 방향)")
            current_api_result = _call_flight_api(
                "arrival",
                search_date=search_date_str,
                from_time=from_time,
                to_time=to_time,
                airport_code=airport_code_for_api,
                flight_id=flight_id
            )
            api_result = current_api_result
        
        retrieved_info = []
        if api_result.get("data"):
            retrieved_info = _extract_flight_info_from_response(
                api_result, 
                info_type=query.get("info_type"), 
                found_date=search_date_str,
                airport_name=airport_name,
                airline_name=airline_name,
                departure_airport_name=departure_airport_name,
                requested_direction=direction
            )
            
        if terminal:
            terminal_code = "P01" if "1" in terminal else "P03" if "2" in terminal else "P02" if "탑승동" in terminal else ""
            retrieved_info = [info for info in retrieved_info if info.get("터미널") == terminal_code]
            print(f"디버그: '{terminal}'으로 필터링 완료. 남은 항목 수: {len(retrieved_info)}")

        if not retrieved_info:
            continue

        for info in retrieved_info:
            info["운항날짜"] = search_date_str

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
        truncated_flight_results = cleaned_results[:3]
        context_for_llm = json.dumps(truncated_flight_results, ensure_ascii=False, indent=2)

        intent_description = (
            "사용자가 요청한 항공편에 대한 운항 현황입니다. 다음 검색 결과를 종합하여 "
            "친절하고 명확하게 답변해주세요. "
            "응답에는 찾은 정보만 포함하고, 정보가 없는 항목은 언급하지 마세요. "
        )
        
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)

    return {**state, "response": final_response}

def regular_schedule_query_handler(state: ChatState) -> ChatState:
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "regular_schedule_query")

    if not query_to_process:
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    parsed_queries_data = _parse_schedule_query_with_llm(query_to_process)
    if not parsed_queries_data or not parsed_queries_data.get('requests'):
        return {**state, "response": "죄송합니다. 스케줄 정보를 파악하는 중 문제가 발생했습니다. 다시 시도해 주세요."}
    
    parsed_queries = parsed_queries_data['requests']
    
    all_retrieved_docs = []
    
    for parsed_query in parsed_queries:
        requested_year = parsed_query.get("requested_year")
        current_year = datetime.now().year

        if requested_year and requested_year != current_year:
            response_text = f"죄송합니다. {requested_year}년 운항 스케줄은 아직 확정되지 않았습니다. 현재는 올해({current_year}년) 정보만 제공 가능합니다."
            return {**state, "response": response_text}
            
        airline_name = parsed_query.get("airline_name")
        airport_name = parsed_query.get("airport_name")
        airport_codes = parsed_query.get("airport_codes", [])
        day_name = parsed_query.get("day_of_week")
        time_period = parsed_query.get("time_period")
        direction = parsed_query.get('direction', '출발')
        
        # 📌 수정된 부분: _get_schedule_from_db에 day_name을 전달
        retrieved_db_docs = _get_schedule_from_db(
            direction=direction,
            airport_codes=airport_codes, 
            day_name=day_name, # 파싱된 day_name을 전달
            time_period=time_period,
            airline_name=airline_name
        )

        if isinstance(retrieved_db_docs, str):
            print(f"디버그: 데이터 조회 오류 - {retrieved_db_docs}")
            continue

        # 📌 수정된 부분: 운항 기간이 유효한 스케줄만 필터링
        active_schedules = [
            doc for doc in retrieved_db_docs
            if doc.get('last_date') and doc['last_date'] >= datetime.now()
        ]

        active_schedules.sort(key=lambda x: x.get("scheduled_time", "99:99"))
        top_5_docs = active_schedules[:5]
        
        # 📌 수정된 부분: 데이터가 없으면 빈 리스트를 추가하여 LLM이 처리하도록 함
        if not top_5_docs:
            query_meta = {
                "query_info": { "airport": airport_name, "day": day_name, "direction": direction },
                "schedules": []
            }
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

    # 📌 수정된 부분: all_retrieved_docs가 비어있을 때만 오류 메시지 반환
    if not all_retrieved_docs:
        final_response_text = "죄송합니다. 요청하신 조건에 맞는 정보를 찾을 수 없습니다."
        return {**state, "response": final_response_text}
    
    context_for_llm = json.dumps(all_retrieved_docs, ensure_ascii=False, indent=2)
    
    intent_description = (
        "사용자가 여러 조건에 대한 정기 운항 스케줄 정보를 요청했습니다. "
        "다음 검색 결과를 종합하여, 각 조건별로 구분하여 친절하고 명확하게 답변해주세요. "
        "각 조건에 해당하는 항공편이 없을 경우, '찾을 수 없습니다'와 같은 명확한 메시지를 포함해 주세요."
    )

    final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)
    
    return {**state, "response": final_response}

def airline_info_query_handler(state: ChatState) -> ChatState:
    """
    'airline_info_query' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 항공사 정보를 검색하고 답변을 생성합니다.
    """
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "airline_info_query")
    slots = state.get("slots", [])

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # 🚀 최적화: 슬롯에서 항공사 이름을 우선 활용, 없으면 LLM fallback
    airline_names = [word for word, slot in slots if slot in ['B-airline_name', 'I-airline_name']]
    
    if not airline_names:
        print("디버그: slot에 항공사 정보 없음, LLM으로 fallback")
        extracted_airline = _extract_airline_name_with_llm(query_to_process)
        if extracted_airline:
            airline_names = [extracted_airline]
        print(f"디버그: LLM을 사용해 추출된 항공사 이름: {airline_names}")
    else:
        print(f"디버그: ⚡ slot에서 항공사 정보 추출 완료 (LLM 호출 생략): {airline_names}")
    
    if not airline_names:
        return {**state, "response": "죄송합니다. 요청하신 항공사 정보를 찾을 수 없습니다."}

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
            # 📌 수정된 로직: 검색 쿼리로 'airline_name' 변수를 사용
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

        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)
        
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
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "airport_info")
    # slots 정보를 가져와서 사용합니다.
    slots = state.get("slots", [])

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # 슬롯에서 'B-airport_name' 태그가 붙은 공항 이름을 모두 추출합니다.
    # 📌 슬롯 추출 로직은 그대로 둡니다.
    airport_names = [word for word, slot in slots if slot in ['B-airport_name', 'I-airport_name']]
    
    if not airport_names:
        # 📌 수정된 부분: 슬롯에 공항 이름이 없으면, 재구성된 쿼리를 사용해 검색을 시도합니다.
        airport_names = [query_to_process]
        print("디버그: 슬롯에서 공항 이름을 찾지 못했습니다. 재구성된 쿼리로 검색을 시도합니다.")

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
            
            # 📌 수정된 부분: 검색을 위해 query_to_process를 사용합니다.
            query_embedding = get_query_embedding(query_to_process)
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
        # 📌 수정된 부분: common_llm_rag_caller에 query_to_process를 전달합니다.
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)
        
        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}