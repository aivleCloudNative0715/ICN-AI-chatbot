import os
from datetime import datetime, timedelta, date
import re
from chatbot.graph.state import ChatState
from dotenv import load_dotenv
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller
from chatbot.rag.airport_congestion_helpers import _get_congestion_level, VALID_AREAS, _parse_query_with_llm, _get_congestion_data_from_db, _get_daily_congestion_data_from_db, _map_area_to_db_key
import json
from zoneinfo import ZoneInfo

load_dotenv()

def airport_congestion_prediction_handler(state: ChatState) -> ChatState:
    print(f"\n--- 공항 혼잡도 예측 핸들러 실행 ---")

    query_to_process = state.get("rephrased_query") or state.get("user_input", "")

    if not query_to_process:
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}
    
    parsed_query = _parse_query_with_llm(query_to_process)
    if parsed_query is None:
        return {**state, "response": "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다. 다시 시도해 주세요."}
    
    requests_list = parsed_query.get("requests", [])
    if not requests_list:
        requests_list = [{"terminal": None, "area": None, "date": "today", "time": "합계", "is_daily": True}]
        
    response_parts_data = []

    # 📌 수정된 부분: 요청 날짜가 'tomorrow' 또는 'unsupported'인 경우를 처리
    for request in requests_list:
        request_date = request.get("date")
        if request_date == "tomorrow" or request_date == "unsupported":
            return {**state, "response": "죄송합니다. 오늘의 혼잡도 예측 정보만 제공되고 있습니다. 그 외의 날은 확인 불가능합니다."}

    # 현재 날짜를 YYYYMMDD 형식의 문자열로 변환하여 사용
    requested_date_str = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    
    try:
        for request in requests_list:
            terminal_number = request.get("terminal")
            area_name = request.get("area")
            requested_time = request.get("time")
            is_daily_request = request.get("is_daily", False)
            
            data = None
            if is_daily_request:
                data = _get_daily_congestion_data_from_db()
            else:
                query_time = requested_time if requested_time is not None else datetime.now(ZoneInfo("Asia/Seoul")).hour
                data = _get_congestion_data_from_db(requested_date_str, query_time)

            if not data:
                continue

            terminals_to_process = [terminal_number] if terminal_number is not None else [1, 2]

            for t_num in terminals_to_process:
                if is_daily_request:
                    total_sum = data.get(f"t{t_num}_arrival_sum", 0.0) + data.get(f"t{t_num}_departure_sum", 0.0)
                    response_parts_data.append({"터미널": t_num, "유형": "하루 전체", "승객수": total_sum, "혼잡도": _get_congestion_level(t_num, total_sum)})
                else:
                    if area_name is not None:
                        mapped_key = _map_area_to_db_key(t_num, area_name)
                        if mapped_key:
                            passenger_count = data.get(mapped_key, 0.0)
                            response_parts_data.append({"터미널": t_num, "구역": area_name, "시간": query_time, "승객수": passenger_count})
                    else:
                        total_count = float(data.get(f"t{t_num}_arrival_sum", 0.0)) + float(data.get(f"t{t_num}_departure_sum", 0.0))
                        congestion = _get_congestion_level(t_num, total_count)
                        response_parts_data.append({"터미널": t_num, "유형": "시간대별", "시간": query_time, "승객수": total_count, "혼잡도": congestion})
        
        context_for_llm = json.dumps(response_parts_data, ensure_ascii=False, indent=2)

        if not response_parts_data:
            final_response_text = f"죄송합니다. {datetime.now(ZoneInfo("Asia/Seoul")).strftime('%Y년 %m월 %d일')}의 혼잡도 정보를 찾을 수 없습니다."
        else:
            final_response_text = common_llm_rag_caller(query_to_process, context_for_llm, "공항 혼잡도 예측 정보", "airport_congestion_prediction")

    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 - {e}")
        final_response_text = "혼잡도 정보를 처리하는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    return {**state, "response": final_response_text}