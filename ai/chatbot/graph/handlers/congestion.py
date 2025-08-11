import os
from datetime import datetime, timedelta
import re
from chatbot.graph.state import ChatState
from dotenv import load_dotenv
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller
from chatbot.rag.airport_congestion_helpers import _get_congestion_level, VALID_AREAS, _parse_query_with_llm, _get_congestion_data_from_db, _get_daily_congestion_data_from_db, _map_area_to_db_key
import json

load_dotenv()

def airport_congestion_prediction_handler(state: ChatState) -> ChatState:
    """
    공항 혼잡도 예측 정보를 MongoDB에서 조회하여 답변을 생성하는 핸들러입니다.
    """
    print(f"\n--- 공항 혼잡도 예측 핸들러 실행 ---")
    
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    
    if not query_to_process:
        response_text = "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."
        return {**state, "response": response_text}
    
    parsed_query = _parse_query_with_llm(query_to_process)

    if parsed_query is None:
        response_text = "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다. 다시 시도해 주세요."
        return {**state, "response": response_text}
    
    requests_list = parsed_query.get("requests", [])
    
    # 📌 수정된 부분: 요청 리스트가 비어있거나, 터미널 정보가 없는 경우를 모두 처리합니다.
    if not requests_list or (requests_list[0].get("terminal") is None and requests_list[0].get("area") is None):
        print("디버그: 터미널 정보가 없어 전체 터미널 혼잡도 검색을 시도합니다.")
        requests_list = [{"terminal": 1, "area": None}, {"terminal": 2, "area": None}]
        
    date_type = requests_list[0].get("date")
    # ... (날짜 처리 로직은 동일) ...

    target_date = datetime.now().date()
    date_label = "오늘"
    
    response_parts_data = []
    
    try:
        hourly_data = _get_congestion_data_from_db(target_date.strftime("%Y%m%d"), datetime.now().hour)
        if not hourly_data:
            return {**state, "response": f"죄송합니다. {date_label} 현재 시각에 대한 혼잡도 예측 정보를 찾을 수 없습니다."}
            
        for request in requests_list:
            terminal_number = request.get("terminal")
            area_name = request.get("area")
            
            is_daily = request.get("is_daily", False) or (request.get("time") == "합계")
            
            if is_daily:
                daily_data = _get_daily_congestion_data_from_db()
                if daily_data:
                    if terminal_number == 1:
                        response_parts_data.append({"터미널": 1, "유형": "하루 전체", "승객수": daily_data.get("t1_arrival_sum", 0.0) + daily_data.get("t1_departure_sum", 0.0)})
                    elif terminal_number == 2:
                        response_parts_data.append({"터미널": 2, "유형": "하루 전체", "승객수": daily_data.get("t2_arrival_sum", 0.0) + daily_data.get("t2_departure_sum", 0.0)})
            else:
                if terminal_number is not None and area_name is not None:
                    mapped_key = _map_area_to_db_key(terminal_number, area_name)
                    if mapped_key:
                        response_parts_data.append({"터미널": terminal_number, "구역": area_name, "시간": datetime.now().hour, "승객수": hourly_data.get(mapped_key, 0.0)})
                
                elif terminal_number is not None and area_name is None:
                    if terminal_number == 1:
                        total_count = float(hourly_data.get("t1_arrival_sum", 0.0)) + float(hourly_data.get("t1_departure_sum", 0.0))
                        congestion = _get_congestion_level(1, total_count)
                        response_parts_data.append({"터미널": 1, "유형": "시간대별", "시간": datetime.now().hour, "승객수": total_count, "혼잡도": congestion})
                    elif terminal_number == 2:
                        total_count = float(hourly_data.get("t2_arrival_sum", 0.0)) + float(hourly_data.get("t2_departure_sum", 0.0))
                        congestion = _get_congestion_level(2, total_count)
                        response_parts_data.append({"터미널": 2, "유형": "시간대별", "시간": datetime.now().hour, "승객수": total_count, "혼잡도": congestion})
        
        context_for_llm = json.dumps(response_parts_data, ensure_ascii=False, indent=2)

        if not response_parts_data:
            final_response_text = "죄송합니다. 요청하신 혼잡도 정보를 찾을 수 없습니다."
        else:
            final_response_text = common_llm_rag_caller(query_to_process, context_for_llm, "공항 혼잡도 예측 정보", "airport_congestion_prediction")

    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 - {e}")
        final_response_text = "혼잡도 정보를 처리하는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    return {**state, "response": final_response_text}