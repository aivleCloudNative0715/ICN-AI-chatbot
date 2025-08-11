import os
from datetime import datetime, timedelta
import re
from chatbot.graph.state import ChatState
from dotenv import load_dotenv

from chatbot.rag.airport_congestion_helpers import _get_congestion_level, VALID_AREAS, _parse_query_with_llm, _get_congestion_data_from_db, _get_daily_congestion_data_from_db, _map_area_to_db_key

load_dotenv()

def airport_congestion_prediction_handler(state: ChatState) -> ChatState:
    """
    공항 혼잡도 예측 정보를 MongoDB에서 조회하여 답변을 생성하는 핸들러입니다.
    복합 질문을 처리하고, '하루 전체' 및 '특정 시간' 데이터를 모두 조회하도록 개선되었습니다.
    """
    print(f"\n--- 공항 혼잡도 예측 핸들러 실행 ---")
    
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    
    if not query_to_process:
        response_text = "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."
        return {**state, "response": response_text}
    
    # 📌 수정된 부분: _parse_query_with_llm 함수에 재구성된 쿼리를 전달합니다.
    parsed_query = _parse_query_with_llm(query_to_process)

    if parsed_query is None:
        response_text = "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다. 다시 시도해 주세요."
        return {**state, "response": response_text}
    
    requests_list = parsed_query.get("requests", [])
    if not requests_list:
        response_text = "죄송합니다. 요청하신 터미널이나 구역 정보를 명확하게 파악할 수 없습니다. 좀 더 구체적으로 말씀해주시겠어요?"
        return {**state, "response": response_text}

    date_type = requests_list[0].get("date")
    if date_type == "unsupported":
        response_text = "죄송합니다. 공항 혼잡도 정보는 현재 날짜를 기준으로 오늘과 내일의 예측 정보만 제공하고 있습니다. 요청하신 날짜는 지원하지 않습니다."
        return {**state, "response": response_text}
    
    # ⭐ '내일' 요청은 즉시 처리하지 않고 오류 응답을 반환합니다.
    if date_type == "tomorrow":
        response_text = "죄송합니다. 현재는 오늘에 대한 혼잡도 정보만 제공하고 있습니다. 다시 질문해 주시겠어요?"
        return {**state, "response": response_text}
    
    # '오늘'에 대한 날짜 및 시간 파라미터 생성
    target_date = datetime.now().date()
    date_label = "오늘"
    
    response_parts = []
    
    try:
        for request in requests_list:
            terminal_number = request.get("terminal")
            area_name = request.get("area")
            
            is_daily = request.get("is_daily", False) or (request.get("time") == "합계")
            
            if is_daily:
                daily_data = _get_daily_congestion_data_from_db()
                if daily_data:
                    if terminal_number == 1:
                        total_count = float(daily_data.get("t1_arrival_sum", 0.0)) + float(daily_data.get("t1_departure_sum", 0.0))
                        congestion = _get_congestion_level(1, total_count)
                        response_parts.append(
                            f"제1여객터미널의 하루 전체 혼잡도는 {congestion}으로 예상됩니다. (총 승객 약 {int(total_count)}명)"
                        )
                    elif terminal_number == 2:
                        total_count = float(daily_data.get("t2_arrival_sum", 0.0)) + float(daily_data.get("t2_departure_sum", 0.0))
                        congestion = _get_congestion_level(2, total_count)
                        response_parts.append(
                            f"제2여객터미널의 하루 전체 혼잡도는 {congestion}으로 예상됩니다. (총 승객 약 {int(total_count)}명)"
                        )
                    else:
                        response_parts.append("요청하신 터미널 번호가 유효하지 않습니다.")
                else:
                    response_parts.append(f"죄송합니다. {date_label} 하루 전체에 대한 혼잡도 정보를 찾을 수 없습니다.")

            else:
                requested_time = request.get("time")
                current_hour = requested_time if isinstance(requested_time, int) and 0 <= requested_time <= 23 else datetime.now().hour
                
                hourly_data = _get_congestion_data_from_db(target_date.strftime("%Y%m%d"), current_hour)
                if not hourly_data:
                    response_parts.append(f"죄송합니다. {date_label} {current_hour}시에 대한 혼잡도 예측 정보를 찾을 수 없습니다.")
                    continue
                
                if terminal_number is not None and area_name is not None:
                    mapped_key = _map_area_to_db_key(terminal_number, area_name)
                    if mapped_key:
                        passenger_count = float(hourly_data.get(mapped_key, 0.0))
                        area_type = "arrival" if "입국장" in area_name else "departure"
                        total_terminal_count = float(hourly_data.get(f"t{terminal_number}_{area_type}_sum", 0.0))
                        terminal_congestion = _get_congestion_level(terminal_number, total_terminal_count)
                        
                        response_parts.append(
                            f"제{terminal_number}터미널 {area_name.replace('입국장', '입국장 ').replace('출국장', '출국장 ')}의 예상 승객은 약 {int(passenger_count)}명입니다.\n"
                            f"전체 {area_name.replace('입국장', '입국장 ').replace('출국장', '출국장 ')}의 혼잡도는 {terminal_congestion}으로 예상됩니다. (총 승객 약 {int(total_terminal_count)}명)"
                        )
                    else:
                        valid_ids_str = ", ".join(sorted(list(VALID_AREAS.get(terminal_number, {}).get(re.sub(r'[^가-힣]', '', area_name), set()))))
                        response_parts.append(f"죄송합니다. 제{terminal_number}터미널에는 {area_name}이(가) 없습니다. 유효한 구역은 다음과 같습니다: {valid_ids_str}")
                
                elif terminal_number is not None and area_name is None:
                    if terminal_number == 1:
                        total_count = float(hourly_data.get("t1_arrival_sum", 0.0)) + float(hourly_data.get("t1_departure_sum", 0.0))
                        congestion = _get_congestion_level(1, total_count)
                    elif terminal_number == 2:
                        total_count = float(hourly_data.get("t2_arrival_sum", 0.0)) + float(hourly_data.get("t2_departure_sum", 0.0))
                        congestion = _get_congestion_level(2, total_count)
                    else:
                        response_parts.append(f"죄송합니다. 제{terminal_number}터미널은 존재하지 않습니다.")
                        continue
                    
                    response_parts.append(
                        f"제{terminal_number}여객터미널의 혼잡도는 {congestion}으로 예상됩니다. (총 승객 약 {int(total_count)}명)"
                    )
                else:
                    response_parts.append("요청하신 구역 또는 터미널 정보를 명확하게 파악할 수 없습니다. 좀 더 구체적으로 말씀해주시겠어요?")

        final_response_text = (
            f"{date_label} 공항 혼잡도 예측 정보입니다.\n\n" +
            "\n\n".join(response_parts)
        )

    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 - {e}")
        final_response_text = "혼잡도 정보를 처리하는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    return {**state, "response": final_response_text}