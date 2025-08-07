import os
import requests
from datetime import datetime
import re
from ai.chatbot.graph.state import ChatState
from dotenv import load_dotenv

from ai.chatbot.rag.airport_congestion_helpers import _get_congestion_level, VALID_AREAS, _parse_query_with_llm

load_dotenv()

SERVICE_KEY = os.getenv("SERVICE_KEY")
if not SERVICE_KEY:
    raise ValueError("SERVICE_KEY 환경 변수가 설정되지 않았습니다.")

API_URL = "http://apis.data.go.kr/B551177/PassengerNoticeKR/getfPassengerNoticeIKR"
API_URL_ARRIVAL = "http://apis.data.go.kr/B551177/StatusOfArrivals/getArrivalsCongestion"

def airport_congestion_prediction_handler(state: ChatState) -> ChatState:
    """
    공항 혼잡도 예측 API를 호출하여 답변을 생성하는 핸들러입니다.
    _parse_query_with_llm 함수를 사용하여 사용자 입력의 모든 정보를 추출하고,
    복합 질문(여러 터미널/구역)도 처리하도록 개선되었습니다.
    """
    print(f"\n--- 공항 혼잡도 예측 핸들러 실행 ---")
    user_query = state.get("user_input", "")
    
    # LLM에게 모든 정보 파싱을 위임
    parsed_query = _parse_query_with_llm(user_query)

    if parsed_query is None:
        response_text = "죄송합니다. 요청을 처리하는 중 문제가 발생했습니다. 다시 시도해 주세요."
        return {**state, "response": response_text}
    
    date_type = parsed_query.get("date")
    if date_type == "unsupported":
        response_text = "죄송합니다. 공항 혼잡도 API는 현재 날짜를 기준으로 오늘과 내일의 예측 정보만 제공하고 있습니다. 요청하신 날짜는 지원하지 않습니다."
        return {**state, "response": response_text}
    
    selectdate_param = "0"
    date_label = "오늘"
    if date_type == "tomorrow":
        selectdate_param = "1"
        date_label = "내일"

    params = {
        "serviceKey": SERVICE_KEY,
        "type": "json",
        "selectdate": selectdate_param
    }

    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()

        response_data = response.json()
        
        items_container = response_data.get("response", {}).get("body", {}).get("items", {})
        if not items_container:
            response_text = "혼잡도 예측 정보를 찾을 수 없습니다. API 응답이 비어있거나 형식이 다릅니다."
            return {**state, "response": response_text}
        
        items = items_container.get("item", []) if isinstance(items_container, dict) else items_container
        if not items:
            response_text = "혼잡도 예측 정보를 찾을 수 없습니다. API 응답 데이터가 비어있습니다."
            return {**state, "response": response_text}
        if isinstance(items, dict): items = [items]

        requested_hour = parsed_query.get("time")
        current_hour = requested_hour if isinstance(requested_hour, int) and 0 <= requested_hour <= 23 else datetime.now().hour

        current_atime_slot = f"{current_hour:02d}_{current_hour+1:02d}"
        current_hour_data = next((item for item in items if item.get("atime") == current_atime_slot), None)
        
        if not current_hour_data:
            response_text = f"죄송합니다. {date_label} {current_hour}시에 대한 혼잡도 예측 정보를 찾을 수 없습니다."
            return {**state, "response": response_text}

        # 새로운 로직: LLM이 반환한 'requests' 리스트를 처리합니다.
        requests_list = parsed_query.get("requests", [])
        response_parts = []
        
        # 'requests' 리스트가 비어있으면 전체 혼잡도 정보를 반환
        if not requests_list:
            total_t1_count = float(current_hour_data.get("t1sumset1", 0.0)) + float(current_hour_data.get("t1sumset2", 0.0))
            total_t2_count = float(current_hour_data.get("t2sumset1", 0.0)) + float(current_hour_data.get("t2sumset2", 0.0))
            
            t1_congestion = _get_congestion_level(1, total_t1_count)
            t2_congestion = _get_congestion_level(2, total_t2_count)
            
            response_parts.append(
                f"[제1여객터미널]\n"
                f" - 혼잡도: {t1_congestion} (예상 총 승객 약 {int(total_t1_count)}명)\n\n"
                f"[제2여객터미널]\n"
                f" - 혼잡도: {t2_congestion} (예상 총 승객 약 {int(total_t2_count)}명)"
            )

        else:
            for request in requests_list:
                terminal_number = request.get("terminal")
                area_name = request.get("area")
                
                # 유효하지 않은 구역 조합에 대한 처리
                if area_name is None and re.search(r'출국장|입국장', user_query) and re.search(r'[A-F]|([1-6])', user_query):
                    area_type_in_query = re.search(r'출국장|입국장', user_query).group(0)
                    area_id_in_query = re.search(r'[A-F]|([1-6])', user_query).group(0)
                    is_invalid = (area_type_in_query == '입국장' and not area_id_in_query.isalpha()) or \
                                 (area_type_in_query == '출국장' and not area_id_in_query.isdigit())
                    if is_invalid:
                        response_parts.append(f"죄송합니다. {area_type_in_query} {area_id_in_query}은(는) 유효하지 않은 구역 조합입니다. 입국장은 알파벳, 출국장은 숫자로 이루어져야 합니다.")
                        continue

                # 특정 터미널 및 구역 정보가 있을 경우
                if terminal_number is not None and area_name is not None:
                    mapped_key = None
                    area_type = re.sub(r'[^가-힣]', '', area_name)
                    area_id = re.sub(r'[^A-Z0-9]', '', area_name)

                    if terminal_number not in VALID_AREAS or area_type not in VALID_AREAS[terminal_number] or area_id not in VALID_AREAS[terminal_number][area_type]:
                        valid_ids_str = ", ".join(sorted(list(VALID_AREAS.get(terminal_number, {}).get(area_type, set()))))
                        response_parts.append(f"죄송합니다. 제{terminal_number}터미널에는 {area_name}이(가) 없습니다. {area_type}은(는) 다음 구역만 존재합니다: {valid_ids_str}")
                        continue

                    if terminal_number == 1:
                        if area_name in ["입국장A", "입국장B"]: mapped_key = "t1sum1"
                        elif area_name in ["입국장E", "입국장F"]: mapped_key = "t1sum2"
                        elif area_name == "입국장C": mapped_key = "t1sum3"
                        elif area_name == "입국장D": mapped_key = "t1sum4"
                        elif area_name in ["출국장1", "출국장2"]: mapped_key = "t1sum5"
                        elif area_name == "출국장3": mapped_key = "t1sum6"
                        elif area_name == "출국장4": mapped_key = "t1sum7"
                        elif area_name in ["출국장5", "출국장6"]: mapped_key = "t1sum8"
                    elif terminal_number == 2:
                        if area_name == "입국장A": mapped_key = "t2sum1"
                        elif area_name == "입국장B": mapped_key = "t2sum2"
                        elif area_name == "출국장1": mapped_key = "t2sum3"
                        elif area_name == "출국장2": mapped_key = "t2sum4"

                    if mapped_key:
                        passenger_count = float(current_hour_data.get(mapped_key, 0.0))
                        total_terminal_count = float(current_hour_data.get(f"t{terminal_number}sumset1", 0.0)) + float(current_hour_data.get(f"t{terminal_number}sumset2", 0.0))
                        terminal_congestion = _get_congestion_level(terminal_number, total_terminal_count)
                        
                        response_parts.append(
                            f"제{terminal_number}터미널 {area_name.replace('입국장', '입국장 ').replace('출국장', '출국장 ')}의 예상 승객은 약 {int(passenger_count)}명입니다.\n"
                            f"전체 혼잡도는 {terminal_congestion}으로 예상됩니다. (총 승객 약 {int(total_terminal_count)}명)"
                        )
                    else:
                        response_parts.append(f"죄송합니다. {area_name}에 대한 세부 혼잡도 정보는 찾을 수 없습니다.")

                # 특정 터미널 정보만 있을 경우
                elif terminal_number is not None and area_name is None:
                    if terminal_number == 1:
                        total_count = float(current_hour_data.get("t1sumset1", 0.0)) + float(current_hour_data.get("t1sumset2", 0.0))
                        congestion = _get_congestion_level(1, total_count)
                    elif terminal_number == 2:
                        total_count = float(current_hour_data.get("t2sumset1", 0.0)) + float(current_hour_data.get("t2sumset2", 0.0))
                        congestion = _get_congestion_level(2, total_count)
                    else:
                        response_parts.append(f"죄송합니다. 제{terminal_number}터미널은 존재하지 않습니다.")
                        continue
                    
                    response_parts.append(
                        f"제{terminal_number}여객터미널의 혼잡도는 {congestion}으로 예상됩니다. (총 승객 약 {int(total_count)}명)"
                    )
                
                else: # terminal_number와 area_name 모두 None인 경우 (복합 질문이지만 구역/터미널 정보가 파악되지 않았을 때)
                    response_parts.append("요청하신 구역 또는 터미널 정보를 명확하게 파악할 수 없습니다. 좀 더 구체적으로 말씀해주시겠어요?")

        final_response_text = (
            f"{date_label} {current_hour}시 기준, 공항 혼잡도 예측 정보입니다.\n\n" +
            "\n\n".join(response_parts) + 
            "\n\n ⚠️ 주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다. "
            "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트에 직접 확인하시기 바랍니다."
        )

    except requests.exceptions.RequestException as e:
        print(f"디버그: API 호출 중 오류 발생 - {e}")
        final_response_text = "혼잡도 정보를 가져오는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 - {e}")
        final_response_text = "혼잡도 정보를 처리하는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    return {**state, "response": final_response_text}