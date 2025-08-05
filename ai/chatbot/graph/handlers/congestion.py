import os
import requests
from datetime import datetime
import re
from ai.chatbot.graph.state import ChatState
from dotenv import load_dotenv
import json

# ai/chatbot/rag/config.py 파일에서 OpenAI 클라이언트를 직접 임포트합니다.
from ai.chatbot.rag.config import client

load_dotenv()

# 환경 변수에서 서비스 키를 가져옵니다.
SERVICE_KEY = os.getenv("SERVICE_KEY")
if not SERVICE_KEY:
    raise ValueError("SERVICE_KEY 환경 변수가 설정되지 않았습니다.")

# 혼잡도 예측 API URL
API_URL = "http://apis.data.go.kr/B551177/PassengerNoticeKR/getfPassengerNoticeIKR"

# 유효한 터미널 및 구역 정보를 정의합니다.
VALID_AREAS = {
    1: {
        "입국장": {"A", "B", "C", "D", "E", "F"},
        "출국장": {"1", "2", "3", "4", "5", "6"}
    },
    2: {
        "입국장": {"A", "B"},
        "출국장": {"1", "2"}
    }
}

def _get_congestion_level(terminal: int, passenger_count: float) -> str:
    """
    터미널과 승객 수에 따라 혼잡도 수준을 판단하는 헬퍼 함수.
    """
    if terminal == 1:
        if passenger_count <= 7000:
            return "원활"
        elif passenger_count <= 7600:
            return "보통"
        elif passenger_count <= 8200:
            return "약간혼잡"
        elif passenger_count <= 8600:
            return "혼잡"
        else:
            return "매우혼잡"
    elif terminal == 2:
        if passenger_count <= 3200:
            return "원활"
        elif passenger_count <= 3500:
            return "보통"
        elif passenger_count <= 3800:
            return "약간혼잡"
        elif passenger_count <= 4000:
            return "혼잡"
        else:
            return "매우혼잡"
    return "정보 없음"

def _parse_query_with_llm(user_query: str) -> dict | None:
    """
    LLM을 사용하여 사용자 쿼리에서 날짜, 시간, 터미널, 구역 정보를 JSON 형식으로 추출하는 함수.
    단일 질문이든 복합 질문이든, 'requests' 리스트에 담아 반환하도록 프롬프트 수정.
    """
    
    prompt_content = (
        "사용자 쿼리에서 인천국제공항의 혼잡도 예측 정보를 추출해줘."
        "오늘, 내일 외의 날짜는 'unsupported'로 응답해줘. 날짜가 언급되지 않으면 오늘로 간주해줘."
        "시간이 언급되지 않으면 null로 추출해줘. 시간은 0부터 23까지의 정수여야 해."
        "터미널 번호는 1 또는 2 중 하나야. '1'처럼 터미널과 출국장 모두에 해당되는 모호한 숫자가 있으면 터미널을 0으로 추출해줘. 터미널 정보가 없으면 null로 추출해줘."
        "구역은 '입국장' 또는 '출국장'과 알파벳/숫자를 포함해야 해."
        "**중요: 입국장은 알파벳(A-F)과 함께, 출국장은 숫자(1-6)와 함께 사용되어야 합니다.**"
        "유효하지 않은 구역 조합(예: '출국장'과 알파벳, '입국장'과 숫자)이 발견되면 'area'는 null로 추출해줘."
        "만약 쿼리에 '입국장' 또는 '출국장'에 대한 언급 없이 'C', 'D', 'E', 'F', '3', '4', '5', '6'이 언급되면, 제1터미널의 해당 구역으로 가정해줘."
        "질문이 혼잡도 전체에 대한 내용이면 'requests' 리스트는 비워줘."
        "만약 여러 터미널이나 구역에 대한 질문이라면, 'requests' 리스트에 여러 객체를 만들어줘."
        "날짜가 'unsupported'이면 'requests' 필드를 포함하지 말고 다른 필드는 모두 null로 추출해줘."
        "**응답 시 다른 설명 없이 오직 JSON 객체만 반환해야 해.**"
        
        "\n\n응답 형식: "
        "```json"
        "{{"
        "  \"date\": \"[today|tomorrow|unsupported]\", "
        "  \"time\": [시간 (0~23 정수), 없으면 null], "
        "  \"requests\": ["
        "    {{"
        "      \"terminal\": [터미널 번호 (1, 2), 겹치면 0, 없으면 null], "
        "      \"area\": \"[구역명 (string), 없으면 null]\""
        "    }}"
        "  ]"
        "}}"
        "```"
        "\n\n예시: "
        "사용자: 내일 11시 1출국장 혼잡해?"
        "응답: ```json\n{{\"date\": \"tomorrow\", \"time\": 11, \"requests\": [{{\"terminal\": 0, \"area\": \"출국장1\"}}]}}```"
        "사용자: 1터미널과 2터미널 혼잡도 알려줘"
        "응답: ```json\n{{\"date\": \"today\", \"time\": null, \"requests\": [{{\"terminal\": 1, \"area\": null}}, {{\"terminal\": 2, \"area\": null}}]}}```"
        "사용자: 1터미널 입국장C, 2터미널 입국장A 혼잡해?"
        "응답: ```json\n{{\"date\": \"today\", \"time\": null, \"requests\": [{{\"terminal\": 1, \"area\": \"입국장C\"}}, {{\"terminal\": 2, \"area\": \"입국장A\"}}]}}```"
        "사용자: 출국장 A 혼잡도가 어떻게 돼?"
        "응답: ```json\n{{\"date\": \"today\", \"time\": null, \"requests\": []}}```"
        "사용자: 혼잡도 알려줘"
        "응답: ```json\n{{\"date\": \"today\", \"time\": null, \"requests\": []}}```"
    )

    messages = [
        {"role": "system", "content": prompt_content},
        {"role": "user", "content": user_query}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0
    )
    
    llm_output = response.choices[0].message.content.strip()

    try:
        if llm_output.startswith("```json") and llm_output.endswith("```"):
            llm_output = llm_output[7:-3].strip()
        
        llm_output = llm_output.replace('{{', '{').replace('}}', '}')

        parsed_data = json.loads(llm_output)
        return parsed_data
    except json.JSONDecodeError:
        print("디버그: LLM 응답이 올바른 JSON 형식이 아닙니다.")
        print(f"디버그: LLM 원본 응답 -> {llm_output}")
    
    return None

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
            "\n\n---\n\n".join(response_parts) + 
            "\n\n⚠️ 주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다."
            "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트에 직접 확인하시기 바랍니다."
        )

    except requests.exceptions.RequestException as e:
        print(f"디버그: API 호출 중 오류 발생 - {e}")
        final_response_text = "혼잡도 정보를 가져오는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 - {e}")
        final_response_text = "혼잡도 정보를 처리하는 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    
    return {**state, "response": final_response_text}