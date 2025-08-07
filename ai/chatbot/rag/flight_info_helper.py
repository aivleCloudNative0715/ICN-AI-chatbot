# ai/chatbot/rag/flight_info_helper.py

import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

from ai.chatbot.rag.config import client

load_dotenv()

SERVICE_KEY = os.getenv("SERVICE_KEY")
if not SERVICE_KEY:
    raise ValueError("SERVICE_KEY 환경 변수가 설정되지 않았습니다.")

FLIGHT_API_BASE_URL = "http://apis.data.go.kr/B551177/StatusOfPassengerFlightsDeOdp"
FLIGHT_ARRIVAL_URL = f"{FLIGHT_API_BASE_URL}/getPassengerArrivalsDeOdp"
FLIGHT_DEPARTURE_URL = f"{FLIGHT_API_BASE_URL}/getPassengerDeparturesDeOdp"

def call_flight_api(params: dict, direction: str):
    """
    항공편 API를 호출하고 결과를 파싱하는 내부 함수
    """
    params_with_key = {
        "serviceKey": SERVICE_KEY,
        "type": "json",
        **params
    }
    
    api_url = FLIGHT_ARRIVAL_URL if direction == "arrival" else FLIGHT_DEPARTURE_URL
    
    try:
        response = requests.get(api_url, params=params_with_key)
        response.raise_for_status()
        response_data = response.json()
        
        body = response_data.get("response", {}).get("body", {})
        total_count = body.get("totalCount", 0)

        if total_count == 0:
            return None
        
        items = body.get("items", {})
        
        flight_info = None
        if isinstance(items, dict) and "item" in items:
            item_data = items["item"]
            flight_info = item_data[0] if isinstance(item_data, list) else item_data
        elif isinstance(items, list) and len(items) > 0:
            flight_info = items[0]
            
        if not flight_info or not isinstance(flight_info, dict):
            return None
            
        return flight_info
    
    except requests.exceptions.RequestException as e:
        print(f"디버그: API 호출 중 오류 발생 ({direction}) - {e}")
        return "api_error"
    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 ({direction}) - {e}")
        return "api_error"

def _format_flight_info(flight_info, date_label, direction, requested_info_keywords=None):
    """
    항공편 정보를 보기 좋게 포맷팅하는 함수
    """
    if requested_info_keywords is None:
        requested_info_keywords = []

    airline = flight_info.get("airline", "정보 없음")
    flight_id_res = flight_info.get("flightId", "정보 없음")
    terminal = flight_info.get("terminalid", "정보 없음")
    gate = flight_info.get("gatenumber", "정보 없음")
    remark = flight_info.get("remark", "정보 없음")
    schedule_time_str = flight_info.get("scheduleDateTime", "")
    estimated_time_str = flight_info.get("estimatedDateTime", "")
    airport_name = flight_info.get("airport", "정보 없음")
    chkinrange = flight_info.get("chkinrange", "정보 없음")
    
    schedule_time = datetime.strptime(schedule_time_str, "%Y%m%d%H%M").strftime("%H시 %M분") if schedule_time_str else "정보 없음"
    estimated_time = datetime.strptime(estimated_time_str, "%Y%m%d%H%M").strftime("%H시 %M분") if estimated_time_str else "정보 없음"

    terminal_map = {
        "P01": "제1여객터미널", "P02": "제1여객터미널 (탑승동)", "P03": "제2여객터미널",
        "C01": "화물터미널 남측", "C02": "화물터미널 북측", "C03": "제2 화물터미널"
    }
    terminal_name = terminal_map.get(terminal, "정보 없음")
    
    # 1. 사용자가 요청한 정보 먼저 제공
    primary_info = []
    if any(kw in requested_info_keywords for kw in ["게이트", "탑승구"]) and gate:
        primary_info.append(f"📌 **{airline} {flight_id_res}편 게이트 번호:** {gate}")
    if any(kw in requested_info_keywords for kw in ["체크인", "카운터"]) and chkinrange:
        primary_info.append(f"📌 **{airline} {flight_id_res}편 체크인 카운터:** {chkinrange}")
    
    # 2. 전체 운항 정보 추가
    full_info = []
    if primary_info:
        full_info.extend(primary_info)
        full_info.append("\n**운항 현황 상세 정보:**")
    else:
        full_info.append(f"**{airline} {flight_id_res}편 운항 정보입니다.**")
    
    full_info.append(f" - {date_label} {schedule_time} 예정 ({estimated_time} 변경)")
    full_info.append(f" - 현황: {remark}")
    full_info.append(f" - {'출발' if direction == 'arrival' else '도착'}지 공항: {airport_name}")
    full_info.append(f" - 터미널: {terminal_name}")
    if gate:
        full_info.append(f" - 게이트 번호: {gate}")
    if direction == "departure" and chkinrange:
        full_info.append(f" - 체크인 카운터: {chkinrange}")

    return full_info

def _parse_flight_query_with_llm(user_query: str) -> list | None:
    """
    LLM을 사용하여 사용자 쿼리에서 항공편 운항 정보를 JSON 리스트 형식으로 추출하는 함수.
    """
    prompt_content = (
        "사용자 쿼리에서 항공편 운항 정보를 추출하여 JSON 배열 형태로 반환해줘."
        "각 배열 요소는 하나의 항공편에 대한 정보를 담고 있어야 해. "
        "조회일 기준 -3일과 +6일 이내의 날짜만 지원하며, 범위를 벗어나면 'unsupported'로 응답해줘. 날짜가 언급되지 않으면 오늘로 간주해줘."
        "운항 방향은 '도착' 또는 '출발' 중 하나여야 해. 쿼리에 정보가 없으면 null로 추출해줘."
        "편명은 'OZ704'와 같이 항공사 코드와 숫자가 조합된 형태여야 해. "
        "도착지/출발지 공항명(한글)이 있다면 추출해줘. 없으면 null로 추출해줘."
        "사용자가 요청한 구체적인 정보 키워드(예: '체크인 카운터', '게이트')가 있다면 리스트로 추출해줘. '수하물 수취대'나 '출구'는 추출 대상이 아니야. 없으면 null로 추출해줘."
        "응답 시 다른 설명 없이 오직 JSON 배열만 반환해야 해."

        "\n\n응답 형식: "
        "```json"
        "["
        "  {{"
        "    \"date_offset\": \"[오늘=0, 내일=1, 3일 전=-3, 6일 뒤=6, 범위를 벗어나면 'unsupported']\", "
        "    \"flight_id\": \"[편명 (string)]\", "
        "    \"direction\": \"[arrival|departure|null]\", "
        "    \"requested_info_keywords\": \"[요청 정보 키워드 리스트, 없으면 null]\""
        "  }},"
        "  {{"
        "    \"date_offset\": \"[오늘=0, 내일=1, 3일 전=-3, 6일 뒤=6, 범위를 벗어나면 'unsupported']\", "
        "    \"flight_id\": \"[편명 (string)]\", "
        "    \"direction\": \"[arrival|departure|null]\", "
        "    \"requested_info_keywords\": \"[요청 정보 키워드 리스트, 없으면 null]\""
        "  }}"
        "]"
        "```"
        "\n\n예시: "
        "사용자: TW281 게이트랑 LJ262 체크인 카운터 좀"
        "응답: ```json\n[{{\"date_offset\": 0, \"flight_id\": \"TW281\", \"direction\": null, \"requested_info_keywords\": [\"게이트\"]}}, {{\"date_offset\": 0, \"flight_id\": \"LJ262\", \"direction\": null, \"requested_info_keywords\": [\"체크인 카운터\"]}}]```"
        "사용자: 3일 전 OZ704편 도착 정보 알려줘"
        "응답: ```json\n[{{\"date_offset\": -3, \"flight_id\": \"OZ704\", \"direction\": \"arrival\", \"requested_info_keywords\": null}}]```"
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
        
        parsed_data = json.loads(llm_output)
            
        # 요청 정보 키워드가 문자열로 반환되는 경우 리스트로 변환하고,
        # 'null'이거나 키가 없는 경우 빈 리스트로 초기화합니다.
        for item in parsed_data:
            if "requested_info_keywords" not in item or item["requested_info_keywords"] is None:
                item["requested_info_keywords"] = []
            elif isinstance(item["requested_info_keywords"], str):
                item["requested_info_keywords"] = [item["requested_info_keywords"]]
            
            # 날짜 오프셋이 문자열일 경우 정수로 변환
            if "date_offset" in item and isinstance(item["date_offset"], str):
                try:
                    item["date_offset"] = int(item["date_offset"])
                except (ValueError, TypeError):
                    item["date_offset"] = 0
                
        return parsed_data
    except json.JSONDecodeError:
        print("디버그: LLM 응답이 올바른 JSON 형식이 아닙니다.")
        print(f"디버그: LLM 원본 응답 -> {llm_output}")
    
    return None
