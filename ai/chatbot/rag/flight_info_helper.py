# 기존 임포트
import requests
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import re

# 수정된 임포트: config.py의 common_llm_rag_caller를 직접 사용합니다.
from chatbot.rag.config import common_llm_rag_caller 
from chatbot.rag.config import client
# 기존 코드는 그대로 유지합니다.
BASE_URL = "http://apis.data.go.kr/B551177/StatusOfPassengerFlightsDeOdp"
SERVICE_KEY = os.getenv("SERVICE_KEY")

def _parse_flight_query_with_llm(user_query: str) -> List[Dict[str, Any]]:
    system_prompt = (
        "사용자의 질문을 분석하여 항공편 정보에 대한 필수 정보를 JSON 형식으로 추출해줘. "
        "응답은 반드시 'flights'라는 키를 가진 JSON 객체여야 해. "
        "각 항공편 정보는 이 'flights' 리스트 안에 객체로 넣어줘. "
        "각 객체는 'direction'(출발 또는 도착), 'airport_name'(공항명), 'departure_airport_name'(출발 공항명), 'info_type'(요청 정보) 등을 포함해야 해. "
        "**'상하이에서 출발해서'와 같이 '~에서 출발'이라는 표현은 'direction'을 'arrival'로, 'departure_airport_name'을 '상하이'로 파싱해야 해.** "
        "**'~행'이라는 표현은 'direction'을 'departure'로, 'airport_name'을 목적지 공항명으로 파싱해야 해.** "
        "예시: {'flights': [{'departure_airport_name': '상하이', 'direction': 'arrival'}]}"
        "또한 '오후 7시'와 같은 시간 정보가 있으면 'scheduleDateTime' 필드에 24시간 형식(예: '19:00')으로 파싱해줘. "
        "만약 사용자가 '게이트', '체크인' 등 특정 정보를 묻는다면, 'info_type'에 해당 정보를 파싱해줘."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        parsed_json_str = response.choices[0].message.content

        parsed_data = json.loads(parsed_json_str)
        parsed_queries = parsed_data.get('flights', [])

        if isinstance(parsed_queries, list):
            for query in parsed_queries:
                if 'flight_id' in query and query['flight_id']:
                    query['flight_id'] = query['flight_id'].upper()
        
        print(f"디버그: 최종 파싱 결과 (대문자 변환 후) - {parsed_queries}")
        return parsed_queries

    except (json.JSONDecodeError, Exception) as e:
        print(f"디버그: LLM 응답 파싱 실패 - {e}")
        return []
    
def _call_flight_api(
    direction: str,
    flight_id: Optional[str] = None,
    f_id: Optional[str] = None,
    airport_code: Optional[str] = None,
    search_date: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None
) -> Dict[str, Any]:
    if direction == "departure":
        url = f"{BASE_URL}/getPassengerDeparturesDeOdp"
    elif direction == "arrival":
        url = f"{BASE_URL}/getPassengerArrivalsDeOdp"
    else:
        return {"error": "Invalid direction"}

    today = datetime.now().strftime("%Y%m%d")
    date_to_search = [today] if not search_date else [search_date]

    all_results = []
    found_date = None
    
    for date in date_to_search:
        params = {
            "serviceKey": SERVICE_KEY,
            "type": "json",
            "numOfRows": 100,
            "pageNo": 1,
            "searchday": date,
            "flight_id": flight_id,
            "f_id": f_id,
            "airport_code": airport_code,
            "from_time": from_time.replace(':', '') if from_time else None,
            "to_time": to_time.replace(':', '') if to_time else None,
        }
        
        call_params = {k: v for k, v in params.items() if v}
        print(f"디버그: API 호출 시도 - {direction} 방향, 날짜: {date}, 파라미터: {call_params}")

        try:
            response = requests.get(url, params=call_params, timeout=5)
            response.raise_for_status()
            data = response.json()

            items = data.get("response", {}).get("body", {}).get("items", {})
            results = items.get("item", []) if isinstance(items, dict) else items
            
            if results:
                all_results.extend(results)
                found_date = date
                print(f"디버그: {date} 날짜에서 정보 발견! 총 {len(results)}건")
                return {"data": all_results, "found_date": found_date, "total_count": len(all_results)}

        except requests.exceptions.RequestException as e:
            print(f"API 호출 오류 (날짜: {date}): {e}")
            continue

    print(f"디버그: {direction} 방향으로 모든 날짜에서 정보를 찾을 수 없습니다.")
    return {"data": [], "total_count": 0}


def _extract_flight_info_from_response(
    api_response: Dict[str, Any], 
    info_type: Optional[str] = None, 
    found_date: Optional[str] = None,
    airport_name: Optional[str] = None,
    airline_name: Optional[str] = None,
    departure_airport_name: Optional[str] = None,
    departure_airport_code: Optional[str] = None
) -> List[Dict[str, Any]]:
    flight_data = api_response.get("data", [])
    if not flight_data:
        return []

    if isinstance(flight_data, dict):
        flight_data = [flight_data]
    
    if departure_airport_code:
        flight_data = [item for item in flight_data if item.get("airportCode") == departure_airport_code]
        print(f"디버그: '{departure_airport_name}' ({departure_airport_code})으로 출발지 정보 필터링 완료. 남은 항목 수: {len(flight_data)}")

    if airport_name:
        flight_data = [item for item in flight_data if airport_name in item.get("airport", "")]
        print(f"디버그: '{airport_name}'으로 도착지 정보 필터링 완료. 남은 항목 수: {len(flight_data)}")

    if airline_name:
        flight_data = [item for item in flight_data if item.get("airline") == airline_name]
        print(f"디버그: '{airline_name}'으로 항공편 정보 필터링 완료. 남은 항목 수: {len(flight_data)}")

    extracted_info = []

    for item in flight_data:
        info = {
            "flightId": item.get("flightId"),
            "direction": "도착" if item.get("carousel") else "출발",
            "airline": item.get("airline"),
            "airport": item.get("airport"),
            "airportCode": item.get("airportCode"),
            "운항날짜": found_date,
            "예정시간": item.get("scheduleDateTime"),
            "변경시간": item.get("estimatedDateTime"),
            "운항현황": item.get("remark"),
            "탑승구": item.get("gatenumber"),
            "출구": item.get("exitnumber"),
            "체크인카운터": item.get("chkinrange"),
            "터미널": item.get("terminalid")
        }
        
        extracted_info.append(info)
    
    return extracted_info


def _get_airport_code_with_llm(airport_name: str) -> Optional[str]:
    system_prompt = (
        f"'{airport_name}'과(와) 관련된 공항의 IATA 코드를 찾아줘. "
        f"만약 국가 이름이라면 해당 국가의 가장 주요한 국제공항 코드를 찾아줘. "
        f"예를 들어, '파리' 또는 '프랑스'는 'CDG'야. '일본'은 'NRT' 또는 'HND' 중 'NRT'를 선택해줘. "
        "오직 공항 코드만 답변해야 하며, 다른 설명은 포함하지 마."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": airport_name}
            ],
            temperature=0.1
        )
        airport_code = response.choices[0].message.content.strip()
        
        if 2 <= len(airport_code) <= 5 and airport_code.isupper() and airport_code.isalnum():
            print(f"디버그: LLM이 '{airport_name}'에 대한 공항 코드로 '{airport_code}'를 반환했습니다.")
            return airport_code
        else:
            print(f"디버그: LLM이 반환한 공항 코드 '{airport_code}'가 유효하지 않습니다.")
            return None
    
    except Exception as e:
        print(f"디버그: LLM을 사용한 공항 코드 추출 실패 - {e}")
        return None

# --- 새로운 헬퍼 함수 추가 ---
def _normalize_time(time_str: str) -> str:
    """
    다양한 시간 문자열을 'HH:MM' 형식으로 표준화합니다.
    """
    time_str = time_str.strip().replace(":", "").replace(" ", "").upper()

    # '오후7시' -> 1900
    if "오후" in time_str:
        hour = int(re.search(r'\d+', time_str).group())
        if hour < 12:
            hour += 12
        return f"{hour:02d}00"
    
    # '7 PM' -> 1900
    if "PM" in time_str:
        hour = int(re.search(r'\d+', time_str).group())
        if hour < 12:
            hour += 12
        return f"{hour:02d}00"

    # '7' -> 0700
    if len(time_str) <= 2:
        return f"{int(time_str):02d}00"
    
    # '19:00' -> 1900
    if len(time_str) == 4 and time_str.isdigit():
        return time_str
    
    # 19:20과 같은 형식도 처리
    if re.match(r'^\d{2}:\d{2}$', time_str):
        return time_str.replace(':', '')

    return time_str
