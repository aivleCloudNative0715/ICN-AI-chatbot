# 기존 임포트
import requests
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

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
        "**'푸동에서 출발해서'와 같이 '~에서 출발'이라는 표현은 'direction'을 'arrival'로, 'departure_airport_name'을 '푸동'으로 파싱해야 해.** "
        "**'~행'이라는 표현은 'direction'을 'departure'로, 'airport_name'을 목적지 공항명으로 파싱해야 해.** "
        "예시: {'flights': [{'departure_airport_name': '푸동', 'direction': 'arrival'}]}"
        "또한 '오후 7시'와 같은 시간 정보가 있으면 'scheduleDateTime' 필드에 24시간 형식으로 파싱해줘. "
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
    departure_airport_code: Optional[str] = None,
    search_date: Optional[str] = None,
    search_time: Optional[str] = None,
    time_range: int = 1
) -> Dict[str, Any]:
    if direction == "departure":
        url = f"{BASE_URL}/getPassengerDeparturesDeOdp"
    elif direction == "arrival":
        url = f"{BASE_URL}/getPassengerArrivalsDeOdp"
    else:
        return {"error": "Invalid direction"}

    today = datetime.now().strftime("%Y%m%d")
    date_to_search = []
    
    if not search_date:
        date_to_search.append(today)
        for i in range(-3, 7):
            search_day = datetime.now() + timedelta(days=i)
            search_day_str = search_day.strftime("%Y%m%d")
            if search_day_str != today:
                date_to_search.append(search_day_str)
    else:
        date_to_search.append(search_date)
    
    all_results = []
    found_date = None
    
    if search_time:
        base_time = datetime.strptime(search_time, "%H:%M")
        time_to_search = [base_time + timedelta(hours=h) for h in range(-time_range, time_range + 1)]
    else:
        time_to_search = [None]

    for date in date_to_search:
        for time_obj in time_to_search:
            current_time = time_obj.strftime("%H:%M") if time_obj else None
            
            params = {
                "serviceKey": SERVICE_KEY,
                "type": "json",
                "numOfRows": 10,
                "pageNo": 1,
                "searchday": date,
                "flight_id": flight_id,
                "f_id": f_id,
                "airport_code": airport_code,
                "dep_airport_code": departure_airport_code,
                "scheduletime": current_time,
            }
            
            call_params = {k: v for k, v in params.items() if v}
            print(f"디버그: API 호출 시도 - {direction} 방향, 날짜: {date}, 시간: {current_time}, 파라미터: {call_params}")

            try:
                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()

                items = data.get("response", {}).get("body", {}).get("items", {})
                results = items.get("item", []) if isinstance(items, dict) else items
                
                if results:
                    all_results.extend(results)
                    found_date = date
                    print(f"디버그: {date} 날짜, {current_time} 시간에서 정보 발견! 총 {len(results)}건")
            except requests.exceptions.RequestException as e:
                print(f"API 호출 오류 (날짜: {date}, 시간: {current_time}): {e}")
                continue
        
        if all_results:
            break
            
    if not all_results:
        print(f"디버그: {direction} 방향으로 모든 날짜와 시간에서 정보를 찾을 수 없습니다.")
        return {"data": [], "total_count": 0}
    else:
        return {"data": all_results, "found_date": found_date, "total_count": len(all_results)}

def _extract_flight_info_from_response(
    api_response: Dict[str, Any], 
    info_type: Optional[str] = None, 
    found_date: Optional[str] = None,
    airport_name: Optional[str] = None,
    airline_name: Optional[str] = None,
    departure_airport_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    flight_data = api_response.get("data", [])
    if not flight_data:
        return []

    if isinstance(flight_data, dict):
        flight_data = [flight_data]

    # 출발지 이름으로 필터링
    if departure_airport_name:
        flight_data = [item for item in flight_data if item.get("airport") == departure_airport_name]
        print(f"디버그: '{departure_airport_name}'으로 출발지 정보 필터링 완료. 남은 항목 수: {len(flight_data)}")

    if airline_name:
        flight_data = [item for item in flight_data if item.get("airline") == airline_name]
        print(f"디버그: '{airline_name}'으로 항공편 정보 필터링 완료. 남은 항목 수: {len(flight_data)}")
    
    extracted_info = []

    for item in flight_data:
        info_map = {
            "gatenumber": "탑승구",
            "chkinrange": "체크인카운터",
            "exitnumber": "출구",
            "remark": "운항현황",
            "terminalid": "터미널",
            "scheduleDateTime": "예정시간",
            "estimatedDateTime": "변경시간"
        }
        
        info = {
            "flightId": item.get("flightId"),
            "direction": "도착" if item.get("carousel") else "출발",
            "airline": item.get("airline"),
            "airport": item.get("airport"),
            "airportCode": item.get("airportCode"),
        }
        
        if found_date:
            info["운항날짜"] = found_date

        if info_type:
            api_key = info_type
            if info_type == '운항정보':
                api_key = 'remark'
            
            display_name = info_map.get(api_key, api_key)
            info[display_name] = item.get(api_key, "정보 없음")
            
            specific_info = {
                "flightId": info["flightId"],
                "direction": info["direction"],
                "airline": info["airline"],
                "airport": info["airport"],
                "airportCode": info["airportCode"],
                display_name: info[display_name],
            }
            if found_date:
                specific_info["운항날짜"] = found_date
            
            extracted_info.append(specific_info)
        else:
            info["예정시간"] = item.get("scheduleDateTime")
            info["변경시간"] = item.get("estimatedDateTime")
            info["운항현황"] = item.get("remark")
            info["탑승구"] = item.get("gatenumber")
            info["출구"] = item.get("exitnumber")
            info["체크인카운터"] = item.get("chkinrange")
            info["터미널"] = item.get("terminalid")
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