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
    """
    LLM을 사용하여 사용자 쿼리에서 운항 정보를 파악하고 JSON 형식으로 추출합니다.
    - 추출된 'flight_id'는 API 호출을 위해 대문자로 변환합니다.
    - '파리행'과 같이 특정 항공편이 아닌 목적지를 묻는 경우 'airport_name'으로 추출합니다.
    """
    system_prompt = (
        "사용자의 질문을 분석하여 항공편 정보에 대한 필수 정보를 JSON 형식으로 추출해줘. "
        "응답은 반드시 'flights'라는 키를 가진 JSON 객체여야 해. "
        "각 항공편 정보는 이 'flights' 리스트 안에 객체로 넣어줘. "
        "각 객체는 'flight_id'(편명), 'direction'(출발 또는 도착), 'info_type'(요청 정보)를 포함해야 해. "
        "만약 질문이 특정 항공편이 아닌 '파리' 또는 '프랑스'와 같이 목적지 도시나 국가를 묻는 경우, 'flight_id' 대신 'airport_name'으로 추출해줘. "
        "방향을 알 수 없으면 'departure'를 기본값으로 사용해. "
        "예시: {'flights': [{'airport_name': '파리', 'direction': 'departure'}]}"
        "또한, '게이트'는 'gatenumber', '출구'는 'exitnumber', '체크인 카운터'는 'chkinrange'와 같이 구체적인 API 항목명으로 변환해줘."
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
    search_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    항공편 운항 정보를 조회하는 공공 API를 호출합니다.
    - 날짜가 명시되지 않은 경우, 오늘 날짜를 가장 먼저 검색합니다.
    - 이후 D-3일 ~ D+6일 범위를 추가로 검색합니다.
    """
    if direction == "departure":
        url = f"{BASE_URL}/getPassengerDeparturesDeOdp"
    elif direction == "arrival":
        url = f"{BASE_URL}/getPassengerArrivalsDeOdp"
    else:
        return {"error": "Invalid direction"}

    today = datetime.now().strftime("%Y%m%d")
    date_to_search = []
    
    # 1. 날짜가 명시되지 않은 경우, 오늘 날짜를 최우선으로 추가
    if not search_date:
        date_to_search.append(today)
        # 2. 이후 D-3일 ~ D+6일 범위를 추가
        for i in range(-3, 7):
            search_day = datetime.now() + timedelta(days=i)
            search_day_str = search_day.strftime("%Y%m%d")
            if search_day_str != today:
                date_to_search.append(search_day_str)
    else:
        date_to_search.append(search_date)

    for date in date_to_search:
        params = {
            "serviceKey": SERVICE_KEY,
            "type": "json",
            "numOfRows": 10,
            "pageNo": 1,
            "searchday": date,
            "flight_id": flight_id,
            "f_id": f_id,
            "airport_code": airport_code,
        }

        try:
            print(f"디버그: '{flight_id}'에 대해 {date} 날짜로 API 호출 시도...")
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            items = data.get("response", {}).get("body", {}).get("items", {})
            results = items.get("item", []) if isinstance(items, dict) else items
            
            if results:
                print(f"디버그: {date} 날짜에서 '{flight_id}' 정보 발견!")
                return {"data": results, "found_date": date, "total_count": len(results)}

        except requests.exceptions.RequestException as e:
            print(f"API 호출 오류 (날짜: {date}): {e}")
            continue

    print(f"디버그: {direction} 방향으로 모든 날짜에서 '{flight_id}'를 찾을 수 없습니다.")
    return {"data": [], "total_count": 0}

def _extract_flight_info_from_response(
    api_response: Dict[str, Any], 
    info_type: Optional[str] = None, 
    found_date: Optional[str] = None,
    airport_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    API 응답에서 필요한 정보를 추출하고 포맷합니다.
    - 'airport_name'이 제공될 경우, 해당 공항과 관련된 정보만 LLM을 활용해 필터링합니다.
    """
    flight_data = api_response.get("data", [])
    if not flight_data:
        return []

    if isinstance(flight_data, dict):
        flight_data = [flight_data]

    # 공항명 필터링이 필요할 경우 LLM 사용
    if airport_name:
        # API 응답 데이터를 문자열로 변환하여 LLM에 전달
        data_to_filter = json.dumps(flight_data, ensure_ascii=False)
        
        system_prompt = (
        f"주어진 JSON 데이터는 인천공항의 항공편 운항 정보 리스트입니다. 이 리스트에서 "
        f"도착/출발 공항명이 '{airport_name}'과(와) 관련된 모든 객체를 찾아 JSON 리스트로 반환해주세요. "
        f"필터링 조건에 맞지 않는 객체는 모두 제거해야 합니다. "
        f"결과는 반드시 유효한 JSON 리스트 형식이어야 합니다. "
        f"예시: [{{'airport': '도쿄', 'flightId': 'KE123'}}, ...]"
        )
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": data_to_filter}
                ],
                temperature=0.1,
                response_format={"type": "json_object"} # JSON 형식 응답 요청
            )
            filtered_json_str = response.choices[0].message.content
            filtered_data = json.loads(filtered_json_str)

            print(f"디버그: LLM으로 '{airport_name}' 관련 정보 필터링 완료.")
            flight_data = filtered_data.get('flights', []) if isinstance(filtered_data, dict) else filtered_data
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"디버그: LLM 필터링 실패 - {e}")
            # 필터링 실패 시, 원본 데이터를 그대로 사용하거나 빈 리스트 반환
            return []

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
            "direction": "도착" if "carousel" in item else "출발",
            "airline": item.get("airline"),
            "airport": item.get("airport"),
            "airportCode": item.get("airportCode"),
        }
        
        if found_date:
            info["운항날짜"] = found_date

        if info_type:
            if info_type == '운항정보':
                api_key = 'remark'
            else:
                api_key = info_type
            
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
    """
    LLM을 사용하여 주어진 공항명의 IATA 코드를 추출합니다.
    """
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
        
        # LLM 응답이 너무 길거나 예상치 못한 형식일 경우를 대비해 간단한 유효성 검사
        if 2 <= len(airport_code) <= 5 and airport_code.isupper() and airport_code.isalnum():
            print(f"디버그: LLM이 '{airport_name}'에 대한 공항 코드로 '{airport_code}'를 반환했습니다.")
            return airport_code
        else:
            print(f"디버그: LLM이 반환한 공항 코드 '{airport_code}'가 유효하지 않습니다.")
            return None
    
    except Exception as e:
        print(f"디버그: LLM을 사용한 공항 코드 추출 실패 - {e}")
        return None