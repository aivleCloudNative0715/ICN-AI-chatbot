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


def _convert_slots_to_query_format(slots: List[tuple], user_query: str) -> List[Dict[str, Any]]:
    """
    의도분류기에서 추출된 slot 정보를 flight API 파라미터 형식으로 변환

    Args:
        slots: [(word, slot_tag), ...] 형식의 slot 정보
        user_query: 사용자 원본 질문

    Returns:
        flight API 호출에 필요한 파라미터 딕셔너리 리스트
    """
    if not slots:
        return []

    # slot에서 정보 추출
    flight_ids = [word for word, slot in slots if slot in ['B-flight_id', 'I-flight_id']]
    airports = [word for word, slot in slots if slot in ['B-airport_name', 'I-airport_name']]
    airlines = [word for word, slot in slots if slot in ['B-airline_name', 'I-airline_name']]
    terminals = [word for word, slot in slots if slot in ['B-terminal', 'I-terminal']]
    departure_airports = [word for word, slot in slots if
                          slot in ['B-departure_airport_name', 'I-departure_airport_name']]
    vague_times = [word for word, slot in slots if slot in ['B-vague_time', 'I-vague_time']]
    time_periods = [word for word, slot in slots if slot in ['B-time_period', 'I-time_period']]

    # vague_time과 time_period에 따른 시간 범위 설정
    from_time, to_time = None, None
    
    # time_period 우선 처리 (더 구체적)
    if time_periods:
        time_period = time_periods[0].lower()
        
        if time_period in ["아침", "오전"]:
            from_time, to_time = "0600", "1200"
        elif time_period in ["점심", "낮", "오후"]:
            from_time, to_time = "1200", "1800"
        elif time_period in ["저녁", "밤", "야간"]:
            from_time, to_time = "1800", "2359"
        elif time_period in ["새벽", "밤늦은"]:
            from_time, to_time = "0000", "0600"
        print(f"디버그: time_period '{time_period}' 감지 → 시간 범위: {from_time}-{to_time}")
    
    # vague_time 처리 (time_period가 없을 때만)
    elif vague_times:
        vague_time = vague_times[0].lower()
        current_time = datetime.now()

        if vague_time in ["곧", "잠깐", "잠시", "조금"]:
            # 현재부터 1시간 후까지
            from_time = current_time.strftime("%H%M")
            to_time = (current_time + timedelta(hours=1)).strftime("%H%M")
        elif vague_time in ["이따가", "나중에", "있다가"]:
            # 1시간 후부터 3시간 후까지
            from_time = (current_time + timedelta(hours=1)).strftime("%H%M")
            to_time = (current_time + timedelta(hours=3)).strftime("%H%M")
        elif vague_time in ["오늘", "금일"]:
            # 현재부터 자정까지
            from_time = current_time.strftime("%H%M")
            to_time = "2359"
        print(f"디버그: vague_time '{vague_time}' 감지 → 시간 범위: {from_time}-{to_time}")

    # 기본 쿼리 구조 생성
    query = {
        "flight_id": flight_ids[0].upper() if flight_ids else None,
        "airport_name": airports[0] if airports else None,
        "airline_name": airlines[0] if airlines else None,
        "departure_airport_name": departure_airports[0] if departure_airports else None,
        "terminal": "T1" if any("1" in str(t) for t in terminals) else "T2" if any(
            "2" in str(t) for t in terminals) else None,
        "direction": "arrival" if departure_airports else None,  # 출발지가 있으면 도착, 없으면 None (두 방향 모두 검색)
        "info_type": "운항 정보",
        "date_offset": 0,
        "from_time": from_time,
        "to_time": to_time,
        "airport_codes": []
    }

    # 유의미한 정보가 하나라도 있으면 쿼리 반환
    has_meaningful_info = any([
        query["flight_id"],
        query["airport_name"],
        query["airline_name"],
        query["departure_airport_name"],
        query["terminal"],
        vague_times,  # vague_time이 있어도 의미있는 정보로 간주
        time_periods  # time_period도 의미있는 정보로 간주
    ])

    if has_meaningful_info:
        print(f"디버그: slot에서 변환된 쿼리 - {query}")
        return [query]
    else:
        print("디버그: slot에 유의미한 항공편 정보가 없음")
        return []


def _parse_flight_query_with_llm(user_query: str) -> List[Dict[str, Any]]:
    system_prompt = (
        "사용자의 질문을 분석하여 항공편 정보에 대한 필수 정보를 JSON 리스트 형식으로 추출해줘. "
        "응답은 반드시 'requests'라는 키를 가진 JSON 객체여야 해. "
        "각 항공편 정보는 이 'requests' 리스트 안에 객체로 넣어줘. "

        "아래 필드들을 추출해줘: "
        "- `flight_id`: 항공편명 (예: 'KE001', 'OZ201'). 정보가 없으면 null로 추출해줘.\n"
        "- `airline_name`: 항공사 이름 (예: '대한항공', '아시아나항공'). 정보가 없으면 null로 추출해줘.\n"
        "- `airport_name`: 도착 도시명 또는 공항 이름. 인천에서 출발하는 경우에만 추출해줘. 정보가 없으면 null로 추출해줘.\n"
        "- `airport_codes`: '일본'처럼 국가명이 포함되면 해당 국가의 주요 공항 IATA 코드 리스트(예: ['NRT', 'HND', 'KIX'])를 추출해줘. '도쿄'처럼 도시명이 포함되면 해당 도시의 주요 공항 IATA 코드 리스트(예: ['NRT', 'HND'])를 추출해줘. **'미국'처럼 국가명이 언급되면 'JFK', 'LAX' 등 주요 공항 코드를 반드시 추출해줘.** 인천을 묻는 질문에서는 이 필드를 비워줘. 정보가 없으면 빈 리스트로 추출해줘.\n"
        "- `departure_airport_name`: 출발 도시명 또는 공항 이름. 인천으로 도착하는 경우에만 추출해줘. 정보가 없으면 null로 추출해줘.\n"
        "- `direction`: 운항 방향 ('arrival' 또는 'departure'). 질문에 명시되어 있지 않으면 null로 추출해줘.\n"
        "- `from_time`: 검색 시작 시간 (HHMM 형식). 정보가 없으면 null로 추출해줘.\n"
        "- `to_time`: 검색 종료 시간 (HHMM 형식). 정보가 없으면 null로 추출해줘.\n"
        "- `info_type`: 사용자가 얻고자 하는 정보의 유형 (예: '체크인 카운터', '탑승구', '운항 정보'). 정보가 없으면 '운항 정보'로 추출해줘.\n"
        "- `date_offset`: '오늘'이면 0, '내일'이면 1, '모레'이면 2, '어제'면 -1처럼 오늘을 기준으로 한 날짜 차이를 정수로 추출해줘. 정보가 없으면 0으로 추출해줘.\n"
        "- `terminal`: 사용자가 요청한 터미널 정보. '1터미널' 또는 '제1터미널'은 'T1'으로, '2터미널' 또는 '제2터미널'은 'T2'로 추출해줘. 정보가 없으면 null로 추출해줘.\n"

        "지침: "
        "1. **시간 모호성**: '3시 반'처럼 모호한 시간은, 오전과 오후를 모두 포함하는 2개의 독립된 요청으로 분리해서 반환해줘. 각 요청에는 from_time과 to_time이 동일하게 추출돼야 해.\n"
        "2. **시간 범위**: '오전 8시 이후'는 from_time을 '0800'으로, to_time을 '2359'로 추출해줘. '오후 8시 이전'은 from_time을 '0000'으로, to_time을 '2000'으로 추출해줘.\n"
        "3. **특정 시간**: '오후 3시'처럼 특정 시점의 시간은 from_time과 to_time에 동일한 시간(예: '1500')을 추출해줘. 핸들러에서 이 값을 기준으로 검색 범위를 계산할 거야.\n"
        "4. **국가/도시명**: '일본'과 같은 국가명은 'airport_codes'에 주요 공항 코드 리스트를 반드시 추가해줘. '도쿄'와 같은 도시명도 마찬가지야. 국가명만 언급되면 'airport_name'은 null로 비워두고, 'airport_codes'에 그 나라의 주요 공항 코드들을 넣어줘.\n"
        "5. **출발지/도착지**: 질문에 도착지만 언급되고 출발지가 명시되지 않으면, 'departure_airport_name'은 '인천국제공항'으로 간주하고, 'direction'은 'departure'로 설정해줘."
        "6. **인천 관련**: '인천 도착'과 같은 질문에서 'airport_name'과 'airport_codes'를 null/빈 리스트로 남겨두고 'direction'을 'arrival'로 설정해줘. '인천 출발'과 같은 질문에서도 마찬가지로 'airport_name'과 'airport_codes'를 비우고 'direction'을 'departure'로 설정해줘."

        "응답 시 다른 설명 없이 오직 JSON 객체만 반환해야 해."
        "\n\n예시: "
        "사용자: 인천에 곧 도착하는 비행기 알려줘"
        "응답: ```json\n{\"requests\": [{\"flight_id\": null, \"airline_name\": null, \"airport_name\": null, \"airport_codes\": [], \"departure_airport_name\": null, \"direction\": \"arrival\", \"from_time\": null, \"to_time\": null, \"info_type\": \"운항 정보\", \"date_offset\": 0, \"terminal\": null}]}```"
        "사용자: 오늘 뉴욕가는거 오후 2시 이후에 어떤거 있어?"
        "응답: ```json\n{\"requests\": [{\"flight_id\": null, \"airline_name\": null, \"airport_name\": \"뉴욕\", \"airport_codes\": [\"JFK\", \"LGA\", \"EWR\"], \"departure_airport_name\": null, \"direction\": \"departure\", \"from_time\": \"1400\", \"to_time\": \"2359\", \"info_type\": \"운항 정보\", \"date_offset\": 0, \"terminal\": null}]}```"
        "사용자: 1터미널 9시 비행기 알려줘"
        "응답: ```json\n{\"requests\": [{\"flight_id\": null, \"airline_name\": null, \"airport_name\": null, \"airport_codes\": [], \"departure_airport_name\": null, \"direction\": \"departure\", \"from_time\": \"0900\", \"to_time\": \"0900\", \"info_type\": \"운항 정보\", \"date_offset\": 0, \"terminal\": \"T1\"}, {\"flight_id\": null, \"airline_name\": null, \"airport_name\": null, \"airport_codes\": [], \"departure_airport_name\": null, \"direction\": \"departure\", \"from_time\": \"2100\", \"to_time\": \"2100\", \"info_type\": \"운항 정보\", \"date_offset\": 0, \"terminal\": \"T1\"}]}```"
        "사용자: 오늘 인천에서 미국 가는 비행기 알려줘"
        "응답: ```json\n{\"requests\": [{\"flight_id\": null, \"airline_name\": null, \"airport_name\": null, \"airport_codes\": [\"JFK\", \"LAX\"], \"departure_airport_name\": null, \"direction\": \"departure\", \"from_time\": null, \"to_time\": null, \"info_type\": \"운항 정보\", \"date_offset\": 0, \"terminal\": null}]}```"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        parsed_json_str = response.choices[0].message.content
        parsed_data = json.loads(parsed_json_str)
        parsed_queries = parsed_data.get('requests', [])

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
            "numOfRows": 1000,
            "pageNo": 1,
            "searchday": date,
            "flight_id": flight_id,
            "f_id": f_id,
            "from_time": from_time.replace(':', '') if from_time else None,
            "to_time": to_time.replace(':', '') if to_time else None,
        }

        # 📌 수정: airport_code가 있을 경우에만 params에 추가
        if airport_code:
            params["airport_code"] = airport_code

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
        departure_airport_code: Optional[str] = None,
        requested_direction: Optional[str] = None  # 📌 추가: 요청 방향 매개변수
) -> List[Dict[str, Any]]:
    flight_data = api_response.get("data", [])
    if not flight_data:
        return []

    if isinstance(flight_data, dict):
        flight_data = [flight_data]

    # 📌 핵심 수정: API 응답의 운항 방향과 요청 방향이 다르면 필터링
    if requested_direction:
        # 응답의 'remark' 필드에 '도착', '출발'이 명시되어 있다고 가정
        flight_data = [
            item for item in flight_data
            if requested_direction == "arrival" and item.get("remark") in ["도착", "지연", "결항", "회항", "착륙"] or \
               requested_direction == "departure" and item.get("remark") in ["출발", "탑승중", "탑승준비", "탑승마감", "마감예정"]
        ]
        print(f"디버그: 요청 방향('{requested_direction}')으로 필터링 완료. 남은 항목 수: {len(flight_data)}")

    if departure_airport_code:
        flight_data = [item for item in flight_data if item.get("airportCode") == departure_airport_code]
        print(
            f"디버그: '{departure_airport_name}' ({departure_airport_code})으로 출발지 정보 필터링 완료. 남은 항목 수: {len(flight_data)}")

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
            "direction": "도착" if requested_direction == "arrival" else "출발",
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