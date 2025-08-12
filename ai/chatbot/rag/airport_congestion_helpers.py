import json
from chatbot.rag.config import db_client, client, db_name
from datetime import datetime
from pymongo.errors import ConnectionFailure, OperationFailure
import re

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

def _map_area_to_db_key(terminal_number: int, area_name: str) -> str | None:
    """
    LLM이 파싱한 구역 이름을 MongoDB 문서의 키로 매핑하는 헬퍼 함수.
    """
    area_type = "arrival" if "입국장" in area_name else "departure"
    area_id = re.sub(r'[^A-Z0-9]', '', area_name)
    
    if terminal_number == 1:
        if area_type == "arrival":
            if area_id in ["A", "B"]: return "t1_arrival_a_b"
            elif area_id == "C": return "t1_arrival_c"
            elif area_id == "D": return "t1_arrival__d"
            elif area_id in ["E", "F"]: return "t1_arrival_e_f"
        elif area_type == "departure":
            if area_id in ["1", "2"]: return "t1_departure_1_2"
            elif area_id == "3": return "t1_departure_3"
            elif area_id == "4": return "t1_departure_4"
            elif area_id in ["5", "6"]: return "t1_departure_5_6"
    elif terminal_number == 2:
        if area_type == "arrival":
            if area_id == "A": return "t2_arrival_a"
            elif area_id == "B": return "t2_arrival_b"
        elif area_type == "departure":
            if area_id == "1": return "t2_departure_1"
            elif area_id == "2": return "t2_departure_2"
    return None

def _get_congestion_data_from_db(date_str: str, hour: int) -> dict | None:
    """
    MongoDB에서 특정 날짜와 시간의 혼잡도 예측 데이터를 가져오는 함수.
    """
    try:
        db = db_client[db_name]
        collection = db.AirportCongestionPredict
        
        next_hour = (hour + 1) % 24
        time_slot = f"{hour:02d}_{next_hour:02d}"

        congestion_data = collection.find_one({
            "date": date_str,
            "time": time_slot
        })
        
        if congestion_data:
            print(f"디버그: MongoDB에서 혼잡도 데이터 조회 성공: {congestion_data.get('congestion_predict_id')}")
            return congestion_data
        else:
            print(f"디버그: MongoDB에서 {date_str} {time_slot}에 대한 혼잡도 데이터를 찾을 수 없습니다.")
            return None
    except ConnectionFailure as e:
        print(f"디버그: MongoDB 연결 실패 - {e}")
        return None
    except OperationFailure as e:
        print(f"디버그: MongoDB 조회 작업 실패 - {e}")
        return None
    except Exception as e:
        print(f"디버그: MongoDB 조회 중 알 수 없는 오류 발생 - {e}")
        return None


def _get_daily_congestion_data_from_db() -> dict | None:
    """
    MongoDB에서 하루 합계 혼잡도 예측 데이터를 가져오는 함수.
    """
    try:
        db = db_client[db_name]
        collection = db.AirportCongestionPredict
        
        daily_data = collection.find_one({
            "date": "합계",
            "time": "합계"
        })
        
        if daily_data:
            print(f"디버그: MongoDB에서 하루 합계 혼잡도 데이터 조회 성공: {daily_data.get('congestion_predict_id')}")
            return daily_data
        else:
            print(f"디버그: MongoDB에서 하루 합계 데이터를 찾을 수 없습니다.")
            return None
    except Exception as e:
        print(f"디버그: 하루 합계 데이터 조회 중 오류 발생 - {e}")
        return None
    
def _parse_query_with_llm(user_query: str) -> dict | None:
    # 📌 수정된 부분: 프롬프트 지시사항을 더 명확하게 강화
    prompt_content = (
        "사용자 쿼리에서 인천국제공항의 혼잡도 예측 정보를 JSON 형식으로 추출해줘."
        "질문에서 복수 터미널, 구역, 날짜, 시간 정보가 있다면, 각각의 요청을 'requests' 리스트의 개별 객체로 만들어줘."
        "만약 '하루', '오늘 전체' 또는 **특정 시간 언급 없이 '공항 혼잡도'와 같이 하루 전체를 묻는 질문이라면, 해당 요청 객체에 'is_daily': true 필드를 반드시 추가하고 'time'은 '합계'라는 문자열로 설정해줘.**"
        "날짜는 'today', 'tomorrow', 'unsupported'로 응답해줘. 날짜가 언급되지 않으면 오늘로 간주해줘."
        "시간은 0~23의 정수여야 해. 언급되지 않으면 null로 추출해줘. 단, 하루 전체에 대한 질문인 경우 '합계'로 설정해줘."
        "터미널 번호는 1 또는 2 중 하나이고, 언급되지 않으면 null로 추출해줘."
        "구역은 '입국장' 또는 '출국장'과 알파벳/숫자를 조합한 형태(예: '입국장A', '출국장1')여야 해. 언급되지 않으면 null로 추출해줘."
        "응답 시 다른 설명 없이 오직 JSON 객체만 반환해야 해."
        
        "\n\n응답 형식: "
        "```json"
        "{"
        "    \"requests\": ["
        "        {"
        "            \"date\": \"[today|tomorrow|unsupported]\", "
        "            \"time\": [시간 (0~23 정수), 또는 '합계', 없으면 null], "
        "            \"terminal\": [터미널 번호 (1, 2), 없으면 null], "
        "            \"area\": \"[구역명 (string), 없으면 null]\", "
        "            \"is_daily\": [true|false, 하루 전체 질문일 경우] "
        "        }"
        "    ]"
        "}"
        "```"
        "\n\n예시: "
        "사용자: 1터미널 하루 전체 혼잡도랑 오늘 2터미널 출국장1 혼잡도 알고싶어"
        "응답: ```json\n{\"requests\": [{\"date\": \"today\", \"time\": \"합계\", \"terminal\": 1, \"area\": null, \"is_daily\": true}, {\"date\": \"today\", \"time\": null, \"terminal\": 2, \"area\": \"출국장1\", \"is_daily\": false}]}```"
        "사용자: 1터미널과 2터미널 혼잡도 알려줘"
        "응답: ```json\n{\"requests\": [{\"date\": \"today\", \"time\": null, \"terminal\": 1, \"area\": null, \"is_daily\": false}, {\"date\": \"today\", \"time\": null, \"terminal\": 2, \"area\": null, \"is_daily\": false}]}```"
        "사용자: 11일 공항혼잡도"
        "응답: ```json\n{\"requests\": [{\"date\": \"unsupported\", \"time\": \"합계\", \"terminal\": null, \"area\": null, \"is_daily\": true}]}```"
        "사용자: 1터미널 하루 전체 혼잡도 알려줘"
        "응답: ```json\n{\"requests\": [{\"date\": \"today\", \"time\": \"합계\", \"terminal\": 1, \"area\": null, \"is_daily\": true}]}```"
        "사용자: 공항 혼잡도"
        "응답: ```json\n{\"requests\": [{\"date\": \"today\", \"time\": \"합계\", \"terminal\": null, \"area\": null, \"is_daily\": true}]}```"
    )

    messages = [
        {"role": "system", "content": prompt_content},
        {"role": "user", "content": user_query}
    ]

    # 📌 수정된 부분: response_format을 사용하여 LLM이 JSON을 반환하도록 강제
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    llm_output = response.choices[0].message.content.strip()
    
    print(f"디버그: LLM 원본 응답 -> {llm_output}")

    try:
        # response_format을 사용하면 ````json`과 같은 마크다운 제거 불필요
        parsed_data = json.loads(llm_output)
        return parsed_data
    except json.JSONDecodeError as e:
        print("디버그: LLM 응답이 올바른 JSON 형식이 아닙니다.")
        print(f"디버그: JSONDecodeError -> {e}")
    except Exception as e:
        print(f"디버그: 알 수 없는 오류 발생 -> {e}")
    
    return None