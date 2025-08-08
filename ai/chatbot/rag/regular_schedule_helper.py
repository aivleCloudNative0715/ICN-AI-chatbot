import json
from chatbot.rag.config import db_client, db_name, client
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime
import re
import locale

# 시스템 로케일 설정 (요일 처리를 위해 필요)
locale.setlocale(locale.LC_TIME, 'ko_KR.UTF-8')

# ⭐ API 호출 함수는 제거하고 MongoDB 관련 함수만 남깁니다.

def _get_day_of_week_field(day_name: str) -> str | None:
    """
    요일 이름을 MongoDB 문서의 필드명으로 변환합니다.
    """
    day_map = {
        "월요일": "monday", "화요일": "tuesday", "수요일": "wednesday",
        "목요일": "thursday", "금요일": "friday", "토요일": "saturday",
        "일요일": "sunday", "오늘": datetime.now().strftime('%A')
    }
    return day_map.get(day_name, datetime.now().strftime('%A')).lower()

def _get_schedule_from_db(
    direction: str,
    airport_codes: list,
    day_name: str,
    time_period: str,
    airline_name: str
) -> list | str:
    """
    MongoDB에서 정기 운항 스케줄 정보를 조회하는 함수.
    """
    try:
        db = db_client[db_name]
        collection = db.FlightSchedule

        query_filter = {}

        # 방향 필터링
        if direction:
            query_filter['direction'] = direction.capitalize()

        # 요일 필터링
        day_field = _get_day_of_week_field(day_name)
        if day_field:
            query_filter[day_field] = True

        # 시간대 필터링
        if time_period:
            time_filter = {}
            if time_period == '오전': time_filter = {"$gte": "06:00", "$lt": "12:00"}
            elif time_period == '오후': time_filter = {"$gte": "12:00", "$lt": "18:00"}
            elif time_period == '저녁': time_filter = {"$gte": "18:00", "$lte": "23:59"}
            elif time_period == '새벽': time_filter = {"$gte": "00:00", "$lt": "06:00"}
            query_filter['scheduled_time'] = time_filter

        # 항공사 필터링
        if airline_name:
            query_filter['airline_name_kor'] = {"$regex": f".*{re.escape(airline_name)}.*", "$options": "i"}

        # 공항 코드 필터링
        if airport_codes:
            query_filter['airport_code'] = {"$in": airport_codes}
            
        print(f"디버그: MongoDB 쿼리 필터 - {query_filter}")
        
        schedules = list(collection.find(query_filter).limit(5))
        
        return schedules

    except ConnectionFailure as e:
        print(f"디버그: MongoDB 연결 실패 - {e}")
        return "MongoDB 연결 실패"
    except OperationFailure as e:
        print(f"디버그: MongoDB 조회 작업 실패 - {e}")
        return "데이터 조회 작업 실패"
    except Exception as e:
        print(f"디버그: 스케줄 조회 중 알 수 없는 오류 발생 - {e}")
        return "알 수 없는 오류 발생"


def _parse_schedule_query_with_llm(user_query: str) -> dict | None:
    prompt_content = (
        "사용자 쿼리에서 정기 운항 스케줄 관련 정보를 JSON 리스트 형식으로 추출해줘."
        "만약 질문에 여러 개의 독립적인 검색 조건이 있다면, 각각의 조건을 하나의 JSON 객체로 만들고, 이들을 리스트에 담아줘."
        "아래 필드들을 추출해줘: "
        "- `airline_name`: 사용자가 언급한 항공사 이름과 가장 유사한 공식 항공사 이름(예: '대한항공', '아시아나항공', '티웨이항공'). 정확한 항공사 이름을 찾을 수 없으면 null로 추출해줘.\n"
        "- `airport_name`: 도시명 또는 공항 이름 (예: '도쿄', '후쿠오카'). 정보가 없으면 null로 추출해줘.\n"
        "- `airport_codes`: **해당 공항의 IATA 코드 리스트**. 아래 예시를 참고하여 정확한 코드를 추출해줘. 정보가 없으면 빈 리스트로 추출해줘.\n"
        "- `day_of_week`: 요일 (예: '월요일', '오늘'). 요일 정보가 없으면 '오늘'로 간주해줘.\n"
        "- `direction`: 운항 방향 ('도착' 또는 '출발'). 정보가 없으면 '출발'로 간주해줘.\n"
        "- `time_period`: 시간대 (예: '오전', '오후', '저녁', '새벽'). 정보가 없으면 null로 추출해줘.\n"
        "응답 시 다른 설명 없이 오직 JSON 리스트만 반환해야 해."
        "\n\n응답 형식: "
        "```json"
        "{"
        "  \"requests\": ["
        "     {{"
        "        \"airline_name\": \"[항공사명 (string), 없으면 null]\", "
        "        \"airport_name\": \"[공항명 (string), 없으면 null]\", "
        "        \"airport_codes\": [\"[IATA 코드 리스트]\"], "
        "        \"day_of_week\": \"[요일 (string), 없으면 '오늘']\", "
        "        \"direction\": \"[도착|출발]\", "
        "        \"time_period\": \"[오전|오후|저녁|새벽|null]\""
        "     }}"
        "  ]"
        "}"
        "```"
        "\n\n예시: "
        "사용자: 일요일에 일본에서 오는거 있어?"
        "응답: ```json\n{\"requests\": [{\"airline_name\": null, \"airport_name\": \"일본\", \"airport_codes\": [\"NRT\", \"HND\", \"KIX\", \"FUK\", \"CTS\", \"OKA\"], \"day_of_week\": \"일요일\", \"direction\": \"도착\", \"time_period\": null}]}```"
        "사용자: 대한항공 월요일 하노이 도착 스케줄"
        "응답: ```json\n{\"requests\": [{\"airline_name\": \"대한항공\", \"airport_name\": \"하노이\", \"airport_codes\": [\"HAN\"], \"day_of_week\": \"월요일\", \"direction\": \"도착\", \"time_period\": null}]}```"
        "사용자: 일요일 도쿄행이랑 월요일 오사카행 스케줄"
        "응답: ```json\n{\"requests\": [{\"airline_name\": null, \"airport_name\": \"도쿄\", \"airport_codes\": [\"NRT\", \"HND\"], \"day_of_week\": \"일요일\", \"direction\": \"출발\", \"time_period\": null}, {\"airline_name\": null, \"airport_name\": \"오사카\", \"airport_codes\": [\"KIX\", \"ITM\"], \"day_of_week\": \"월요일\", \"direction\": \"출발\", \"time_period\": null}]}```"
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
        
        return parsed_data
    except json.JSONDecodeError as e:
        print("디버그: LLM 응답이 올바른 JSON 형식이 아닙니다.")
        print(f"디버그: LLM 원본 응답 -> {llm_output}")
        print(f"디버그: JSONDecodeError -> {e}")
    except Exception as e:
        print(f"디버그: 알 수 없는 오류 발생 -> {e}")
        print(f"디버그: LLM 원본 응답 -> {llm_output}")
        
    return None