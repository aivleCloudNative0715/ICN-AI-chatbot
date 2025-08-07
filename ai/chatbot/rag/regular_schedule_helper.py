import json
from .config import client
from datetime import datetime
import locale
import requests

locale.setlocale(locale.LC_TIME, 'ko_KR.UTF-8')

def _parse_schedule_query_with_llm(user_query: str) -> list[dict] | None:
    """
    LLM을 사용하여 사용자 쿼리에서 정기 스케줄 정보를 JSON 리스트 형식으로 추출하는 함수.
    """
    prompt_content = (
        "사용자 쿼리에서 정기 운항 스케줄 관련 정보를 추출해줘."
        "만약 질문에 여러 개의 독립적인 검색 조건이 있다면, 각각의 조건을 하나의 JSON 객체로 만들고, 이들을 리스트에 담아줘."
        "아래 필드들을 추출해줘: "
        "- `airline_name`: 항공사 이름 (예: '대한항공', '아시아나')\n"
        "- `airport_name`: **국가명이 아닌 도시명 또는 공항 이름** (예: '도쿄', '후쿠오카'). 국가명만 있다면 해당 국가의 주요 공항명으로 추출해줘 (예: '일본' -> '도쿄').\n"
        "- `day_of_week`: 요일 (예: '월요일', '오늘'). 요일 정보가 없으면 '오늘'로 간주해줘.\n"
        "- `direction`: 운항 방향 ('arrival' 또는 'departure'). 정보가 없으면 null로 추출해줘.\n"
        "- `time_period`: 시간대 (예: '오전', '오후', '저녁'). 정보가 없으면 null로 추출해줘.\n"
        "응답 시 다른 설명 없이 오직 JSON 리스트만 반환해야 해."

        "\n\n응답 형식: "
        "```json"
        "["
        "  {{"
        "    \"airline_name\": \"[항공사명 (string), 없으면 null]\", "
        "    \"airport_name\": \"[공항명 (string), 없으면 null]\", "
        "    \"day_of_week\": \"[요일 (string), 없으면 '오늘']\", "
        "    \"direction\": \"[arrival|departure|null]\", "
        "    \"time_period\": \"[오전|오후|저녁|null]\""
        "  }}"
        "]"
        "```"
        "\n\n예시: "
        "사용자: 일요일 도쿄행이랑 월요일 오사카행 스케줄 알려줘"
        "응답: ```json\n[\n  {{\"airline_name\": null, \"airport_name\": \"도쿄\", \"day_of_week\": \"일요일\", \"direction\": \"departure\", \"time_period\": null}},\n  {{\"airline_name\": null, \"airport_name\": \"오사카\", \"day_of_week\": \"월요일\", \"direction\": \"departure\", \"time_period\": null}}\n]```"
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
        
        start_index = llm_output.find('[')
        end_index = llm_output.rfind(']') + 1
        
        if start_index == -1 or end_index == 0:
            raise json.JSONDecodeError("JSON 리스트 시작/끝을 찾을 수 없습니다.", llm_output, 0)
            
        json_string = llm_output[start_index:end_index]
        parsed_json_string = json_string.replace('"null"', 'null')
        parsed_data = json.loads(parsed_json_string)

        if not isinstance(parsed_data, list):
            parsed_data = [parsed_data]
            
        return parsed_data
    except json.JSONDecodeError as e:
        print("디버그: LLM 응답이 올바른 JSON 형식이 아닙니다.")
        print(f"디버그: LLM 원본 응답 -> {llm_output}")
        print(f"디버그: JSONDecodeError -> {e}")
    
    return None

def _get_day_of_week_field(day_name: str):
    day_map = {
        '월요일': 'monday', '화요일': 'tuesday', '수요일': 'wednesday',
        '목요일': 'thursday', '금요일': 'friday', '토요일': 'saturday',
        '일요일': 'sunday', '오늘': datetime.now().strftime('%A')
    }
    return day_map.get(day_name, datetime.now().strftime('%A')).lower()

def _call_schedule_api(params: dict, direction: str):
    """
    공공 데이터 포털 API를 호출하고 결과를 파싱하는 내부 함수
    """
    api_url = f"{FLIGHT_SCHEDULE_API_BASE_URL}/getPaxFltSched{direction.capitalize()}s"
    
    # API 요청 파라미터
    params_with_key = {
        "serviceKey": SERVICE_KEY,
        "type": "json",
        "numOfRows": 100,  # 충분한 데이터 확보
        **params
    }
    
    try:
        response = requests.get(api_url, params=params_with_key)
        response.raise_for_status()
        response_data = response.json()
        
        items = response_data.get("response", {}).get("body", {}).get("items", {})
        
        if not items:
            return []
            
        return items.get("item", []) if isinstance(items, dict) else items
    
    except requests.exceptions.RequestException as e:
        print(f"디버그: API 호출 중 오류 발생 ({direction}) - {e}")
        return "api_error"
    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 ({direction}) - {e}")
        return "api_error"