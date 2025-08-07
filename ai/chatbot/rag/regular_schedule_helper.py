import os
import requests
import json
from datetime import datetime
import locale
from dotenv import load_dotenv

# ai/chatbot/rag/config.py 에 있는 client 객체
# ai/chatbot/rag/db_connector.py 에 있는 get_mongo_collection 함수를 가정
# 실제 경로에 맞게 수정 필요
from chatbot.rag.config import client
from chatbot.rag.utils import get_mongo_collection
from chatbot.rag.config import common_llm_rag_caller
from chatbot.graph.state import ChatState

# 시스템 로케일 설정 (요일 처리를 위해 필요)
locale.setlocale(locale.LC_TIME, 'ko_KR.UTF-8')
load_dotenv()

# 환경 변수에서 서비스 키 로드
SERVICE_KEY = os.getenv("SERVICE_KEY")
FLIGHT_SCHEDULE_API_BASE_URL = "http://apis.data.go.kr/B551177/PaxFltSched"

def _get_day_of_week_field(day_name: str) -> str:
    """
    한글 요일명을 API 응답 필드명으로 변환합니다.
    """
    day_map = {
        '월요일': 'monday', '화요일': 'tuesday', '수요일': 'wednesday',
        '목요일': 'thursday', '금요일': 'friday', '토요일': 'saturday',
        '일요일': 'sunday', '오늘': datetime.now().strftime('%A')
    }
    return day_map.get(day_name, datetime.now().strftime('%A')).lower()

def _parse_schedule_query_with_llm(user_query: str) -> list[dict] | None:
    """
    LLM을 사용하여 사용자 쿼리에서 정기 스케줄 정보를 JSON 리스트 형식으로 추출하는 함수입니다.
    도시명에 대한 IATA 코드도 함께 추출하도록 지시합니다.
    """
    prompt_content = (
        "사용자 쿼리에서 정기 운항 스케줄 관련 정보를 추출해줘."
        "만약 질문에 여러 개의 독립적인 검색 조건이 있다면, 각각의 조건을 하나의 JSON 객체로 만들고, 이들을 리스트에 담아줘."
        "아래 필드들을 추출해줘: "
        "- `airline_name`: 사용자가 언급한 항공사 이름과 가장 유사한 공식 항공사 이름(예: '대한항공', '아시아나항공', '티웨이항공'). 정확한 항공사 이름을 찾을 수 없으면 null로 추출해줘.\n"
        "- `airport_name`: 도시명 또는 공항 이름 (예: '도쿄', '후쿠오카'). 정보가 없으면 null로 추출해줘.\n"
        "- `airport_codes`: **해당 공항의 IATA 코드 리스트**. 예를 들어 '파리'라고 말하면 ['CDG', 'ORY']를, '도쿄'라고 말하면 ['NRT', 'HND']를, '후쿠오카'라고 말하면 ['FUK']을 넣어줘. 정보가 없으면 빈 리스트로 추출해줘.\n"
        "- `day_of_week`: 요일 (예: '월요일', '오늘'). 요일 정보가 없으면 '오늘'로 간주해줘.\n"
        "- `direction`: 운항 방향 ('arrival' 또는 'departure'). 정보가 없으면 'departure'로 간주해줘.\n"
        "- `time_period`: 시간대 (예: '오전', '오후', '저녁', '새벽'). 정보가 없으면 null로 추출해줘.\n"
        "응답 시 다른 설명 없이 오직 JSON 리스트만 반환해야 해."
        "\n\n응답 형식: "
        "```json"
        "["
        "  {{"
        "    \"airline_name\": \"[항공사명 (string), 없으면 null]\", "
        "    \"airport_name\": \"[공항명 (string), 없으면 null]\", "
        "    \"airport_codes\": [\"[IATA 코드 리스트]\"], "
        "    \"day_of_week\": \"[요일 (string), 없으면 '오늘']\", "
        "    \"direction\": \"[arrival|departure]\", "
        "    \"time_period\": \"[오전|오후|저녁|새벽|null]\""
        "  }}"
        "]"
        "```"
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

def _call_schedule_api(direction: str) -> list | str:
    """
    공공 데이터 포털 API를 호출하여 전체 데이터를 가져오고 결과를 파싱합니다.
    """
    api_url = f"{FLIGHT_SCHEDULE_API_BASE_URL}/getPaxFltSched{direction.capitalize()}s"
    
    all_items = []
    page_no = 1
    total_count = 1
    num_of_rows = 1000 # 한 번에 최대한 많은 데이터를 가져오도록 설정

    while len(all_items) < total_count:
        params_with_key = {
            "serviceKey": SERVICE_KEY,
            "type": "json",
            "numOfRows": num_of_rows,
            "pageNo": page_no,
        }
        
        try:
            response = requests.get(api_url, params=params_with_key)
            response.raise_for_status()
            response_data = response.json()
            
            body = response_data.get("response", {}).get("body", {})
            items = body.get("items", {})
            
            if not items:
                return all_items
            
            total_count = body.get("totalCount", 0)
            
            current_items = items.get("item", []) if isinstance(items, dict) else items
            
            if not current_items:
                return all_items
                
            all_items.extend(current_items)
            page_no += 1
            
            print(f"디버그: {direction.capitalize()} API 호출 (page {page_no-1}, {len(current_items)} items retrieved)")
            
        except requests.exceptions.RequestException as e:
            print(f"디버그: API 호출 중 오류 발생 ({direction}) - {e}")
            return "api_error"
        except Exception as e:
            print(f"디버그: 응답 처리 중 오류 발생 ({direction}) - {e}")
            return "api_error"
    
    return all_items

def regular_schedule_query_handler(state: ChatState) -> ChatState:
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "regular_schedule_query")

    if not user_query:
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    parsed_queries = _parse_schedule_query_with_llm(user_query)
    if not parsed_queries:
        return {**state, "response": "죄송합니다. 스케줄 정보를 파악하는 중 문제가 발생했습니다. 다시 시도해 주세요."}
    
    final_responses = []

    # API 호출은 한 번만 수행
    direction = parsed_queries[0].get('direction', 'departure')
    
    api_result = _call_schedule_api(direction)
    if isinstance(api_result, str):
      return {**state, "response": f"API 호출 중 오류가 발생했습니다: {api_result}"}
    
    retrieved_api_docs = api_result

    for parsed_query in parsed_queries:
        airline_name = parsed_query.get("airline_name")
        # LLM이 추출한 IATA 코드 리스트
        airport_codes = parsed_query.get("airport_codes", [])
        day_name = parsed_query.get("day_of_week")
        time_period = parsed_query.get("time_period")
        
        filtered_docs = []
        day_field_name = _get_day_of_week_field(day_name)
        
        for doc in retrieved_api_docs:
            if not isinstance(doc, dict):
                continue
            
            # ✅ 도착지 공항 코드(IATA) 필터링 추가
            # LLM이 추출한 공항 코드가 있고, 현재 문서의 공항 코드가 리스트에 없으면 건너뜁니다.
            if airport_codes and doc.get("airportcode") not in airport_codes:
                continue

            # 요일 필터링
            if day_field_name and doc.get(day_field_name, 'N') != 'Y':
                continue

            # 시간대 필터링
            if time_period:
                scheduled_time_str = doc.get("st", "0000")
                if time_period == '오전' and not ('0600' <= scheduled_time_str < '1200'):
                    continue
                if time_period == '오후' and not ('1200' <= scheduled_time_str < '1800'):
                    continue
                if time_period == '저녁' and not ('1800' <= scheduled_time_str <= '2359'):
                    continue
                if time_period == '새벽' and not ('0000' <= scheduled_time_str < '0600'):
                    continue

            # 항공사명 필터링 (정확한 매칭)
            normalized_airline_name = airline_name.replace(' ', '') if airline_name else None
            normalized_doc_airline = doc.get("airline", "").replace(' ', '')
            
            if normalized_airline_name and normalized_airline_name not in normalized_doc_airline:
                continue

            filtered_docs.append(doc)
        
        # 필터링된 문서를 출발 시간 순으로 정렬
        filtered_docs.sort(key=lambda x: x.get("st", "9999"))
        
        # 상위 5개만 선택
        top_5_docs = filtered_docs[:5]
        
        print(f"디버그: 필터링된 문서 총 {len(filtered_docs)}개 중, 상위 5개 문서: {top_5_docs}")
        
        if not top_5_docs:
            response_text = f"죄송합니다. '{airline_name}' 항공사의 {parsed_query.get('airport_name', '특정 목적지')} {day_name} {time_period} 출발 스케줄 정보를 찾을 수 없습니다."
            final_responses.append(response_text)
            continue
        
        # RAG 호출을 위한 컨텍스트 생성
        context_for_llm = json.dumps(top_5_docs, ensure_ascii=False, indent=2)
        intent_description = f"사용자가 요청한 '{airline_name}' 항공사의 {parsed_query.get('airport_name', '특정 목적지')} {day_name} {time_period} 출발 스케줄 정보를 요약하여 친절하게 답변해줘. 여러 항공편 정보를 구조화된 목록 형태로 보기 좋게 정리해줘."

        final_response_part = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)
        final_responses.append(final_response_part)

    if not final_responses:
      return {**state, "response": "죄송합니다. 요청하신 조건에 맞는 정보를 찾을 수 없습니다."}

    return {**state, "response": "\n\n".join(final_responses)}