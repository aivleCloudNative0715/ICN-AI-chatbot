import os
import requests
import json
from dotenv import load_dotenv

from chatbot.rag.config import client
from chatbot.graph.utils.formatting_utils import get_enhanced_prompt

load_dotenv()

SERVICE_KEY = os.getenv("SERVICE_KEY")
if not SERVICE_KEY:
    raise ValueError("SERVICE_KEY 환경 변수가 설정되지 않았습니다.")

FLIGHT_API_BASE_URL = "http://apis.data.go.kr/B551177/StatusOfPassengerFlightsDeOdp"
FLIGHT_ARRIVAL_URL = f"{FLIGHT_API_BASE_URL}/getPassengerArrivalsDeOdp"



def call_arrival_flight_api(params: dict):
    """
    항공편 API를 호출하고 결과를 파싱하는 내부 함수
    """
    params_with_key = {
        "serviceKey": SERVICE_KEY,
        "type": "json",
        **params
    }
    print(params_with_key)
    api_url = FLIGHT_ARRIVAL_URL
    
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
        print(f"디버그: API 호출 중 오류 발생 - {e}")
        return "api_error"
    except Exception as e:
        print(f"디버그: 응답 처리 중 오류 발생 - {e}")
        return "api_error"


def _parse_flight_baggage_query_with_llm(user_query: str):
    """
    LLM을 사용하여 사용자 쿼리에서 항공편 운항 정보를 JSON 리스트 형식으로 추출하는 함수.
    """
    prompt_content = (
        "사용자 쿼리에서 정보를 추출하여 JSON 배열 형태로 반환해줘."
        "각 배열 요소는 하나의 항공편에 대한 정보를 담고 있어야 해. "
        "조회일 기준 -3일과 +6일 이내의 날짜만 지원하며, 범위를 벗어나면 'unsupported'로 응답해줘. 언급이 없으면 0으로 처리해줘."
        "편명은 'OZ704'와 같이 항공사 코드와 숫자가 조합된 형태여야 해. 대문자와 숫자만 사용돼. 언급되지 않는다면 null 처리해줘."
        "searchday는 '20231001'과 같이 YYYYMMDD 형식으로 반환해줘. 날짜가 추정되지 않는다면 null 처리해줘."
        "airport_code은 출발지 공항에 대한 IATA 코드야. 언급되지 않으면 null 처리해줘."
        "만약 너가 알기 IATA 코드를 모른다면 그냥 공항 이름을 한글로 반환해줘."
        "from_time과 to_time은 '0000'과 같이 HHMM 형식으로 반환해줘. "
        "from_time을 유추할 언급이 없다면 null 처리해줘. 유추 가능하다면 그 시각 -1시간으로 설정해줘."
        "to_time은 from_time +2시간으로 설정해줘."
        "응답 시 다른 설명 없이 오직 JSON 배열만 반환해야 해."

        "\n\n응답 형식: "
        "```json"
        "["
        "  {{"
        "    \"date_offset\": \"[오늘=0, 내일=1, 3일 전=-3, 6일 뒤=6, 범위를 벗어나면 'unsupported', 언급이 없으면 0]\", "
        "    \"flight_id\": \"[편명 (string), 없으면 null]\", "
        "    \"searchday\": \"[일자 (string)], 없으면 null\", "
        "    \"from_time\": \"[추정되는 도착 예정 시각 시작점(string), 없으면 null]\", "
        "    \"to_time\": \"[추정되는 도착 예정 시각 끝점 (string), 없으면 null]\", "
        "    \"airport_code\": \"[출발한 공항 IACA 코드명 (string), 없으면 null]\", "
        "  }}"
        "]"
        "```"
        "\n\n예시: "
        "사용자: 25년 8월 8일 암스테르담에서 출발해서 12시에 도착하는 항공편인데 수하물 수취대 정보 알려줘"
        "응답: ```json\n[{{\"date_offset\": 0, \"flight_id\": null, \"searchday\": \"20250808\", \"from_time\": \"1100\", \"to_time\": \"1300\", \"airport_name\": \"AMS\"}}]```"
        "사용자: 오늘 5시에 도착하는 비행긴데, 수하물 어디서 받아?"
        "응답: ```json\n[{{\"date_offset\": 0, \"flight_id\": null, \"searchday\": null, \"from_time\": \"1400\", \"to_time\": \"1800\", \"airport_name\": null}}]```"
        "사용자: 내일 나리타 공항에서 출발해서 오전 7시에 도착하는 비행긴데, 수하물 어디서 받아? 편명은 KE211이야"
        "응답: ```json\n[{{\"date_offset\": 1, \"flight_id\": \"KE211\", \"searchday\": null, \"from_time\": \"0600\", \"to_time\": \"0800\", \"airport_name\": \"NRT\"}}]```"
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
    print(f"디버그: LLM 응답 - {llm_output}")

    try:
        if llm_output.startswith("```json") and llm_output.endswith("```"):
            llm_output = llm_output[7:-3].strip()
        
        parsed_data = json.loads(llm_output)
        print(f"디버그: 파싱된 데이터 - {parsed_data}")
            
        # 요청 정보 키워드가 문자열로 반환되는 경우 리스트로 변환하고,
        # 'null'이거나 키가 없는 경우 빈 리스트로 초기화합니다.
        for item in parsed_data:
            
            # 날짜 오프셋이 문자열일 경우 정수로 변환
            if "date_offset" in item and isinstance(item["date_offset"], str):
                try:
                    item["date_offset"] = int(item["date_offset"])
                except (ValueError, TypeError):
                    item["date_offset"] = 0
            if "searchday" in item and isinstance(item["searchday"], str):
                try:
                    item["searchday"] = int(item["searchday"])
                except (ValueError, TypeError):
                    item["searchday"] = None
            if "from_time" in item and isinstance(item["from_time"], str):
                try:
                    item["from_time"] = int(item["from_time"])
                except (ValueError, TypeError):
                    item["from_time"] = 0000
            if "to_time" in item and isinstance(item["to_time"], str):
                try:
                    item["to_time"] = int(item["to_time"])
                except (ValueError, TypeError):
                    item["to_time"] = 2359
                
        return parsed_data
    except json.JSONDecodeError:
        print("디버그: LLM 응답이 올바른 JSON 형식이 아닙니다.")
        print(f"디버그: LLM 원본 응답 -> {llm_output}")
    
    return None

def _parse_airport_code_with_llm(document: str):
    """
    LLM을 사용하여 공항 코드를 추출하는 함수
    """
    prompt_content = (
        "RAG에서 검색된 공항의 코드를 추출해줘."
        "공항 코드는 IATA 3자리 코드로, 예를 들어 인천국제공항은 'ICN'이야."
        "\n\n응답 형식: "
        "오직 공항 코드만 반환해줘. string 형태로 반환해야 해."

        "\n\n예시: "
        "입력: {공항 코드 LCG는 스페인에 있는 아코루냐 공항입니다.}"
        "\n\n출력: "
        "LCG"
    )

    messages = [
        {"role": "system", "content": prompt_content},
        {"role": "user", "content": document}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0
    )
    try:
        llm_output = response.choices[0].message.content.strip()
    
        return llm_output
    except:
        print("디버그: LLM 응답이 올바른 형식이 아닙니다.")
        print(f"디버그: LLM 원본 응답 -> {llm_output}")
    
        return None
    
def _generate_final_answer_with_llm(document: dict, user_query: str) -> str:
    """
    LLM을 수하물 수취대 정보를 알려주는 최종 답변을 생성하는 함수
    """
    prompt_content = (
        "너는 인천국제공항의 수하물 수취대 정보를 제공하는 친절한 챗봇이야."
        "{document} 정보를 기반으로 수하물 수취대 정보를 알려줘."
        "도착시간이나, 게이트 같은 항공편에 대한 정보도 제공하면 좋지만, 반드시 수하물 수취대 정보에 집중해야 해."
        "만약 후보가 여러 개라면, 가장 적합한 수하물 수취대 정보를 선택해서 알려줘. 우열이 명확하지 않다면 후보를 5개 이하로 제공해"
        "만약 수하물 수취대 정보가 없다면, 수하물 수취대 정보가 없다고 답변해줘."
        "답변은 사용자에게 친절하고 유용하게 작성해줘."
        "만약 답변에 필요한 정보가 부족하다면, 사용자에게 추가 정보를 요청하는 메시지를 작성해줘."
    )
    
    # 📌 기존 프롬프트와 HTML 지침을 결합
    final_prompt_with_formatting = prompt_content

    formatted_prompt = final_prompt_with_formatting.format(
        document=json.dumps(document, ensure_ascii=False, indent=2, default=str)
    )
    
    # 포맷팅 지침이 포함된 프롬프트 사용
    enhanced_prompt = get_enhanced_prompt(formatted_prompt, "baggage_claim_info")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": enhanced_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.5,
        max_tokens=600
    )
    
    return response.choices[0].message.content