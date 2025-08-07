import json
from ai.chatbot.rag.config import client

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

def _parse_query_with_llm(user_query: str) -> dict | None:
    """
    LLM을 사용하여 사용자 쿼리에서 날짜, 시간, 터미널, 구역 정보를 JSON 형식으로 추출하는 함수.
    """
    prompt_content = (
        "사용자 쿼리에서 인천국제공항의 혼잡도 예측 정보를 추출해줘."
        "오늘, 내일 외의 날짜는 'unsupported'로 응답해줘. 날짜가 언급되지 않으면 오늘로 간주해줘."
        "시간은 0부터 23까지의 정수여야 해. 시간이 언급되지 않으면 null로 추출해줘."
        "터미널 번호는 1 또는 2 중 하나야. 터미널 정보가 없으면 null로 추출해줘."
        "구역은 '입국장' 또는 '출국장'과 알파벳/숫자를 포함해야 해."
        "유효하지 않은 구역 조합(예: '출국장'과 알파벳, '입국장'과 숫자)이 발견되면 'area'는 null로 추출해줘."
        "만약 쿼리에 '입국장' 또는 '출국장'에 대한 언급 없이 'C', 'D', 'E', 'F', '3', '4', '5', '6'이 언급되면, 제1터미널의 해당 구역으로 가정해줘."
        "질문이 혼잡도 전체에 대한 내용이면 'requests' 리스트는 비워줘."
        "만약 여러 터미널이나 구역에 대한 질문이라면, 'requests' 리스트에 여러 객체를 만들어줘."
        "각 요청 객체는 'date', 'time', 'terminal', 'area' 필드를 모두 포함해야 해."
        "**응답 시 다른 설명 없이 오직 JSON 객체만 반환해야 해.**"
        
        "\n\n응답 형식: "
        "```json"
        "{"
        "   \"requests\": ["
        "      {"
        "         \"date\": \"[today|tomorrow|unsupported]\", "
        "         \"time\": [시간 (0~23 정수), 없으면 null], "
        "         \"terminal\": [터미널 번호 (1, 2), 없으면 null], "
        "         \"area\": \"[구역명 (string), 없으면 null]\""
        "      }"
        "   ]"
        "}"
        "```"
        "\n\n예시: "
        "사용자: 내일 11시 1출국장 혼잡해?"
        "응답: ```json\n{\"requests\": [{\"date\": \"tomorrow\", \"time\": 11, \"terminal\": 1, \"area\": \"출국장1\"}]}```"
        "사용자: 1터미널과 2터미널 혼잡도 알려줘"
        "응답: ```json\n{\"requests\": [{\"date\": \"today\", \"time\": null, \"terminal\": 1, \"area\": null}, {\"date\": \"today\", \"time\": null, \"terminal\": 2, \"area\": null}]}```"
        "사용자: 1터미널 입국장C, 2터미널 입국장A 혼잡해?"
        "응답: ```json\n{\"requests\": [{\"date\": \"today\", \"time\": null, \"terminal\": 1, \"area\": \"입국장C\"}, {\"date\": \"today\", \"time\": null, \"terminal\": 2, \"area\": \"입국장A\"}]}```"
        "사용자: 출국장 A 혼잡도가 어떻게 돼?"
        "응답: ```json\n{\"requests\": []}```"
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
        # 1. '```json'과 '```' 부분을 안전하게 제거
        if llm_output.startswith("```json"):
            llm_output = llm_output.lstrip("```json").rstrip("```").strip()

        # 2. 파이썬 문자열로 변환할 때 발생할 수 있는 문제 방지
        # 예: JSON 내부에 잘못된 문자나 포맷이 있을 경우
        parsed_data = json.loads(llm_output)
        return parsed_data
    except json.JSONDecodeError as e:
        print("디버그: LLM 응답이 올바른 JSON 형식이 아닙니다.")
        print(f"디버그: JSONDecodeError -> {e}")
        print(f"디버그: LLM 원본 응답 -> {llm_output}")
    except Exception as e:
        print(f"디버그: 알 수 없는 오류 발생 -> {e}")
        print(f"디버그: LLM 원본 응답 -> {llm_output}")
    
    return None