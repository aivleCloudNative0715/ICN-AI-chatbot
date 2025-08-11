import json
from chatbot.rag.config import client

def _parse_transfer_route_query_with_llm(user_query: str) -> dict | None:
    """
    LLM을 사용하여 복합 환승 경로 질문을 개별 요청으로 분해하는 함수.
    """
    prompt_content = (
        "사용자 쿼리에서 '터미널', '탑승동' 등 공항 내 주요 시설 간의 환승 경로와 관련된 개별 요청을 JSON 형식으로 추출해줘."
        "만약 복합 질문이라면, 각각의 요청을 'requests' 리스트의 개별 객체로 만들어줘."
        "각 요청 객체는 'query' 필드를 포함해야 해. 'query'에는 RAG 검색에 사용할 구체적인 질문을 담아줘."
        "주차장이나 다른 시설로 가는 길에 대한 질문은 환승 경로에 해당하지 않으므로 포함하지 않아도 돼."
        "응답 시 다른 설명 없이 오직 JSON 객체만 반환해야 해."
        
        "\n\n응답 형식: "
        "```json"
        "{"
        "    \"requests\": ["
        "        {"
        "            \"query\": \"[검색용 질문]\""
        "        }"
        "    ]"
        "}"
        "```"
        "\n\n예시: "
        "사용자: 제1터미널에서 탑승동으로, 그리고 탑승동에서 제2터미널로 가는 길을 알려줘."
        "응답: ```json\n{\"requests\": [{\"query\": \"제1터미널에서 탑승동으로 가는 길\"}, {\"query\": \"탑승동에서 제2터미널로 가는 길\"}]}```"
        "사용자: 탑승동에서 제1터미널로 가는 길, 제1터미널에서 제2터미널로 환승하는데 걸리는 시간 알려줘"
        "응답: ```json\n{\"requests\": [{\"query\": \"탑승동에서 제1터미널로 가는 길\"}, {\"query\": \"제1터미널에서 제2터미널로 환승 시간\"}]}```"
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
        if llm_output.startswith("```json"):
            llm_output = llm_output.lstrip("```json").rstrip("```").strip()
        
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