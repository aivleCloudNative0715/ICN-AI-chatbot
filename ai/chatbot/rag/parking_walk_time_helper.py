import json
from chatbot.rag.config import client

def _parse_parking_walk_time_query_with_llm(user_query: str) -> dict | None:
    """
    LLM을 사용하여 복합 주차장 도보 시간 질문을 개별 요청으로 분해하는 함수.
    """
    prompt_content = (
        "사용자 쿼리에서 주차장 도보 시간과 관련된 개별 요청을 JSON 형식으로 추출해줘."
        "만약 복합 질문이라면, 각각의 요청을 'requests' 리스트의 개별 객체로 만들어줘."
        "각 요청 객체는 'query' 필드를 포함해야 해. 'query'에는 RAG 검색에 사용할 구체적인 질문을 담아줘."
        "응답 시 다른 설명 없이 오직 JSON 객체만 반환해야 해."
        
        "\n\n응답 형식: "
        "```json"
        "{"
        "   \"requests\": ["
        "      {"
        "        \"query\": \"[검색용 질문]\""
        "      }"
        "   ]"
        "}"
        "```"
        "\n\n예시: "
        "사용자: 제1여객터미널 장기주차장에서 탑승동까지 얼마나 걸려?"
        "응답: ```json\n{\"requests\": [{\"query\": \"제1여객터미널 장기주차장에서 탑승동까지 도보 시간\"}]}```"
        "사용자: 제2터미널 주차장에서 출발장까지 가는 시간과, 제1터미널에서 장기주차장까지 가는 시간 알려줘."
        "응답: ```json\n{\"requests\": [{\"query\": \"제2터미널 주차장에서 출발장까지 가는 시간\"}, {\"query\": \"제1터미널에서 장기주차장까지 가는 시간\"}]}```"
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