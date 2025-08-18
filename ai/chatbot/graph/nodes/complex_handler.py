import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from openai import OpenAI
from functools import partial
from chatbot.graph.state import ChatState
from langchain_core.messages import HumanMessage, AIMessage
from chatbot.graph.nodes.classifiy_intent import classify_intent
import re
from chatbot.rag.llm_tools import _format_and_style_with_llm

# 환경 변수를 로드합니다.
from dotenv import load_dotenv
load_dotenv()

# 직접 openai 클라이언트를 사용합니다.
from openai import OpenAI
client = OpenAI()

DISCLAIMER = (
    "\n\n"
    "주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다."
    "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트 또는 해당 항공사/기관/시설에 직접 확인하시기 바랍니다."
)

# 주의 문구를 제거하는 헬퍼 함수
def _remove_disclaimer(text: str) -> str:
    """텍스트에서 미리 정의된 주의 문구를 제거합니다."""
    return re.sub(re.escape(DISCLAIMER), "", text, flags=re.DOTALL).strip()

def _decompose_and_classify_queries(user_query: str, supported_intents: List[str], messages: List[Any]) -> List[Dict[str, str]]:
    """
    LLM을 사용하여 복합 의도 질문을 단일 질문으로 분해하고 의도를 분류합니다.
    이전 대화 맥락을 활용하여 후속 질문을 처리합니다.
    """
    client = OpenAI() # OpenAI 클라이언트 초기화
    supported_intents_str = ", ".join(supported_intents)

    system_prompt = f"""
    당신은 사용자의 복합적인 질문을 단일 의도 질문으로 분해하는 전문가입니다. 
    전체 대화 기록과 사용자의 마지막 질문을 참고하여 질문을 분해하고, 각 부분에 가장 적합한 단일 의도를 찾아 JSON 형식으로 반환하세요.

    사용 가능한 의도 목록:
    {supported_intents_str}

    지침:
    1. 현재 질문이 이전 대화의 후속 질문이라면, 이전 맥락을 포함하여 질문을 재구성하세요. 
       예시: "장기주차장 현황" -> "요금은?" -> "장기주차장 요금은?"
    2. 핵심 시설 키워드 우선 분류: '약국', '은행', '환전소', '식당' 등 시설이나 업체를 묻는 질문은 `facility_guide` 의도로 우선적으로 분류하세요.
    3. 분해된 각 질문에 가장 적절한 단일 의도를 할당하세요.
    4. 질문을 분해할 필요가 없다면, 전체 질문과 하나의 의도만 반환하세요.
    5. 다른 설명이나 문장은 추가하지 말고, 오직 JSON 객체만 반환하세요.

    JSON 응답 형식:
    {{
      "decomposed_queries": [
        {{"question": "질문 1", "intent": "의도명1"}},
        {{"question": "질문 2", "intent": "의도명2"}}
      ]
    }}
    """
    
    messages_for_llm = [
        {"role": "system", "content": system_prompt}
    ]
    # 전체 대화 기록을 프롬프트에 추가
    for msg in messages:
        if isinstance(msg, HumanMessage):
            messages_for_llm.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages_for_llm.append({"role": "assistant", "content": msg.content})
    
    # 현재 질문을 가장 마지막에 추가
    messages_for_llm.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_llm,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        parsed_result = json.loads(result)
        return parsed_result.get("decomposed_queries", [])
    except Exception as e:
        print(f"디버그: LLM 질문 분해 및 의도 분류 실패 - {e}")
        # 실패 시 원래 질문과 기본 의도를 반환
        return [{"question": user_query, "intent": "default"}]

def _initial_dispatch(state: ChatState, handlers: Dict[str, Any]) -> ChatState:
    """서브그래프 내에서 핸들러로 라우팅하기 위한 더미 노드입니다."""
    return state

def _combine_responses(original_query: str, responses: List[str]) -> str:
    """여러 핸들러의 응답을 하나의 응답으로 결합합니다."""
    if len(responses) == 1:
        return responses[0]
    
    combined_text = ""
    for response in responses:
        combined_text += f"{response.strip()}\n\n"
    
    return combined_text.strip()

# ----------------------------------------------------------------------
# 챗봇의 메인 그래프에서 호출되는 함수
def handle_complex_intent(state: ChatState, handlers: Dict[str, Any], supported_intents: List[str]):
    """복합 의도 질문을 분리하고 처리하는 메인 함수"""
    user_input = state["user_input"]
    messages = state.get("messages", [])
    print("--- 복합 의도 처리 시작 ---")

    # 📌 핵심 변경점: LLM을 사용하여 전체 맥락을 고려한 질문 분해
    decomposed_queries = _decompose_and_classify_queries(user_input, supported_intents, messages)
    print(f"분해된 질문: {decomposed_queries}")

    all_responses = []
    
    # 단일 의도 처리를 위한 서브그래프 생성
    subgraph_builder = StateGraph(ChatState)
    subgraph_builder.add_node("entry_point", partial(_initial_dispatch, handlers=handlers))
    for name, handler in handlers.items():
        subgraph_builder.add_node(name, handler)
        subgraph_builder.add_edge(name, END)
    
    subgraph_builder.set_entry_point("entry_point")
    
    def subgraph_router(state):
        intent = state.get("intent")
        return f"{intent}_handler" if f"{intent}_handler" in handlers else "default_handler"

    subgraph_builder.add_conditional_edges("entry_point", subgraph_router)
    subgraph = subgraph_builder.compile()

    for item in decomposed_queries:
        question = item["question"]
        intent = item["intent"]
        
        # 분해된 각 질문에 대해 새로운 상태를 만들고 서브그래프 호출
        sub_state = {"user_input": question, "intent": intent, "response": None}
        result = subgraph.invoke(sub_state)
        response_content = result.get("response", "")
        if response_content:
            cleaned_response = _remove_disclaimer(response_content)
            all_responses.append(cleaned_response)

    final_response_text = _combine_responses(user_input, all_responses)
    
    final_response = _format_and_style_with_llm(final_response_text, "complex_intent")
    final_response += DISCLAIMER
    
    print("--- 복합 의도 처리 완료 ---")
    
    state["response"] = final_response
    state["intent"] = "complex_intent"
    
    return state
