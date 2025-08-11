from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from chatbot.graph.state import ChatState
from chatbot.graph.nodes.classifiy_intent import classify_intent

# 환경 변수를 로드합니다.
from dotenv import load_dotenv
load_dotenv()

# 직접 openai 클라이언트를 사용합니다.
from openai import OpenAI
client = OpenAI()


def _create_subgraph_for_intent(handlers: Dict[str, Any]):
    """개별 질문 처리를 위한 서브그래프를 생성합니다."""
    subgraph = StateGraph(ChatState)
    subgraph.add_node("classify_intent", classify_intent)
    
    for name, handler in handlers.items():
        subgraph.add_node(name, handler)
        subgraph.add_edge(name, END)
    
    def route_single_intent_to_handler(state):
        intent = state.get("intent")
        if intent:
            node_name = f"{intent}_handler"
            if node_name in handlers:
                return node_name
        return None 
    
    subgraph.set_entry_point("classify_intent")
    all_handler_names = list(handlers.keys())
    
    subgraph.add_conditional_edges(
        "classify_intent", 
        route_single_intent_to_handler,
        all_handler_names
    )
    
    return subgraph.compile()


def _split_intents(user_input: str, supported_intents: List[str]) -> List[str]:
    """LLM을 사용하여 복합 의도 질문을 독립적인 하위 질문으로 분해합니다."""
    
    prompt = f"""
    당신은 사용자의 질문을 분석하여, 여러 의도가 포함된 질문을 독립적인 하위 질문들로 분해하는 데 능숙한 전문가입니다.

    ### 지시사항
    1. 사용자의 질문이 **하나의 주제에 대한 여러 요청**인지, 아니면 **서로 다른 주제에 대한 여러 요청**인지 판단하세요.
    2. 주제가 서로 다른 경우에만 질문을 분리하세요.
    3. 질문을 분리할 때, 주차장, 시설, 비행, 날씨 등의 키워드를 기준으로 주제를 판단하세요.
    4. 분리된 각 질문은 완전한 문장 형태여야 하며, 쉼표(,)로 구분하여 한 줄로 반환하세요.
    5. 질문이 분리될 수 없다면 (단일 의도라면), 원래 질문을 그대로 반환하세요.
    6. 절대 다른 설명이나 문장은 추가하지 말고, 오직 분리된 질문들만 반환하세요.
    7. 지원되는 의도 목록은 {', '.join(supported_intents)}입니다.

    ### 예시
    - 사용자 질문: "주차 요금이랑 카페 위치 알려줘"
    - 출력: 주차 요금 알려줘, 카페 위치 알려줘

    - 사용자 질문: "지금 공항 날씨 어때요? 내일은요?"
    - 출력: 지금 공항 날씨 어때요? 내일은요?

    - 사용자 질문: "제1터미널 주차장 요금 알려줘"
    - 출력: 제1터미널 주차장 요금 알려줘

    사용자 질문: "{user_input}"
    출력:
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 사용자의 질문 의도를 정확하게 분석하고, 복합적인 질문을 여러 개의 단일 의도로 분리하는 데 능숙한 전문가입니다. 특히 인천국제공항과 관련된 질문을 가장 잘 처리합니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    result_text = response.choices[0].message.content
    split_questions = [q.strip() for q in result_text.split(',') if q.strip()]
    print("분리된 질문 :", split_questions)
    
    if not split_questions:
        return [user_input]
        
    return split_questions

def _combine_responses(original_question: str, responses: List[str]) -> str:
    """LLM을 사용하여 여러 답변을 하나의 자연스러운 문장으로 종합합니다."""
    if not responses:
        return "죄송합니다. 요청하신 정보에 대한 답변을 찾을 수 없습니다."
        
    prompt = f"""
    당신은 사용자의 원래 질문 '{original_question}'과 그에 대한 여러 정보를 종합하여 하나의 자연스러운 답변을 만듭니다.

    ### 지시사항
    1. 제공된 정보들을 분석하여 답변 간의 관계와 사용자의 숨겨진 의도를 파악하세요.
    2. 파악된 의도를 바탕으로, 모든 정보를 유기적으로 연결하여 하나의 완성된 답변을 만드세요.
    3. 만약 정보 간에 연결점이 없다면, 각 답변을 명확하게 분리하여 나열하되, 답변이 자연스럽게 이어지도록 정리하세요.
    4. 제공된 정보에 없는 내용은 절대로 추가하거나 추론하지 마세요.

    ### 예시
    - 사용자 질문: '도착하자마자 화장실에 가고싶은데 어디에 주차하는게 좋아?'
    - 제공된 정보: ['탑승동 3층 중앙 안내소 부근에 화장실이 있습니다.', '제1터미널에는 P1, P2 장기주차장이 있습니다.']
    - 예상 답변: '탑승동 3층 중앙 안내소 부근에 화장실이 있습니다. 이와 가까운 제1터미널 P1, P2 장기주차장을 이용하시면 편리합니다.'

    ### 제공된 정보:
    {'- ' + '\\n- '.join(responses)}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 사용자의 질문에 대해 가장 핵심적인 정보를 바탕으로 간결하고 정확하게 답변을 종합하는 전문가입니다. 특히 복합 의도가 있는 질문의 경우, 여러 정보를 연결하여 하나의 완성된 답변을 만드는 데 능숙합니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    combined_response_text = response.choices[0].message.content

    common_disclaimer = (
        "\n\n---"
        "\n주의: 이 정보는 인천국제공항 웹사이트(공식 출처)를 기반으로 제공되지만, 실제 공항 운영 정보와 다를 수 있습니다."
        "가장 정확한 최신 정보는 인천국제공항 공식 웹사이트 또는 해당 항공사/기관/시설에 직접 확인하시기 바랍니다."
    )
    
    return combined_response_text + common_disclaimer

def handle_complex_intent(state: ChatState, handlers: Dict[str, Any], supported_intents: List[str]):
    """복합 의도 질문을 분리하고 처리하는 메인 함수"""
    user_input = state["user_input"]
    print("--- 복합 의도 처리 시작 ---")

    print(f"디버그: handle_complex_intent에 전달된 핸들러 목록: {handlers.keys()}")

    split_questions: List[str] = _split_intents(user_input, supported_intents)
    print(f"분해된 질문: {split_questions}")

    all_responses = []
    subgraph = _create_subgraph_for_intent(handlers)
    
    for question in split_questions:
        sub_state = {"user_input": question, "response": None}
        result = subgraph.invoke(sub_state)
        response_content = result.get("response", "")
        if response_content:
            all_responses.append(response_content)
    
    final_response = _combine_responses(user_input, all_responses)
    
    print("--- 복합 의도 처리 완료 ---")
    
    state["response"] = final_response
    state["intent"] = "complex_intent"
    
    return state