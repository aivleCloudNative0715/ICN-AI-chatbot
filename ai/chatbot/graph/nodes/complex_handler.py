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
    
    # 핸들러를 추가하고, 핸들러에서 END로 바로 연결
    # 이렇게 하면 핸들러의 return 값이 최종 result에 담깁니다.
    for name, handler in handlers.items():
        subgraph.add_node(name, handler)
        subgraph.add_edge(name, END)
    
    def route_single_intent_to_handler(state):
        intent = state.get("intent")
        if intent:
            # 핸들러의 이름(handler_func)이 아닌, 노드 이름(name)을 반환
            node_name = f"{intent}_handler" # 핸들러 이름으로 라우팅
            if node_name in handlers:
                return node_name
        return None 
    
    subgraph.set_entry_point("classify_intent")
    all_handler_names = list(handlers.keys())
    
    # conditional_edges의 두 번째 인자로 라우팅 함수를 전달
    subgraph.add_conditional_edges(
        "classify_intent", 
        route_single_intent_to_handler,
        # conditional_edges의 세 번째 인자로 유효한 노드 이름 목록을 전달합니다.
        all_handler_names
    )
    
    return subgraph.compile()


def _split_intents(user_input: str, supported_intents: List[str]) -> List[str]:
    """LLM을 사용하여 복합 의도 질문을 독립적인 하위 질문으로 분해합니다."""
    
    prompt = f"""
    당신은 사용자의 질문을 분석하여, 여러 의도가 포함된 질문을 독립적인 하위 질문들로 분해하는 데 능숙한 전문가입니다.

    ### 지시사항
    1. 사용자의 질문이 하나 이상의 독립적인 질문으로 분리될 수 있는지 판단하세요.
    2. 질문을 분리할 때, 주차장(주차, 주차요금), 시설(카페, 식당, 라운지), 비행(도착, 출발, 항공편) 등의 키워드를 기준으로 분리하세요.
    3. 분리된 각 질문은 완전한 문장 형태여야 하며, 쉼표(,)로 구분하여 한 줄로 반환하세요.
    4. 질문이 분리될 수 없다면 (단일 의도라면), 원래 질문을 그대로 반환하세요.
    5. 절대 다른 설명이나 문장은 추가하지 말고, 오직 분리된 질문들만 반환하세요.
    6. 지원되는 의도 목록은 {', '.join(supported_intents)}입니다.

    ### 예시
    사용자 질문: "주차 요금이랑 카페 위치 알려줘"
    출력: 주차 요금 알려줘, 카페 위치 알려줘

    사용자 질문: "도착하자마자 카페가고싶은데 어디에 주차하는게 좋아?"
    출력: 카페 위치 알려줘, 도착하자마자 어디에 주차하는게 좋아?

    사용자 질문: "제1터미널 주차장 요금 알려줘"
    출력: 제1터미널 주차장 요금 알려줘

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
    1. 제공된 정보들을 분석하여 두 답변 간에 공통된 터미널이나 위치 정보가 있는지 확인하세요.
    2. 만약 공통된 정보(예: '제1여객터미널')가 있다면, 이를 바탕으로 두 정보를 논리적으로 연결하여 하나의 통합 답변을 만드세요.
    3. 공통된 정보가 없다면, 두 답변을 각각 명확하게 분리하여 나열하되, 답변이 자연스럽게 이어지도록 정리하세요.
    4. 제공된 정보에 없는 내용은 절대로 추가하거나 추론하지 마세요.

    ### 제공된 정보:
    {'- ' + '\n- '.join(responses)}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 사용자의 질문에 대해 가장 핵심적인 정보를 바탕으로 간결하고 정확하게 답변을 종합하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    combined_response_text = response.choices[0].message.content

    # 모든 답변에 공통적으로 추가될 주의 문구
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

    # 딕셔너리 내용 확인용 디버그 코드 추가
    print(f"디버그: handle_complex_intent에 전달된 핸들러 목록: {handlers.keys()}")

    # 1. LLM을 사용하여 복합 의도 질문을 분리
    split_questions: List[str] = _split_intents(user_input, supported_intents)
    print(f"분해된 질문: {split_questions}")

    # 2. 분리된 각 질문을 개별적으로 처리 (서브그래프 활용)
    all_responses = []
    subgraph = _create_subgraph_for_intent(handlers)
    
    for question in split_questions:
        # 서브그래프에 전달할 초기 상태
        sub_state = {"user_input": question, "response": None}
        
        # 서브그래프 실행
        result = subgraph.invoke(sub_state)
        
        # 여기서 서브그래프가 반환한 최종 응답만 추출
        response_content = result.get("response", "")
        
        if response_content:
            all_responses.append(response_content)
    
    # 3. 모든 답변을 하나로 종합
    final_response = _combine_responses(user_input, all_responses)
    
    print("--- 복합 의도 처리 완료 ---")
    
    # 최종 상태 업데이트
    state["response"] = final_response
    state["intent"] = "complex_intent" 
    
    return state