# ai/chatbot/graph/nodes/complex_handler.py

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from ai.chatbot.graph.state import ChatState
from ai.chatbot.graph.nodes.classifiy_intent import classify_intent
from ai.chatbot.graph.utils.llm_utils import split_intents, combine_responses


def _create_subgraph_for_intent(handlers: Dict[str, Any]):
    """개별 질문 처리를 위한 서브그래프 생성 함수"""
    subgraph = StateGraph(ChatState)
    subgraph.add_node("classify_intent", classify_intent)
    for name, handler in handlers.items():
        subgraph.add_node(name, handler)
        subgraph.add_edge(name, END)

    def route_single_intent_to_handler(state):
        """의도 이름을 핸들러 노드 이름으로 변환하여 라우팅"""
        intent = state.get("intent")
        if intent:
            node_name = f"{intent}_handler"
            if node_name in handlers:
                return node_name
        return "default_handler"

    subgraph.set_entry_point("classify_intent")
    subgraph.add_conditional_edges("classify_intent", route_single_intent_to_handler)

    return subgraph.compile()


def handle_complex_intent(state: ChatState, handlers: Dict[str, Any], supported_intents: List[str]):
    """복합 의도 질문을 분리하고 처리하는 메인 함수"""
    user_input = state["user_input"]

    # 1. LLM을 사용하여 복합 의도 질문을 분리
    split_questions: List[str] = split_intents(user_input, supported_intents)

    # 2. 분리된 각 질문을 개별적으로 처리 (서브그래프 활용)
    all_responses = []
    subgraph = _create_subgraph_for_intent(handlers)

    for question in split_questions:
        result = subgraph.invoke({"user_input": question})
        all_responses.append(result.get("response", ""))

    # 3. 모든 답변을 하나로 종합
    final_response = combine_responses(all_responses)

    return {"response": final_response, "intent": "complex_intent"}