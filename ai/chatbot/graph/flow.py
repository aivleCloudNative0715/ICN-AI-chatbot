import importlib
import pkgutil
from functools import partial
from langgraph.graph import StateGraph, END

from chatbot.graph.state import ChatState
from chatbot.graph.nodes.classifiy_intent import classify_intent
from chatbot.graph.nodes.complex_handler import handle_complex_intent
from chatbot.graph.nodes.llm_verify_intent import llm_verify_intent_node
import chatbot.graph.handlers


def build_chat_graph():
    builder = StateGraph(ChatState)
    handlers = {}
    supported_intents = []

    # classify_intent 노드를 가장 먼저 추가
    builder.add_node("classify_intent", classify_intent)

    # 핸들러 노드들을 동적으로 추가하고 엣지를 연결
    for importer, modname, ispkg in pkgutil.iter_modules(chatbot.graph.handlers.__path__):
        if modname.startswith("__"):
            continue
        
        module = importlib.import_module(f"chatbot.graph.handlers.{modname}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if callable(attribute) and attribute_name.endswith("_handler"):
                node_name = attribute_name
                builder.add_node(node_name, attribute)
                builder.add_edge(node_name, END)
                handlers[node_name] = attribute
                supported_intents.append(node_name.replace("_handler", ""))

    # 복합 의도 처리 노드 추가
    complex_handler_node = partial(handle_complex_intent, handlers=handlers, supported_intents=supported_intents)
    builder.add_node("handle_complex_intent", complex_handler_node)
    builder.add_edge("handle_complex_intent", END)

    # LLM 검증 노드 추가
    builder.add_node("llm_verify_intent", llm_verify_intent_node)
    
    def route_final_intent_to_handler(state):
        final_intent = state.get("intent")
        if final_intent:
            if final_intent == "complex_intent":
                return "handle_complex_intent"
            node_name = f"{final_intent}_handler"
            if node_name in handlers:
                return node_name
        return "fallback_handler"
    
    def route_after_initial_classification(state: ChatState) -> str:
        top_k_intents = state.get('top_k_intents_and_probs', [])
        slots = state.get("slots", [])
        user_query = state.get("user_input", "")

        # 이전 대화 감지 로직을 최상단으로 이동 (기존 로직 유지)
        if len(state.get("messages", [])) > 1:
            print("DEBUG: 이전 대화 감지 -> llm_verify_intent로 라우팅")
            return "llm_verify_intent"

        # 1. 복합 의도 감지 (classify_intent에서 이미 판별됨)
        if state.get("is_multi_intent", False) or state.get("intent") == "complex_intent":
            detected_intents = [intent for intent, _ in state.get("detected_intents", [])]
            print(f"복합 의도 감지: {detected_intents} -> handle_complex_intent로 라우팅")
            return "handle_complex_intent"
            
        # 2. 단일 의도인 경우 직접 핸들러로 라우팅
        intent = state.get("intent")
        if intent and intent != "complex_intent":
            handler_name = f"{intent}_handler"
            print(f"DEBUG: 단일 의도 감지 -> {handler_name}로 라우팅")
            return handler_name

        # 3. 신뢰도가 낮거나 모호한 경우 LLM 재확인
        print("DEBUG: 낮은 신뢰도 또는 모호한 의도 감지 -> llm_verify_intent로 라우팅")
        return "llm_verify_intent"

    # 그래프의 시작점과 엣지 연결
    builder.set_entry_point("classify_intent")
    
    all_nodes = list(handlers.keys()) + ["handle_complex_intent", "llm_verify_intent"]
    
    builder.add_conditional_edges(
        "classify_intent",
        route_after_initial_classification,
        all_nodes
    )
    
    builder.add_conditional_edges(
        "llm_verify_intent",
        route_final_intent_to_handler,
        all_nodes
    )

    return builder.compile()