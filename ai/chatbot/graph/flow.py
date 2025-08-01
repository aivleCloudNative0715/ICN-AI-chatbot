import importlib
import pkgutil
from functools import partial
from langgraph.graph import StateGraph, END

from ai.chatbot.graph.state import ChatState
from ai.chatbot.graph.router import route_by_intent
from ai.chatbot.graph.nodes.classifiy_intent import classify_intent
from ai.chatbot.graph.nodes.complex_handler import handle_complex_intent
import ai.chatbot.graph.handlers


def build_chat_graph():
    builder = StateGraph(ChatState)
    handlers = {}
    supported_intents = []

    # classify_intent 노드를 가장 먼저 추가
    builder.add_node("classify_intent", classify_intent)

    # 핸들러 노드들을 동적으로 추가 (노드 이름은 '_handler'를 포함한 원래 이름으로)
    for importer, modname, ispkg in pkgutil.iter_modules(ai.chatbot.graph.handlers.__path__):
        module = importlib.import_module(f"ai.chatbot.graph.handlers.{modname}")
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

    # 엣지 추가
    # 라우팅 함수는 classify_intent의 반환값을 명확하게 처리하도록 수정합니다.
    def route_to_complex_or_single(state: ChatState) -> str:
        # 이 코드는 이미 state가 병합된 상태를 가정하고 있습니다.
        # 이전에 제안했던 코드는 옳았고, 문제는 state가 제대로 병합되지 않는 데 있었습니다.
        # 따라서, 여기서는 디버깅 출력만 다시 한번 확인해 보겠습니다.
        print(f"\n--- DEBUGGING STATE IN ROUTER ---")
        print(f"User Input: {state.get('user_input')}")
        print(f"Predicted Intent: {state.get('intent')}")
        print(f"Confidence: {state.get('confidence')}")
        print(f"Top-K Intents: {state.get('top_k_intents_and_probs')}")
        print(f"-----------------------------------\n")

        confidence = state.get("confidence")
        top_k_intents = state.get("top_k_intents_and_probs", [])

        # top_k_intents가 비어있을 경우를 대비해 예외 처리
        if not top_k_intents or len(top_k_intents) < 2:
            print("DEBUG: Top-K 의도 정보가 불충분하여 단일 의도로 처리합니다.")
            return route_by_intent(state)

        top1_intent, top1_prob = top_k_intents[0]
        top2_intent, top2_prob = top_k_intents[1]

        CONFIDENCE_THRESHOLD = 0.7
        PROB_DIFF_THRESHOLD = 0.15

        if top1_prob < CONFIDENCE_THRESHOLD or (top1_prob - top2_prob) < PROB_DIFF_THRESHOLD:
            print(f"DEBUG: 복합 의도 감지 - TOP1({top1_prob:.2f}), TOP2({top2_prob:.2f})")
            return "handle_complex_intent"

        print("DEBUG: 단일 의도 감지")
        return route_by_intent(state)

    builder.set_entry_point("classify_intent")
    builder.add_conditional_edges("classify_intent", route_to_complex_or_single)

    return builder.compile()