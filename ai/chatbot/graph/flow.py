import importlib
import pkgutil
from langgraph.graph import StateGraph, END

from chatbot.graph.state import ChatState
from chatbot.graph.router import route_by_intent
from chatbot.graph.nodes.classifiy_intent import classify_intent
import chatbot.graph.handlers


def build_chat_graph():
    builder = StateGraph(ChatState)

    # 핸들러 노드 동적 등록
    for importer, modname, ispkg in pkgutil.iter_modules(chatbot.graph.handlers.__path__):
        module = importlib.import_module(f"chatbot.graph.handlers.{modname}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if callable(attribute) and attribute_name.endswith("_handler"):
                node_name = attribute_name
                builder.add_node(node_name, attribute)
                builder.add_edge(node_name, END)

    # 분류기 노드 등록
    builder.add_node("classify_intent", classify_intent)

    # 플로우 구성
    builder.set_entry_point("classify_intent")
    builder.add_conditional_edges("classify_intent", route_by_intent)

    return builder.compile()
