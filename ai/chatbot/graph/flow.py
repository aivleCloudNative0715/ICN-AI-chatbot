import importlib
import pkgutil
from functools import partial
from langgraph.graph import StateGraph, END

from chatbot.graph.state import ChatState
from chatbot.graph.router import route_by_intent
from chatbot.graph.nodes.classifiy_intent import classify_intent
from chatbot.graph.nodes.complex_handler import handle_complex_intent
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

    # 엣지 추가: 복합 의도 감지 로직이 포함된 라우팅 함수
    def route_to_complex_or_single(state: ChatState) -> str:
        print(f"\n--- DEBUGGING STATE IN ROUTER ---")
        print(f"User Input: {state.get('user_input')}")
        print(f"Predicted Intent: {state.get('intent')}")
        print(f"Confidence: {state.get('confidence')}")
        print(f"Top-K Intents: {state.get('top_k_intents_and_probs')}")
        print(f"Slots: {state.get('slots')}")
        print(f"-----------------------------------\n")

        slots = state.get("slots", [])

        # 슬롯 태그 그룹 정의
        slot_groups = {
            'parking': {'B-parking_type', 'B-parking_lot', 'B-parking_area', 'B-vehicle_type', 'B-payment_method', 'B-availability_status'},
            'facility_info': {'B-facility_name', 'B-location_keyword'},
            'flight_info': {'B-airline_flight', 'B-flight_status', 'B-airline_name', 'B-airport_name', 'B-airport_code', 'B-arrival_type', 'B-departure_type', 'B-destination', 'B-gate', 'B-terminal'},
            'baggage_info': {'B-baggage_type', 'B-luggage_term'},
            'policy': {'B-document', 'B-organization', 'B-person_type', 'B-item', 'B-transfer_topic'},
            'weather': {'B-weather_topic'},
            'time': {'B-date', 'B-time', 'B-season', 'B-day_of_week'},
            'general_topic': {'B-topic'}
        }

        found_groups = set()
        for _, tag in slots:
            # "B-" 태그만 검사하여 주요 키워드만 파악
            if tag.startswith('B-'):
                for group_name, tags in slot_groups.items():
                    if tag in tags:
                        found_groups.add(group_name)

        # general_topic 그룹에 해당하는 슬롯이 있는지 확인
        has_general_topic = 'general_topic' in found_groups

        # general_topic을 제외한 다른 슬롯 그룹의 개수를 세기
        specific_groups = found_groups - {'general_topic'}
        
        # 특정 슬롯 그룹이 2개 이상일 경우에만 복합 의도로 판단
        if len(specific_groups) > 1:
            print("DEBUG: 복수 슬롯 그룹 기반 복합 의도 감지")
            return "handle_complex_intent"
        
        # 그렇지 않은 경우(specific_groups가 0개 또는 1개)는 단일 의도로 판단
        else:
            print("DEBUG: 단일 슬롯 그룹 기반 단일 의도 감지")
            return route_by_intent(state)

    # def route_by_intent(state: ChatState):
    #     intent = state.get("intent")
    #     if intent:
    #         return f"{intent}_handler"
    #     return "fallback_handler"

    builder.set_entry_point("classify_intent")
    
    all_handler_names = list(handlers.keys()) + ["handle_complex_intent"]
    
    builder.add_conditional_edges(
        "classify_intent",
        route_to_complex_or_single,
        all_handler_names
    )

    return builder.compile()