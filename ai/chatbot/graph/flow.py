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

    # 핸들러 노드들을 동적으로 추가하고 엣지를 연결
    for importer, modname, ispkg in pkgutil.iter_modules(ai.chatbot.graph.handlers.__path__):
        if modname.startswith("__"):
            continue
        
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

    # 엣지 추가: 복합 의도 감지 로직이 포함된 라우팅 함수
    def route_to_complex_or_single(state: ChatState) -> str:
        print(f"\n--- DEBUGGING STATE IN ROUTER ---")
        print(f"User Input: {state.get('user_input')}")
        print(f"Predicted Intent: {state.get('intent')}")
        print(f"Confidence: {state.get('confidence')}")
        print(f"Top-K Intents: {state.get('top_k_intents_and_probs')}")
        print(f"Slots: {state.get('slots')}")
        print(f"-----------------------------------\n")

        confidence = state.get("confidence")
        top_k_intents = state.get("top_k_intents_and_probs", [])
        slots = state.get("slots", [])

        # 슬롯 태그 그룹 정의
        slot_groups = {
            'parking': {'B-parking_type', 'B-parking_lot', 'B-parking_area', 'B-vehicle_type', 'B-payment_method', 'B-availability_status'},
            'facility_info': {'B-facility_name', 'B-location_keyword', 'B-gate', 'B-terminal'},
            'flight_info': {'B-airline_flight', 'B-flight_status', 'B-airline_name', 'B-airport_name', 'B-airport_code', 'B-arrival_type', 'B-departure_type', 'B-destination'},
            'baggage_info': {'B-baggage_type', 'B-luggage_term'},
            'policy': {'B-document', 'B-organization', 'B-person_type', 'B-item', 'B-transfer_topic'},
            'weather': {'B-weather_topic'},
            'time': {'B-date', 'B-time', 'B-season', 'B-day_of_week'},
            'general_topic': {'B-topic'}
        }

        found_groups = set()
        for _, tag in slots:
            for group_name, tags in slot_groups.items():
                if tag in tags:
                    found_groups.add(group_name)

        # 복수 개의 슬롯 그룹이 감지되면 복합 의도로 간주
        if len(found_groups) > 1:
            print("DEBUG: 슬롯 그룹 기반 복합 의도 감지")
            return "handle_complex_intent"
        
        # 기존 확신도 기반 로직 (슬롯 감지에 실패했을 경우만 실행)
        if not top_k_intents or len(top_k_intents) < 2:
            print("DEBUG: Top-K 의도 정보가 불충분하여 단일 의도로 처리합니다.")
            return route_by_intent(state)

        top1_intent, top1_prob = top_k_intents[0]
        top2_intent, top2_prob = top_k_intents[1]

        CONFIDENCE_THRESHOLD = 0.7
        PROB_DIFF_THRESHOLD = 0.15

        if top1_prob < CONFIDENCE_THRESHOLD or (top1_prob - top2_prob) < PROB_DIFF_THRESHOLD:
            print(f"DEBUG: 확신도 기반 복합 의도 감지 - TOP1({top1_prob:.2f}), TOP2({top2_prob:.2f})")
            return "handle_complex_intent"

        print("DEBUG: 단일 의도 감지")
        return route_by_intent(state)

    builder.set_entry_point("classify_intent")
    
    all_handler_names = list(handlers.keys()) + ["handle_complex_intent"]
    
    builder.add_conditional_edges(
        "classify_intent",
        route_to_complex_or_single,
        all_handler_names
    )

    return builder.compile()