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
        confidence = state.get("confidence", 0.0)

        print(f"DEBUG: 의도별 확신도 점수: {top_k_intents}")
        
        # 🚀 Smart Routing: 높은 신뢰도 + 충분한 slot이 있으면 검증 스킵
        def has_sufficient_slots(intent, slots):
            """의도별로 충분한 slot 정보가 있는지 확인"""
            if intent == "flight_info":
                # flight_id, airport_name, airline_name, terminal 중 하나라도 있으면 충분
                flight_slots = [word for word, slot in slots if slot in ['B-flight_id', 'I-flight_id', 'B-airport_name', 'I-airport_name', 'B-airline_name', 'I-airline_name', 'B-terminal', 'I-terminal']]
                return len(flight_slots) > 0
            elif intent == "airline_info_query":
                airline_slots = [word for word, slot in slots if slot in ['B-airline_name', 'I-airline_name']]
                return len(airline_slots) > 0
            elif intent == "airport_info":
                airport_slots = [word for word, slot in slots if slot in ['B-airport_name', 'I-airport_name']]
                return len(airport_slots) > 0
            elif intent == "facility_guide":
                # facility_name, terminal, area 중 하나라도 있으면 충분
                facility_slots = [word for word, slot in slots if slot in ['B-facility_name', 'I-facility_name', 'B-terminal', 'I-terminal', 'B-area', 'I-area']]
                return len(facility_slots) > 0
            elif intent == "airport_weather_current":
                # weather_topic이 있으면 충분
                weather_slots = [word for word, slot in slots if slot in ['B-weather_topic', 'I-weather_topic']]
                return len(weather_slots) > 0
            elif intent == "baggage_rule_query":
                # baggage_type, rule_type, item 중 하나라도 있으면 충분
                baggage_slots = [word for word, slot in slots if slot in ['B-baggage_type', 'I-baggage_type', 'B-luggage_term', 'I-luggage_term', 'B-rule_type', 'I-rule_type', 'B-item', 'I-item']]
                return len(baggage_slots) > 0
            elif intent == "parking_fee_info":
                # parking 관련 slot이 있으면 충분
                parking_slots = [word for word, slot in slots if slot in ['B-fee_topic', 'I-fee_topic', 'B-vehicle_type', 'I-vehicle_type', 'B-parking_area', 'I-parking_area', 'B-time_period', 'I-time_period']]
                return len(parking_slots) > 0
            elif intent == "parking_location_recommendation":
                # parking 위치 관련 slot이 있으면 충분
                parking_location_slots = [word for word, slot in slots if slot in ['B-parking_lot', 'I-parking_lot', 'B-parking_area', 'I-parking_area', 'B-terminal', 'I-terminal']]
                return len(parking_location_slots) > 0
            elif intent == "parking_walk_time_info":
                # parking 도보시간 관련 slot이 있으면 충분
                walk_time_slots = [word for word, slot in slots if slot in ['B-parking_lot', 'I-parking_lot', 'B-parking_area', 'I-parking_area', 'B-terminal', 'I-terminal', 'B-location', 'I-location']]
                return len(walk_time_slots) > 0
            elif intent == "transfer_info":
                # transfer 관련 slot이 있으면 충분
                transfer_slots = [word for word, slot in slots if slot in ['B-transfer_topic', 'I-transfer_topic', 'B-transport_type', 'I-transport_type', 'B-location', 'I-location', 'B-terminal', 'I-terminal']]
                return len(transfer_slots) > 0
            return False
        
        # 1. 복합 의도 감지 우선 처리 (가장 먼저 체크)
        if state.get("is_multi_intent", False) or state.get("intent") == "complex_intent":
            detected_intents = [intent for intent, _ in state.get("detected_intents", [])]
            print(f"복합 의도 감지: {detected_intents} -> handle_complex_intent로 라우팅")
            return "handle_complex_intent"

        # 2. 이전 대화 감지 로직
        if len(state.get("messages", [])) > 1:
            intent = state.get("intent", "")
            
            # 🚀 스마트 라우팅: 신뢰도 높고 slot 충분하면 바로 핸들러로
            if confidence > 0.85 and has_sufficient_slots(intent, slots):
                handler_name = f"{intent}_handler"
                print(f"DEBUG: ⚡ 스마트 라우팅 - 높은 신뢰도({confidence:.3f}) + 충분한 slot -> {handler_name} 직접 호출")
                return handler_name
            else:
                print(f"DEBUG: 이전 대화 감지 -> llm_verify_intent로 라우팅 (신뢰도: {confidence:.3f})")
                return "llm_verify_intent"

        # 4. 단일 의도인 경우 직접 핸들러로 라우팅
        intent = state.get("intent")
        if intent and intent != "complex_intent":
            handler_name = f"{intent}_handler"
            print(f"DEBUG: 단일 의도 감지 -> {handler_name}로 라우팅")
            return handler_name

        # 5. 신뢰도가 낮거나 모호한 경우 LLM 재확인
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