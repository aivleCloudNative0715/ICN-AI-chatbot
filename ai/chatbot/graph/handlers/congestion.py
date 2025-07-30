from chatbot.graph.state import ChatState

def arrival_congestion_query_handler(state: ChatState) -> ChatState:
    return {**state, "response": "입국장 혼잡도 정보입니다."}

def departure_congestion_prediction_handler(state: ChatState) -> ChatState:
    return {**state, "response": "출국장 혼잡도 예측입니다."}

def arrival_congestion_prediction_handler(state: ChatState) -> ChatState:
    return {**state, "response": "입국장 혼잡도 예측입니다."}
