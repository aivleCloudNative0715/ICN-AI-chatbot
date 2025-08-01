from chatbot.graph.state import ChatState

def parking_fee_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "주차 요금 안내입니다."}

def parking_congestion_prediction_handler(state: ChatState) -> ChatState:
    return {**state, "response": "주차 혼잡도 예측입니다."}

def parking_location_recommendation_handler(state: ChatState) -> ChatState:
    return {**state, "response": "추천 주차장 위치입니다."}

def parking_availability_query_handler(state: ChatState) -> ChatState:
    return {**state, "response": "주차장 이용 가능 여부입니다."}

def parking_walk_time_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "주차장까지 도보 시간입니다."}
