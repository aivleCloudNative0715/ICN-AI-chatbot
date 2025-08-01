from chatbot.graph.state import ChatState

def flight_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "항공편 정보입니다."}

def regular_schedule_query_handler(state: ChatState) -> ChatState:
    return {**state, "response": "정기 운항 스케줄입니다."}

def airline_info_query_handler(state: ChatState) -> ChatState:
    return {**state, "response": "항공사 정보입니다."}
