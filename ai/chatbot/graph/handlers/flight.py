from ai.chatbot.graph.state import ChatState
from ai.chatbot.rag import get_airline_info

def flight_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "항공편 정보입니다."}

def regular_schedule_query_handler(state: ChatState) -> ChatState:
    return {**state, "response": "정기 운항 스케줄입니다."}

def airline_info_query_handler(state: ChatState) -> ChatState:
    user_input = state.get("user_input", "")
    answer = get_airline_info(user_input)
    return {**state, "response": answer}