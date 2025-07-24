from ai.chatbot.graph.state import ChatState

def transfer_info_handler(state: ChatState) -> ChatState:
    return {**state, "response": "환승 절차 안내입니다."}

def transfer_route_guide_handler(state: ChatState) -> ChatState:
    return {**state, "response": "환승 경로 안내입니다."}
