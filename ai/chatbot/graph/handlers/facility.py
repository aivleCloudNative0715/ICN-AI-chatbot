from chatbot.graph.state import ChatState

def facility_guide_handler(state: ChatState) -> ChatState:
    return {**state, "response": "공항 시설 안내입니다."}
