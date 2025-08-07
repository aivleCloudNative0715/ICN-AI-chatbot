from chatbot.graph.state import ChatState

def airport_weather_current_handler(state: ChatState) -> ChatState:
    return {**state, "response": "현재 인천국제공항의 날씨 정보입니다."}
