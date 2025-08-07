from chatbot.graph.state import ChatState

def default_greeting_handler(state: ChatState) -> ChatState:
    return {**state, "response": "안녕하세요! 무엇을 도와드릴까요?"}
