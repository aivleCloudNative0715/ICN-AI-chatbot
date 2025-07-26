from ai.chatbot.graph.state import ChatState

def default_handler(state: ChatState) -> ChatState:
    return {**state, "response": "죄송해요, 이해하지 못했어요. 다시 말씀해 주세요."}
