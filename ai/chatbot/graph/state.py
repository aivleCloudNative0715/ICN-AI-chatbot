# graph/state.py
from typing import TypedDict
from langchain_core.messages import BaseMessage

class ChatState(TypedDict, total=False):
    user_input: str
    intent: str
    response: str
    messages: list[BaseMessage]
