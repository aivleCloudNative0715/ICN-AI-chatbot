# graph/state.py
from typing import TypedDict
from langchain_core.messages import BaseMessage

class ChatState(TypedDict, total=False):
    user_input: str
    intent: str
    slots: list
    response: str
    pre_message_id: str
    messages: list[BaseMessage]
