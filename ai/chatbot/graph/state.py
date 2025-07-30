# graph/state.py
from typing import TypedDict

class ChatState(TypedDict, total=False):
    user_input: str
    intent: str
    response: str
