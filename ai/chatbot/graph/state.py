from typing import TypedDict, List, Tuple
from langchain_core.messages import BaseMessage

class ChatState(TypedDict, total=False):
    user_input: str
    intent: str
    slots: list
    response: str
    confidence: float
    top_k_intents_and_probs: List[Tuple[str, float]]
    pre_message_id: str
    messages: list[BaseMessage]
    # 📌 수정된 부분: rephrased_query 키 추가
    rephrased_query: str 