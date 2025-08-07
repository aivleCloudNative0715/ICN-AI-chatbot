from typing import TypedDict, List, Tuple

class ChatState(TypedDict, total=False):
    user_input: str
    intent: str
    slots: list
    response: str
    confidence: float  # confidence 키 추가
    top_k_intents_and_probs: List[Tuple[str, float]]  # top_k_intents_and_probs 키 추가
