from ai.shared.config import MODEL_PATH, INTENT2IDX_PATH, SLOT2IDX_PATH
from ai.intent_classifier.inference import predict_top_k_intents_and_slots
from ai.chatbot.graph.utils.kobert_classifier import KoBERTPredictor

from typing import List, Tuple
from ai.chatbot.graph.state import ChatState
import sys

# KoBERT 분류기 인스턴스 생성
# my_kobert_classifier = KoBERTPredictor(
#     model_path=MODEL_PATH,
#     intent2idx_path=INTENT2IDX_PATH,
#     slot2idx_path=SLOT2IDX_PATH
# 

def classify_intent(state: ChatState) -> ChatState:
    print("✅ classify_intent 함수 진입")
    text = state["user_input"]
    top_k_intents_and_probs, slots = predict_top_k_intents_and_slots(text, k=3)

    if top_k_intents_and_probs:
        top_intent, confidence = top_k_intents_and_probs[0]
    else:
        top_intent, confidence = "default", 0.0

    print(f"\n📌 classify_intent 실행 결과:")
    print(f"Intent: {top_intent}, Confidence: {confidence}")
    print(f"Top-K Intents: {top_k_intents_and_probs}", file=sys.stderr)
    print(f"Slots: {slots}", file=sys.stderr)

    # 모든 상태 값을 키-값으로 업데이트
    state["intent"] = top_intent
    state["confidence"] = confidence
    state["top_k_intents_and_probs"] = top_k_intents_and_probs
    state["slots"] = slots
    return state