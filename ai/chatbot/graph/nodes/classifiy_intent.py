from chatbot.graph.state import ChatState
from shared.config import MODEL_PATH, INTENT2IDX_PATH, SLOT2IDX_PATH
from intent_classifier.inference import predict_top_k_intents_and_slots


def classify_intent(state: ChatState) -> ChatState:
    text = state["user_input"]
    top_k_intents_and_probs, slots = predict_top_k_intents_and_slots(text, k=3)

    if top_k_intents_and_probs:
        top_intent, confidence = top_k_intents_and_probs[0]
    else:
        top_intent, confidence = "default", 0.0

    state["intent"] = top_intent
    state["confidence"] = confidence
    state["top_k_intents_and_probs"] = top_k_intents_and_probs
    state["slots"] = slots
    return state

