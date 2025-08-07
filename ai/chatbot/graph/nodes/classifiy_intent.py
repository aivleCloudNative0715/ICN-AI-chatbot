from ai.shared.config import MODEL_PATH, INTENT2IDX_PATH, SLOT2IDX_PATH
from ai.intent_classifier.inference import predict_top_k_intents_and_slots
from ai.chatbot.graph.utils.kobert_classifier import KoBERTPredictor

# KoBERT 분류기 인스턴스 생성
my_kobert_classifier = KoBERTPredictor(
    model_path=MODEL_PATH,
    intent2idx_path=INTENT2IDX_PATH,
    slot2idx_path=SLOT2IDX_PATH
)


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

