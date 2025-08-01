from shared.config import MODEL_PATH, INTENT2IDX_PATH, SLOT2IDX_PATH
from intent_classifier.inference import predict_top_k_intents_and_slots
from chatbot.graph.utils.kobert_classifier import KoBERTPredictor

# KoBERT 분류기 인스턴스 생성
my_kobert_classifier = KoBERTPredictor(
    model_path=MODEL_PATH,
    intent2idx_path=INTENT2IDX_PATH,
    slot2idx_path=SLOT2IDX_PATH
)


def classify_intent(state):
    text = state["user_input"]

    intents, slots = predict_top_k_intents_and_slots(text, k=1)
    top_intent, prob = intents[0]

    state["intent"] = top_intent
    state["slots"] = slots

    return state

