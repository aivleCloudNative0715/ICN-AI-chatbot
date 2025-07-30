from chatbot.graph.state import ChatState
from chatbot.graph.utils.kobert_classifier import KoBERTClassifier, KoBERTPredictor

from pathlib import Path

# 모델 및 라벨 인코더 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent  # ai/
MODEL_DIR = BASE_DIR / "intent_classifier" / "best_models" / "intent-kobert-v1"
MODEL_PATH = MODEL_DIR / "best_kobert_model.pt"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# KoBERT 분류기 인스턴스 생성
my_kobert_classifier = KoBERTPredictor(
    model_path=MODEL_PATH,
    label_encoder_path=LABEL_ENCODER_PATH
)

def classify_intent(state: ChatState) -> ChatState:
    user_input = state["user_input"]
    predicted_intent = my_kobert_classifier.predict(user_input)
    return {**state, "intent": predicted_intent}
