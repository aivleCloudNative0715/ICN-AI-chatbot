import os
from pathlib import Path

# 현재 파일(__file__)의 절대 경로를 기준으로 디렉토리 경로를 계산합니다.
# config.py -> shared -> ai
# 이 코드는 ai 디렉토리의 상위 디렉토리(ICN-AI-chatbot)를 루트로 잡습니다.
# config.py -> shared -> ai -> ICN-AI-chatbot
# 따라서 .parents[2]를 사용해야 합니다.
ROOT_DIR = Path(__file__).resolve().parents[2]

# 모델 파일이 위치한 실제 경로로 수정합니다.
# 의도 분류 모델이 intent_classifier 디렉토리 내에 있으므로 해당 경로를 추가합니다.
INTENT_SLOT_MODEL_DIR = os.path.join(ROOT_DIR, "ai", "intent_classifier", "best_models", "intent-kobert-v1")

# 개별 파일 경로
MODEL_PATH = os.path.join(INTENT_SLOT_MODEL_DIR, "best_model.pt")
INTENT2IDX_PATH = os.path.join(INTENT_SLOT_MODEL_DIR, "intent2idx.pkl")
SLOT2IDX_PATH = os.path.join(INTENT_SLOT_MODEL_DIR, "slot2idx.pkl")

# 저장 디렉토리 경로만 필요할 경우
SAVE_PATH = INTENT_SLOT_MODEL_DIR