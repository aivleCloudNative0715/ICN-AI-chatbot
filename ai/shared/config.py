import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# best_models는 config.py의 한 디렉토리 위에 있음
ROOT_DIR = os.path.dirname(BASE_DIR)

# 모델 저장/불러오기 공통 디렉토리
INTENT_SLOT_MODEL_DIR = os.path.join(ROOT_DIR, "best_models", "intent-kobert-v2")

# 개별 파일 경로
MODEL_PATH = os.path.join(INTENT_SLOT_MODEL_DIR, "best_model.pt")
INTENT2IDX_PATH = os.path.join(INTENT_SLOT_MODEL_DIR, "intent2idx.pkl")
SLOT2IDX_PATH = os.path.join(INTENT_SLOT_MODEL_DIR, "slot2idx.pkl")

# 저장 디렉토리 경로만 필요할 경우
SAVE_PATH = INTENT_SLOT_MODEL_DIR