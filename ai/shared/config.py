import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# best_models는 config.py의 한 디렉토리 위에 있음
ROOT_DIR = os.path.dirname(BASE_DIR)

# 모델 저장/불러오기 공통 디렉토리
INTENT_SLOT_MODEL_DIR = os.path.join(ROOT_DIR, "best_models", "intent-kobert-v3")

# 개별 파일 경로
MODEL_PATH = os.path.join(INTENT_SLOT_MODEL_DIR, "best_model.pt")
INTENT2IDX_PATH = os.path.join(INTENT_SLOT_MODEL_DIR, "intent2idx.pkl")
SLOT2IDX_PATH = os.path.join(INTENT_SLOT_MODEL_DIR, "slot2idx.pkl")

# 저장 디렉토리 경로만 필요할 경우
SAVE_PATH = INTENT_SLOT_MODEL_DIR

# 의도 분류 관련 설정
INTENT_CLASSIFICATION = {
    # BCE 기반 예측에서 사용할 기본 임계값
    "DEFAULT_THRESHOLD": 0.7,
}