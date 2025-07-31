import torch
import pickle

from ai.shared.config import INTENT2IDX_PATH, SLOT2IDX_PATH, MODEL_PATH
from ai.shared.model import KoBERTIntentSlotModel
from ai.shared.utils import device

# 인텐트/슬롯 라벨 딕셔너리 로드
with open(INTENT2IDX_PATH, "rb") as f:
    intent2idx = pickle.load(f)

with open(SLOT2IDX_PATH, "rb") as f:
    slot2idx = pickle.load(f)

idx2intent = {v: k for k, v in intent2idx.items()}
idx2slot = {v: k for k, v in slot2idx.items()}

# ✅ 모델 로드
model = KoBERTIntentSlotModel(num_intents=len(intent2idx), num_slots=len(slot2idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()