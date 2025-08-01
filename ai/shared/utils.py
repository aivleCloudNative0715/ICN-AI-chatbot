# ai/intent_classifier/utils.py

import torch
from transformers import AutoTokenizer

# 공통 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 공통 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)
