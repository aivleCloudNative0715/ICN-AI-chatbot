import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
import pickle


class KoBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(KoBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(self.dropout(pooled_output))


class KoBERTPredictor:
    def __init__(self, model_path, intent2idx_path, slot2idx_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)

        # 모델 가중치 로드
        state_dict = torch.load(model_path, map_location=self.device)
        num_labels = state_dict["classifier.weight"].shape[0]

        # 모델 구성 및 가중치 적용
        self.model = KoBERTClassifier(num_labels)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 의도 라벨 인코더 로드 (intent2idx_path 사용)
        with open(intent2idx_path, "rb") as f:
            self.intent_label_encoder = pickle.load(f)

        # 슬롯 라벨 인코더 로드 (slot2idx_path 사용)
        # 만약 슬롯 예측 기능이 필요하다면 추가
        # with open(slot2idx_path, "rb") as f:
        #     self.slot_label_encoder = pickle.load(f)

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs)
            predicted_idx = logits.argmax(dim=-1).item()

        return self.label_encoder.inverse_transform([predicted_idx])[0]
