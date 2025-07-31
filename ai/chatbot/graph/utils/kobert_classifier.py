import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
import pickle


class KoBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(KoBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.dropout = nn.Dropout(0.3)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.intent_classifier(self.dropout(pooled_output))


class KoBERTPredictor:
    def __init__(self, model_path, label_encoder_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)

        state_dict = torch.load(model_path, map_location=self.device)
        filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith("bert") or k.startswith("intent_classifier")}
        num_labels = filtered_state_dict["intent_classifier.weight"].shape[0]

        self.model = KoBERTClassifier(num_labels)
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # label_encoder가 dict로 로드됨을 가정
        with open(label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)

        # dict가 {label: idx} 형태면 역매핑 dict 생성
        self.idx2label = {v: k for k, v in self.label_encoder.items()}

    def predict(self, text: str) -> dict:
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs)
            probabilities = torch.softmax(logits, dim=1)
            confidence_score, predicted_idx = torch.max(probabilities, dim=1)

        intent = self.idx2label[predicted_idx.item()]
        confidence = confidence_score.item()

        return {
            "intent": intent,
            "confidence": confidence
        }