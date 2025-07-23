import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import BertModel, BertTokenizer
import pickle

# 🧠 모델 클래스 정의
class KoBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(KoBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(self.dropout(pooled_output))

# 🔧 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 토크나이저 및 라벨 인코더 로드
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ✅ 모델 로드
model = KoBERTClassifier(num_labels=len(label_encoder.classes_))
model.load_state_dict(torch.load("best_kobert_model.pt", map_location=device))
model.to(device)
model.eval()

# 🔮 예측 함수
def predict_intent(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    intent = label_encoder.inverse_transform([predicted.item()])[0]
    return intent, conf.item()

# 🔎 테스트
if __name__ == "__main__":
    while True:
        text = input("✉️ 질문을 입력하세요 (종료하려면 'exit'): ")
        if text.lower() == 'exit':
            break
        intent, confidence = predict_intent(text)
        print(f"🔖 예측 인텐트: {intent} | 🔍 Confidence: {confidence:.4f}")
