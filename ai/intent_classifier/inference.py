import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import BertModel, AutoTokenizer
import pickle


# 📌 KoBERTIntentSlotModel 로드
class KoBERTIntentSlotModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits


# 🔧 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 토크나이저 및 인텐트 인덱스 로드
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)
with open("best_models/intent-slot-kobert/intent2idx.pkl", "rb") as f:
    intent2idx = pickle.load(f)
idx2intent = {v: k for k, v in intent2idx.items()}

# ✅ 모델 로드
model = KoBERTIntentSlotModel(num_intents=len(intent2idx), num_slots=19)
model.load_state_dict(torch.load("best_models/intent-slot-kobert/best_model.pt", map_location=device))
model.to(device)
model.eval()


# 🔮 예측 함수
def predict_top_k_intents_and_slots(text, k=3):
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
        intent_logits, slot_logits = model(input_ids, attention_mask)
        intent_probs = softmax(intent_logits, dim=1)

        # 🎯 인텐트 Top-K
        topk_probs, topk_indices = torch.topk(intent_probs, k, dim=1)
        intents = [(idx2intent[topk_indices[0][i].item()], topk_probs[0][i].item()) for i in range(k)]

        # 🎯 슬롯 예측
        slot_pred = torch.argmax(slot_logits, dim=2)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

        # 슬롯 인덱스 로드
        with open("best_models/intent-slot-kobert/slot2idx.pkl", "rb") as f:
            slot2idx = pickle.load(f)
        idx2slot = {v: k for k, v in slot2idx.items()}

        # 🏷️ 실제 텍스트 단어 단위에 대응하는 토큰 + 슬롯만 추출 (special token 제외)
        words_with_slots = []
        for token, slot_id in zip(tokens, slot_pred):
            if token not in ["[CLS]", "[SEP]", "[PAD]"]:
                words_with_slots.append((token, idx2slot[slot_id]))

    return intents, words_with_slots


if __name__ == "__main__":
    while True:
        text = input("✉️ 질문을 입력하세요 (종료하려면 'exit'): ")
        if text.lower() == 'exit':
            break
        intents, word_slots = predict_top_k_intents_and_slots(text, k=3)

        print("🔍 예측된 인텐트 TOP 3:")
        for intent, conf in intents:
            print(f" - {intent}: {conf:.4f}")

        print("🎯 예측된 슬롯 정보:")
        for word, slot in word_slots:
            print(f" - {word}: {slot}")