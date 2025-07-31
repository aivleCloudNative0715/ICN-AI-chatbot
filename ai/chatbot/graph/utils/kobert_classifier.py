import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer
import pickle

from ai.shared.model import KoBERTIntentSlotModel


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

        # 인덱스 로드
        with open(intent2idx_path, "rb") as f:
            self.intent2idx = pickle.load(f)
        with open(slot2idx_path, "rb") as f:
            self.slot2idx = pickle.load(f)

        self.idx2intent = {v: k for k, v in self.intent2idx.items()}
        self.idx2slot = {v: k for k, v in self.slot2idx.items()}

        # 모델 초기화 및 가중치 로드
        self.model = KoBERTIntentSlotModel(
            num_intents=len(self.intent2idx),
            num_slots=len(self.slot2idx)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def merge_tokens_and_slots(self, tokens, slot_ids):
        merged = []
        word = ''
        current_slot = ''

        for token, slot_id in zip(tokens, slot_ids):
            slot = self.idx2slot.get(slot_id, 'O')
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            if token.startswith("▁"):  # 새 단어
                if word:
                    merged.append((word, current_slot))
                word = token[1:]
                current_slot = slot
            else:
                word += token.replace("▁", "")

        if word:
            merged.append((word, current_slot))

        return merged

    def predict(self, text: str, k: int = 1):
        # 토크나이징
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=64
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # 예측
        with torch.no_grad():
            intent_logits, slot_logits = self.model(input_ids, attention_mask)
            intent_probs = softmax(intent_logits, dim=1)
            topk_probs, topk_indices = torch.topk(intent_probs, k, dim=1)

            # 인텐트
            intents = [(self.idx2intent[idx.item()], prob.item()) for idx, prob in zip(topk_indices[0], topk_probs[0])]

            # 슬롯
            slot_ids = torch.argmax(slot_logits, dim=2)[0].tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            slots = self.merge_tokens_and_slots(tokens, slot_ids)

        return intents, slots
