# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% colab={"base_uri": "https://localhost:8080/", "height": 435} executionInfo={"elapsed": 17757, "status": "ok", "timestamp": 1753706324109, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="18563153" outputId="732c71e5-c59f-4ecd-b7ce-94aacb2d9e0c"
# !pip install -q transformers
# !pip install -q tqdm scikit-learn pandas
# !pip install numpy==1.26.4


# %% executionInfo={"elapsed": 5, "status": "ok", "timestamp": 1753779574905, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="a9577f8d"

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, AutoTokenizer
from torch.optim import AdamW
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from collections import Counter

from config import MODEL_PATH, INTENT2IDX_PATH, SLOT2IDX_PATH, SAVE_PATH

tqdm.pandas()


# %% executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1753776679450, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="Wm6xqcWKRjUD"
df = pd.read_csv("intent_slot_dataset_cleaned.csv")
df_extra = pd.read_csv("keyword_boost_slot.csv")
df_combined = pd.concat([df_extra, df]).reset_index(drop=True)


# %%
def convert_to_bio_word_level(sentence, slot_json):
    tokens = sentence.split()  # 단어 단위로 토큰화
    labels = ['O'] * len(tokens)

    for slot_name, slot_value in slot_json.items():
        slot_values = slot_value if isinstance(slot_value, list) else [slot_value]
        for val in slot_values:
            if not isinstance(val, str): continue
            val_tokens = val.split()

            for i in range(len(tokens) - len(val_tokens) + 1):
                if tokens[i:i+len(val_tokens)] == val_tokens:
                    labels[i] = f'B-{slot_name}'
                    for j in range(1, len(val_tokens)):
                        labels[i + j] = f'I-{slot_name}'
                    break  # 중복 방지
    return tokens, labels


# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 443, "status": "ok", "timestamp": 1753776680148, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="Q__xPnvpRh5n" outputId="3b5c8b07-ce80-4d40-f1a3-c942616cf037"
import ast

bio_data = []

for idx, (_, row) in enumerate(df_combined.iterrows()):
    sentence = str(row['question']).strip()

    if not sentence or sentence.lower() == 'nan':
        continue

    raw_slots = row['slots']

    try:
        if isinstance(raw_slots, str):
            if raw_slots.strip() in ['', '[]', '{}', 'nan']:
                slot_dict = {}
            else:
                # 🔁 문자열 기반 JSON → Python 호환 치환
                cleaned_slot_str = (
                    raw_slots
                    .replace('null', 'None')
                    .replace('true', 'True')
                    .replace('false', 'False')
                )
                slot_dict = ast.literal_eval(cleaned_slot_str)
        elif isinstance(raw_slots, dict):
            slot_dict = raw_slots
        else:
            slot_dict = {}

        tokens, labels = convert_to_bio_word_level(sentence, slot_dict)
        bio_data.append({
            "intent": row["intent"],
            "tokens": tokens,
            "labels": labels
        })

    except Exception as e:
        print("⚠️ Error at row:", idx)
        print("Sentence:", sentence)
        print("Raw slots:", raw_slots)
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))
        print("=" * 60)
        continue

df_bio = pd.DataFrame(bio_data)
df_bio.head()


# %% executionInfo={"elapsed": 3, "status": "ok", "timestamp": 1753776680182, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="d9505fea"
intent_labels = sorted(df_bio['intent'].unique())
intent2idx = {label: i for i, label in enumerate(intent_labels)}
idx2intent = {i: label for label, i in intent2idx.items()}

# 1. string → list 변환
df_bio['labels'] = df_bio['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 2. flatten + B/I 확장
original_labels = set()
for label_list in df_bio['labels']:
    for label in label_list:
        original_labels.add(label)
        # B-xxx가 있으면 I-xxx도 추가
        if label.startswith('B-'):
            original_labels.add('I-' + label[2:])

# 3. 정렬 및 매핑 생성
slot_labels = sorted(original_labels)
slot2idx = {label: i for i, label in enumerate(slot_labels)}
idx2slot = {i: label for label, i in slot2idx.items()}


# %% executionInfo={"elapsed": 316, "status": "ok", "timestamp": 1753776680691, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="48dd4c68"
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", use_fast=False)

MAX_LEN = 64

class IntentSlotDataset(Dataset):
    def __init__(self, encodings, slot_labels, intents):
        self.encodings = encodings
        self.slot_labels = slot_labels
        self.intents = intents

    def _mask_pad_tokens(self, slot_labels):
        # PAD 토큰은 보통 O 또는 지정된 pad_index로 처리되어 있음
        # 아래에서는 O의 index를 pad처럼 간주한다면 slot2idx['O']로 처리 가능
        pad_token_label_id = slot2idx.get('PAD', slot2idx['O'])  # 혹시 PAD가 정의되어 있으면 우선
        masked = []
        for seq in slot_labels:
            masked_seq = [label if label != pad_token_label_id else -100 for label in seq]
            masked.append(masked_seq)
        return masked

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.slot_labels[idx])
        item['intent'] = torch.tensor(self.intents[idx])
        return item



# %% colab={"base_uri": "https://localhost:8080/"} id="AIRu6JZaOyJx" executionInfo={"status": "ok", "timestamp": 1753776681506, "user_tz": -540, "elapsed": 12, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}} outputId="35903579-4329-415d-9f66-163b1c178957"
print(df["intent"].value_counts())

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 112, "status": "ok", "timestamp": 1753776760624, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="LGqicPqrV4J1" outputId="461ac4eb-c267-457f-dc06-53a5ee6fac05"
# 설정
TARGET_INTENT_SAMPLES = 300
TARGET_SLOT_SAMPLES = 300  # 희귀 slot label의 최소 등장 횟수

# 먼저 intent 기준 균형 맞추기
balanced_intent_dfs = []

for intent in df_bio['intent'].unique():
    intent_df = df_bio[df_bio['intent'] == intent]

    if len(intent_df) >= TARGET_INTENT_SAMPLES:
        sampled_df = resample(intent_df, replace=False, n_samples=TARGET_INTENT_SAMPLES, random_state=42)
    else:
        sampled_df = resample(intent_df, replace=True, n_samples=TARGET_INTENT_SAMPLES, random_state=42)

    balanced_intent_dfs.append(sampled_df)

df_balanced_intent = pd.concat(balanced_intent_dfs).reset_index(drop=True)

# 슬롯 라벨 등장 횟수 계산
slot_counter = Counter(label for labels in df_balanced_intent['labels'] for label in labels)

# 부족한 슬롯 라벨 찾기
rare_slots = [label for label, count in slot_counter.items() if count < TARGET_SLOT_SAMPLES]

# 해당 rare slot을 포함하는 샘플만 추출해서 오버샘플링
slot_augmented_dfs = []
for rare_label in rare_slots:
    slot_df = df_bio[df_bio['labels'].apply(lambda x: rare_label in x)]

    if len(slot_df) == 0:
        continue  # 해당 slot이 아예 없는 경우는 건너뜀

    needed = TARGET_SLOT_SAMPLES - slot_counter[rare_label]
    duplicated_df = resample(slot_df, replace=True, n_samples=needed, random_state=42)
    slot_augmented_dfs.append(duplicated_df)

# intent 기반 + slot 보완 데이터 합치기
df_balanced = pd.concat([df_balanced_intent] + slot_augmented_dfs).reset_index(drop=True)

# 결과 확인
print("✅ Intent 분포:")
print(df_balanced['intent'].value_counts())

print("\n✅ Slot label 분포 (상위 20개):")
from collections import Counter
slot_counter_final = Counter(label for labels in df_balanced['labels'] for label in labels)
print(dict(sorted(slot_counter_final.items(), key=lambda x: x[1], reverse=True)[:20]))


# %% executionInfo={"elapsed": 25, "status": "ok", "timestamp": 1753776762340, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="2OzUKUPXuLGS"
def align_labels_with_tokenizer(tokens, labels):
    bert_tokens = []
    aligned_labels = []

    for token, label in zip(tokens, labels):
        sub_tokens = tokenizer.tokenize(token)

        if not sub_tokens:
            sub_tokens = [tokenizer.unk_token]

        bert_tokens.extend(sub_tokens)

        # 첫 sub-token에는 원래 라벨 사용
        aligned_labels.append(label)

        # 나머지 sub-token에 대해서는 label 확장
        for _ in range(1, len(sub_tokens)):
            if label == "O":
                aligned_labels.append("O")
            elif label.startswith("B-"):
                aligned_labels.append("I-" + label[2:])
            elif label.startswith("I-"):
                aligned_labels.append(label)
            else:
                # 예외 처리: label이 'B-', 'I-', 'O' 중 아무 것도 아닐 경우
                aligned_labels.append("O")

    assert len(bert_tokens) == len(aligned_labels), f"Mismatch: {bert_tokens} vs {aligned_labels}"
    return bert_tokens, aligned_labels



# %% executionInfo={"elapsed": 2638, "status": "ok", "timestamp": 1753776765191, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="8ec4dcf5"
def encode_data(df):
    input_ids, attention_masks, slot_label_ids, intent_ids = [], [], [], []

    for _, row in df.iterrows():
        tokens = row["tokens"]
        labels = row["labels"]

        bert_tokens, aligned_labels = align_labels_with_tokenizer(tokens, labels)
        label_ids = [slot2idx[label] for label in aligned_labels]

        if len(bert_tokens) > MAX_LEN - 2:
            bert_tokens = bert_tokens[:MAX_LEN-2]
            label_ids = label_ids[:MAX_LEN-2]

        tokens_input = ['[CLS]'] + bert_tokens + ['[SEP]']
        label_ids = [slot2idx['O']] + label_ids + [slot2idx['O']]

        input_id = tokenizer.convert_tokens_to_ids(tokens_input)
        attention_mask = [1] * len(input_id)

        pad_len = MAX_LEN - len(input_id)
        input_id += [0] * pad_len
        attention_mask += [0] * pad_len
        label_ids += [slot2idx['O']] * pad_len

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        slot_label_ids.append(label_ids)
        intent_ids.append(intent2idx[row["intent"]])

    encodings = {
        "input_ids": input_ids,
        "attention_mask": attention_masks
    }
    return encodings, slot_label_ids, intent_ids

# ✅ Train / Val split
train_df, val_df = train_test_split(df_balanced, test_size=0.1, stratify=df_balanced['intent'])

# ✅ Encode
train_encodings, train_slot_labels, train_intents = encode_data(train_df)
val_encodings, val_slot_labels, val_intents = encode_data(val_df)

# ✅ Dataset / DataLoader
train_dataset = IntentSlotDataset(train_encodings, train_slot_labels, train_intents)
val_dataset = IntentSlotDataset(val_encodings, val_slot_labels, val_intents)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


# %% id="KohMCMpxh0_k" executionInfo={"status": "ok", "timestamp": 1753777134018, "user_tz": -540, "elapsed": 768, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}}
from collections import Counter
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전체 slot label 모으기 (train_dataset 기준)
all_slot_labels = [
    label for data in train_dataset
    for label in data['labels'].tolist()
    if label != -100
]

slot_counts = Counter(all_slot_labels)
total = sum(slot_counts.values())

# slot2idx 기준으로 weight 생성
slot_weights = []
for idx in range(len(slot2idx)):
    count = slot_counts.get(idx, 0)
    if count == 0:
        slot_weights.append(0.0)  # 안 쓰인 경우 0
    else:
        slot_weights.append(1.0 / count)

slot_weights = torch.tensor(slot_weights).float().to(device)

# %% colab={"base_uri": "https://localhost:8080/"} id="vt52PCI6QSuB" executionInfo={"status": "ok", "timestamp": 1753778224751, "user_tz": -540, "elapsed": 106, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}} outputId="86c0c03a-23cc-4444-9a0f-da341846fdb5"
print(f"Slot label 개수: {len(slot2idx)}")
print(f"slot_weights shape: {slot_weights.shape}")
slot_weights

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1753777136100, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="oUXgoCngx-GP" outputId="6e4d3a48-cab6-4bd8-b10b-c0970a38b78b"
# ✅ 🔍 디버깅 출력 (encode_data 밖에 위치)
sample_idx = 6
print("🧾 Original Tokens:", train_df.iloc[sample_idx]['tokens'])
print("🔖 Labels:", train_df.iloc[sample_idx]['labels'])
print("🧠 BERT Tokens:", tokenizer.tokenize(" ".join(train_df.iloc[sample_idx]['tokens'])))
print("📦 Tokens after adding [CLS]/[SEP]:", train_encodings['input_ids'][sample_idx])
print("🧱 Attention Mask:", train_encodings['attention_mask'][sample_idx])
print("🎯 Intent:", train_intents[sample_idx])

# %% executionInfo={"elapsed": 8, "status": "ok", "timestamp": 1753777137120, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="ed435501"
import torch.nn as nn

class KoBERTIntentSlotModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.dropout = nn.Dropout(0.1)

        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)

        self.slot_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, num_slots)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits


# %% colab={"base_uri": "https://localhost:8080/"} id="CRiF7Q7Bkrks" executionInfo={"status": "ok", "timestamp": 1753777138346, "user_tz": -540, "elapsed": 884, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}} outputId="845124df-39d2-48c7-e787-57bd25e6f78f"
from collections import Counter

all_slot_labels = []
for data in train_dataset:
    all_slot_labels.extend(data['labels'].tolist())

label_counts = Counter(all_slot_labels)
total = sum(label_counts.values())
for label_id, count in label_counts.items():
    label_name = list(slot2idx.keys())[list(slot2idx.values()).index(label_id)]
    print(f"{label_name:10s}: {count:5d} ({count/total:.2%})")


# %% id="TSz99X9TU5si"

# %% executionInfo={"elapsed": 495, "status": "ok", "timestamp": 1753778602525, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="69054996"
from sklearn.metrics import classification_report, accuracy_score
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
import pickle
import os


model = KoBERTIntentSlotModel(len(intent2idx), len(slot2idx)).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
intent_loss_fn = CrossEntropyLoss()
slot_loss_fn = nn.CrossEntropyLoss(weight=slot_weights.to(device))

best_val_intent_acc = 0
best_report = None
save_path = SAVE_PATH
os.makedirs(save_path, exist_ok=True)

def train_epoch(dataloader):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="🛠️ Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        slot_labels = batch['labels'].to(device)
        intent_label = batch['intent'].to(device)

        optimizer.zero_grad()
        intent_logits, slot_logits = model(input_ids, attention_mask)

        loss_intent = intent_loss_fn(intent_logits, intent_label)
        loss_slot = slot_loss_fn(slot_logits.view(-1, len(slot2idx)), slot_labels.view(-1))
        loss = loss_intent + loss_slot

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(dataloader):
    model.eval()
    intent_preds, intent_trues = [], []
    slot_preds, slot_trues = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🔍 Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            slot_labels = batch['labels'].to(device)
            intent_label = batch['intent'].to(device)

            intent_logits, slot_logits = model(input_ids, attention_mask)
            intent_pred = torch.argmax(intent_logits, dim=1)

            # 인텐트 정확도용
            intent_preds.extend(intent_pred.cpu().numpy())
            intent_trues.extend(intent_label.cpu().numpy())

            # 슬롯 정확도용
            slot_pred = torch.argmax(slot_logits, dim=2)
            for i in range(slot_labels.size(0)):
                true_seq = slot_labels[i].cpu().tolist()
                pred_seq = slot_pred[i].cpu().tolist()
                for t, p in zip(true_seq, pred_seq):
                    if t != -100:  # PAD 제거
                        slot_trues.append(t)
                        slot_preds.append(p)

    # 🔢 인텐트 결과
    intent_acc = accuracy_score(intent_trues, intent_preds)
    intent_report = classification_report(
        intent_trues,
        intent_preds,
        target_names=idx2intent.values(),
        zero_division=0
    )

    # 🔢 슬롯 결과 (support > 0인 라벨만 포함)
    support_counter = Counter(slot_trues)
    nonzero_labels = [i for i in slot2idx.values() if support_counter[i] > 0]
    target_names_nonzero = [key for key, val in slot2idx.items() if val in nonzero_labels]

    slot_acc = accuracy_score(slot_trues, slot_preds)
    slot_report = classification_report(
        slot_trues,
        slot_preds,
        labels=nonzero_labels,
        target_names=target_names_nonzero,
        zero_division=0
    )

    return intent_acc, intent_report, slot_acc, slot_report



# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 798995, "status": "ok", "timestamp": 1753779403297, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="7de0556b" outputId="61cd4545-11bb-4f50-aea8-eae1cd7d9cc3"
EPOCHS = 5
train_losses = []
val_intent_accuracies = []
val_slot_accuracies = []

best_val_slot_acc = 0
best_intent_report = ""
best_slot_report = ""

for epoch in range(EPOCHS):
    print(f"\n📚 Epoch {epoch+1}")
    train_loss = train_epoch(train_loader)
    train_losses.append(train_loss)
    print(f"📉 Train Loss: {train_loss:.4f}")

    val_intent_acc, intent_report, val_slot_acc, slot_report = evaluate(val_loader)
    val_intent_accuracies.append(val_intent_acc)
    val_slot_accuracies.append(val_slot_acc)
    print(f"🎯 Val Intent Accuracy: {val_intent_acc:.4f}")
    print(f"🏷 Val Slot Accuracy: {val_slot_acc:.4f}")

    # ✅ 인텐트 + 슬롯 기준 모두 개선 시 저장
    if val_intent_acc > best_val_intent_acc or val_slot_acc > best_val_slot_acc:
        best_val_intent_acc = max(best_val_intent_acc, val_intent_acc)
        best_val_slot_acc = max(best_val_slot_acc, val_slot_acc)
        best_intent_report = intent_report
        best_slot_report = slot_report

        torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))
        with open(os.path.join(save_path, "intent2idx.pkl"), "wb") as f:
            pickle.dump(intent2idx, f)
        with open(os.path.join(save_path, "slot2idx.pkl"), "wb") as f:
            pickle.dump(slot2idx, f)
        print("✅ Best model saved!")

# 🔚 학습 완료 후 최종 성능 출력
print("\n📈 Best Intent Accuracy:", best_val_intent_acc)
print("📊 Best Intent Classification Report:\n", best_intent_report)
print("\n📈 Best Slot Accuracy:", best_val_slot_acc)
print("📊 Best Slot Classification Report:\n", best_slot_report)


# %% colab={"base_uri": "https://localhost:8080/"} id="dKUZOy__px47" executionInfo={"status": "ok", "timestamp": 1753779487905, "user_tz": -540, "elapsed": 8, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}} outputId="2cff12b9-8bf0-4e32-f6b5-ced2b96cd9d2"
import pickle

with open(SLOT2IDX_PATH, "rb") as f:
    slot2idx = pickle.load(f)
    print(slot2idx)
    print(len(slot2idx))

# %% colab={"base_uri": "https://localhost:8080/", "height": 507} executionInfo={"elapsed": 386, "status": "ok", "timestamp": 1753779751415, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}, "user_tz": -540} id="6eANrSDadoWO" outputId="a04c9cef-bdc4-4aa6-81ff-9e298452a65c"
fig, ax1 = plt.subplots(figsize=(10, 5))

# 왼쪽 Y축: Train Loss
ax1.plot(range(1, EPOCHS+1), train_losses, color='blue', label='Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 오른쪽 Y축: Accuracy
ax2 = ax1.twinx()
ax2.plot(range(1, EPOCHS+1), val_intent_accuracies, color='orange', label='Intent Accuracy')
ax2.plot(range(1, EPOCHS+1), val_slot_accuracies, color='green', label='Slot Accuracy')
ax2.set_ylabel('Accuracy', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# 범례 처리
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center')

plt.title('📊 Training Loss (Left) and Accuracy (Right) over Epochs')
plt.grid(True)
plt.tight_layout()
plt.show()


# %% colab={"base_uri": "https://localhost:8080/", "height": 997} id="oym377wZgyjK" outputId="04e4b4c6-d5fe-44a2-d878-f98c7ef7dc17" executionInfo={"status": "error", "timestamp": 1753779780340, "user_tz": -540, "elapsed": 23380, "user": {"displayName": "Hyun Jung Oh", "userId": "14841286664462358370"}}
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import BertModel
import pickle

# ✅ 인덱스 맵 로딩
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


# 🧱 토큰 → 단어 병합 + 슬롯 정렬
def merge_tokens_and_slots(tokens, slot_ids, idx2slot):
    merged = []
    word = ''
    current_slot = ''

    for token, slot_id in zip(tokens, slot_ids):
        slot = idx2slot.get(slot_id, 'O')

        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue

        if token.startswith("▁"):  # 새 단어 시작
            if word:
                merged.append((word, current_slot))
            word = token[1:]
            current_slot = slot
        else:
            word += token.replace("▁", "")

    if word:
        merged.append((word, current_slot))

    return merged


# 🔮 예측 함수
def predict_top_k_intents_and_slots(text, k=3):
    encoding = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=64
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        intent_logits, slot_logits = model(input_ids, attention_mask)
        intent_probs = softmax(intent_logits, dim=1)

        # 토큰 디버깅용
        # for i, logit in enumerate(slot_logits[0]):
        #   top_id = logit.argmax().item()
        #   top_slot = idx2slot[top_id]
        #   print(f"{i}번째 토큰 → {top_slot} ({logit[top_id].item():.4f})")

        # 🎯 인텐트 예측 (Top-k)
        topk_probs, topk_indices = torch.topk(intent_probs, k, dim=1)
        intents = [(idx2intent[idx.item()], prob.item()) for idx, prob in zip(topk_indices[0], topk_probs[0])]

        # 🎯 슬롯 예측
        slot_pred_ids = torch.argmax(slot_logits, dim=2)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        merged = merge_tokens_and_slots(tokens, slot_pred_ids, idx2slot)

    return intents, merged


# 🧪 실행 루프
if __name__ == "__main__":
    print("✨ KoBERT 기반 인텐트/슬롯 예측기입니다.")
    while True:
        text = input("\n✉️ 입력 (exit 입력 시 종료): ").strip()
        if text.lower() == "exit":
            print("👋 종료합니다.")
            break

        intents, slots = predict_top_k_intents_and_slots(text, k=3)

        print("\n🔍 예측된 인텐트 TOP 3:")
        for i, (intent, prob) in enumerate(intents, 1):
            print(f" {i}. {intent} ({prob:.4f})")

        print("\n🎯 슬롯 태깅 결과:")
        for word, slot in slots:
            print(f" - {word}: {slot}")


# %% [markdown] id="niz6FsslXNwF"
#
