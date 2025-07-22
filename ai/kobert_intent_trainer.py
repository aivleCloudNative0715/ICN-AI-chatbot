import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, logging as transformers_logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 🤫 경고 숨기기
transformers_logging.set_verbosity_error()

# 📁 Dataset 클래스 정의
class IntentDataset(Dataset):
    def __init__(self, df, tokenizer, label_encoder, max_len=64):
        self.sentences = df['question'].tolist()
        self.labels = label_encoder.transform(df['intent'].tolist())
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 🧠 KoBERT 분류 모델
class KoBERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(KoBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("monologg/kobert")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# 📄 데이터 로드
df = pd.read_csv("intent_dataset_cleaned.csv")

# 📊 라벨 인코딩
label_encoder = LabelEncoder()
label_encoder.fit(df['intent'])

# 🧪 학습/검증 데이터 분할
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['intent'],
    random_state=42
)

# 🔤 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

# 🧷 Dataset 및 DataLoader 준비
train_dataset = IntentDataset(train_df, tokenizer, label_encoder)
val_dataset = IntentDataset(val_df, tokenizer, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 🚀 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KoBERTClassifier(num_labels=len(label_encoder.classes_)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 🔁 학습 루프
EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📚 Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # 🎯 검증
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)

            probs = softmax(outputs, dim=1)  # 💡 각 클래스에 대한 확률
            confs, predicted = torch.max(probs, 1)  # 💡 최대 확률값(confidence)과 클래스 index

            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    avg_conf = confs.mean().item()
    print(f"🎯 Validation Accuracy: {acc:.4f} | 🔍 Avg Confidence: {avg_conf:.4f}")

