# AI 모델 코드 설명

이 문서는 `preprocess_intent_data.py`와 `kobert_intent_trainer.py` 파일의 코드에 대한 자세한 설명을 제공합니다.

---

## 1. `preprocess_intent_data.py`

이 스크립트는 KoBERT 모델 학습에 사용될 `intent_dataset.csv`를 전처리하는 역할을 합니다.

### 주요 기능

- **텍스트 정제:** 한글, 영어, 숫자, 공백을 제외한 모든 특수문자를 제거하고, 여러 개의 공백을 하나로 통합하여 텍스트를 정제합니다.
- **결측치 및 중복 제거:** 데이터에서 비어있는 행과 중복되는 행을 제거하여 데이터의 품질을 높입니다.

### 함수 설명

#### `clean_text(text)`

- **입력:** 정제할 텍스트 문자열
- **기능:**
    1. `re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", str(text))`: 정규식을 사용하여 한글, 영문, 숫자, 공백을 제외한 모든 문자를 제거합니다.
    2. `re.sub(r"\s+", " ", text)`: 여러 개의 연속된 공백을 하나의 공백으로 변환합니다.
- **반환:** 정제된 텍스트 문자열

#### `preprocess_intent_csv(input_path: str, output_path: str)`

- **입력:**
    - `input_path`: 원본 CSV 파일 경로 (`intent_dataset.csv`)
    - `output_path`: 전처리 후 저장할 CSV 파일 경로 (`intent_dataset_cleaned.csv`)
- **기능:**
    1. `pd.read_csv(input_path)`: CSV 파일을 Pandas DataFrame으로 불러옵니다.
    2. `df.dropna(subset=['intent', 'question'], inplace=True)`: 'intent' 또는 'question' 열에 결측치가 있는 행을 제거합니다.
    3. `df['question'].astype(str).apply(clean_text)`: 'question' 열의 모든 텍스트에 `clean_text` 함수를 적용하여 정제합니다.
    4. `df.drop_duplicates(subset=['intent', 'question'], inplace=True)`: 'intent'와 'question'이 모두 동일한 중복 행을 제거합니다.
    5. `df.to_csv(output_path, index=False, encoding='utf-8-sig')`: 처리된 데이터를 새로운 CSV 파일로 저장합니다.

### 실행

스크립트를 직접 실행하면 `intent_dataset.csv`를 읽어 `intent_dataset_cleaned.csv`를 생성합니다.

```bash
python preprocess_intent_data.py
```

---

## 2. `kobert_intent_trainer.py`

이 스크립트는 전처리된 데이터를 사용하여 KoBERT 기반의 의도 분류 모델을 학습하고 평가합니다.

### 주요 구성 요소

#### `IntentDataset(Dataset)` 클래스

- PyTorch의 `Dataset`을 상속받아 KoBERT 모델에 맞는 데이터셋 형태로 변환하는 클래스입니다.
- **`__init__`**: 데이터프레임, 토크나이저, 라벨 인코더를 초기화합니다.
- **`__len__`**: 데이터셋의 총 길이를 반환합니다.
- **`__getitem__`**: 특정 인덱스(`idx`)에 해당하는 문장을 토크나이징하고, `input_ids`, `attention_mask`, `label`을 텐서(Tensor) 형태로 반환합니다.

#### `KoBERTClassifier(nn.Module)` 클래스

- KoBERT를 기반으로 하는 의도 분류 모델의 아키텍처를 정의하는 클래스입니다.
- **`__init__`**: 모델의 구성 요소를 초기화합니다.
    - `self.bert`: `monologg/kobert` 사전 학습 모델을 불러옵니다.
    - `self.dropout`: 과적합을 방지하기 위한 드롭아웃 레이어 (30% 드롭아웃)
    - `self.classifier`: BERT 모델의 출력을 받아 최종적으로 의도를 분류하는 선형 레이어
- **`forward`**: 모델의 순전파 과정을 정의합니다. 입력으로 `input_ids`와 `attention_mask`를 받아 각 의도에 대한 확률(logits)을 출력합니다.

### 학습 과정

1.  **데이터 로드 및 분할:**
    - `pd.read_csv("intent_dataset_cleaned.csv")`: 전처리된 데이터를 로드합니다.
    - `LabelEncoder()`: 문자열 형태의 의도(intent)를 숫자 라벨로 변환합니다.
    - `train_test_split()`: 데이터를 학습용(80%)과 검증용(20%)으로 분할합니다.

2.  **토크나이저 및 데이터로더 준비:**
    - `BertTokenizer.from_pretrained("monologg/kobert")`: KoBERT 토크나이저를 로드합니다.
    - `IntentDataset`과 `DataLoader`를 사용하여 학습 및 검증 데이터를 모델에 배치(batch) 단위로 공급할 수 있도록 준비합니다.

3.  **모델 및 학습 설정:**
    - `KoBERTClassifier`를 초기화하고 GPU 사용을 설정합니다.
    - `optimizer`: 모델의 가중치를 최적화하기 위해 `AdamW` 옵티마이저를 사용합니다.
    - `criterion`: 손실 함수로 다중 클래스 분류에 적합한 `CrossEntropyLoss`를 사용합니다.

4.  **학습 및 검증 루프:**
    - **학습 (`model.train()`):**
        - `train_loader`로부터 배치 단위로 데이터를 받아 모델에 입력합니다.
        - 모델의 출력과 실제 라벨 간의 손실(loss)을 계산합니다.
        - `loss.backward()`와 `optimizer.step()`을 통해 모델의 가중치를 업데이트합니다.
    - **검증 (`model.eval()`):**
        - `val_loader`로부터 데이터를 받아 모델의 성능을 평가합니다.
        - `softmax` 함수를 통해 모델의 출력을 각 클래스(의도)에 대한 확률로 변환합니다.
        - `torch.max`를 사용하여 가장 높은 확률을 가진 의도를 예측값으로 선택하고, 해당 확률값(confidence)도 함께 얻습니다.
        - `accuracy_score`를 사용하여 모델의 정확도를 계산하고 출력합니다.

### 실행

스크립트를 직접 실행하면 설정된 `EPOCHS` 만큼 학습과 검증을 반복합니다.

```bash
python inference.py
```
