# AI 모델 코드 설명

이 문서는 `preprocess_intent_data.py`, `train_kobert.ipynb`, `inference.py` 파일의 코드에 대한 자세한 설명을 제공합니다.

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

## 2. `create_slot_dataset.py`

이 스크립트는 슬롯 태깅 모델 학습에 사용될 데이터를 생성합니다.

### 주요 기능

- **데이터셋 생성:** 주어진 의도와 질문 쌍을 기반으로 슬롯 태깅을 위한 데이터셋을 생성합니다.


### 함수 설명

#### `create_slot_data(intent, question)`

- **입력:**
    - `intent`: 의도 문자열 (예: '날씨')
    - `question`: 질문 문자열 (예: '오늘 날씨 어때?')
- **기능:**
    1. 질문을 단어 단위로 분리합니다.
    2. 질문 내에서 특정 키워드나 패턴을 기반으로 슬롯 정보를 추출합니다.
- **반환:** 추출된 슬롯 정보 (딕셔너리 형태)

### 실행

스크립트를 직접 실행하면 슬롯 태깅 데이터셋을 생성하고 저장합니다.

```bash
python create_slot_dataset.py
```

---

## 3. `train_kobert.ipynb`

이 Jupyter Notebook은 전처리된 데이터를 사용하여 KoBERT 기반의 의도 분류 모델을 학습하고 평가합니다. 기존 `kobert_intent_trainer.py`에서 데이터 보강 및 균형 조정 기능이 추가되었습니다.

### 주요 구성 요소

#### `IntentDataset(Dataset)` 클래스

- PyTorch의 `Dataset`을 상속받아 KoBERT 모델에 맞는 데이터셋 형태로 변환하는 클래스입니다.
- **`__init__`**: 데이터프레임, 토크나이저, 라벨 인코더를 초기화합니다.
- **`__len__`**: 데이터셋의 총 길이를 반환합니다.
- **`__getitem__`**: 특정 인덱스(`idx`)에 해당하는 문장을 토크나이징하고, `input_ids`, `attention_mask`, `label`을 텐서(Tensor) 형태로 반환합니다.

#### `KoBERTClassifier(nn.Module)` 클래스

- KoBERT를 기반으로 하는 의도 분류 모델의 아키텍처를 정의하는 클래스입니다.
- **`__init__`**: 모델의 구성 요소를 초기화합니다.
    - `self.bert`: `skt/kobert-base-v1` 사전 학습 모델을 불러옵니다.
    - `self.dropout`: 과적합을 방지하기 위한 드롭아웃 레이어 (30% 드롭아웃)
    - `self.classifier`: BERT 모델의 출력을 받아 최종적으로 의도를 분류하는 선형 레이어
- **`forward`**: 모델의 순전파 과정을 정의합니다. 입력으로 `input_ids`와 `attention_mask`를 받아 각 의도에 대한 확률(logits)을 출력합니다.

### 학습 과정

1.  **데이터 로드 및 보강:**
    - `pd.read_csv("intent_dataset_cleaned.csv")`: 전처리된 데이터를 로드합니다.
    - `pd.read_csv("keyword_boost.csv")`: 반드시 포함되어야 하는 키워드 기반의 문장 데이터를 추가로 로드합니다.
    - 두 데이터프레임을 병합하여 학습 데이터를 구성합니다.

2.  **데이터 균형 조정:**
    - 각 의도(intent)별 데이터가 불균형한 문제를 해결하기 위해, 각 의도별 샘플 수를 200개로 맞춥니다.
    - `sklearn.utils.resample`을 사용하여 샘플 수가 부족한 경우 오버샘플링(중복 추출)을, 샘플 수가 많은 경우 언더샘플링(무작위 추출)을 수행합니다.

3.  **데이터 분할 및 인코딩:**
    - `LabelEncoder()`: 문자열 형태의 의도(intent)를 숫자 라벨로 변환합니다.
    - `train_test_split()`: 균형이 맞춰진 데이터를 학습용(80%)과 검증용(20%)으로 분할합니다.

4.  **토크나이저 및 데이터로더 준비:**
    - `AutoTokenizer.from_pretrained("skt/kobert-base-v1")`: KoBERT 토크나이저를 로드합니다.
    - `IntentDataset`과 `DataLoader`를 사용하여 학습 및 검증 데이터를 모델에 배치(batch) 단위로 공급할 수 있도록 준비합니다.

5.  **모델 학습 및 검증:**
    - `AdamW` 옵티마이저와 `CrossEntropyLoss` 손실 함수를 사용하여 모델을 학습합니다.
    - 매 에폭(epoch)마다 학습 손실(loss)과 검증 정확도(accuracy)를 계산하여 출력합니다.
    - 검증 정확도가 가장 높은 모델을 `best_models/intent-kobert-v1/best_kobert_model.pt` 파일로 저장하고, `classification_report`를 통해 상세 성능을 출력합니다.
    - 학습에 사용된 `label_encoder`는 `best_models/intent-kobert-v1/label_encoder.pkl` 파일로 저장하여 추론 시 일관된 라벨을 사용하도록 합니다.

### 실행

Jupyter Notebook 환경에서 각 셀을 순서대로 실행하여 모델 학습을 진행합니다.

---

## 3. `inference.py`

이 스크립트는 학습된 KoBERT 모델(`best_kobert_model.pt`)을 불러와 새로운 사용자 입력에 대한 의도를 예측하는 역할을 합니다.

### 주요 구성 요소

#### `KoBERTClassifier(nn.Module)` 클래스

- 학습 스크립트(`train_kobert.ipynb`)에 정의된 것과 동일한 모델 아키텍처입니다.

#### `predict_intent(text)` 함수

- **입력:** 예측할 텍스트 문자열
- **기능:**
    1. `monologg/kobert` 토크나이저를 사용하여 입력 텍스트를 토크나이징합니다.
    2. 저장된 `best_kobert_model.pt`와 `label_encoder.pkl`을 로드합니다.
    3. 토크나이징된 입력을 모델에 전달하여 각 의도별 확률을 계산합니다.
    4. `softmax` 함수를 통해 가장 확률이 높은 의도와 해당 신뢰도(confidence)를 찾습니다.
- **반환:** 예측된 의도(문자열)와 신뢰도(실수)

### 실행

스크립트를 직접 실행하면 사용자로부터 질문을 입력받아 예측된 의도와 신뢰도를 출력하는 상호작용 세션이 시작됩니다. 'exit'를 입력하면 종료됩니다.

```bash
python inference.py
```
