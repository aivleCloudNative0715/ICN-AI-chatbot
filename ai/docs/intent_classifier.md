# AI 모델 코드 설명

이 문서는 `create_slot_dataset.py`, `preprocess_intent_data.py`, `train_kobert.ipynb`, `inference.py` 파일의 코드에 대한 자세한 설명을 제공합니다.

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
    3. **konlpy 기반 normalize_with_morph 함수**를 이용하여 질문을 형태소 단위로 나눕니다. 
- **반환:** 정제된 텍스트 문자열

#### `preprocess_intent_csv(input_path: str, output_path: str)`

- **입력:**
    - `input_path`: 원본 CSV 파일 경로 (`intent_dataset.csv`)
    - `output_path`: 전처리 후 저장할 CSV 파일 경로 (`intent_dataset_cleaned.csv`)
- **기능:**
    1. `pd.read_csv(input_path)`: CSV 파일을 Pandas DataFrame으로 불러옵니다.
    2. `df.dropna(subset=['intent_list', 'question'], inplace=True)`: 'intent' 또는 'question' 열에 결측치가 있는 행을 제거합니다.
    3. `df['question'].astype(str).apply(clean_text)`: 'question' 열의 모든 텍스트에 `clean_text` 함수를 적용하여 정제합니다.
    4. `df['question'] = df['question'].apply(normalize_with_morph)`: **konlpy 기반 normalize_with_morph 함수**에서 질문을 형태소 단위로 나눕니다.
    5. `df.drop_duplicates(subset=['intent', 'question'], inplace=True)`: 'intent'와 'question'이 모두 동일한 중복 행을 제거합니다.
    6. `df.to_csv(output_path, index=False, encoding='utf-8-sig')`: 처리된 데이터를 새로운 CSV 파일로 저장합니다.

### 실행

스크립트를 직접 실행하면 `intent_slot_dataset.csv`를 읽어 `intent_slot_dataset_cleaned.csv`를 생성합니다.

```bash
python preprocess_intent_data.py
```
---

## 2. `create_slot_dataset.py`

이 스크립트는 슬롯 태깅 모델 학습에 사용될 데이터를 생성합니다.

### 주요 기능

- **데이터셋 생성:** 주어진 의도와 질문 쌍을 기반으로 슬롯 태깅을 위한 데이터셋을 생성합니다.
    - **해당 코드는 현재 학습데이터셋에 특화되어 개발되었습니다.**


### 함수 설명

#### `create_slot_data(question, intent_list)`

- **입력:**
    - `intent_list`: 의도 문자열 리스트(예: ['airport_weather_current'])
    - `question`: 질문 문자열 (예: '지금 인천공항 비 오나요?')
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


---

## 3. `train_kobert.ipynb`

이 Jupyter Notebook은 전처리된 데이터를 사용하여 **KoBERT 기반의 의도 분류 및 슬롯 태깅 모델**을 학습하고 평가합니다.

> ⚠️ **본 노트북은 Google Colab 환경에서 최적의 성능을 발휘하도록 구성되어 있습니다.**  
> 필요한 패키지 설치, 경로 설정, GPU 사용 등이 Colab 기준으로 설정되어 있어 빠른 실행과 손쉬운 실험이 가능합니다.

---

### 📌 주요 구성 요소

#### 🧩 `IntentSlotDataset(Dataset)`
- PyTorch의 `Dataset`을 상속받아 KoBERT 입력 형식에 맞게 데이터를 가공합니다.
- 각 샘플은 다음 정보를 포함합니다:
  - `input_ids`
  - `attention_mask`
  - `intent_label`
  - `slot_labels` (`-100`으로 패딩 토큰 제외)

#### 🧠 `KoBERTIntentSlotModel(nn.Module)`
- `skt/kobert-base-v1`를 기반으로 의도 분류와 슬롯 태깅을 동시에 수행하는 멀티태스크 구조입니다.
- 구성:
  - `intent_classifier`: 문장 단위 분류기
  - `slot_classifier`: 토큰 단위 분류기

---

### ⚙️ 학습 파이프라인

1. **데이터 로드 및 보강**
   - `intent_slot_dataset.csv`: 일반 학습 데이터
   - `keyword_boost_slot.csv`: 핵심 키워드 기반 데이터
   - 병합하여 다양한 문장 구조 학습 가능

2. **의도 라벨 클래스 균형화**
   - `resample`로 각 intent 클래스 샘플 수를 200개로 맞춤
   - 클래스 불균형으로 인한 학습 편향 최소화

3. **라벨 인코딩 및 분할**
   - `LabelEncoder`로 intent 및 slot 라벨 인코딩
   - `train_test_split`으로 8:2 비율 분할

4. **토크나이저 및 DataLoader 구성**
   - `skt/kobert-base-v1` 전용 토크나이저 사용
   - `IntentSlotDataset` → `DataLoader`로 변환하여 batch 학습 지원

5. **모델 학습 및 검증**
   - 옵티마이저: `AdamW`
   - 손실 함수
     - `BCEWithLogitsLoss` (intent)
     - `CrossEntropyLoss` (slot)
   - 매 epoch마다 loss 및 accuracy 출력
   - **검증 정확도 기반으로 best 모델 저장**

6. **모델 및 인코더 저장**
   - `best_models/intent-kobert-v3/best_kobert_model.pt`
   - `intent2idx.pkl`, `slot2idx.pkl` → 추론 시 라벨 일관성 유지
### 실행

Jupyter Notebook 환경에서 각 셀을 순서대로 실행하여 모델 학습을 진행합니다.

---

## 4. `inference.py`

이 스크립트는 학습된 KoBERT 모델(`best_model.pt`)을 불러와 새로운 사용자 입력에 대한 의도와 슬롯을 예측하는 역할을 합니다.

> 모든 주요 기능은 `ai/shared/` 모듈을 통해 모듈화되어 관리됩니다.

---

### 📌 `normalize_with_morph`  
- **위치:** `ai/shared/normalize_with_morph.py`  
- **역할:** 사용자 입력 문장을 **형태소 분석 기반으로 전처리**  
- 예: 띄어쓰기 보정, 조사/어미 분리 등 자연어 처리 전처리에 적합

---

### 📌 `predict_with_bce`  
- **위치:** `ai/shared/predict_intent_and_slots.py`  
- **역할:** 전처리된 텍스트에 대해 KoBERT 모델을 통해  
  - **의도(Intents)** 상위 k개 예측 (softmax에서 sigmoid로 변경하여 각 의도별 독립적인 확률 계산)
  - **슬롯(Slots)** BIO 태그 기반으로 예측  
- 내부에서 `best_model.pt`, `intent2idx.pkl`, `slot2idx.pkl`을 자동 로드하여 사용
- **복합의도 처리:** 다중 라벨 분류를 위해 BCE(Binary Cross Entropy) 손실 함수 적용
- **출력 예시:** 임계값 기반 의도 필터링, 슬롯 태깅 결과, 상세 예측 분석 포함

---

### 🧪 실행 루프 (`__main__`)  
- 사용자 입력을 받아 위 두 함수를 호출하고 **예측 결과를 실시간으로 출력**
- `exit` 입력 시 종료되는 **간단한 CLI 인터페이스**
### 실행

스크립트를 직접 실행하면 사용자로부터 질문을 입력받아 예측된 상위 3개의 의도와 신뢰도, 그리고 슬롯 태깅 결과를 출력하는 상호작용 세션이 시작됩니다. 'exit'를 입력하면 종료됩니다.

```bash
python inference.py
```