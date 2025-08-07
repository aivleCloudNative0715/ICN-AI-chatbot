# 🔍 의도 분류기 (Intent Classifier)

이 모듈은 KoBERT 기반의 멀티태스크 학습 모델을 사용하여 **사용자의 질문 의도(Intents)** 를 분류하고 **관련 슬롯(Slots)** 을 동시에 추출합니다.  
Google Colab 환경에 최적화된 학습용 노트북과 CLI 기반 실시간 추론 스크립트를 포함합니다.


---

## 💡 핵심 구성 요소

| 파일/디렉토리 | 설명 |
|---------------|------|
| `train_kobert.ipynb` | **KoBERT 기반 멀티태스크 모델** 학습을 위한 Colab 전용 Jupyter 노트북입니다. 데이터 로딩, 전처리, 학습, 평가 과정을 포함합니다. |
| `inference.py` | 학습된 모델을 사용하여 **실시간으로 의도 및 슬롯을 예측**하는 CLI 스크립트입니다. |
| `create_slot_dataset.py` | 원본 `intent_dataset.csv`로부터 **규칙 기반 슬롯 태깅 데이터셋**을 생성합니다. |
| `preprocess_intent_data.py` | 질문 텍스트 전처리 및 정제용 스크립트입니다. |
| `data/` | 모든 학습/실험용 CSV 파일이 저장되어 있습니다. (`intent_slot_dataset.csv` 등) |
| `Old_data/` | 이전 버전 코드 백업 또는 실험 중 코드 보관용 디렉토리입니다. |
| `shared/` *(외부 디렉토리)* | `normalize_with_morph.py`, `predict_intent_and_slots.py` 등의 **전처리 및 모델 예측 유틸리티**를 포함하는 공통 코드 모듈입니다. |
| `best_models/` *(외부 디렉토리)* | 학습된 KoBERT 모델 가중치(`.pt`)와 라벨 인코더(`.pkl`)가 저장되는 위치입니다. |

---

## 🛠 데이터 준비 (선 실행됨)

> 이 단계는 이미 완료되어 있습니다. 아래는 참고용 설명입니다.

### 1. 슬롯 데이터 생성

```bash
python create_slot_dataset.py
```

- 입력: `data/intent_dataset.csv`
- 출력: `data/intent_slot_dataset.csv`

### 2. 데이터 전처리

```bash
python preprocess_intent_data.py
```

- 불필요한 기호 제거, 질문 전처리 등

---

## 🏁 모델 학습 (선 실행됨)

> 모델 학습도 완료되어 있습니다. 아래는 참고용 설명입니다.

- `train_kobert.ipynb` 노트북에서 KoBERT 모델 학습을 진행하였으며,  
  가장 성능이 우수한 모델은 다음 경로에 저장되었습니다. 높은 버전은 가장 최신 모델을 뜻합니다. 

```
best_models/intent-kobert-v2/
├── best_kobert_model.pt
├── intent2idx.pkl
└── slot2idx.pkl
```

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 가상 환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

---

### 2. 실시간 추론

```bash
python inference.py
```

- 실행 시 입력을 받는 CLI 인터페이스가 시작됩니다.
- 형태소 분석 기반 정규화 + KoBERT 모델 예측을 수행합니다.
- 상위 3개 인텐트와 각 토큰별 슬롯 예측 결과를 출력합니다.

---

## 📌 예시 출력

```
✨ KoBERT 기반 인텐트/슬롯 예측기입니다.

✉️ 입력 (exit 입력 시 종료): 인천공항 주차장 지금 어디가 여유있어?

🔍 예측된 인텐트 TOP 3:
 1. parking_availability_query (0.7651)
 2. parking_location_recommendation (0.2293)
 3. parking_congestion_prediction (0.0036)

🎯 슬롯 태깅 결과:
 - 인천공항: O
 - 주차장: B-parking_lot
 - 지금: B-time
 - 어디가: O
 - 여유: B-availability_status
 - 있어: O
 - ?: O
```

---

## 📎 참고

- `inference.py`는 다음 `shared/` 모듈에 의존합니다:
  - `ai/shared/normalize_with_morph.py`
  - `ai/shared/predict_intent_and_slots.py`
- 학습된 모델 가중치(`.pt`) 및 인코더 파일(`.pkl`)은 반드시 존재해야 추론이 가능합니다.
- `train_kobert.ipynb`는 Google Colab 환경에서 가장 안정적이며 빠르게 실행됩니다.
