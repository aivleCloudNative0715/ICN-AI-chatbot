# 🔍 의도 분류기 (Intent Classifier)

이 모듈은 KoBERT 기반의 멀티태스크 학습 모델을 사용하여 **사용자의 질문 의도(Intents)** 를 분류하고 **관련 슬롯(Slots)** 을 동시에 추출합니다.  
**konlpy**를 사용한 형태소 분석 기반 전처리와 **복합의도 처리**를 지원하며, Google Colab 환경에 최적화된 학습용 노트북과 CLI 기반 실시간 추론 스크립트를 포함합니다.


---

## 💡 핵심 구성 요소

| 파일/디렉토리 | 설명 |
|---------------|------|
| `train_kobert.ipynb` | **KoBERT 기반 멀티태스크 모델** 학습을 위한 Colab 전용 Jupyter 노트북입니다. 데이터 로딩, 전처리, 학습, 평가 과정을 포함하며 **복합의도 처리**를 지원합니다. |
| `inference.py` | 학습된 모델을 사용하여 **실시간으로 의도 및 슬롯을 예측**하는 CLI 스크립트입니다. |
| `create_slot_dataset.py` | 원본 `intent_dataset.csv`로부터 **규칙 기반 슬롯 태깅 데이터셋**을 생성합니다. |
| `preprocess_intent_data.py` | **konlpy 기반 형태소 분석**을 통한 질문 텍스트 전처리 및 정제용 스크립트입니다. |
| `classify_recommend_question.py` | 추천 질문 데이터셋(`recommend_question_data.csv`) 테스트용 **단순 테스트 스크립트**입니다. |
| `data/` | 모든 학습/실험용 CSV 파일이 저장되어 있습니다. (`intent_slot_dataset.csv`, `recommend_question_data.csv` 등) |
| `Old_data/` | 이전 버전 코드 백업 또는 실험 중 코드 보관용 디렉토리입니다. |
| `shared/` *(외부 디렉토리)* | `normalize_with_morph.py`, `predict_intent_and_slots.py` 등의 **전처리 및 모델 예측 유틸리티**를 포함하는 공통 코드 모듈입니다. |
| `best_models/` *(외부 디렉토리)* | 학습된 KoBERT 모델 가중치(`.pt`)와 라벨 인코더(`.pkl`)가 저장되는 위치입니다. |

---

## 🛠 데이터 준비 (선 실행됨)

> 이 단계는 이미 완료되어 있습니다. 아래는 참고용 설명입니다.

### 1. 슬롯 데이터 생성

```bash
python preprocess_intent_data.py
```

- 불필요한 기호 제거, 질문 전처리 등


### 2. 데이터 전처리

```bash
python create_slot_dataset.py
```

- 입력: `data/intent_dataset_cleaned.csv`
- 출력: `data/intent_slot_dataset_cleaned.csv`

---

## 🏁 모델 학습 (선 실행됨)

> 모델 학습도 완료되어 있습니다. 아래는 참고용 설명입니다.

- `train_kobert.ipynb` 노트북에서 KoBERT 모델 학습을 진행하였으며,  
  가장 성능이 우수한 모델은 다음 경로에 저장되었습니다. 높은 버전은 가장 최신 모델을 뜻합니다. 

```
best_models/intent-kobert-v3/
├── best_kobert.pt
├── intent2idx.pkl
├── slot2idx.pkl
└── train_kobert.ipynb
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
- **konlpy 기반 형태소 분석** + KoBERT 모델 예측을 수행합니다.
- 상위 3개 인텐트와 각 토큰별 슬롯 예측 결과를 출력합니다.
- **복합의도 감지** 시 의도 리스트 형태로 결과를 반환합니다.

---

## 📌 예시 출력

```
✉️ 입력 (Analyze Thresh=0.50, Multi Thresh=0.50): 주차장 혼잡도 알려줘 

--- 라우팅 결정 ---
🎯 결정: ROUTE
📊 최대 신뢰도: 0.9984
🔄 액션: ✅ 직접 라우팅: parking_availability_query 핸들러 호출
🏷️ 예측 의도 (임계값 0.50 이상): parking_availability_query(0.998)

--- 상세 예측 분석 ---

📝 입력: 주차장 혼잡도 알려줘
🎯 임계값: 0.5
🔢 복합 의도 여부: No

🏆 임계값 이상 인텐트 (1개):
   1. parking_availability_query: 0.9984

📊 전체 Top-3 인텐트:
   1. parking_availability_query: 0.9984
   2. airport_congestion_prediction: 0.0055
   3. parking_congestion_prediction: 0.0029

🎭 슬롯 태깅 결과:
   - 주차장: B-parking_lot
   - 혼잡: B-availability_status
   - 도: O
   - 알려줘: O
```

## 📊 모델 성능

### 🎯 Intent Classification 성능

- **정확도 (Exact Match)**: **99.77%** (0.9977)

```
                                 precision    recall  f1-score   support

             airline_info_query       1.00      1.00      1.00       703
  airport_congestion_prediction       1.00      0.99      1.00       652
                   airport_info       1.00      1.00      1.00       480
        airport_weather_current       1.00      1.00      1.00       694
             baggage_claim_info       1.00      1.00      1.00       936
             baggage_rule_query       1.00      1.00      1.00       622
               default_greeting       1.00      0.99      1.00       116
                 facility_guide       1.00      1.00      1.00      1396
                    flight_info       1.00      1.00      1.00      1093
             immigration_policy       1.00      0.99      1.00       527
                  out_of_domain       0.99      1.00      1.00       116
     parking_availability_query       1.00      1.00      1.00       505
  parking_congestion_prediction       1.00      1.00      1.00       326
               parking_fee_info       1.00      1.00      1.00       895
parking_location_recommendation       1.00      1.00      1.00       479
         parking_walk_time_info       1.00      1.00      1.00       577
         regular_schedule_query       0.99      1.00      1.00       365
                  transfer_info       1.00      1.00      1.00       837
           transfer_route_guide       1.00      1.00      1.00       356

                      micro avg       1.00      1.00      1.00     11675
                      macro avg       1.00      1.00      1.00     11675
                   weighted avg       1.00      1.00      1.00     11675
                    samples avg       0.98      0.98      0.98     11675
```

### 🏷️ Slot Tagging 성능

- **정확도**: **99.93%** (0.9993)

```
                          precision    recall  f1-score   support

          B-airline_info       1.00      1.00      1.00       629
          B-airline_name       1.00      1.00      1.00       168
          B-airport_code       1.00      1.00      1.00        92
          B-airport_name       0.98      0.97      0.98       255
       B-arrival_airport       0.90      1.00      0.95        54
          B-arrival_type       1.00      1.00      1.00        43
   B-availability_status       0.99      1.00      1.00       127
         B-baggage_issue       1.00      1.00      1.00        33
          B-baggage_type       1.00      1.00      1.00       116
   B-congestion_keywords       1.00      1.00      1.00       354
                  B-date       1.00      1.00      1.00       225
           B-day_of_week       1.00      1.00      1.00        22
     B-departure_airport       1.00      1.00      1.00        30
B-departure_airport_name       1.00      0.90      0.95        29
        B-departure_type       1.00      1.00      1.00        33
         B-facility_name       1.00      1.00      1.00      1388
             B-fee_topic       1.00      1.00      1.00       514
             B-flight_id       1.00      1.00      1.00       140
         B-flight_status       1.00      0.99      1.00       298
                  B-gate       1.00      1.00      1.00        24
                  B-item       1.00      1.00      1.00       108
      B-location_keyword       1.00      1.00      1.00        32
          B-luggage_term       0.99      1.00      1.00      1240
          B-organization       1.00      1.00      1.00        29
          B-parking_area       1.00      1.00      1.00        44
 B-parking_duration_unit       1.00      1.00      1.00        22
B-parking_duration_value       1.00      1.00      1.00        32
           B-parking_lot       1.00      1.00      1.00       949
          B-parking_type       1.00      1.00      1.00       284
        B-payment_method       1.00      1.00      1.00        29
           B-person_type       1.00      1.00      1.00        33
         B-relative_time       0.99      1.00      0.99        98
             B-rule_type       1.00      1.00      1.00        31
                B-season       1.00      1.00      1.00        32
              B-terminal       1.00      1.00      1.00       623
                  B-time       1.00      1.00      1.00        57
           B-time_period       0.97      0.99      0.98       118
        B-transfer_topic       1.00      1.00      1.00        26
            B-vague_time       1.00      0.99      0.99       398
         B-weather_topic       0.99      1.00      0.99       146
          I-airline_info       1.00      1.00      1.00      1169
          I-airline_name       1.00      1.00      1.00       116
          I-airport_code       1.00      1.00      1.00       162
          I-airport_name       0.99      0.95      0.97       353
       I-arrival_airport       0.95      1.00      0.97        72
          I-arrival_type       1.00      1.00      1.00        43
   I-availability_status       1.00      1.00      1.00        79
         I-baggage_issue       1.00      1.00      1.00        33
          I-baggage_type       1.00      1.00      1.00       101
   I-congestion_keywords       1.00      1.00      1.00        76
                  I-date       1.00      1.00      1.00       181
           I-day_of_week       1.00      1.00      1.00        45
     I-departure_airport       0.00      0.00      0.00         0
I-departure_airport_name       0.81      0.83      0.82        30
        I-departure_type       1.00      1.00      1.00        33
         I-facility_name       1.00      1.00      1.00       754
             I-fee_topic       1.00      1.00      1.00       892
             I-flight_id       1.00      1.00      1.00       416
         I-flight_status       1.00      1.00      1.00        12
                  I-gate       0.00      0.00      0.00         0
                  I-item       1.00      1.00      1.00       270
      I-location_keyword       1.00      1.00      1.00        32
          I-luggage_term       1.00      1.00      1.00       898
          I-organization       1.00      1.00      1.00        28
          I-parking_area       1.00      1.00      1.00        18
 I-parking_duration_unit       0.00      0.00      0.00         0
I-parking_duration_value       0.00      0.00      0.00         0
           I-parking_lot       1.00      1.00      1.00      1560
          I-parking_type       0.00      0.00      0.00         0
        I-payment_method       0.00      0.00      0.00         0
           I-person_type       0.00      0.00      0.00         0
         I-relative_time       0.98      1.00      0.99       138
             I-rule_type       0.00      0.00      0.00         2
                I-season       1.00      1.00      1.00         8
              I-terminal       1.00      1.00      1.00       623
                  I-time       1.00      1.00      1.00        76
           I-time_period       0.00      0.00      0.00         0
        I-transfer_topic       1.00      1.00      1.00        52
            I-vague_time       1.00      1.00      1.00        45
         I-weather_topic       0.99      1.00      0.99        68
                       O       1.00      1.00      1.00     96843

                accuracy                           1.00    114133
               macro avg       0.88      0.88      0.88    114133
            weighted avg       1.00      1.00      1.00    114133
```

---

## 📎 참고

- `inference.py`는 다음 `shared/` 모듈에 의존합니다:
  - `ai/shared/normalize_with_morph.py`
  - `ai/shared/predict_intent_and_slots.py`
- 학습된 모델 가중치(`.pt`) 및 인코더 파일(`.pkl`)은 반드시 존재해야 추론이 가능합니다.
- `train_kobert.ipynb`는 Google Colab 환경에서 가장 안정적이며 빠르게 실행됩니다.
