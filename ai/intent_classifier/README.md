# 🔍 의도 분류기 (Intent Classifier)

이 문서는 KoBERT 모델을 사용하여 사용자 의도를 이해하고 분류하며, 질문에서 핵심 엔티티(슬롯)를 추출하는 챗봇 프로젝트의 `intent_classifier` 모듈에 대한 포괄적인 설명을 제공합니다.

## 1. 모듈 개요

`intent_classifier` 모듈은 챗봇이 사용자의 질문 의도를 정확하게 파악하고, 해당 질문에서 필요한 정보를 추출하여 다음 단계의 응답 생성 또는 기능 실행을 위한 기반을 마련합니다. 이는 크게 데이터 전처리, 모델 학습, 슬롯 추출, 그리고 추론 단계로 구성됩니다.

## 2. 프로젝트 구조 및 주요 파일

### 2.1. 🧹 데이터 전처리

-   `preprocess_intent_data.py`
    -   **목적**: 원본 의도 데이터셋을 정제(cleaning)하여 `intent_dataset_cleaned.csv`로 저장합니다.
    -   **기능**: 결측치 제거, 중복 제거, 텍스트 정제 등의 전처리 작업을 수행합니다.
-   `intent_slot_dataset.csv`
    -   **목적**: `create_slot_dataset.py`에 의해 생성되며, 질문, 의도, 그리고 추출된 슬롯 정보가 포함된 데이터셋입니다.
-   `intent_slot_dataset_cleaned.csv`
    -   **목적**: 전처리가 완료된 의도 및 슬롯 데이터셋입니다.
-   `keyword_boost_slot.csv`
    -   **목적**: `create_slot_dataset.py`에 의해 생성되며, 키워드와 해당 키워드에서 추출된 슬롯 정보가 포함된 데이터셋입니다.
-   `Old_data/intent_dataset.csv`
    -   **목적**: 전처리가 되지 않은 원본 의도 데이터셋입니다.  
- `Old_data/keyword_boost.csv`
    -   **목적**: 슬롯 추출에 사용되는 원본 키워드 데이터셋입니다.

### 2.2. 🔧 학습용

-   `train_kobert.ipynb`
    -   **목적**: KoBERT 모델을 학습하는 Jupyter Notebook입니다.
    -   **기능**: 학습 데이터 불러오기, 전처리, 학습, 시각화, 모델 저장 과정을 포함합니다.
-   `best_models/intent-kobert-v1`
    -   **목적**: 학습된 KoBERT 모델 파라미터가 저장되는 디렉토리입니다 (가장 정확도가 높은 모델).

### 2.3. 🚀 배포용

-   `inference.py`
    -   **목적**: 저장된 모델과 라벨 인코더를 사용하여 문장의 의도를 예측하는 배포용 코드입니다


### 2.4. 🧩 슬롯 추출 (`create_slot_dataset.py`)

이 스크립트는 사용자 질문에서 핵심 엔티티(슬롯)를 식별하고 추출하는 데 사용됩니다.

-   **`load_keywords(keyword_file)`**
    -   **목적**: CSV 파일에서 키워드를 로드하여 특정 의도와 연결합니다. 이 키워드는 슬롯 추출 정확도를 높이고 문맥별 정보를 제공하는 데 사용됩니다.
-   **`remove_korean_josa(text)`**
    -   **목적**: 대문자 영어 텍스트(예: 공항 코드)에서 한국어 조사를 제거하여 더 깔끔한 일치를 용이하게 합니다.
-   **`extract_slots(question, intent, keywords_by_intent)`**
    -   **목적**: 주어진 `question`과 `intent`를 분석하여 관련 슬롯을 추출하는 핵심 함수입니다. 정규 표현식, 키워드 일치 및 의도별 로직을 활용합니다.
-   **`create_full_slot_dataset(input_file, output_file, keyword_file)`**
    -   **목적**: 전체 슬롯 추출 프로세스를 조정합니다. 입력 CSV에서 질문과 의도를 읽고, 각 질문에 `extract_slots`를 적용하고, 결과를 (추출된 슬롯 포함) 출력 CSV에 씁니다.

## 3. 슬롯 추출 로직 상세 (`extract_slots` 함수)

`extract_slots` 함수는 다양한 유형의 슬롯을 식별하고 추출하기 위해 일련의 규칙을 사용합니다.

### 3.1. 일반 슬롯 (여러 의도에 적용 가능)

-   **`terminal`**: "1터미널", "T1", "제1터미널" 등과 같은 키워드를 기반으로 "T1" 또는 "T2"를 추출합니다.
-   **`area`**: "입국장", "출국장", "도착층", "출발층"과 같은 키워드를 기반으로 "arrival" 또는 "departure"를 식별합니다. 의도가 동점자 처리 역할을 할 수 있습니다.
-   **`gate`**: 정규 표현식을 사용하여 게이트 번호/문자(예: "A12", "23번 게이트")를 추출합니다.
-   **`airline_flight`**: 정규 표현식을 사용하여 항공편 번호(예: "KE123", "OZ4567")를 추출합니다.
-   **`airport_code`**: 정규 표현식을 사용하여 2자 또는 3자 공항 코드(예: "ICN", "JFK")를 캡처합니다.
-   **`date`**: "1월 2일", "오늘", "내일", "모레", "어제"와 같은 날짜를 추출합니다.
-   **`day_of_week`**: 요일(예: "월요일", "화요일")을 추출합니다.
-   **`time`**: 특정 시간(예: "오전 10시 30분", "오후 3시"), 상대 시간(예: "2시간 뒤"), 또는 모호한 시간("지금", "이따가")을 추출합니다.
-   **`season`**: 키워드를 기반으로 계절(예: "봄", "여름")을 추출합니다.
-   **`airline_name`**: 미리 정의된 목록에서 항공사 이름(예: "대한항공", "아시아나항공")을 추출합니다.
-   **`facility_name`**: 미리 정의된 목록 및 의도별 키워드에서 시설 이름(예: "약국", "은행", "면세점")을 추출합니다.
-   **`departure_type` / `arrival_type`**: 문맥을 기반으로 "국내선" 또는 "국제선"을 식별합니다.
-   **`location_keyword`**: "이동통로", "탑승동", "수속"과 같은 일반적인 위치 키워드를 추출합니다.
-   **`luggage_term`**: "수하물", "짐", "가방"과 같은 수하물 관련 일반 용어를 추출합니다.

### 3.2. 의도별 슬롯

#### `flight_info` 의도

-   **`flight_status`**: 키워드를 기반으로 항공편 상태(예: "지연", "결항", "취소", "도착", "출발")를 추출합니다.
-   **`origin` / `destination`**: 정규 표현식을 사용하여 "인천발 파리행"과 같은 구문에서 출발 및 도착 도시/공항을 추출합니다.

#### `airline_info_query` 의도

-   **`topic`**: "로고 이미지", "로고 링크" 또는 "이미지"와 같은 항공사 정보와 관련된 특정 주제를 추출합니다.

#### `airport_info` 의도

-   **`airport_name`**: 정규 표현식을 사용하여 전체 공항 이름(예: "인천공항", "김해 국제공항")을 추출합니다.
-   **`airport_code`**: IATA 3자 공항 코드(예: "CDG", "JFK")를 추출합니다.

#### `airport_weather_current` 의도

-   **`weather_topic`**: 특정 날씨 관련 주제(예: "기온", "습도", "바람", "강수량", "미세먼지", "TAF")를 추출합니다.

#### 주차 관련 의도 (`parking_walk_time_info`, `parking_fee_info`, `parking_availability_query`)

-   **`parking_lot`**: 특정 주차장 이름(예: "P1", "장기주차장", "주차타워")을 추출합니다.
-   **`parking_type`**: "장기", "단기", "화물", "예약"과 같은 주차 유형을 식별합니다.
-   **`vehicle_type`**: 차량 유형(예: "대형", "소형", "장애인")을 추출합니다.
-   **`destination`**: `parking_walk_time_info`의 경우 터미널 또는 구역(예: "1터미널", "출국장")을 추출합니다.
-   **`parking_duration_value` / `parking_duration_unit`**: `parking_fee_info`의 경우 기간(예: "3시간", "2일")을 추출합니다.
-   **`payment_method`**: `parking_fee_info`의 경우 결제 방법(예: "카드", "현금", "하이패스")을 추출합니다.
-   **`parking_area`**: `parking_availability_query`의 경우 특정 주차 구역(예: "P1", "지하 1층", "동편")을 추출합니다.
-   **`availability_status`**: `parking_availability_query`의 경우 가용성 상태(예: "만차", "혼잡", "여유")를 추출합니다.

#### `baggage_rule_query` 의도

-   **`baggage_type`**: 수하물 유형(예: "기내", "위탁", "특수")을 식별합니다.
-   **`rule_type`**: 규칙 유형(예: "크기", "무게", "개수", "요금", "금지 품목")을 추출합니다.
-   **`item`**: 수하물 규칙과 관련된 특정 품목(예: "액체", "전자담배", "배터리")을 추출합니다.
-   **`self_bag_drop`**: "셀프 백드랍"이 언급되었는지 여부를 나타내는 부울 슬롯입니다.

#### `baggage_claim_info` 의도

-   **`baggage_belt_number`**: 수하물 벨트 번호를 추출합니다.
-   **`baggage_issue`**: 수하물 문제(예: "lost", "damaged", "delayed")를 식별합니다.
-   **`baggage_type`**: "general", "special", "excess" 수하물을 구분합니다.

#### `transfer_info` 의도

-   **`transfer_topic`**: 특정 환승 관련 주제(예: "스탑오버", "관광 프로그램", "셔틀", "LAGs", "건강상태질문서", "세관", "항공사")를 추출합니다.

#### `departure_policy_info` 의도

-   **`passport_type`**: 여권 유형(예: "관용여권", "일반여권")을 추출합니다.
-   **`person_type`**: 인물 유형(예: "국외체류자", "외국인")을 추출합니다.
-   **`item`**: 출국 정책과 관련된 품목(예: "귀금속", "화폐")을 추출합니다.
-   **`facility_name`**: 시설 이름(예: "여권민원센터")을 추출합니다.
-   **`organization`**: 기관 이름(예: "외교부", "병무청")을 추출합니다.
-   **`document`**: 문서 유형(예: "비행기표", "여권")을 추출합니다.
-   **`topic`**: 출국 정책과 관련된 일반적인 주제(예: "수수료", "반납", "신청")를 추출합니다.

### 3.3. 주제 슬롯 (일반)

-   **`topic`**: `keyword_boost.csv` 파일에 정의된 대로 의도와 관련된 키워드를 추출하는 일반 슬롯입니다. 이는 의도별 주제에 대한 대체 또는 보조 슬롯입니다.

## 4. 실행하기

### 4.1. 설정

1.  **가상 환경 생성:**
    ```bash
    python -m venv .venv
    ```
2.  **가상 환경 활성화:**
    -   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```
3.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```

### 4.2. 사용법

1.  **슬롯 데이터셋 생성 및 보강 (`create_slot_dataset.py` 실행):**
    이 스크립트는 원본 `intent_dataset.csv`와 `keyword_boost.csv`를 입력으로 받아, 질문에서 슬롯 정보를 추출하고 이를 포함하는 `intent_slot_dataset.csv`를 생성합니다. 또한 `keyword_boost_slot.csv`도 생성됩니다.
    ```bash
    python create_slot_dataset.py
    ```
    > 📥 입력: `intent_dataset.csv`, `keyword_boost.csv`
    > 
    > 💾 출력: `intent_slot_dataset.csv`, `keyword_boost_slot.csv`

2.  **초기 데이터 전처리 (`preprocess_intent_data.py` 실행):**
    이 스크립트는 원본 `intent_dataset.csv` 파일을 정제하여 `intent_dataset_cleaned.csv` 파일을 생성합니다. 이는 모델 학습을 위한 기본적인 데이터 클리닝 단계입니다.
    ```bash
    python preprocess_intent_data.py
    ```
    > 📥 입력: `intent_slot_dataset.csv`
    > 
    > 💾 출력: `intent_slot_dataset_cleaned.csv`

3.  **의도 분류 모델 학습 (`train_kobert.ipynb` 실행):**
    Jupyter 환경에서 `train_kobert.ipynb`를 실행하여 KoBERT 모델을 학습하고, 가장 정확도가 높은 모델을 저장합니다. 학습 과정 중 정확도 및 confidence 시각화도 함께 수행됩니다.
    > 📥 입력: `intent_slot_dataset.csv` (또는 `intent_dataset_cleaned.csv`)
    > 
    > 💾 출력:
    > - `best_models/intent-kobert-v1` : 최고 성능 모델 파라미터
    > - `intent2idx.pkl` : 의도 레이블과 해당 인덱스 매핑 정보 (모델 학습에 사용) 
    > - `slot2idx.pkl` : 슬롯 레이블과 해당 인덱스 매핑 정보 (모델 학습에 사용)
    > - 학습 그래프: 학습 손실, 검증 정확도, confidence 시각화

4.  **모델 추론 (`inference.py` 실행):**
    ```bash
    python inference.py
    ```
    > 📥 입력: 사용자 입력 문장
    > 
    > 💾 출력: 예측된 intent 라벨 및 confidence score
