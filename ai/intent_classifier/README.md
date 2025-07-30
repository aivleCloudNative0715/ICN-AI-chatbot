# 🔍의도 분류기

KoBERT 모델을 사용하여 사용자 의도를 이해하고 분류하는 챗봇 프로젝트입니다.

## 프로젝트 구조

### 🧹 데이터 전처리
- `preprocess_intent_data.py`  
  - 원본 의도 데이터셋을 정제(cleaning)하여 `intent_dataset_cleaned.csv`로 저장  
  - 결측치 제거, 중복 제거, 텍스트 정제 등의 전처리 작업 수행

- `intent_dataset.csv`  
  - 전처리가 되지 않은 데이터셋

### 🔧 학습용
- `train_kobert.ipynb`  
  - KoBERT 모델을 학습하는 Jupyter Notebook  
  - 학습 데이터 불러오기, 전처리, 학습, 시각화, 모델 저장 포함

- `intent_dataset_cleaned.csv`  
  - 질문(question)과 의도(intent)가 포함된 학습용 데이터셋
- 
- `intent_keywords.csv`  
  - 학습용 키워드 데이터셋


### 🚀 배포용
- `inference.py`  
  - 저장된 모델과 라벨 인코더를 사용하여 문장의 의도를 예측하는 배포용 코드

- `best_kobert_model.pt`  
  - 학습된 KoBERT 모델 파라미터 (가장 정확도가 높은 모델)

- `label_encoder.pkl`  
  - 라벨 인코딩 정보를 저장한 파일 (예측 결과 해석에 필요)

### 📦 기타
- `requirements.txt`  
  - 프로젝트 실행에 필요한 패키지 목록



## 실행하기

### 설정

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

### 사용법

1.  **데이터 전처리:**

    이 스크립트는 `intent_dataset.csv` 파일을 정리하여 `intent_dataset_cleaned.csv` 파일을 생성합니다.

    ```bash
    python preprocess_intent_data.py
    ```
    >    📥 입력: `intent_dataset.csv`

    >    💾 출력: `intent_dataset_cleaned.csv`

   2.  **의도 분류 모델 학습 (Jupyter Notebook 실행)**

       Jupyter 환경에서 train_kobert.ipynb를 실행하여 KoBERT 모델을 학습하고, 가장 정확도가 높은 모델을 저장합니다.
       학습 과정 중 정확도 및 confidence 시각화도 함께 수행됩니다.

        >    📥 입력: `intent_dataset_cleaned.csv`
    
        >    💾 출력:
        > - `best_kobert_model.pt` : 최고 성능 모델 파라미터
        >   - `label_encoder.pkl` : 클래스 이름과 인덱스 정보를 포함하는 인코더
        >   - 학습 그래프: 학습 손실, 검증 정확도, confidence 시각화
    
   3.  **모델 추론 (배포용 코드 실행)**

       ```bash
       python inference.py
       ```
       >    📥 입력: 사용자 입력 문장
       > 
       >    💾 출력: 예측된 intent 라벨 및 confidence score
