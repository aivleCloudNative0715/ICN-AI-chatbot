# ICN-AI-Chatbot 🤖

KoBERT 모델을 사용하여 사용자 의도를 이해하고 분류하는 챗봇 프로젝트입니다.

## 프로젝트 구조

- `ai/`: AI 모델, 학습 스크립트 및 데이터가 포함되어 있습니다.

## AI 모델 (`ai/`)

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

2.  **의도 분류 모델 학습:**

    이 스크립트는 전처리된 데이터를 사용하여 KoBERT 모델을 학습합니다.

    ```bash
    python kobert_intent_trainer.py
    ```

### 파일 설명

-   `intent_dataset.csv`: 사용자 질문과 의도에 대한 원시 데이터셋입니다.
-   `intent_dataset_cleaned.csv`: 학습 준비가 완료된 정리된 데이터셋입니다.
-   `preprocess_intent_data.py`: 원시 데이터를 정리하고 전처리하는 스크립트입니다.
-   `kobert_intent_trainer.py`: KoBERT 기반 의도 분류 모델을 학습하는 스크립트입니다.
-   `requirements.txt`: 이 프로젝트에 필요한 Python 패키지 목록입니다.
