
# 챗봇 아키텍처 문서

이 문서는 ICN AI 챗봇의 아키텍처를 설명합니다.

## 개요

챗봇은 `langgraph`를 기반으로 한 상태 그래프(State Graph)를 사용하여 사용자 입력을 처리하고 응답을 생성합니다. 전체적인 흐름은 다음과 같습니다.

1.  **의도 분류 (Intent Classification)**: 사용자의 입력을 받아 KoBERT 모델을 사용하여 의도를 분류합니다.
2.  **라우팅 (Routing)**: 분류된 의도에 따라 적절한 핸들러(Handler)로 작업을 라우팅합니다.
3.  **핸들러 실행 (Handler Execution)**: 각 의도에 맞는 핸들러가 실행되어 최종 응답을 생성합니다.

## 디렉토리 구조

```
chatbot/
├── main.py
└── graph/
    ├── __init__.py
    ├── flow.py
    ├── router.py
    ├── state.py
    ├── handlers/
    │   ├── congestion.py
    │   ├── default.py
    │   └── ... (각 의도별 핸들러)
    ├── nodes/
    │   └── classifiy_intent.py
    └── utils/
        └── kobert_classifier.py
```

## 주요 파일 및 구성 요소

### `main.py`

-   챗봇 애플리케이션의 진입점(Entry Point)입니다.
-   `build_chat_graph()` 함수를 호출하여 `langgraph` 기반의 챗봇 그래프를 생성합니다.
-   사용자 입력을 받아 그래프를 실행하고 최종 응답을 출력합니다.

### `graph/state.py`

-   `ChatState` TypedDict를 정의합니다.
-   `ChatState`는 그래프의 상태를 나타내며, `user_input`, `intent`, `response` 등의 정보를 포함합니다.

### `graph/nodes/classifiy_intent.py`

-   `classify_intent` 노드를 정의합니다.
-   이 노드는 그래프의 시작점(Entry Point)입니다.
-   `KoBERTPredictor`를 사용하여 사용자의 입력(`user_input`)으로부터 의도(`intent`)를 예측하고, 그 결과를 `ChatState`에 저장합니다.

### `graph/utils/kobert_classifier.py`

-   `KoBERTClassifier`: `skt/kobert-base-v1` 모델을 기반으로 하는 PyTorch 모델 클래스입니다.
-   `KoBERTPredictor`: 미리 학습된 KoBERT 모델과 라벨 인코더를 로드하여, 주어진 텍스트의 의도를 예측하는 `predict` 메서드를 제공합니다.
    -   📌**중요**: `KoBERTPredictor`가 올바르게 작동하려면, 학습된 모델 파일(`best_kobert_model.pt`)과 라벨 인코더 파일(`label_encoder.pkl`)이 `intent_classifier/best_models/intent-kobert-v1/` 디렉토리 내에 위치해야 합니다.

### `graph/router.py`

-   `route_by_intent` 함수를 정의합니다.
-   `classify_intent` 노드에서 예측된 의도(`intent`)에 따라 다음에 실행할 핸들러 노드를 결정합니다.
-   의도 이름에 `_handler`를 붙여 해당 핸들러 노드의 이름을 반환합니다.

### `graph/handlers/`

-   각 의도에 대한 핸들러 함수들을 포함하는 디렉토리입니다.
-   각 파일은 특정 의도와 관련된 핸들러들을 정의합니다. (예: `parking.py`, `flight.py`)
-   핸들러 함수의 이름은 반드시 `_handler`로 끝나야 합니다. (예: `parking_fee_info_handler`)
-   각 핸들러는 `ChatState`를 입력으로 받아, 처리 후 `response`를 포함한 `ChatState`를 반환합니다.

### `graph/flow.py`

-   `build_chat_graph` 함수를 정의하여 전체 `langgraph`를 구성합니다.
-   `StateGraph(ChatState)`를 사용하여 그래프 빌더를 초기화합니다.
-   `handlers` 디렉토리 내의 모든 `_handler`로 끝나는 함수들을 동적으로 찾아 그래프 노드로 추가합니다.
-   `classify_intent`를 시작점으로 설정하고, `route_by_intent`를 사용하여 조건부 엣지(Conditional Edge)를 추가합니다.
-   최종적으로 컴파일된 그래프를 반환합니다.

## 실행 흐름

1.  `main.py`에서 사용자 입력과 함께 `chat_graph.invoke()`가 호출됩니다.
2.  `classify_intent` 노드가 실행되어 사용자의 의도를 예측합니다.
3.  `route_by_intent` 라우터가 예측된 의도에 따라 다음 노드를 결정합니다. (예: 의도가 "parking_fee_info"이면 "parking_fee_info_handler" 노드로 라우팅)
4.  해당 의도 핸들러(예: `parking_fee_info_handler`)가 실행되어 응답을 생성하고 `ChatState`를 업데이트합니다.
5.  핸들러 노드는 `END`로 연결되어 있어, 실행이 완료되면 그래프 실행이 종료되고 최종 `ChatState`가 반환됩니다.
