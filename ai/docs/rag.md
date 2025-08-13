### 의도 처리 흐름
1. classify_intent: 질문을 받으면 챗봇이 1차로 의도와 슬롯을 분류합니다. 이 단계에서 여러 의도가 감지될 수도 있고, 단일 의도의 확신도 점수가 함께 계산됩니다.
2. route_after_initial_classification: 이 함수가 챗봇의 다음 단계를 결정하는 핵심 라우터입니다.
   - 이전 대화 감지: 만약 사용자의 질문이 이전 대화에 이어서 나온 것이라면, 질문의 맥락을 고려하기 위해 llm_verify_intent 노드로 보냅니다.
   - 복합 의도 판단: classify_intent가 복합 의도를 감지하거나, state에 복합 의도 정보가 이미 있는 경우 handle_complex_intent 노드로 보냅니다.
   - 단일 의도 판단: 위 조건에 해당하지 않고, state에 의도 정보가 있다면 가장 높은 점수의 의도에 맞는 핸들러로 바로 라우팅합니다.
   - 모호하거나 신뢰도가 낮은 경우: 위 모든 조건에 해당하지 않을 경우, 의도 분류가 모호하거나 신뢰도가 낮다고 판단하고 llm_verify_intent 노드로 보내서 LLM이 의도를 재검증하도록 합니다.
3. llm_verify_intent: 이 노드는 모호하거나 이전 대화의 맥락이 있는 질문을 받아 LLM에게 의도를 재검증받고, 확정된 의도를 바탕으로 핸들러로 라우팅합니다.
4. handle_complex_intent: 이 노드는 복합 의도 질문을 받아 LLM을 통해 질문을 개별 의도로 분해하고, 각 의도에 맞는 핸들러를 호출하여 답변을 통합합니다.

---
### 1. 전체 아키텍처 개요
- 사용자의 질문이 들어오면, 챗봇은 다음과 같은 단계로 답변을 생성합니다.
1. **상태 초기화**: ChatState 객체를 생성하여 사용자 입력, 의도, 슬롯 등 대화 정보를 저장합니다.
2. **의도 분류 및 라우팅**: classifier를 통해 사용자의 의도를 파악하고, router를 통해 해당 의도에 맞는 핸들러(Handler)로 대화 흐름을 전환합니다. 이 단계에서 복합 의도, 모호한 의도, 단일 의도를 구분하여 최적의 경로를 찾습니다.
3. **핸들러 실행**: 특정 의도에 맞는 핸들러가 실행됩니다. 이 핸들러는 RAG(검색 증강 생성) 기술을 활용하여 답변을 만듭니다.
4. **RAG 파이프라인**: 핸들러는 쿼리 임베딩, 벡터 검색, Python 기반 필터링, LLM 호출 과정을 거쳐 최종 답변을 생성합니다.
5. **상태 업데이트**: 핸들러가 생성한 답변으로 ChatState를 업데이트하고, 사용자에게 응답을 보냅니다.

### 2. 주요 컴포넌트 및 로직 흐름 상세
`main.py`
- 역할: 챗봇의 시작점입니다.
- 로직:
  - ChatState를 초기화합니다.
  - router와 flow를 호출하여 사용자 입력을 처리합니다.

**의도 분류 및 슬롯 추출 로직**
- `graph/nodes/classifiy_intent.py`: 이 파일은 챗봇이 사용자의 질문을 처음 받았을 때, 질문의 의도(예: 항공편, 혼잡도)를 예측하고, 질문에 포함된 중요한 정보(예: 터미널 번호, 항공사 이름)를 슬롯(Slot)으로 추출하는 역할을 합니다.
- `graph/utils/kobert_classifier.py`: classifiy_intent.py가 사용하는 실제 의도 분류 모델이 정의된 파일입니다. KoBERT 모델을 사용하여 한국어 텍스트를 분석하고, 의도와 슬롯 정보를 반환하는 기능을 수행합니다.
- 이 두 파일은 함께 작동하여 사용자의 의도를 정확히 파악하고, 이어질 핸들러가 질문의 핵심 정보를 활용할 수 있도록 준비하는 역할을 합니다.

**복합 질문 처리 로직**
- `graph/nodes/complex_handler.py`: 이 파일은 하나의 쿼리에 여러 의도나 질문이 포함된 경우를 처리하는 로직을 담고 있습니다.
  - 질문 분해: LLM을 사용하여 "대한항공 전화번호랑 인천공항 혼잡도 알려줘"와 같은 질문을 `_decompose_and_classify_queries` 함수를 통해 두 개의 개별 질문으로 분리합니다.
  - 개별 핸들러 실행: 분해된 각 질문에 대해 해당 핸들러(예: airline_info_handler, parking_congestion_handler)를 호출합니다.
  - 답변 통합 및 정리: 각 핸들러의 답변을 받은 후, 중복되는 주의 문구(Disclaimer)를 제거하고, 최종 답변을 하나로 통합합니다.
 
**`graph/handlers/` 디렉토리**
- 역할: 각 의도에 대한 구체적인 답변 생성 로직을 담당합니다.
- 핵심 로직:
  - RAG 활용: 대부분의 핸들러는 RAG를 사용하여 외부 데이터(예: MongoDB, 공공 API)를 검색하고 답변을 생성합니다.
  - 상세 필터링: facility_guide_handler는 '제2터미널'과 같은 위치 키워드를 추출하여 직접 필터링을 수행합니다.
  - API 맞춤 연동: flight_info_handler는 API 명세에 따라 항공편의 방향(departure / arrival)과 상대 공항 코드의 유무를 파악하여 API 호출 방식을 동적으로 변경합니다.
  - 의도 구분: 총 18개의 의도

**`rag/` 디렉토리**
- 역할: RAG 파이프라인의 핵심 기능을 제공합니다.
- 주요 파일:
  - `config.py`: MongoDB, OpenAI API 키 등 RAG에 필요한 각종 설정과 LLM 호출을 위한 공통 함수(common_llm_rag_caller)를 정의합니다.
  - `utils.py`: 쿼리 임베딩(get_query_embedding) 및 MongoDB 벡터 검색(perform_vector_search)과 같은 핵심적인 기능을 수행합니다.
  - `llm_tools.py`: RAG 파이프라인 내에서 특정 정보를 추출하기 위해 LLM을 호출하는 보조 함수(예: _extract_facility_names_with_llm, _parse_flight_query_with_llm)를 포함합니다. 여기서 LLM에게 명확한 지침을 줘서 의도 및 정보 추출 오류를 방지합니다.

### 3. 전체 로직 흐름 요약
1. 사용자 입력 발생 (main.py)
2. `classify_intent` 호출: 의도와 슬롯을 파악합니다.
3. `router.py`가 라우팅 결정: classify_intent 결과에 따라 다음 노드를 결정합니다.
    - 단일 의도 → 해당 핸들러로 직행
    - 복합 의도 → complex_handler.py로 이동
    - 모호한 의도(confident < 0.7) → llm_verify_intent.py로 이동
4. 핸들러 실행: 라우팅된 핸들러가 실행됩니다.
5. RAG 파이프라인 실행:
    - `rag/utils.py`로 쿼리를 임베딩합니다.
    - rag/utils.py로 MongoDB에서 관련 문서를 검색합니다.
    - rag/config.py의 `common_llm_rag_caller`로 최종 답변을 생성합니다.
6. 핸들러가 답변 반환: 생성된 답변을 ChatState에 담아 반환합니다.
7. main.py가 최종 답변 전달: 최종 답변을 사용자에게 전달합니다.

---
##### 실행 과정
가상 환경 생성 및 활성화
```
# 가상 환경 생성
python -m venv .venv

# 가상 환경 활성화 (macOS/Linux)
source .venv/bin/activate

# 가상 환경 활성화 (Windows)
.venv\Scripts\activate
```
필요한 패키지 설치

```
pip install -r requirements.txt
```
환경 변수 설정
- 프로젝트 루트 디렉토리에 .env 파일을 생성, 필요 API 키와 정보 입력

```
MONGO_URI = "MONGO_URI 넣기"
MONGO_DB_NAME="AirBot"
EMBEDDING_MODEL_PATH="dragonkue/snowflake-arctic-embed-l-v2.0-ko"
OPENAI_API_KEY="노션에 올라가있는 api key 넣기"
SERVICE_KEY = "공공데이터 활용 신청 후 decoding key 넣기"
```