### 의도 처리 흐름
1. classify_intent: 질문을 받으면 챗봇이 1차로 의도와 슬롯을 분류합니다.
2. route_after_initial_classification: 이 함수가 슬롯 그룹을 분석하여 복합 의도인지, 아니면 LLM에게 검증을 맡길 단일 의도인지 판단합니다.
  - len(specific_groups) > 1 이 조건이 만족하면, 복합 의도라고 판단하고 handle_complex_intent 노드로 보냅니다.
  - 그렇지 않으면, llm_verify_intent 노드로 보내서 LLM이 의도를 확정하도록 합니다.
3. llm_verify_intent: 이 노드는 위 단계에서 넘어온 단일 의도를 받아서, 해당 의도가 정말 맞는 의도인지 검증합니다.

#### 모호한 질문의 기준
- 단일 질문에서 confidenct 점수가 0.7 이하인 경우 -> 확인 절차


#### 1. 전체 아키텍처 개요
- 사용자의 질문이 들어오면, 챗봇은 다음과 같은 단계로 답변을 생성합니다.

1. **상태 초기화**: ChatState 객체를 생성하여 사용자 입력, 의도, 슬롯 등 대화 정보를 저장
2. **의도 분류 및 라우팅**: classifier를 통해 사용자의 의도를 파악하고, router를 통해 해당 의도에 맞는 핸들러(Handler)로 대화 흐름을 전환
3. **핸들러 실행**: 특정 의도에 맞는 핸들러(예: congestion.py, facility.py)가 실행/ 이 핸들러는 RAG(검색 증강 생성) 기술을 활용하여 답변을 만듭니다.
4. **RAG 파이프라인**: 핸들러는 쿼리 임베딩, 벡터 검색, LLM 호출 과정을 거쳐 최종 답변을 생성
5. **상태 업데이트**: 핸들러가 생성한 답변으로 ChatState를 업데이트하고, 사용자에게 응답을 보냄

<br/>

#### 2. 주요 컴포넌트 및 로직 흐름 상세
`main.py`
- 역할: 챗봇의 시작점입니다.
- 로직:
  - ChatState를 초기화합니다.
  - router와 flow를 호출하여 사용자 입력을 처리합니다.
---
**의도 분류 및 슬롯 추출 로직**
- `graph/nodes/classifiy_intent.py`: 이 파일은 챗봇이 사용자의 질문을 처음 받았을 때, 질문의 의도(예: 항공편, 혼잡도)를 예측하고, 질문에 포함된 중요한 정보(예: 터미널 번호, 항공사 이름)를 슬롯(Slot)으로 추출하는 역할을 합니다.

- `graph/utils/kobert_classifier.py`: classifiy_intent.py가 사용하는 실제 의도 분류 모델이 정의된 파일입니다. KoBERT 모델을 사용하여 한국어 텍스트를 분석하고, 의도와 슬롯 정보를 반환하는 기능을 수행합니다.

- 이 두 파일은 함께 작동하여 사용자의 의도를 정확히 파악하고, 이어질 핸들러가 질문의 핵심 정보를 활용할 수 있도록 준비하는 역할을 합니다.
---
**복합 질문 처리 로직**
- `graph/nodes/complex_handler.py`: 이 파일은 하나의 쿼리에 여러 의도나 질문이 포함된 경우를 처리하는 로직을 담고 있습니다. 
  - 예를 들어, "대한항공 전화번호랑 인천공항 혼잡도 알려줘"와 같은 질문을 받았을 때, complex_handler.py가 이를 두 개의 개별 질문으로 분리하고, 각각의 핸들러(e.g., airline_info_handler, congestion_handler)를 호출하여 답변을 통합하는 역할을 할 수 있습니다.
---
`graph/router.py`
- 역할: classifiy_intent의 결과를 바탕으로 대화의 다음 단계(즉, 어떤 핸들러를 호출할지)를 결정합니다.
- 로직:
  - ChatState에 저장된 의도(intent)에 따라 미리 정의된 핸들러 함수(예: facility_guide_handler, airport_info_handler)를 호출합니다.
---
`graph/handlers/` 디렉토리
- 역할: 각 의도에 대한 구체적인 답변 생성 로직을 담당합니다.
- 핵심 로직:
  - 대부분의 핸들러는 RAG(검색 증강 생성)를 사용합니다.
  - 복합 질문 처리: 특히 `facility.py`와 `congestion.py` 핸들러는 classifiy_intent에서 추출된 슬롯(예: 여러 시설명, 터미널 번호)을 활용하여 단일 의도 내의 여러 질문을 개별적으로 처리하도록 설계되었습니다.
    - 예: "1터미널과 2터미널 혼잡도 알려줘" -> slots에서 '1터미널'과 '2터미널'을 추출 -> 각 터미널에 대한 혼잡도 정보를 따로 검색 -> 두 결과를 하나의 응답으로 통합합니다.
  - 외부 API 연동: congestion.py 핸들러는 외부 공공 데이터 API를 호출하여 실시간 혼잡도 예측 데이터를 가져옵니다.

(추후 복합 질문 처리와 외부 API 하나씩 연동해야됨)

---
`rag/ `디렉토리
- 역할: RAG 파이프라인의 핵심 기능을 제공합니다.
- 주요 파일:
  - `config.py`: MongoDB, OpenAI API 키 등 RAG에 필요한 각종 설정과 LLM 호출을 위한 공통 함수(common_llm_rag_caller)를 정의합니다.
  - `utils.py`: 쿼리 임베딩(get_query_embedding) 및 MongoDB 벡터 검색(perform_vector_search)과 같은 핵심적인 기능을 수행합니다.
  - `llm_tools.py`: RAG 파이프라인 내에서 특정 정보를 추출하기 위해 LLM을 호출하는 보조 함수(예: extract_location_with_llm)를 포함합니다.
---
#### 3. 전체 로직 흐름 요약
1. 사용자 입력 발생 (main.py)
2. router.py에서 kobert_classifier.py를 호출하여 의도(intent)와 슬롯(slots)을 파악 
3. router.py가 의도에 맞는 핸들러 함수(예: facility_guide_handler in facility.py)로 라우팅
4. 핸들러는 slots 정보를 바탕으로 rag/ 디렉토리의 함수를 호출
- rag/utils.py로 쿼리를 임베딩
- rag/utils.py로 MongoDB에서 관련 문서 검색
- rag/config.py의 common_llm_rag_caller로 최종 답변 생성
5. 핸들러가 생성한 답변을 ChatState에 담아 반환
6. main.py가 최종 답변을 사용자에게 전달

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
