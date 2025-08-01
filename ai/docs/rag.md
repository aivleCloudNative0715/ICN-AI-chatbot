## 동작 흐름
사용자 질문은 다음과 같은 단계를 거쳐 최종 답변으로 변환됩니다.

1. 사용자 입력: main 파일로 사용자의 질문이 들어옵니다.
2. 의도 및 슬롯 감지: 입력된 질문의 의도(Intent)와 핵심 정보(Slots)를 분석합니다.
3. 라우팅: flow.py의 라우터가 분석된 정보를 바탕으로 질문이 단일 의도인지 복합 의도인지 판단합니다.
4. 핸들러 실행: 라우터의 결정에 따라 각 의도에 맞는 핸들러(단일 또는 복합)가 실행됩니다.
5. 답변 생성: 핸들러는 RAG(검색 증강 생성) 시스템을 통해 최종 답변을 생성하고 사용자에게 반환합니다.

### 핵심 구성 요소
- `main.py`
챗봇의 실행 진입점입니다. 사용자의 질문을 받아 LangGraph 상태(State)를 초기화하고, 정의된 그래프를 실행하는 역할을 합니다.

- `flow.py`
챗봇의 가장 중요한 라우터(Router) 로직을 담고 있습니다. route_to_complex_or_single 함수가 핵심이며, 이 함수는 의도 및 슬롯 분석 결과를 기반으로 다음 실행 노드를 결정합니다.
- 동작 방식: 슬롯에 감지된 슬롯 그룹의 개수를 기준으로 판단합니다.
    - 단일 의도: parking, flight_info 등 특정 슬롯 그룹이 하나만 감지되면, 해당 의도에 맞는 단일 핸들러(예: parking_fee_info_handler)로 바로 라우팅합니다.
    - 복합 의도: parking과 facility_info처럼 두 개 이상의 특정 슬롯 그룹이 감지되면, handle_complex_intent 노드로 라우팅합니다.

- `complex_handler.py`
복합 의도 질문을 처리하는 노드입니다. 라우터가 복합 의도로 판단했을 때만 실행됩니다.
    - handle_complex_intent 함수:
        1. 질문 분해: _split_intents 함수를 호출하여 "주차 요금이랑 카페 위치 알려줘" 같은 질문을 "주차 요금 알려줘", "카페 위치 알려줘"와 같이 독립적인 하위 질문으로 분해합니다.
        2. 서브그래프 실행: 분해된 각 하위 질문을 LangGraph의 서브그래프로 보내어 다시 처음부터 의도 분류 및 단일 핸들러 실행 과정을 거치게 합니다.
        3. 답변 종합: 각 하위 질문에 대한 답변을 모아 하나의 최종 답변으로 종합하여 반환합니다.

### 상세 동작 시나리오
- **시나리오 1: 단일 의도 질문 (주차 요금이 얼마인지 알려줘)**
이 질문은 parking 슬롯 그룹만 감지되므로 단일 의도로 처리됩니다.
1. 입력: main.py에 "주차 요금이 얼마인지 알려줘" 입력.
2. 의도/슬롯 감지: parking_fee_info 의도와 B-parking_type, B-parking_lot 등의 슬롯이 감지됩니다.
3. 라우팅: flow.py의 라우터가 parking 슬롯 그룹만 있음을 확인하고 **parking_fee_info_handler**로 라우팅합니다.
4. 핸들러 실행: parking_fee_info_handler가 직접 RAG 시스템을 통해 주차 요금 정보를 검색하고 답변을 생성합니다.

- **시나리오 2: 복합 의도 질문 (주차 요금이랑 카페 위치 알려줘)**
이 질문은 parking과 facility_info 두 슬롯 그룹이 감지되므로 복합 의도로 처리됩니다.
1. 입력: main.py에 "주차 요금이랑 카페 위치 알려줘" 입력.
2. 의도/슬롯 감지: parking_fee_info와 facility_info 관련 슬롯이 감지됩니다.
3. 라우팅: flow.py의 라우터가 두 개 이상의 슬롯 그룹을 확인하고 handle_complex_intent 노드로 라우팅합니다.
4. 핸들러 실행:
    - complex_handler.py의 _split_intents가 질문을 "주차 요금 알려줘", "카페 위치 알려줘"로 분해합니다.
    - 각 질문은 다시 classify_intent부터 시작하는 서브그래프를 통해 parking_fee_info_handler, facility_info_handler로 각각 전달됩니다.
    - 두 핸들러의 답변이 생성되면, complex_handler가 이 답변들을 종합하여 하나의 최종 응답을 만듭니다.

---

## 디렉토리 구조
```
chatbot/
├── main.py                   # 챗봇 애플리케이션의 메인 진입점
├── graph/                    # 챗봇의 대화 흐름(LangGraph) 및 의도별 핸들러 관리
│   ├── handlers/             # 각 사용자 의도를 처리하는 핸들러 함수 모음
│   │   ├── parking.py        # 주차 핸들러
│   │   ├── facility.py       # 시설 안내 핸들러
│   │   ├── flight.py         # 항공편/항공사 핸들러
│   │   ├── policy.py         # 입국/출국 정책 핸들러
│   │   └── transfer.py       # 환승 절차 핸들러
└── rag/ 
    ├── config.py             # RAG 검색 설정 (컬렉션, 인덱스 매핑 등)
    └── utils.py              # RAG 검색에 필요한 공통 유틸리티 (DB 연결, 임베딩, 벡터 검색 등)
```

### `handlers/ 폴더`
- 각 핸들러는 주로 rag/ 폴더의 유틸리티를 활용하여 필요한 데이터를 검색하고 가공하여 사용자에게 답변을 제공합니다.

- parking.py: 주차 요금, 주차장 위치 추천, 주차장-터미널 간 도보 시간 등 주차 관련 사용자 질의를 처리합니다.
- facility.py: 공항 내 특정 시설(식당, 약국, 라운지 등)의 위치, 운영 시간, 제공 서비스 등 시설 안내 관련 질의를 처리합니다.
- flight.py: 항공편 정보(운항 스케줄, 지연/결항 여부)나 항공사 고객센터 연락처 등 항공편 및 항공사 관련 질의를 처리합니다.
- policy.py: 공항의 입국 절차, 출국 절차, 수하물 규정(제한 물품 등) 등 공항 정책 관련 질의를 처리합니다.
- transfer.py: 공항 내 환승 경로, 환승 절차, 최소 환승 시간 등 환승 절차 관련 질의를 처리합니다.
---
### `rag/ 폴더`
- 챗봇의 RAG(Retrieval Augmented Generation) 기능 구현을 위한 공통 설정과 유틸리티 함수들을 모아둔 곳입니다.

- config.py:
    - RAG 검색에 필요한 설정 정보를 정의합니다.
    - 각 핸들러(의도)가 어떤 MongoDB 컬렉션(collection_name), 어떤 벡터 인덱스(vector_index_name)를 사용해야 하는지, 그리고 어떤 추가 필터링(query_filter)이 필요한지 등을 매핑해 둡니다.
    - common_llm_rag_caller와 같이 모든 RAG 핸들러에서 공통으로 사용할 LLM 호출 함수의 기본적인 로직을 포함합니다.

- utils.py:
    - RAG 검색 프로세스에서 필요한 하위 레벨의 공통 유틸리티 함수들을 제공합니다.
    - MongoDB 연결 관리(get_mongo_client, get_mongo_collection, close_mongo_client), 임베딩 모델 로딩(get_embedding_model), 사용자 쿼리 임베딩 생성(get_query_embedding), 그리고 MongoDB Atlas에서 벡터 검색을 수행하는(perform_vector_search, perform_multi_collection_search) 핵심 기능들이 이곳에 구현되어 있습니다.
---
