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
