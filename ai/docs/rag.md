## 디렉토리 구조
```
chatbot/
├── main.py
├── graph/
│   ├── handlers/
│   │   ├── congestion.py
│   │   ├── default.py
│   │   ├── facility.py     # 시설 안내 핸들러
│   │   ├── flight.py       # 항공편/항공사 핸들러
│   │   ├── policy.py       # 입국/출국 정책 핸들러
│   │   ├── transfer.py     # 환승 절차 핸들러
│   │   └── ... (각 의도별 핸들러)
├── embedding/
│   ├── airline_embed_data.py
│   ├── airportprocedure_embed_data.py
│   ├── country_embed_data.py
│   └── facility_embed_data.py
└── rag/
    ├── airline_info.py
    ├── arrival_policy.py
    ├── client.py
    ├── departure_policy.py
    ├── facility_guide.py
    └── transfer_policy.py
```

### `embedding/ 폴더`
- MongoDB 컬렉션에 저장된 원본 텍스트 데이터를 text_embedding(텍스트의 벡터 표현)으로 변환하여 추가하는 스크립트들을 담은 폴더
#### `airline_embed_data.py`
- `Airline` 컬렉션의 항공사 이름, 연락처, 코드 등의 정보를 기반으로 임베딩을 생성하고 업데이트

#### `airportprocedure_embed_data.py`
- `AirportProcedure` 컬렉션의 절차 유형, 단계 이름, 설명 등을 기반으로 임베딩을 생성하고 업데이트합니다. (예: 입국, 출국, 환승 절차)

#### `country_embed_data.py`
- `Country` 컬렉션의 국가 이름, 비자 요구 사항 등의 정보를 기반으로 임베딩을 생성하고 업데이트

#### `facility_embed_data.py`
- `AirportFacility` (공항 시설) 및 `AirportEnterprise` (입점업체) 컬렉션의 시설/업체 이름, 위치, 운영 시간, 서비스 설명 등의 정보를 기반으로 임베딩을 생성하고 업데이트
---
### `rag/ 폴더`
- 특정 정보 도메인에 특화된 RAG 검색 로직을 구현하며, client.py를 활용하여 실제 MongoDB Atlas에서 데이터를 검색

#### `client.py`
- rag 폴더의 모든 모듈이 MongoDB Atlas와 SentenceTransformer 모델을 효율적으로 사용하기 위한 중앙 유틸리티 파일
- MongoDB 연결 관리, 임베딩 모델 로딩, MongoDB Atlas Vector Search를 수행하는 핵심 query_vector_store 함수를 제공
- 각 컬렉션의 벡터 인덱스 이름을 매핑하는 VECTOR_INDEX_NAMES를 정의

#### `airline_info.py`
- 항공사 관련 쿼리(예: "대한항공 연락처 뭐야?")를 처리하기 위한 RAG 로직을 포함
- client.py를 사용하여 Airline 컬렉션에서 관련 정보를 검색

#### `arrival_policy.py`
- 공항 입국 절차 관련 쿼리(예: "입국 심사 어떻게 해?")를 처리하기 위한 RAG 로직을 포함
- client.py를 사용하여 AirportProcedure 컬렉션에서 procedure_type이 '입국'인 문서를 필터링하여 검색

#### `departure_policy.py`
- 공항 출국 절차 관련 쿼리(예: "출국할 때 보안 심사 받아야 해?")를 처리하기 위한 RAG 로직을 포함
- client.py를 사용하여 AirportProcedure 컬렉션에서 procedure_type이 '출국'인 문서를 필터링하여 검색

#### `facility_guide.py`
- 공항 시설 및 입점업체 관련 쿼리(예: "약국 어디 있어?", "스타벅스 운영 시간 알려줘?")를 처리하기 위한 RAG 로직을 포함
- client.py를 사용하여 AirportFacility와 AirportEnterprise 두 컬렉션에서 동시에 정보를 검색하여 결합

#### `transfer_policy.py`
- 환승 절차 관련 쿼리(예: "환승 절차 알려줘?")를 처리하기 위한 RAG 로직을 포함
- client.py를 사용하여 AirportProcedure 컬렉션에서 procedure_type이 '환승'인 문서를 필터링하여 검색
---
### 수정한 `handlers/ 폴더`
#### `facility.py`
- rag/facility_guide.py의 get_facility_guide_info 함수를 호출하여 관련 시설/업체 정보를 검색

#### `flight.py`
- rag/airline_info.py의 get_airline_info 함수를 호출하여 항공사 정보를 검색

#### `policy.py`
- rag/arrival_policy.py
- rag/departure_policy.py의 해당 함수를 호출하여 정책 정보를 검색

#### `transfer.py`
- rag/transfer_policy.py의 get_transfer_policy_info 함수를 호출하여 환승 절차 정보를 검색

---
### 실행 흐름
1. 시작 (main.py)
2. 의도 분류
3. 라우팅
4. 핸들러 실행 (예: facility_guide_handler)
- langgraph 라우터가 facility_guide_handler 노드로 제어권을 넘김
- facility_guide_handler는 ChatState에서 user_input을 가져옴
5. get_facility_guide_info 호출 (ai/chatbot/rag/facility_guide.py)
- facility_guide_handler는 get_facility_guide_info(user_input) 함수를 호출
- 이 함수 내부에서:
  - ai/chatbot/rag/client.py의 get_model()을 통해 임베딩 모델을 로드
  - user_input을 임베딩 벡터로 변환
  - ai/chatbot/rag/client.py의 query_vector_store() 함수를 호출하여 실제 MongoDB Atlas에서 AirportFacility 및 AirportEnterprise 컬렉션에 대한 벡터 검색을 수행
  - query_vector_store()는 검색 결과를 get_facility_guide_info에 반환
  - get_facility_guide_info는 이 결과를 combined_results에 담아 facility_guide_handler로 반환
6. ChatState 업데이트 (facility_guide_handler)
- facility_guide_handler는 get_facility_guide_info로부터 반환된 retrieved_docs (검색된 문서 리스트)를 받음
- 이 retrieved_docs 값을 updated_state 딕셔너리에 담아 ChatState의 retrieved_documents 필드를 업데이트

---
facility_guide_handler ( ai/chatbot/graph/handlers/facility.py 내)<br>
⬇️ 호출<br>
get_facility_guide_info ( ai/chatbot/rag/facility_guide.py 내)<br>
⬇️ 호출<br>
query_vector_store ( ai/chatbot/rag/client.py 내)<br>
⬇️ 검색 결과를 반환<br>
get_facility_guide_info ( ai/chatbot/rag/facility_guide.py 내)<br>
⬇️ 처리된(결합된) 결과를 반환<br>
facility_guide_handler ( ai/chatbot/graph/handlers/facility.py 내)<br>
⬇️ <br>
최종 결과를 ChatState에 업데이트하고 반환
