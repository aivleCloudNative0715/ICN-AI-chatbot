## 디렉토리 구조
```
ai
 ┣ chatbot
 ┃ ┣ embedding
 ┃ ┃ ┣ airline_embed_data.py
 ┃ ┃ ┣ airportprocedure_embed_data.py
 ┃ ┃ ┣ country_embed_data.py
 ┃ ┃ ┗ facility_embed_data.py
 ┃ ┣ graph
 ┃ ┃ ┣ handlers
 ┃ ┃ ┃ ┣ congestion.py
 ┃ ┃ ┃ ┣ default.py
 ┃ ┃ ┃ ┣ facility.py
 ┃ ┃ ┃ ┣ flight.py
 ┃ ┃ ┃ ┣ greeting.py
 ┃ ┃ ┃ ┣ parking.py
 ┃ ┃ ┃ ┣ policy.py
 ┃ ┃ ┃ ┣ transfer.py
 ┃ ┃ ┃ ┣ weather.py
 ┃ ┃ ┃ ┗ __init__.py
 ┃ ┃ ┣ nodes
 ┃ ┃ ┃ ┗ classifiy_intent.py
 ┃ ┃ ┣ utils
 ┃ ┃ ┃ ┗ kobert_classifier.py
 ┃ ┃ ┣ flow.py
 ┃ ┃ ┣ router.py
 ┃ ┃ ┗ state.py
 ┃ ┣ rag
 ┃ ┃ ┣ config.py
 ┃ ┃ ┣ utils.py
 ┃ ┃ ┗ __init__.py
 ┃ ┣ main.py
 ┃ ┗ __init__.py
 ┣ chatbot_app
 ┃ ┣ migrations
 ┃ ┃ ┗ __init__.py
 ┃ ┣ admin.py
 ┃ ┣ apps.py
 ┃ ┣ models.py
 ┃ ┣ tests.py
 ┃ ┣ urls.py
 ┃ ┣ views.py
 ┃ ┗ __init__.py
 ┣ chatbot_core
 ┃ ┣ asgi.py
 ┃ ┣ settings.py
 ┃ ┣ urls.py
 ┃ ┣ wsgi.py
 ┃ ┗ __init__.py
 ┣ docs
 ┃ ┣ API.md
 ┃ ┣ chatbot_architecture.md
 ┃ ┣ intent_classifier.md
 ┃ ┗ rag.md
 ┣ intent_classifier
 ┃ ┣ best_models
 ┃ ┃ ┗ intent-kobert-v1
 ┃ ┃ ┃ ┣ best_kobert_model.pt
 ┃ ┃ ┃ ┣ label_encoder.pkl
 ┃ ┃ ┃ ┗ train_kobert_report.ipynb
 ┃ ┣ inference.py
 ┃ ┣ preprocess_intent_data.py
 ┃ ┣ README.md
 ┃ ┗ train_kobert.ipynb
 ┣ .env
 ┣ .gitignore
 ┣ db.sqlite3
 ┣ intent_dataset_cleaned.csv
 ┣ keyword_boost.csv
 ┣ manage.py
 ┣ requirements.txt
 ┗ __init__.py
```

### `manage.py`
- Django 프로젝트를 관리하고 다양한 작업을 수행하는 데 사용되는 명령어들의 집합
- makemigrations, migrate, runserver등 실행 가능

---
### `chatbot_core/ 폴더`
- Django 프로젝트 폴더

#### `settings.py`
- Django 프로젝트의 모든 설정을 담고 있는 핵심 설정 파일
- 최상위 디렉토리, 디버그 여부(개발단계에서만), 서버 호스트, 앱, URL, 데이터베이스 설정 

#### `urls.py`
- Django 프로젝트에서 URL 라우팅을 담당

#### `wsgi.py`
- Django 앱과 웹 서버를 연결하는 진입점 설정 파일 동기(Synchronous) 통신을 지원하는 객체 생성

#### `asgi.py`
- Django 앱과 웹 서버를 연결하는 진입점 설정 파일 비동기(Asynchronous) 통신을 지원하는 객체 생성

---
### `chatbot_app/ 폴더`
- Django 앱 폴더

#### `migrations/ 폴더`
- 데이터베이스 스키마(스키마) 변경 기록을 담고 있는 디렉토리

#### `admin.py`
- 앱의 데이터 모델을 Django 관리자(admin) 페이지에 등록

#### `apps.py`
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
