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
---
## 사용법
0. 가상환경 활성화 및 의존성 설치
```bash
.venv\Scripts\activate
pip install -r RAG_requirements.txt
```

1. 로컬 Redis 서버 설치
WSL 안에서
```bash
sudo apt update
sudo apt install redis-server
sudo service redis-server start
redis-cli
```

1.5 (필요시) redis-cli로 나오는 주소가 127.0.0.1:6379가 아니라면
`ai/chatbot_core/settings.py`의 CACHES 파라미터 중 LOCATION을 자신의 redis-cli 주소/1로 변경  

2. 다음의 코드 실행
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
3. http://127.0.0.1:8000/chatbot/generate 로 챗봇 응답 POST 요청
    - Body 예시
    ```bash
    {
    "session_id": "test-user-1234",
    "message_id": "1",
    "parent_id": null,
    "user_id": "1234",
    "content": "주차 요금 안내해줘",
    "context": ""
    }
    ```
    - 응답 예시
    ```bash
    {
    "answer": "수하물은 도착층에서 찾을 수 있습니다.",  # 실제 챗봇 응답으로 변경
    "metadata": {
        "source": "retrieval",
        "confidence": 0.92      
    }
    }
    ```
4. http://127.0.0.1:8000/chatbot/recommend 로 추천 질문 POST 요청
    - Body 예시
    ```bash
    {
    "message_id": "1",
    "user_id": "1234",
    "content": "주차 요금 안내해줘"
    }
    ```
    - 응답 예시
    ```bash
    {
    "recommended_questions": [
        "수하물 분실 시 어떻게 하나요?",
        "환승 시 수하물은 자동으로 옮겨지나요?",
        "국내선 수하물 규정은 어떻게 되나요?"
    ]
    }
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
- 현재 캐싱은 기본적인 메모리 캐시 사용중
- 나중에 Redis로 변경 고려

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
- 앱의 설정(configuration)을 정의하는 파일. 앱의 이름('chatbot_app')이 이곳에 정의
- settings.py의 INSTALLED_APPS에 'chatbot_app.apps.ChatbotAppConfig'를 추가하여 앱을 활성화할 때 사용

#### `models.py`
- 앱의 핵심 데이터 구조를 정의하는 파일.
- 현재는 대화 기록을 캐싱해서 저장하므로, 별도 DB 사용 안 함

#### `tests.py`
- 앱의 기능을 테스트하는 단위 코드를 작성하는 파일

#### `views.py`
- 사용자의 요청을 받아 처리하고 응답을 반환하는 View 함수 또는 클래스를 작성
- GenerateAPIView : 백엔드 서버에서 POST 요청이 오면, 챗봇 응답을 보내는 클래스. 캐시를 활용해서 세션 id별로 채팅 구분
- RecommendAPIView : POST 요청이 오면 content 필드를 의도분류해서, 해당하는 추천 질문을 n개 보내는 클래스

#### `urls.py`
- 앱의 로직(views)에 접근 가능한 URL 경로 설정 파일

---
### 수정한 것들
- 챗봇의 대화 기록 저장을 위해서 `ChatState`에 `messages`라는 필드 추가
- 최상위 디렉토리를 `ai`로 수정함에 따라서 `chatbot`내 파일들의 import를 ai.chatbot.에서 -> chatbot. 으로 변경함


---
### 실행 흐름
1. 백엔드 서버에서 POST 요청 (views.py)
2. 필드 데이터 추출 
3. `session_id`로 대화 내용 캐싱
- `chatbot_session_{session_id}`를 key로 캐시에 저장된 대화 내용을 불러옴
- 캐시에 해당 세션이 없으면 새로운 `ChatState`생성.
4. `chat_graph` 실행
- 의도 분류 + MongoDB에서 문서 검색 + LLM 응답
- 새로운 state를 응답으로 받음
5. 챗봇 응답과 metadata를 꺼내옴.(metadata는 현재 ChatState에 구현이 안 되어서 default값이 됨)
6. 업데이트 된 state를 캐시에 다시 저장
7. Response 생성 



---
### 현재 상황
- LLM이 없어서 의도분류기만 동작 확인
- 질문의 의도를 분류해서, 추천질문 DB에서 가져와 응답하는 로직 추가
- 로컬 Redis 캐시 서버를 사용하도록 변경
- LLM 및 State와 연동 예정(차주)