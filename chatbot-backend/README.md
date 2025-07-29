# 🤖 AI 스마트 공항 서비스 - 백엔드

인천국제공항 이용객을 위한 AI 챗봇 서비스의 백엔드 서버입니다.

## ✨ 주요 기능

* **일반/소셜 로그인**: JWT 기반의 안전한 사용자 인증 시스템.
* **실시간 AI 챗봇**: 항공편, 주차, 시설 등 공항 이용에 대한 실시간 질의응답.
* **문의/건의 게시판**: 사용자의 의견을 수렴하고 관리자가 답변하는 기능.
* **관리자 대시보드**: 사용자 문의, 지식 데이터 등을 관리하는 시스템.

## 🛠️ 기술 스택

* **Backend**: Java 21, Spring Boot 3.x, Spring Security, Spring Data JPA, WebSocket
* **Database**: PostgreSQL, MongoDB, Redis
* **DevOps & Tools**: Docker, Docker Compose, Gradle, JWT

## 🚀 시작하기

### 전제 조건

* Git
* JDK 21
* Docker Desktop

### 실행 방법

1.  **Git 저장소 복제**

    ```bash
    git clone [저장소_URL]
    cd [프로젝트_폴더]
    ```

2.  **환경 변수 설정 (가장 중요\!)**
    프로젝트의 민감한 정보는 `.env` 파일로 관리합니다. 이 파일은 Git에 포함되지 않으므로, 예시 파일을 복사하여 직접 생성해야 합니다.

    * 프로젝트 최상위 폴더에 있는 `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.
    * `.env` 파일을 열고, 아래 항목에 자신의 로컬 환경에 맞는 값을 입력합니다.
        * `DB_USERNAME`: enter\_your\_db\_username
        * `DB_PASSWORD`: enter\_your\_db\_password
        * `GOOGLE_CLIENT_ID`: (선택 사항) 구글 로그인 API를 테스트할 경우 자신의 클라이언트 ID 입력
        * `GOOGLE_CLIENT_SECRET`: (선택 사항) 구글 로그인 API를 테스트할 경우 자신의 클라이언트 시크릿 입력

3.  **데이터베이스 실행**
    프로젝트에 필요한 PostgreSQL, MongoDB, Redis를 Docker Compose로 한 번에 실행합니다.

    ```bash
    docker-compose up -d
    ```

    * `-d` 옵션은 컨테이너들을 백그라운드에서 실행합니다.

4.  **백엔드 애플리케이션 실행**

    * IntelliJ 또는 선호하는 IDE에서 프로젝트를 엽니다.
    * `src/main/java/com/incheonai/chatbotbackend/` 경로에 있는 `ChatbotBackendApplication.java` 파일을 찾아 `main` 메소드를 실행합니다.
    * 서버는 기본적으로 `8080` 포트에서 실행됩니다.

### API 테스트

* **Swagger UI**: 애플리케이션 실행 후, 웹 브라우저에서 `http://localhost:8080/swagger-ui/index.html` 로 접속하면 모든 API 문서를 확인하고 직접 테스트할 수 있습니다.

* **Postman**:

    * **회원가입**:
        * **Method**: `POST`
        * **URL**: `http://localhost:8080/api/users/signup`
        * **Body** (`raw`, `JSON`):
          ```json
          {
              "userId": "testuser",
              "password": "password1234"
          }
          ```