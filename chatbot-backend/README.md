# 🤖 AI 스마트 공항 서비스 - 백엔드

인천국제공항 이용객을 위한 AI 챗봇 서비스의 백엔드 서버입니다. 이 문서는 프로젝트의 설정, 실행, 그리고 API 사용법에 대해 안내합니다.

## ✨ 주요 기능

* **통합 인증 시스템**: 로컬 회원가입/로그인 및 구글 소셜 로그인을 지원하며, JWT 토큰 기반으로 안전하게 인증을 관리합니다.
* **실시간 AI 챗봇**: 항공편, 주차, 시설 등 공항 이용에 대한 실시간 질의응답.
* **문의/건의 게시판**: 사용자의 의견을 수렴하고 관리자가 답변하는 기능.
* **관리자 대시보드**: 사용자 문의, 지식 데이터 등을 관리하는 시스템.

## 🛠️ 기술 스택

* **Backend**: Java 21, Spring Boot 3.x, Spring Security, Spring Data JPA, WebSocket
* **Database**: PostgreSQL, MongoDB, Redis
* **DevOps & Tools**: Docker, Docker Compose, Gradle, JWT, OAuth 2.0

## 🚀 시작하기

### 전제 조건

* Git
* JDK 21
* Docker Desktop
* OpenSSL (macOS, Linux, Git Bash for Windows에 기본적으로 포함되어 있습니다.)

### 1. Git 저장소 복제

```bash
git clone [저장소_URL]
cd [프로젝트_폴더]
```

### 2. 환경 변수 설정 (가장 중요!)

프로젝트의 민감한 정보는 `.env` 파일로 관리합니다. 이 파일은 Git에 포함되지 않으므로, 예시 파일을 복사하여 직접 생성해야 합니다.

1.  프로젝트 최상위 폴더에 있는 `.env.example` 파일을 복사하여 `.env` 파일을 생성합니다.

2.  **JWT 시크릿 키 생성**:
    보안을 위해 강력한 JWT 시크릿 키를 생성해야 합니다. 터미널(Git Bash, WSL 등)에서 아래 명령어를 실행하여 64바이트 길이의 랜덤 문자열을 생성하세요.
    ```bash
    openssl rand -hex 64
    ```
    생성된 긴 문자열을 복사하여 `.env` 파일의 `JWT_SECRET_KEY` 값으로 붙여넣습니다.
    ```env
    # .env 파일 예시
    JWT_SECRET_KEY=여기에_생성된_키를_붙여넣으세요
    ```

3.  `.env` 파일의 나머지 항목(DB 계정, 구글 클라이언트 정보 등)을 자신의 환경에 맞게 수정합니다.

### 3. 데이터베이스 실행

Docker Compose로 프로젝트에 필요한 모든 데이터베이스(PostgreSQL, MongoDB, Redis)를 한 번에 실행합니다.

```bash
docker-compose up -d
```
* `-d` 옵션은 컨테이너들을 백그라운드에서 실행합니다.

### 4. 백엔드 애플리케이션 실행

* IntelliJ 또는 선호하는 IDE에서 프로젝트를 엽니다.
* `src/main/java/com/incheonai/chatbotbackend/` 경로에 있는 `ChatbotBackendApplication.java` 파일을 찾아 `main` 메소드를 실행합니다.
* 서버는 기본적으로 `http://localhost:8080` 에서 실행됩니다.

### 5. (선택) 로컬 HTTPS 환경 설정

로컬에서 HTTPS로 테스트하려면, 아래 과정을 따라 SSL 인증서를 생성하고 적용해야 합니다.

1.  **인증서 생성**:
    프로젝트 루트 폴더의 터미널에서 아래 명령어를 실행하여 SSL 인증서를 생성합니다.
    ```bash
    keytool -genkeypair -alias tomcat -keyalg RSA -keysize 2048 -keystore keystore.p12 -storetype PKCS12 -validity 365
    ```
    * 명령어 실행 후, 비밀번호는 `password`로 입력하고 나머지 질문은 Enter 키를 눌러 넘어가세요.
    * 생성된 `keystore.p12` 파일을 `src/main/resources` 폴더로 이동시키세요. **(주의: 이 파일은 `.gitignore`에 등록하여 GitHub에 올리지 마세요!)**

2.  **HTTPS 프로필 활성화**:
    IntelliJ의 실행 구성(`실행` > `실행 구성 편집...`)으로 이동하여, `환경 변수` 필드에 `SPRING_PROFILES_ACTIVE=local-ssl` 을 추가합니다.
    * 이제 서버를 실행하면 `https://localhost:8443` 에서 HTTPS로 실행됩니다.

## 📝 API 명세

애플리케이션 실행 후, `http://localhost:8080/swagger-ui.html` (또는 HTTPS 설정 시 `https://localhost:8443/swagger-ui.html`) 에서 모든 API 문서를 확인하고 직접 테스트할 수 있습니다.

### 인증 (Authentication)

* **아이디 중복 확인**
    * `POST /api/auth/check-id`
* **회원가입 (후 자동 로그인)**
    * `POST /api/users/signup`
* **로그인 (사용자/관리자)**
    * `POST /api/auth/login`
* **로그아웃**
    * `POST /api/users/logout`
    * **Header**: `Authorization: Bearer [JWT_TOKEN]`
* **구글 소셜 로그인**
    * `GET /oauth2/authorization/google`
    * (브라우저에서 직접 호출하면 구글 로그인 페이지로 이동합니다.)
    * **주의**: 구글 클라우드 콘솔의 '승인된 리디렉션 URI'에 자신의 로컬 주소(`http://localhost:8080/login/oauth2/code/google` 또는 `https://localhost:8443/login/oauth2/code/google`)가 등록되어 있어야 합니다.
