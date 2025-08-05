# 베이스 이미지 지정 (Python 3.12)
FROM python:3.12-slim

# 작업 디렉토리 생성
WORKDIR /app

RUN apt-get update

RUN apt-get update && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt RAG_requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r RAG_requirements.txt && \
    pip install sentencepiece --prefer-binary

# 프로젝트 파일 전체 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 컨테이너 시작 시 실행할 명령어
CMD ["gunicorn", "chatbot_core.wsgi:application", "--bind", "0.0.0.0:8000"]