FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    default-jdk \
    locales \
    && rm -rf /var/lib/apt/lists/*

# ko_KR.UTF-8 로케일 생성
RUN sed -i '/ko_KR.UTF-8/s/^# //g' /etc/locale.gen && locale-gen

ENV LANG=ko_KR.UTF-8
ENV LANGUAGE=ko_KR:ko
ENV LC_ALL=ko_KR.UTF-8

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sentencepiece --prefer-binary

COPY . .

EXPOSE 8000

CMD ["gunicorn", "chatbot_core.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "8", "--timeout", "120"]

