# BILLIARD COACH 서버 — 영구 클라우드 배포용
# (Render / Railway / Fly.io 등 Docker 지원 호스팅에서 사용)
FROM python:3.12-slim

# opencv-python-headless 런타임 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY detection ./detection
COPY shots ./shots
COPY main.py .

ENV PYTHONIOENCODING=utf-8
EXPOSE 8000

# Render 등은 PORT 환경변수로 포트를 지정한다
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
