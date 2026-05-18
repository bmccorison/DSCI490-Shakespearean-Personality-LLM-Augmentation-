# syntax=docker/dockerfile:1

FROM node:20-bookworm-slim AS frontend

WORKDIR /build/interface
COPY interface/package.json interface/package-lock.json ./
RUN npm ci

COPY interface/ ./
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_PORT=8000 \
    PORT=8000 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logging /app/.cache/huggingface

COPY app.py ./
COPY pipeline ./pipeline
COPY models ./models
COPY data ./data
COPY --from=frontend /build/interface/dist ./interface/dist

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

VOLUME ["/app/logging", "/app/.cache/huggingface"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"BACKEND_PORT\", \"8000\")}/api/get_models', timeout=3)" || exit 1

CMD ["python", "app.py"]
