FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[platform]"

COPY backend ./backend
COPY infrastructure ./infrastructure

ENV PYTHONPATH=/app

EXPOSE 8000
