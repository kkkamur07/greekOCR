FROM python:3.12-slim

ARG APP_VERSION=0.3.2
LABEL org.opencontainers.image.title="annote-api" \
      org.opencontainers.image.version="${APP_VERSION}"

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends libpq-dev gcc fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

COPY README.md VERSION /app/
COPY backend/pyproject.toml /app/backend/pyproject.toml

RUN python - <<'PY'
import subprocess
import sys
import tomllib

with open("/app/backend/pyproject.toml", "rb") as handle:
    metadata = tomllib.load(handle)

dependencies = metadata["project"]["dependencies"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *dependencies])
PY

COPY backend /app/backend
COPY infrastructure /app/infrastructure

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.core.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
