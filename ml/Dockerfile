FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml

RUN python - <<'PY'
import subprocess
import sys
import tomllib

with open("/app/pyproject.toml", "rb") as handle:
    metadata = tomllib.load(handle)

deps = metadata["dependency-groups"]["ml"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *deps])
PY

COPY ml /app/ml

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "ml.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
