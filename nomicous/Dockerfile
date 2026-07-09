FROM python:3.11-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/
COPY nomicous/README.md nomicous/VERSION /app/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --only-group platform-prod


FROM python:3.11-slim AS runtime

ARG APP_VERSION=0.3.2
LABEL org.opencontainers.image.title="nomicous-api" \
      org.opencontainers.image.version="${APP_VERSION}"

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml uv.lock /app/
COPY nomicous/README.md nomicous/VERSION /app/
COPY nomicous/backend /app/backend
COPY nomicous/infrastructure /app/infrastructure
COPY scripts/platform /app/scripts/platform
COPY inference/__init__.py /app/inference/__init__.py
COPY inference/contracts /app/inference/contracts

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.core.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
