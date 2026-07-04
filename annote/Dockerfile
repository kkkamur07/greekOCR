FROM python:3.12-slim

ARG APP_VERSION=0.3.2
LABEL org.opencontainers.image.title="annote-api" \
      org.opencontainers.image.version="${APP_VERSION}"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/

RUN apt-get update \
  && apt-get install -y --no-install-recommends libpq-dev gcc fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/
COPY annote/README.md annote/VERSION /app/

RUN uv sync --frozen --only-group platform

COPY annote/backend /app/backend
COPY annote/infrastructure /app/infrastructure

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.core.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
