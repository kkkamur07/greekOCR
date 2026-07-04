FROM python:3.11-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen --only-group ml


FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml /app/pyproject.toml

COPY ml_service /app/ml_service
COPY _support_repo/calamari /app/_support_repo/calamari

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app

RUN groupadd --system appuser \
  && useradd --system --gid appuser --home-dir /home/appuser --create-home --shell /usr/sbin/nologin appuser \
  && mkdir -p /app/ml_service/weights/cache \
  && chown -R appuser:appuser /app/ml_service/weights/cache

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()"

CMD ["uvicorn", "ml_service.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
