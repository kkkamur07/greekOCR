FROM python:3.11-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --only-group inference


FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml /app/pyproject.toml

COPY inference /app/inference
COPY src/hf /app/src/hf

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app

RUN groupadd --system appuser \
  && useradd --system --gid appuser --home-dir /home/appuser --create-home --shell /usr/sbin/nologin appuser \
  && mkdir -p /app/inference/weights/cache \
  && chown -R appuser:appuser /app/inference/weights/cache

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()"

CMD ["uvicorn", "inference.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
