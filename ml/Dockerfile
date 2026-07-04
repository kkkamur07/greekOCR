FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen --only-group ml


FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY ml /app/ml

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app

RUN groupadd --system appuser \
  && useradd --system --gid appuser --home-dir /home/appuser --create-home --shell /usr/sbin/nologin appuser \
  && mkdir -p /app/ml/weights/cache \
  && chown -R appuser:appuser /app/ml/weights/cache

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()"

CMD ["poe", "ml-api"]
