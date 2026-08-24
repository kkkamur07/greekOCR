# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/
COPY nomikos/README.md nomikos/VERSION /app/

# `--only-group` is deliberate: this image must NOT pick up
# [project].dependencies (numpy, onnxruntime, opencv-python-headless). The
# platform API imports none of them: it only needs inference.contracts and
# inference.admission, which are pure pydantic, so installing the full set
# would add hundreds of megabytes of unused ML wheels. If the backend ever
# imports one of them, add it to the platform-prod group explicitly.
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --only-group platform-prod


FROM python:3.11-slim AS runtime

ARG APP_VERSION=0.3.3
LABEL org.opencontainers.image.title="nomikos-api" \
      org.opencontainers.image.version="${APP_VERSION}"

WORKDIR /app

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y --no-install-recommends fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/* \
  && groupadd --system appuser \
  && useradd --system --gid appuser --home-dir /home/appuser --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml uv.lock /app/
COPY nomikos/README.md nomikos/VERSION /app/
COPY nomikos/backend /app/backend
COPY nomikos/infrastructure /app/infrastructure
COPY scripts/platform /app/scripts/platform
COPY inference /app/inference

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app

RUN mkdir -p /app/backend/media \
  && chown -R appuser:appuser /app/backend/media /home/appuser \
  && JWT_SECRET=container-build-validation-secret \
     INFERENCE_WEBHOOK_SECRET=container-build-validation-secret \
     python -c "from backend.core.main import app; assert app.title" \
  && /usr/local/bin/python -m pip uninstall -y pip setuptools wheel \
  && rm -rf /usr/local/lib/python3.11/ensurepip

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "backend.core.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
