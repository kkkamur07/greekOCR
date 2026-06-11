FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Third-party deps only — no local package build; app code copied after install.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-install-project --group platform --system

COPY backend ./backend
COPY infrastructure ./infrastructure

ENV PYTHONPATH=/app

EXPOSE 8000
