FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Third-party deps only — no local package build; app code copied after install.
COPY requirements-platform.txt ./
RUN pip install --no-cache-dir -r requirements-platform.txt

COPY backend ./backend
COPY infrastructure ./infrastructure

ENV PYTHONPATH=/app

EXPOSE 8000
