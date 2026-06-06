FROM python:3.12-slim

ARG APP_VERSION=0.1.0
LABEL org.opencontainers.image.title="annote-backend" \
      org.opencontainers.image.version="${APP_VERSION}"

WORKDIR /app/backend

COPY README.md /app/README.md
COPY VERSION /app/VERSION
COPY backend/pyproject.toml ./
COPY backend/annote ./annote

RUN pip install --no-cache-dir . \
  && python -c "import annote"

ENV ANNOTE_HOST=0.0.0.0
ENV ANNOTE_PORT=5050
ENV ANNOTE_DATA_ROOT=/data
ENV ANNOTE_RELOAD=false

EXPOSE 5050

CMD ["annote"]
