FROM python:3.12-slim

WORKDIR /app/backend

COPY README.md /app/README.md
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
