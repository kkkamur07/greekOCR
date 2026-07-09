FROM node:20-alpine AS dev

WORKDIR /app

ENV VITE_API_BASE_URL=http://localhost:8000

COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
  npm ci

EXPOSE 5173

CMD ["sh", "-c", "rm -rf node_modules/.vite && npm run dev -- --host 0.0.0.0 --force"]


FROM node:20-alpine AS builder

# Keep default in sync with ../VERSION; docker compose passes APP_VERSION from NOMICOUS_VERSION.
ARG APP_VERSION=0.3.2
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_APP_VERSION=$APP_VERSION
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

WORKDIR /app

COPY VERSION ./VERSION
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
  npm ci

COPY frontend/ .

RUN npm run build


FROM nginx:1.27-alpine AS runner

# Keep default in sync with ../VERSION; docker compose passes APP_VERSION from NOMICOUS_VERSION.
ARG APP_VERSION=0.3.2
LABEL org.opencontainers.image.title="nomicous-frontend" \
      org.opencontainers.image.version="${APP_VERSION}"

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 5173

CMD ["nginx", "-g", "daemon off;"]
