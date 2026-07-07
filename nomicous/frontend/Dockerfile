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

FROM node:20-alpine AS runner

# Keep default in sync with ../VERSION; docker compose passes APP_VERSION from NOMICOUS_VERSION.
ARG APP_VERSION=0.3.2
LABEL org.opencontainers.image.title="nomicous-frontend" \
      org.opencontainers.image.version="${APP_VERSION}"

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=5173

RUN npm install -g serve@14 \
  && addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 vite

COPY --from=builder --chown=vite:nodejs /app/dist ./dist

USER vite

EXPOSE 5173

CMD ["serve", "-s", "dist", "-l", "5173"]
