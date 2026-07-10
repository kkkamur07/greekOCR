# syntax=docker/dockerfile:1

FROM node:20-alpine AS dev

WORKDIR /app

ENV NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
  npm ci

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0"]


FROM node:20-alpine AS builder

# Keep default in sync with ../VERSION; docker compose passes APP_VERSION from NOMICOUS_VERSION.
ARG APP_VERSION=0.3.2
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_APP_VERSION=$APP_VERSION
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL

WORKDIR /app

COPY VERSION ./VERSION
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
  npm ci

COPY frontend/ .

RUN --mount=type=cache,target=/app/.next/cache \
  npm run build


FROM node:20-alpine AS runner

# Keep default in sync with ../VERSION; docker compose passes APP_VERSION from NOMICOUS_VERSION.
ARG APP_VERSION=0.3.2
LABEL org.opencontainers.image.title="nomicous-frontend" \
      org.opencontainers.image.version="${APP_VERSION}"

WORKDIR /app
ENV NODE_ENV=production \
    PORT=5173 \
    HOSTNAME=0.0.0.0

RUN addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 5173

CMD ["node", "server.js"]
