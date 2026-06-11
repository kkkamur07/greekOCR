FROM node:20-alpine AS builder

# Keep default in sync with ../VERSION; docker compose passes APP_VERSION from ANNOTE_VERSION.
ARG APP_VERSION=0.3.2
ENV NEXT_PUBLIC_APP_VERSION=$APP_VERSION

WORKDIR /app

COPY VERSION ./VERSION
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ .

ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:5050
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL

RUN npm run build

FROM node:20-alpine AS runner

# Keep default in sync with ../VERSION; docker compose passes APP_VERSION from ANNOTE_VERSION.
ARG APP_VERSION=0.3.2
LABEL org.opencontainers.image.title="nomicous-frontend" \
      org.opencontainers.image.version="${APP_VERSION}"

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

RUN addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]
