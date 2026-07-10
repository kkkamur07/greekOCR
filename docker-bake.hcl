# Parallel image builds with persistent local cache.
#
#   docker buildx bake
#   docker compose up
#
# Compose reuses the tagged images; bake builds all three targets at once.

variable "APP_VERSION" {
  default = "0.3.2"
}

variable "NEXT_PUBLIC_API_BASE_URL" {
  default = "http://localhost:8000"
}

group "default" {
  targets = ["api", "inference", "frontend-dev"]
}

target "api" {
  context    = "."
  dockerfile = "nomicous/Dockerfile"
  target     = "runtime"
  tags       = ["nomicous-api:latest"]
  args = {
    APP_VERSION = APP_VERSION
  }
  cache-from = ["type=local,src=.docker-cache/api"]
  cache-to   = ["type=local,dest=.docker-cache/api,mode=max"]
}

target "inference" {
  context    = "."
  dockerfile = "inference/Dockerfile"
  target     = "runtime"
  tags       = ["nomicous-inference:latest"]
  cache-from = ["type=local,src=.docker-cache/inference"]
  cache-to   = ["type=local,dest=.docker-cache/inference,mode=max"]
}

target "frontend-dev" {
  context    = "nomicous"
  dockerfile = "frontend/Dockerfile"
  target     = "dev"
  tags       = ["nomicous-frontend:latest"]
  args = {
    APP_VERSION              = APP_VERSION
    NEXT_PUBLIC_API_BASE_URL = NEXT_PUBLIC_API_BASE_URL
  }
  cache-from = ["type=local,src=.docker-cache/frontend"]
  cache-to   = ["type=local,dest=.docker-cache/frontend,mode=max"]
}

target "frontend-prod" {
  inherits = ["frontend-dev"]
  target   = "runner"
  tags     = ["nomicous-frontend:prod"]
}
