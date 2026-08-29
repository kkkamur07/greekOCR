# Parallel image builds with persistent local cache.
#
#   cd infrastructure
#   docker buildx bake --allow=fs.read=..
#   docker compose -f docker-compose.yml up
#
# Run bake from this directory, not the repository root. Unlike Compose, bake
# resolves the `..` contexts below against the working directory rather than
# against this file, and the entitlement grant is what lets it read outside that
# directory at all. `--print` will happily render a plan for an invocation that
# cannot build, so it does not verify either point.
#
# Compose reuses the tagged images; bake builds both targets at once.

variable "APP_VERSION" {
  default = "0.3.3"
}

variable "NEXT_PUBLIC_API_BASE_URL" {
  default = "http://localhost:8000"
}

group "default" {
  targets = ["api", "frontend-dev"]
}

target "api" {
  context    = ".."
  dockerfile = "nomikos/Dockerfile"
  target     = "runtime"
  tags       = ["nomikos-api:latest"]
  args = {
    APP_VERSION = APP_VERSION
  }
  cache-from = ["type=local,src=.docker-cache/api"]
  cache-to   = ["type=local,dest=.docker-cache/api,mode=max"]
}

target "frontend-dev" {
  context    = "../nomikos"
  dockerfile = "frontend/Dockerfile"
  target     = "dev"
  tags       = ["nomikos-frontend:latest"]
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
  tags     = ["nomikos-frontend:prod"]
}
