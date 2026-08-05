# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/

# `--only-group` installs ONLY the named group and drops [project].dependencies,
# which is where torch and safetensors are declared -- the image then builds and
# imports the FastAPI app fine but fails every real inference request.
# `--no-default-groups --group inference` installs the project runtime
# dependencies plus the group, while still keeping the `dev` group (ruff,
# pre-commit, pip-audit, poethepoet) out of the runtime image.
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-default-groups --group inference \
  && find /app/.venv -type d -name __pycache__ -prune -exec rm -rf {} + \
  && find /app/.venv -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete


FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/* \
  && groupadd --system appuser \
  && useradd --system --gid appuser --home-dir /home/appuser --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml /app/pyproject.toml

COPY inference /app/inference
# The runtime resolves Hub weights but never publishes them.  Copy the small
# resolver surface explicitly so local caches, staged uploads, and model
# artifacts cannot enter the image through a broad src/hf copy.
COPY src/hf/__init__.py src/hf/paths.py /app/src/hf/
COPY src/hf/resolve /app/src/hf/resolve

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app

# `from inference.api.main import app` alone does NOT exercise the model
# runtime: the architecture modules are imported lazily, so an image without
# Torch still builds and still answers /health. The second check loads Torch and
# both architecture modules, so a missing or broken Torch fails the build
# instead of every inference request. It also asserts CPU-only (ADR 0004): a
# CUDA wheel would multiply the image size for no benefit on the target.
RUN mkdir -p /app/src/hf/cache \
  && chown -R appuser:appuser /app/src/hf/cache /home/appuser \
  && python -c "from inference.api.main import app; assert app.title" \
  && python -c "import torch; import inference.architectures.calamari.adapter; import inference.architectures.blla.blla; assert not torch.version.cuda, torch.version.cuda; print('torch', torch.__version__)" \
  && /usr/local/bin/python -m pip uninstall -y pip setuptools wheel \
  && rm -rf /usr/local/lib/python3.11/ensurepip

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()"

CMD ["uvicorn", "inference.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
