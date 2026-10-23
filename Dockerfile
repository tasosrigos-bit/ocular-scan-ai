# The application image: the FastAPI service and its web page, on CPU-only PyTorch.
#
# The checkpoint, the index and the API key are not baked in. docker-compose.yml
# mounts the checkpoint and the index read-only and passes the key from .env, so the
# image holds code only and contains nothing private.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first, in their own layer, so a code change does not reinstall them.
# sync_cpu.sh installs the lock with CPU-only PyTorch and none of the CUDA packages.
COPY pyproject.toml uv.lock ./
COPY scripts/sync_cpu.sh scripts/sync_cpu.sh
RUN scripts/sync_cpu.sh --no-dev --no-install-project

COPY README.md ./
COPY ocular ocular
COPY app app
COPY scripts scripts
RUN scripts/sync_cpu.sh --no-dev

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
