#!/bin/sh
# Install the locked environment with CPU-only PyTorch, for the Docker image and CI.
#
# On Linux the locked PyTorch is the CUDA build, which pulls in torch, triton and a
# set of nvidia-* and cuda-* packages that together weigh many gigabytes. Neither the
# image nor CI has a GPU, so those packages are skipped and the same torch and
# torchvision versions are installed from the CPU-only index instead.
#
# Arguments are passed to `uv sync`, for example `--no-dev --no-install-project`.
set -eu

LOCKED=$(uv export --locked --format requirements-txt --no-hashes)
SKIP=$(printf '%s\n' "$LOCKED" \
    | grep -oE '^(torch|torchvision|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==' \
    | tr -d '=' | sed 's/^/--no-install-package /')

# --inexact keeps the CPU PyTorch from an earlier run, which the lock does not list.
# shellcheck disable=SC2086
uv sync --locked --inexact $SKIP "$@"

printf '%s\n' "$LOCKED" | grep -E '^(torch|torchvision)==' > /tmp/torch-cpu.txt
uv pip install -r /tmp/torch-cpu.txt --index-url https://download.pytorch.org/whl/cpu
