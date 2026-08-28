#!/usr/bin/env bash
# VRAM smoke test: submit to the idle RTX 3090 node (cc-gpu-n03).
# Usage: sbatch scripts/cluster/smoke_test.sh
#
#SBATCH --job-name=smoke-test
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=all
#SBATCH --nodelist=cc-gpu-n03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail

# ── Ensure uv and user-local binaries are on PATH ────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"

# ── Locate repo root ─────────────────────────────────────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"

command -v uv >/dev/null 2>&1 || { echo "Error: uv not on PATH" >&2; exit 127; }

export PYTHONUNBUFFERED=1

# ── Print environment info ────────────────────────────────────────────────────
echo "================================================================"
echo "Repo     : ${REPO_ROOT}"
echo "Job      : ${SLURM_JOB_ID:-not-slurm}"
echo "Node     : $(hostname)"
echo "CVDs     : ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || true
echo "================================================================"

# ── Background VRAM monitor (logs every 2 s) ──────────────────────────────
VRAM_LOG="slurm-vram-${SLURM_JOB_ID:-local}.log"
nvidia-smi dmon -s mu -d 2 -o DT > "${VRAM_LOG}" &
MONITOR_PID=$!
trap "kill ${MONITOR_PID} 2>/dev/null || true; echo 'VRAM log: ${VRAM_LOG}'" EXIT

# ── Run the smoke test ────────────────────────────────────────────────────────
exec uv run --extra train python scripts/smoke_test.py
