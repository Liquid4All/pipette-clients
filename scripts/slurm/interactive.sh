#!/usr/bin/env bash
# Open an interactive Slurm shell on one defq node with N GPUs.
#
# Usage: ./scripts/slurm/interactive.sh [GPUS [HOURS]]
#   GPUS  number of GPUs to request (default: 1)
#   HOURS wall-time limit in hours    (default: 24)
# Run from the head node; srun's default --export=ALL carries the env below
# into the compute-node shell.
#
#   * CUDA toolkit: vLLM JIT-compiles some kernels (LFM2/Mamba, Qwen3 GDN) at
#     startup and dies with "Could not find nvcc" without it.
#   * TMPDIR, UV_CACHE_DIR: compute-node /tmp has ~700 M free; nvcc's .ii files
#     fill it and ninja then fails with "No such file or directory". $HOME (NFS)
#     has room.
#   * HF_HOME: keep model snapshots in the pipette workspace's model dir.
set -euo pipefail

GPUS="${1:-1}"
HOURS="${2:-24}"

source /etc/profile.d/modules.sh
module load slurm/slurm/24.11 cuda12.9/toolkit/12.9.1

mkdir -p "$HOME/.tmp" "$HOME/.cache/uv" "$HOME/pipette-clients/.pipette/models"
export TMPDIR="$HOME/.tmp"
export UV_CACHE_DIR="$HOME/.cache/uv"
export HF_HOME="$HOME/pipette-clients/.pipette/models"

exec srun \
    --partition=defq \
    --gres=gpu:"${GPUS}" \
    --nodes=1 \
    --time="${HOURS}:00:00" \
    --job-name="interactive-${USER}" \
    --pty bash -l
