#!/usr/bin/env bash
# Open an interactive Slurm shell on one defq node with N GPUs.
#
# Usage: ./scripts/slurm/interactive.sh [GPUS [HOURS]]
#   GPUS  number of GPUs to request (default: 1)
#   HOURS wall-time limit in hours    (default: 2)
#
# Sources the modules profile, loads slurm + CUDA 12.9 toolkit, redirects
# scratch dirs off the full /tmp, then srun --pty bash -l.
#
# Why each piece:
#   * cuda12.9 toolkit  — vLLM JIT-compiles some kernels (LFM2/Mamba,
#     Qwen3 GDN) at server startup; without nvcc / CUDA_HOME the engine
#     dies with `Could not find nvcc and default cuda_home='/usr/local/cuda'`.
#   * TMPDIR redirect   — /tmp on compute nodes is ~128 G total with only
#     ~700 M free; nvcc writes preprocessed .ii files there and the
#     ninja build fails with "No such file or directory" once tmp fills.
#     Redirecting to $HOME (NFS, multi-TB free) avoids that.
#   * UV_CACHE_DIR      — same /tmp pressure for uv venv installs.
#   * HF_HOME           — keep model snapshots inside the pipette
#     workspace so `pipette` picks them up via its auto-mount.
#
# srun's default `--export=ALL` carries these env vars into the
# compute-node shell.
#
# Run from the head node.
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
