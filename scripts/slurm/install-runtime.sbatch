#!/usr/bin/env bash
#SBATCH --job-name=pipette-install-runtime
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log
#
# Install a pipette-torch-oai runtime from the bundled uv catalog.
#
# Usage:
#   sbatch ~/pipette-clients/scripts/slurm/install-runtime.sbatch \
#       [SLUG] [-- EXTRA_UV_FLAGS]
#
# Example:
#   sbatch ~/pipette-clients/scripts/slurm/install-runtime.sbatch \
#       vllm@0.22.0+cu129.py3.12
#
# Default SLUG is vllm@0.22.0+cu129.py3.12 (matches driver 580 / H100).
# Logs land next to the submit dir as `<job-name>-<jobid>.log`.
#
# Why a Slurm job (instead of running on the head node):
#   - head node has no GPU and may have stricter networking;
#   - compute nodes are where the runtime will actually be used, so the
#     uv cache warms up where it matters.
#
# IMPORTANT: /tmp on compute nodes is full (~128G, <1G free). We redirect
# TMPDIR + UV_CACHE_DIR to $HOME (NFS, ~8T free) before invoking uv.

set -euo pipefail

SLUG="${1:-vllm@0.22.0+cu129.py3.12}"
shift || true   # drop SLUG; remaining args (if any) pass through

# ---- environment ----
source /etc/profile.d/modules.sh
module load slurm/slurm/24.11 python312

export PATH="$HOME/.cargo/bin:$PATH"   # picks up existing uv
mkdir -p "$HOME/.cache/uv" "$HOME/.tmp" "$HOME/.cache/huggingface"
export UV_CACHE_DIR="$HOME/.cache/uv"
export TMPDIR="$HOME/.tmp"
export HF_HOME="$HOME/.cache/huggingface"

cd "$HOME/pipette-clients"

# ---- diagnostics for the log ----
echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-n/a}"
echo "[$(date -Is)] slug=${SLUG}"
echo "[$(date -Is)] uv=$(uv --version 2>&1)"
echo "[$(date -Is)] python3.12=$(python3.12 --version 2>&1)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
df -h "$HOME" "$TMPDIR" | tail -2

# ---- the install ----
./target/release/pipette-torch-oai runtimes install-uv-catalog \
    "${SLUG}" \
    --uv-install-flags "--index-strategy unsafe-best-match" \
    "$@"

echo "[$(date -Is)] installed; current runtimes:"
./target/release/pipette-torch-oai runtimes list
