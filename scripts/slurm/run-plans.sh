#!/usr/bin/env bash
set -euo pipefail

# Emit sharded `pipette-plan run` command lines (one per shard) to stdout.
# Pipe into ./scripts/slurm/schedule to submit them as parallel SLURM jobs.
#
# Each shard runs only its disjoint slice of the missing + failed cells
# (`run --include-failed --shard i/N`) and records state, so a plain
# `pipette-plan status` reflects completion. Sharding is by position in
# the full ordered matrix, so already-complete cells keep their slot and
# are simply skipped — re-running the same shards only picks up the work
# that is left.
#
# Usage:
#   run-plans.sh [--shards N] [--work-dir DIR] [--bin PATH] PLAN[:N] ...
#
#   PLAN[:N]     plan TOML path; optional :N overrides the shard count for
#                that plan (small plans want fewer shards, big ones more)
#   --shards N   default shard count for plans without :N (default 1)
#   --work-dir   passed to `pipette-plan --work-dir` (default: omitted/cwd)
#   --bin PATH   pipette-plan binary (default ./target/release/pipette-plan)
#
# Example — the incomplete gen6 plans, then submit on the GPU partition:
#   ./scripts/slurm/run-plans.sh --work-dir /home/yuri/pipette-clients \
#       internal-pipette-plans/release_v1/gen6.gemma-4-e2b-it.toml:2 \
#       internal-pipette-plans/release_v1/gen6.gemma-4-e4b-it.toml:4 \
#       internal-pipette-plans/release_v1/gen6.qwen3.5-0.8b.toml:4 \
#       internal-pipette-plans/release_v1/gen6.qwen3.5-2b.toml:4 \
#       internal-pipette-plans/release_v1/gen6.qwen3.5-4b.toml:6 \
#     | PARTITION=gpu GPUS=1 CPUS=8 JOB_PREFIX=gen6 ./scripts/slurm/schedule

BIN=./target/release/pipette-plan
WORK_DIR=""
DEFAULT_SHARDS=1
plans=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shards) DEFAULT_SHARDS=$2; shift 2 ;;
        --work-dir) WORK_DIR=$2; shift 2 ;;
        --bin) BIN=$2; shift 2 ;;
        -h | --help) sed -n '4,33p' "$0" >&2; exit 0 ;;
        --*) echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
        *) plans+=("$1"); shift ;;
    esac
done

if [[ ${#plans[@]} -eq 0 ]]; then
    echo "no plans given (see --help)" >&2
    exit 2
fi

for entry in "${plans[@]}"; do
    plan=${entry%%:*}
    shards=$DEFAULT_SHARDS
    [[ "$entry" == *:* ]] && shards=${entry##*:}
    if [[ ! "$shards" =~ ^[1-9][0-9]*$ ]]; then
        echo "invalid shard count '$shards' for $plan" >&2
        exit 2
    fi
    for ((i = 0; i < shards; i++)); do
        echo "$BIN${WORK_DIR:+ --work-dir $WORK_DIR} run --plan $plan --include-failed --shard $i/$shards"
    done
done
