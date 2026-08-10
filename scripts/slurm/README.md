# SLURM helpers

Two small scripts for running `pipette-plan` work on a SLURM cluster.

## `schedule`

Reads command lines from stdin and submits one `sbatch` job per line.
Blank lines and `#` comments are skipped. Tunables are env vars
(`PARTITION`, `ACCOUNT`, `TIME`, `GPUS`, `CPUS`, `MEM`, `JOB_PREFIX`,
`LOG_DIR`, `SERIAL`, `DRY_RUN`, `EXTRA_SBATCH_ARGS`). See the script
header. A failed submission is logged and skipped; the rest continue.

## `run-plans.sh`

Generates sharded `pipette-plan run` command lines (missing + failed
cells, one line per shard) for one or more plans. Pipe it into
`schedule` to run them as parallel jobs.

## Recommended workflow: run incomplete plans in parallel

Each shard records its own state, so `pipette-plan status` shows the
aggregate afterward. Shards are disjoint and stable, so already-complete
cells are skipped and the same shards can be re-launched to mop up
what's left.

```sh
./scripts/slurm/run-plans.sh --work-dir /home/yuri/pipette-clients \
    internal-pipette-plans/release_v1/gen6.qwen3.5-4b.toml:6 \
    internal-pipette-plans/release_v1/gen6.qwen3.5-2b.toml:4 \
  | PARTITION=gpu GPUS=1 CPUS=8 JOB_PREFIX=gen6 ./scripts/slurm/schedule
```

Do **not** set `SERIAL=1` here. That chains jobs sequentially and
defeats the point of sharding. Preview a shard's exact work first with:

```sh
pipette-plan --work-dir /home/yuri/pipette-clients \
    commands --plan <plan.toml> --state all --shard 0/6
```

## Raw per-cell jobs (no plan state)

`pipette-plan commands --plan <plan.toml> | ./scripts/slurm/schedule`
submits one job per cell running the raw client. It syncs results to the
warehouse but does **not** update the plan's `state.jsonl`, so `status`
won't change. Prefer `run-plans.sh` when you want status to track it.
