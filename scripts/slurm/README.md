# SLURM helpers

Scripts for running pipette work on a SLURM cluster as detached `sbatch`
jobs, so nothing breaks when you log out. For a single
`pipette-ctl run --plan` driving SLURM itself, use the slurm transports
in [plan-runner.md](../../docs/pipette-ctl/plan-runner.md#slurm-transports).

- `schedule`: reads command lines from stdin and submits one `sbatch`
  job per line, skipping blank lines and `#` comments. A failed
  submission is logged and the rest continue. Resources and job naming
  are env vars listed in the script header.
- `run-plans.sh`: prints one `pipette-ctl run --include-failed --shard
  i/N` line per shard for each plan given as `PLAN[:N]`. Its flags:
  `--shards N` (the shard count for a plan without `:N`, default 1),
  `--work-dir DIR` (passed to `pipette-ctl --work-dir`) and `--bin PATH`
  (the `pipette-ctl` binary, default `./target/release/pipette-ctl`).
- `interactive.sh`: opens an interactive shell on a GPU node with the
  CUDA toolkit loaded and scratch dirs moved off the compute nodes'
  full `/tmp`.
- `install-runtime.sbatch`: meant to install a uv runtime from the bundled
  torch-oai catalog on a compute node. As written it calls a
  `pipette-torch-oai` binary that this repository does not build, so it
  fails. Run `pipette runtimes pull --runtime
  'uv-vllm://server=<version>&build=<build>&python=<python>'` on the compute
  node instead (see [torch-oai](../../docs/pipette-cli/torch-oai.md)).

`interactive.sh` and `install-runtime.sbatch` are written for one cluster:
they request partition `defq`, load that cluster's `slurm`, `cuda12.9` and
`python312` environment modules, and expect the checkout at
`$HOME/pipette-clients`. Edit those values for another cluster.

## Run incomplete plans in parallel

```sh
./scripts/slurm/run-plans.sh --work-dir <work-dir> \
    <plan-a.toml>:6 \
    <plan-b.toml>:4 \
  | PARTITION=gpu GPUS=1 CPUS=8 ./scripts/slurm/schedule
```

Each shard records its own state, so `pipette-ctl status` shows the
aggregate. Shards are stable as cells complete, so re-launching the same
shards picks up only what is left. Do not set `SERIAL=1` here: it chains
the jobs one after another and defeats the sharding. Preview one shard's
work with:

```sh
pipette-ctl --work-dir <work-dir> \
    commands --plan <plan.toml> --state outstanding --shard 0/6
```

See [plan-runner.md](../../docs/pipette-ctl/plan-runner.md#commands) for
`--shard` and `--state`.

## Raw per-cell jobs

`pipette-ctl commands --plan <plan.toml> | ./scripts/slurm/schedule`
submits one job per missing cell running the `pipette` client directly.
Results still sync, but the plan's `state.jsonl` is not updated, so
`status` does not change. Use `run-plans.sh` when status should track
the work.
