# pipette-plan

Drive a matrix of benchmarks (× models × runtimes) across one or more
remote devices, resumably. A plan is a typed TOML document: models
declare their artifact format, runtimes declare the serving/runtime
kind, and transports point at the runner binary installed on each
target.

This page covers **local dispatch**: driving devices directly. To hand
the matrix to the management server instead, so it leases each cell to a
matching registered client, see
[job generation](job-generation.md) and its scheduler-mode plan format.

## Setup

1. Install `pipette-plan` locally.
2. Install the client binary (`pipette`) on each target device, register
   it with the management server, and install the runtime and benchmarks.
   The plan runner calls the remote binary. It doesn't deploy it. The
   same `pipette` binary serves every runtime; it dispatches on the
   `runtime.type` declared in the plan.
3. Create the local state workspace:

```bash
pipette-plan init
```

## Start from an example

Copy one of these to `plans/my-plan.toml` and edit the host/models/etc.
The comments in each file explain the fields.

- [`examples/plans/llamacpp-android-dual.toml`](../../examples/plans/llamacpp-android-dual.toml):
  llamacpp on two Android devices over ADB.
- [`examples/plans/llamacpp-android-vl.toml`](../../examples/plans/llamacpp-android-vl.toml):
  llamacpp with vision-language GGUF models.
- [`examples/plans/mlx-macstudio.toml`](../../examples/plans/mlx-macstudio.toml):
  mlx on a single Mac over SSH.
- [`examples/plans/torch-oai-vllm-linux.toml`](../../examples/plans/torch-oai-vllm-linux.toml):
  torch-oai with vLLM on one Linux host over SSH.
- [`examples/plans/llamacpp-slurm.toml`](../../examples/plans/llamacpp-slurm.toml):
  llamacpp dispatched as SLURM jobs (`slurm_local`, with a commented
  `slurm_over_ssh` alternative).

Transports supported: `adb`, `ssh`, `local` (spawns the runner binary
directly on the driver host (no SSH/ADB wrap, no sshd required),
`slurm_local` / `slurm_over_ssh` (dispatch each cell as its own `srun`
allocation) see [SLURM transports](#slurm-transports) below), and
`ios` (drive an iOS device from a host Mac; see [iOS transport](#ios-transport)).

Each `[[transports]]` entry can carry `parallelism = N` (default `1`).
`N > 1` spawns `N` concurrent workers against that target: useful
for I/O-bound eval benchmarks. **Do not use on perf benchmarks**:
concurrent execution destroys throughput/latency measurements.

The `ssh` transport accepts a `port = <n>` field (defaults to 22).
The `adb` transport accepts an optional `port = <n>` for the *local
adb server* (`adb -P <port>`): set this when the adb server runs on
another machine reached through an SSH tunnel. Omit to use the adb
default (5037) or whatever `ANDROID_ADB_SERVER_PORT` is set to. The
global `--adb-port <n>` flag overrides every adb transport's `port`
for one invocation without editing the plan.

## Reaching a device through another box

`adb_over_ssh` runs the `adb` command **on an intermediate host** instead
of on the driver: the driver needs neither adb nor a tunnel, only ssh to
the box that holds the pairing keys. This is `adb`'s counterpart the way
`slurm_over_ssh` is `slurm_local`'s, and it suits a phone rig where the
handsets are paired to one controller.

```toml
[[transports]]
type        = "adb_over_ssh"
client_id   = "ev1_…"
host        = "boston-linux-belink"   # the box whose adb server owns the device
user        = "liquid"
serial      = "R3GL30CRBGM"
binary_path = "/data/local/tmp/pipette/pipette"
work_dir    = "/data/local/tmp/pipette"
```

Two ports, deliberately named apart: `port` is the **ssh** port (as on
every ssh-reached transport), `adb_port` the adb server port on `host`.
`--adb-port` does **not** apply. It exists to retarget a tunnel from the
driver, which this transport removes the need for.

`pre_exec` runs on the intermediate host before `adb`, joined with `&&`;
the same role it plays for slurm. Non-interactive ssh skips the login
profile, so a host that only puts `adb` on PATH there needs it:

```toml
pre_exec = "export PATH=$PATH:$HOME/Android/Sdk/platform-tools"
```

The device command is quoted before it crosses ssh, so a cell's JSON
`--model` / `--runtime` descriptors arrive at `adb shell` intact. `shell`
still describes the *device* shell and must be `posix`: `powershell` is
rejected when the plan loads. The **intermediate host** is assumed posix
too, since that is what hosts an adb server; a Windows box running adb
would pass the quotes through literally and garble the command, which
needs a separate `host_shell` field to express.

The host budget keys on the **serial**, not on `host`: the contended
resource is the handset, so several phones behind one controller run
concurrently rather than serializing on the ssh hop.

`ios_over_ssh` applies the same hop to iOS: `xcrun devicectl` only reaches
devices paired with the machine it runs on, so without it a plan with iOS
variants must be driven from that Mac. See [`ios_over_ssh`](#ios_over_ssh).
Scheduler mode ([job-generation.md](job-generation.md)) remains the other
option, where the device claims its own jobs and no intermediate host exists.

## SLURM transports

`slurm_local` and `slurm_over_ssh` dispatch **each matrix cell as its
own `srun` allocation**. `srun` runs synchronously (it blocks until
the allocation finishes and propagates the job's exit code), so each
cell behaves like any other transport `exec`, just scheduled onto a
compute node. The two types differ only in how the `srun` command is
reached:

- `slurm_local`: pipette-plan runs **on the cluster login node** and
  shells out to `srun` directly via `sh -c`.
- `slurm_over_ssh`: pipette-plan runs **elsewhere** (laptop, CI box)
  and runs `ssh [user@]host "<srun cmd>"`, using the same ssh
  invocation style as the `ssh` transport (`-o BatchMode=yes`, optional
  `-p PORT`).

Under the hood each cell is wrapped as `srun <flags> sh -c '<payload>'`
(the payload single-quoted so it survives exactly one shell parse), with
an optional `pre_exec` prefix joined by `&&`.

### Fields

Common to both (`client_id`, `binary_path`, `work_dir`, `shell`,
`parallelism` work exactly as for the other transports):

| field | type | notes |
|-------|------|-------|
| `client_id` | string | unique routing handle (referenced by `variants.clients`). |
| `binary_path` | string | runner binary on the compute node. |
| `work_dir` | string | runner working directory. |
| `shell` | `posix` \| `powershell` | defaults to `posix`. |
| `parallelism` | int | defaults to `1`; see semantics below. |

`slurm_over_ssh` additionally takes `host` (required), `user`
(optional), and `port` (optional, defaults to 22): same as the `ssh`
transport.

Resource / setup fields, **all optional**, on both types:

| field | type | maps to |
|-------|------|---------|
| `pre_exec` | string | shell command run before `srun` (joined with `&&`). |
| `partition` | string | `--partition` |
| `account` | string | `--account` |
| `gpus` | int | `--gres=gpu:N` |
| `cpus` | int | `--cpus-per-task` |
| `time_limit` | string | `--time`, e.g. `"02:00:00"` |
| `mem` | string | `--mem`, e.g. `"32G"` |
| `log_dir` | string | when set, `srun` writes per-job `--output`/`--error` files here (`%x-%j` patterns); omit to stream task output to the driver. |
| `extra_srun_args` | string[] | appended to the `srun` command line verbatim. |

### Job names and logs

Each cell's `srun` is given a `--job-name` derived (sanitized) from its
benchmark and model, so `squeue` shows what each job is running rather
than all jobs appearing as `sh`. Set `log_dir` to also capture per-job
`--output`/`--error` files (named by `%x-%j` = job-name + id); without
it, `srun` streams task output back to the `pipette-plan run` driver.

### `parallelism` = concurrent SLURM jobs

For slurm transports, `parallelism = N` means the runner spawns `N`
workers, each holding one blocking `srun` allocation, so **`N`
concurrent SLURM jobs**. Unlike co-located `ssh`/`local` transports
(which serialize through a shared per-box budget keyed on the physical
host), each slurm transport gets a per-transport `physical_id`
(`slurm:<client_id>`): its jobs are scheduled onto separate compute
nodes and do **not** serialize against each other. `parallelism` is
therefore exactly the cap on how many jobs are RUNNING at once: with
`parallelism = 3` and 5 cells, `squeue` never shows more than 3
RUNNING.

### `pre_exec`: setting up the slurm environment

The `srun` command runs in a **non-login** shell. On a typical cluster
that shell does NOT have `srun` on PATH or its config set up. The
login profile does that, and it is not run for non-interactive ssh or
non-login `sh -c`. `pre_exec` is how you establish it. A typical value
for a Bright / Base Command Manager cluster:

```toml
pre_exec = ". /etc/profile.d/modules.sh && module load slurm"
```

Use the POSIX `.` builtin, **not** bash `source`: `slurm_local` runs
the command under `sh -c`, which may be dash, where `source` is
undefined.

### Driver-process caveat

One `pipette-plan run` process holds all `N` allocations for the
duration. If that process dies (an ssh drop, a logout, a closed laptop
lid for `slurm_over_ssh`) the jobs die with it. Run it under `tmux` or
`nohup` so it survives disconnects.

If you want a disconnect-resilient path with **no** long-lived driver,
use the shard + `sbatch` helpers in [`scripts/slurm/`](../../scripts/slurm/README.md)
instead: those submit detached jobs that keep running after you log
out, at the cost of the single `pipette-plan run --plan` UX. Both
approaches exist; pick the slurm transport for the integrated
single-command run, and the shard path when disconnect-resilience
matters more.

## iOS transport

The `ios` transport drives an iOS device from a host Mac via
`xcrun devicectl`, launching the Pipette app in its `headlessrun` mode.
It is unlike the others: there is no pre-provisioned remote binary or
work dir. Each cell becomes one process launch
(`xcrun devicectl device process launch --device <udid> --console
<bundle> headlessrun runtime=<r> benchmarks=<id> submit=1`), and the
device uploads its own results inline via `submit=1`, so the runner
does not append `--sync`, and there is no final sync. Success is read
from the app's `BENCH_DONE <status>` console line rather than the
`devicectl` exit code.

```toml
[[transports]]
type = "ios"
client_id = "iphone15-lab"   # warehouse join key
device_udid = "00008130-000A1B2C3D4E5F6"
# bundle_id defaults to "ai.liquid.liquid-pipette"
```

Intended for on-device runtimes: chiefly Apple Foundation Models
(`{ type = "apple_foundation" }` × `{ type = "apple_foundation_text" }`).
The plan's `benchmark` id is passed straight through as
`benchmarks=<id>`, so it must match an on-device catalog id.

Prerequisites the runner cannot bootstrap: install a code-signed device
build once via Xcode, and register the device on-device
(`headlessrun register …`) so `submit=1` uploads are accepted.

### `ios_over_ssh`

`devicectl` only talks to devices paired with the machine it runs on, so a
plain `ios` transport forces the whole plan onto that Mac. `ios_over_ssh`
renders the same `devicectl` argv as a shell command and runs it on an
intermediate Mac over ssh, so any driver can reach the phones:

```toml
[[transports]]
type        = "ios_over_ssh"
host        = "boston-macstudio-m3-1"   # the Mac the devices are paired to
user        = "liquid"
device_udid = "018CEA67-06AF-5B21-9763-F32CA2B2A044"
client_id   = "iphone17pro-1"
# port and bundle_id are optional, as for `ios`
```

Everything else matches `ios`: same headless grammar, same `submit=1`, no
`--sync`, and success still read from the app's `BENCH_DONE` line, which is
scanned out of the ssh stream rather than a local pipe.

The intermediate host must be posix: the argv is quoted for its shell, the same
constraint `adb_over_ssh` carries.

## Plan TOML Shape

Root fields:

- `plan_id` names the resumable state directory.
- `benchmarks` is an optional default benchmark list.
- `[[transports]]` declares the runnable targets.
- `[[variants]]` declares model/runtime/client groups.

A variant inherits root `benchmarks` unless it sets its own
`benchmarks` list. If root `benchmarks` is omitted, every variant must
provide `benchmarks`.

```toml
plan_id = "mixed-smoke"
benchmarks = ["prefill_throughput_512"]

[[transports]]
client_id = "mac-1"
type = "ssh"
host = "boston-macstudio"
user = "yuri"
binary_path = "/Users/yuri/bin/pipette"
work_dir = "/Users/yuri/edge-evals"
shell = "posix"

[[variants]]
clients = ["mac-1"]
models = [
  { type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-350M-MLX-4bit" },
]
runtimes = [
  { type = "mlx_macos_pipette", version = "0.31.3", requirements = { type = "catalog" }, flavor = "macos-arm64" },
]
```

Variant-local benchmarks replace the root list for that variant:

```toml
[[variants]]
clients = ["mac-1"]
benchmarks = ["end_to_end_latency_512_256"]
models = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-350M-MLX-4bit" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.31.3", requirements = { type = "catalog" }, flavor = "macos-arm64" }]
```

Model and runtime compatibility is validated when the plan loads:
GGUF models pair with `llamacpp_cli_stock_tools` (or the mobile
`llamacpp_apk_pipette` / `llamacpp_ios_pipette`), MLX models with
`mlx_macos_pipette` / `mlx_ios_pipette`, Torch/HF models with Docker or uv
OpenAI-compatible runtimes, and the `apple_foundation_text` model with the
`apple_foundation` runtime (Apple Foundation Models on iOS). AFM is a bare marker. The
model and runtime carry no fields, since the model ships with the OS:

```toml
models = [{ type = "apple_foundation_text" }]
runtimes = [{ type = "apple_foundation" }]
```

Preview before running with `pipette-plan status --plan plans/my-plan.toml`.

## Commands

```bash
# Preview counts and list missing/failed cells.
pipette-plan status --plan plans/my-plan.toml

# List raw cells by state (tab-separated, script-friendly).
pipette-plan list --plan plans/my-plan.toml --state missing

# Execute. Resumes from state.jsonl — done cells skip, missing ones run.
pipette-plan run --plan plans/my-plan.toml

# Cap the batch size, retry previous failures, or emit JSON per cell.
pipette-plan run --plan plans/my-plan.toml --limit 10
pipette-plan run --plan plans/my-plan.toml --include-failed
pipette-plan run --plan plans/my-plan.toml --json

# Force-run cells whose pinned device is no longer in the plan.
# Loses their on-device sample checkpoint — see "Device affinity" below.
pipette-plan run --plan plans/my-plan.toml --reassign-stranded

# Talk to an adb server reached over an SSH tunnel without baking the
# port into the plan. Affects `run`, `status`, and `kill`.
pipette-plan --adb-port 5038 run --plan plans/my-plan.toml

# Kill the pipette client process on every transport in the plan.
# Uses `taskkill /F /IM` on powershell hosts, `pkill -x` on posix.
pipette-plan kill --plan plans/my-plan.toml
```

## Device affinity

When the plan runs against multiple transports, each cell sticks to
the device that first started it. The reason is the on-device sample
checkpoint (`docs/pipette-cli/eval-checkpoint.md`): if device A got 3,000/5,000
samples into a cell before the operator hit Ctrl-C, restarting that
cell on device B would throw away A's progress and start from sample
zero.

How it works:

- The first time a worker claims a cell it appends a `started` event
  to `state.jsonl` carrying the worker's transport label
  (`adb:R5CY…`, `adb:R5CY…@5038` when an adb port is set,
  `ssh:host:port`, `local`).
- On the next `run` invocation, cells with a recorded label go to the
  matching worker's queue; cells without a label go to a shared
  unassigned pool that any worker can drain.
- Within one run, each worker pops from its pinned queue first and
  only steals from the unassigned pool when its own queue is empty.
- Workers with no pinned cells still drain the unassigned pool, so
  adding a new device to the plan doesn't leave it idle.

Stranded cells (pinned to a label that's no longer in
`[[transports]]`) are skipped by default and reported on stderr. The
correct fix is usually to bring the original device back. Pass
`--reassign-stranded` to send them to whichever worker grabs them; the
on-device sample checkpoint on the original device is then orphaned
and the cell restarts from sample zero on the new one.

`--limit` is applied before partitioning. If `--limit 10` is set and
4 of those 10 cells turn out to be stranded, only 6 actually run this
invocation. The limit caps total work considered, not work executed.
Re-run without `--limit`, or with `--reassign-stranded`, to drain the
remainder.

A worker killed mid-cell leaves a `started`-without-terminal record;
the next `run` re-queues the cell to the same device. Missing cells
(never run, or interrupted mid-run) always re-run: only `success`
takes a cell out of the queue, and only an explicit `--include-failed`
re-queues a cell whose latest terminal status is `failed`.

Progress logs stream to stderr; summary tables (or JSON events with
`--json`) land on stdout.

## State

Each plan's progress lives at:

```
<work-dir>/.pipette-plan/plans/<plan_id>/state.jsonl
```

Append-only JSONL, one event per cell attempt. Re-running a plan
picks up where it left off. Moving or renaming the TOML is fine:
state is keyed by `plan_id`, not by file path.

## `--json` event stream

For automation. One JSON object per line on stdout:

```jsonc
{ "event": "start", "plan_id": "...", "total": 120, "transports": [...],
                    "stranded": {"adb:OLD_DEVICE": 4} }
{ "event": "cell",  "plan_id": "...", "transport": "adb:R5CY…", "attempt": 1,
                    "status": "success", "exit_code": 0,
                    "benchmark": "...", "model": "...", "runtime": "...",
                    "mmproj": null }
{ "event": "end",   "plan_id": "...", "done": 118, "failed": 1, "missing": 1 }
```

`"event": "nothing_to_run"` replaces `start`/`end` when the matrix
was already complete. Per-cell `started` records are written to the
state file (for affinity tracking) but intentionally not surfaced on
the JSON event stream; only terminal `success` / `failed` outcomes
are emitted as `"event": "cell"` lines.

`stranded` on the `start` event is a `{label: count}` map of cells
pinned to a device not in this run's `[[transports]]`. Always
present, possibly empty. `total` excludes them: when stranded > 0,
fewer cells will run than `state.jsonl` shows as Missing.

## Secrets in the remote command line

**A plan's `auth_token` is visible to other users on the machines the
runner reaches. Treat a plan-carried HF token as disclosed to anyone with
an account on those hosts.**

The runner reaches a remote host by handing one shell command to `ssh`,
`adb shell`, or `srun`. Environment values ride in that string as
`KEY=value` prefixes, so `PIPETTE_HF_TOKEN` appears in the remote
process's argv. Anything that can list processes on the far side can read
it: `ps` for any local user, and on SLURM also `squeue`/`sacct`, which
expose the submitted command to other tenants of the cluster.

This is a property of the transport, not a bug in a particular one. The
alternative would be feeding the value over stdin per transport, which
`ssh`, `adb`, and `srun` each solve differently; until that exists, the
exposure is the documented cost of carrying the token in the plan.

What follows from it:

- **Scope the token.** Use a read-only HF token limited to the gated
  repos a run needs, not an account-wide one.
- **Rotate it after a shared-host run.** Assume any run on a
  multi-tenant SLURM cluster or a shared lab box has disclosed it.
- **Prefer a public mirror** where one exists, and omit `auth_token`
  entirely.
- **Do not paste a plan containing `auth_token` into a ticket or a
  chat.** `pipette-plan generate` already strips it from generated job
  specs (see [job-generation.md](job-generation.md)), and
  `pipette-plan commands` prints the argv without the env wrapper for
  the same reason. The plan file on disk still holds it.

The driver's own environment is not the exposure path: the token is read
from the plan, not from the driver env, so a token that never enters a
plan never reaches a remote command line.

## Troubleshooting

**`unknown field 'X'` at load.** Most common: a top-level key placed
after a `[section]` header. TOML parses top-level keys as members of
the preceding section. Move all top-level keys to the top of the
file, before any `[…]`.

**Runtime-specific field on the wrong runtime.** Load errors name the
unknown field. Move the field into the matching runtime entry, or use
the typed model/runtime variant that owns it.

**Gated/private HF models.** Set `auth_token = "hf_…"` on the model so
the runner forwards it to every transport as `PIPETTE_HF_TOKEN`. The token is
carried in the plan, not read from the driver env. Omit `auth_token`
for public models. Note the token reaches the remote command line and is
readable by other users on that host: see
[Secrets in the remote command line](#secrets-in-the-remote-command-line).

**Doom-loop overrides not applied.** Doom-loop overrides are per-cell
run-driving settings on the eval cell's `benchmark_flags` (keyed by the
`(benchmark_type, runtime_type, model_type)` triple); not inside runtime
entries and not a top-level `[doomloop]` table. For example:
`benchmark_flags = [{ benchmark_type = "eval", runtime_type = "mlx_macos_pipette", model_type = "mlx", doomloop = { exact_repeat = { window = 256 } } }]`.

**One device hangs: others should keep going.** They do. Each
transport has its own worker and its own consecutive-failure counter.
Watch stderr; every line is prefixed with the device label.

## See also

- [`docs/architecture.md`](../architecture.md): crate layout and
  workspace model.
- [`docs/pipette-cli/eval-checkpoint.md`](../pipette-cli/eval-checkpoint.md): how
  orchestrator-level state relates to per-sample state on the device.
- `crates/pipette-plan-types/src/`: the canonical schema, with
  `#[serde(deny_unknown_fields)]` enforcing it.
