# pipette-ctl

`pipette-ctl` runs a matrix of benchmarks × models × runtimes on one or
more devices, and resumes an interrupted run. A plan is a typed TOML
document:

- each model declares its artifact format;
- each runtime declares its serving or runtime kind;
- each transport points at the runner binary installed on a device.

This page covers **local dispatch**: driving devices directly. To hand
the matrix to the management server instead, so it leases each cell to a
matching registered client, see
[job generation](job-generation.md) and its scheduler-mode plan format.

## Setup

1. Install `pipette-ctl` locally.
2. Install the client binary (`pipette`) on each target device, register
   it with the management server, and install the runtime and benchmarks.
   The plan runner calls the remote binary. It doesn't deploy it. The
   same `pipette` binary serves every runtime; it dispatches on the
   `runtime.type` declared in the plan.
3. Create the local state workspace:

```bash
pipette-ctl init
```

The workspace is `.pipette-ctl/` under the work dir. Older releases named
this directory `.pipette-plan/`. A work dir that holds only `.pipette-plan/`
keeps using it.

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

## Transports

The plan schema lives in pipette-mgmt: the fields, types and
defaults of each `[[transports]]` type are in
[`crates/pipette-plan-types/src/plan/transport.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/plan/transport.rs).
The types:

- `adb`: an Android device on the driver's adb server.
- `adb_over_ssh`: an Android device on the adb server of an intermediate
  host ([below](#reaching-a-device-through-an-intermediate-host)).
- `ssh`: a host reached over ssh.
- `local`: starts the runner binary on the driver host through its local
  shell (`sh -c`, or PowerShell for `shell = "powershell"`), with no ssh or
  adb hop and no sshd.
- `slurm_local` / `slurm_over_ssh`: each cell as its own `srun`
  allocation ([below](#slurm-transports)).
- `ios` / `ios_over_ssh`: an iOS device driven through `xcrun devicectl`
  ([below](#ios-transport)).

`parallelism = N` (default `1`) starts `N` concurrent workers against
that device. Use it for I/O-bound eval benchmarks. **Do not use it on
performance benchmarks**: concurrent execution corrupts throughput and
latency measurements. Transports that reach one device (the same ssh host,
adb serial or iOS UDID, the unique device identifier) share one budget. The largest `parallelism` among them sets the size of that budget.

The `adb` transport's optional `port` is the *local adb server* port
(`adb -P <port>`): set it when the adb server runs on another machine
reached through an SSH tunnel. Omit it to use the adb default (5037) or
whatever `ANDROID_ADB_SERVER_PORT` is set to. `--adb-port` overrides
every adb transport's `port` for one invocation without editing the plan.

## Reaching a device through an intermediate host

`adb_over_ssh` runs the `adb` command **on an intermediate host**, not on
the driver. The driver needs only ssh to the host that holds the pairing
keys. It needs no adb and no tunnel. `adb_over_ssh` relates to `adb` as
`slurm_over_ssh` relates to `slurm_local`. Use it when the phones are
paired to one controller host.

```toml
[[transports]]
type        = "adb_over_ssh"
client_id   = "ev1_9f2c1d3e4b5a6c7d"
host        = "phone-controller"   # the host whose adb server owns the device
user        = "eval"
serial      = "<serial>"
binary_path = "/data/local/tmp/pipette/pipette"
work_dir    = "/data/local/tmp/pipette"
```

The transport has two port fields. `port` is the **ssh** port, as on
every transport that uses ssh. `adb_port` is the adb server port on `host`.
`--adb-port` does **not** apply: it changes the port of a tunnel from the
driver, and this transport uses no tunnel.

`pre_exec` runs on the intermediate host before `adb`, joined with `&&`;
the same role it plays for slurm. Non-interactive ssh skips the login
profile, so a host that only puts `adb` on PATH there needs it:

```toml
pre_exec = "export PATH=$PATH:$HOME/Android/Sdk/platform-tools"
```

The runner quotes the device command before it crosses ssh, so the JSON
`--model` and `--runtime` descriptors of a cell reach `adb shell` intact.
`shell` describes the *device* shell and must be `posix`: the plan load
refuses `powershell`. The **intermediate host** must also run a POSIX shell.

The budget for this transport keys on the **serial**, not on `host`. The
phone is the shared resource, so several phones behind one controller run
at the same time.

`ios_over_ssh` applies the same hop to iOS; see [`ios_over_ssh`](#ios_over_ssh).
Scheduler mode ([job-generation.md](job-generation.md)) remains the other
option, where the device claims its own jobs and no intermediate host exists.

## SLURM transports

`slurm_local` and `slurm_over_ssh` dispatch **each matrix cell as its
own `srun` allocation**. `srun` runs synchronously (it blocks until
the allocation finishes and propagates the job's exit code), so each
cell behaves like any other transport `exec`, just scheduled onto a
compute node. The two types differ only in how the `srun` command is
reached:

- `slurm_local`: pipette-ctl runs **on the cluster login node** and
  shells out to `srun` directly via `sh -c`.
- `slurm_over_ssh`: pipette-ctl runs **elsewhere** (a laptop, a CI host)
  and runs `ssh [user@]host "<srun cmd>"`, using the same ssh
  invocation style as the `ssh` transport (`-o BatchMode=yes -o
  ConnectTimeout=15`, optional `-p PORT`).

Each cell is wrapped as `srun <flags> sh -c '<payload>'` (the payload
single-quoted so it survives exactly one shell parse), with an optional
`pre_exec` prefix joined by `&&`. The optional resource fields
(`partition`, `account`, `gpus`, `cpus`, `time_limit`, `mem`,
`extra_srun_args`) map onto `srun` flags as documented on the
`SlurmLocal` / `SlurmOverSsh` variants in
[`transport.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/plan/transport.rs).

### Job names and logs

Each cell's `srun` is given a `--job-name` derived (sanitized) from its
benchmark and model, so `squeue` shows what each job is running rather
than all jobs appearing as `sh`. Set `log_dir` to also capture per-job
`--output`/`--error` files (named by `%x-%j` = job-name + id); without
it, `srun` streams task output back to the `pipette-ctl run` driver.

### `parallelism` = concurrent SLURM jobs

For slurm transports, `parallelism = N` means the runner starts `N`
workers, each holding one blocking `srun` allocation, so **`N`
concurrent SLURM jobs**. `ssh` and `local` transports that reach one host
share one budget, keyed on that host. Each slurm transport gets its own
budget (`slurm:<client_id>`): SLURM schedules its jobs onto separate
compute nodes, so they do **not** wait for each other. `parallelism` is therefore exactly the cap
on how many jobs are RUNNING at once: with `parallelism = 3` and 5
cells, `squeue` never shows more than 3 RUNNING.

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

One `pipette-ctl run` process holds all `N` allocations for the
duration. If that process dies (an ssh drop, a logout, a closed laptop
lid for `slurm_over_ssh`) the jobs die with it. Run it under `tmux` or
`nohup` so it survives disconnects.

If you want a disconnect-resilient path with **no** long-lived driver,
use the shard + `sbatch` helpers in [`scripts/slurm/`](../../scripts/slurm/README.md)
instead: those submit detached jobs that keep running after you log
out, at the cost of the single `pipette-ctl run --plan` UX. Pick the
slurm transport for the integrated single-command run, and the shard
path when disconnect-resilience matters more.

## iOS transport

The `ios` transport drives an iOS device from a host Mac via
`xcrun devicectl`, launching the Pipette app in its `headlessrun` mode.
It has no installed remote binary and no work dir. Each cell becomes one
process launch:

```text
xcrun devicectl device process launch --device <udid> --terminate-existing
  --console <bundle> headlessrun runtime=<runtime JSON> [model=<model JSON>]
  benchmarks=<id> [runtime-flags=<JSON>] submit=1
```

- `model=` is absent only for Apple Foundation, whose model ships with the OS.
- With `submit=1`, the app uploads its own results. The runner does not
  append `--sync`, and the sync passes skip iOS.
- The runner reads success from the app's `BENCH_DONE <status>` console
  line, not from the `devicectl` exit code.
- `--terminate-existing` stops a live instance of the app. Without it,
  `devicectl` waits for that instance and does not fail.

`--readiness-max-wait-secs` and `--readiness-skip-thermal` reach the app as
its environment, through `devicectl`'s `--environment-variables`, which the
launch omits when neither is set. The plan's `auth_token` is not forwarded:
the app does not read it, and `--environment-variables` sits on the Mac's
command line.

```toml
[[transports]]
type = "ios"
client_id = "iphone15-lab"   # warehouse join key
device_udid = "00008130-000A1B2C3D4E5F6"
# bundle_id defaults to "ai.liquid.liquid-pipette"
```

The transport is for on-device runtimes, mainly Apple Foundation Models
(`{ type = "apple_foundation" }` × `{ type = "apple_foundation_text" }`,
both bare markers with no fields, since the model ships with the OS).
The plan's `benchmark` id is passed straight through as
`benchmarks=<id>`, so it must match an on-device catalog id. The
headless grammar is in [pipette-ios/usage.md](../pipette-ios/usage.md).

Prerequisites the runner cannot bootstrap: install a code-signed device
build once via Xcode, and register the device on-device
(`headlessrun auth register`, see
[pipette-ios/usage.md](../pipette-ios/usage.md#identity-auth)) so
`submit=1` uploads are accepted.

### `ios_over_ssh`

`devicectl` only talks to devices paired with the machine it runs on, so a
plain `ios` transport forces the whole plan onto that Mac. `ios_over_ssh`
renders the same `devicectl` argv as a shell command and runs it on an
intermediate Mac over ssh, so any driver can reach the phones:

```toml
[[transports]]
type        = "ios_over_ssh"
host        = "mac-studio-1"   # the Mac the devices are paired to
user        = "pipette"
device_udid = "<device-udid>"
client_id   = "iphone17pro-1"
# port and bundle_id are optional, as for `ios`
```

Everything else matches `ios`: same headless grammar, same `submit=1`, no
`--sync`, and success still read from the app's `BENCH_DONE` line, which is
scanned out of the ssh stream rather than a local pipe.

The intermediate host must run a POSIX shell, because the runner quotes the
argv for that shell. `adb_over_ssh` has the same rule.

## Plan TOML shape

The schema is
[`pipette-plan-types`](https://github.com/Liquid4All/pipette-mgmt/tree/main/crates/pipette-plan-types/src)
in pipette-mgmt (`Plan` in
[`plan/mod.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/plan/mod.rs)),
with `deny_unknown_fields` enforcing it. A variant inherits the root
`benchmarks` unless it sets its own; with no root list, every variant
must set one.

```toml
plan_id = "mixed-smoke"
benchmarks = ["prefill_throughput_512"]

[[transports]]
client_id = "mac-1"
type = "ssh"
host = "mac-studio-1"
user = "pipette"
binary_path = "/Users/pipette/bin/pipette"
work_dir = "/Users/pipette/edge-evals"
shell = "posix"

[[variants]]
clients = ["mac-1"]
models = [
  { type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-350M-MLX-4bit" },
]
runtimes = [
  { type = "mlx_macos_pipette", version = "0.31.3", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.31.3\n" } },
]
```

A local-dispatch plan states a runtime's `source` in full. The plan thus
carries the environment it measured, and a later catalog row cannot change
it. `version` labels the body beside it; it does not select it. A
scheduler-mode plan can also name a catalog entry
([job generation](job-generation.md)). `pipette runtimes
catalog <type>` prints the rows, and `examples/plans/mlx-macos-vl-spec-decode.toml`
shows a long one written as a sub-table so the pins stay readable.

When the plan loads, each variant expands to its compatible (model,
runtime) pairs, as
[`is_compatible`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/mod.rs)
decides them. The expansion leaves out an incompatible pair. A model or a
runtime with no compatible partner in its variant stops the load.

## Commands

Each command's flags and defaults are in `pipette-ctl <command> --help`.

- `status --plan`: completion counts, then the outstanding cells as a
  ladder (see [Reading the ladder](#reading-the-ladder)); `--state`
  picks another set. Use it to preview a plan before running it.
- `run --plan`: execute. Resumes from `state.jsonl`: done cells skip,
  missing ones run. `--include-failed` and `--include-done` re-queue
  those states, `--shard INDEX/COUNT` splits the ordered matrix across
  independent invocations (one SLURM job each, say), and
  `--reassign-stranded` / `--ignore-pins` override
  [device affinity](#device-affinity).
- `commands --plan`: print the exact `pipette` invocation each cell would
  get, including the appended `--sync` (not on iOS) but not the ssh/adb/srun wrapper or
  its env, without running anything. Takes `--state` (default `missing`)
  and `--shard` as `run` does.
- `kill --plan`: kill the pipette client, and the llama.cpp tools it
  spawns (`llama-server`, `llama-bench`, `llama-batched-bench`), on every
  transport in the plan; on iOS it terminates the app on the device.
- `reset --plan`: delete the plan's state file, so every cell is `missing`
  again. `--client <ID>` wipes only the cells that device ran. The flag is
  repeatable, and part of a `client_id` is enough. Neither form touches
  remote per-sample state on the device.

`--adb-port`, `--readiness-max-wait-secs`, `--readiness-skip-thermal` and
`--readiness-held-touch` go before the subcommand or after `run`
(`--adb-port` also after `kill`). `--readiness-held-touch off|wait|fail` reaches
each runner as `PIPETTE_READINESS_HELD_TOUCH`; only the Android CLI gate reads
it. `off` disables touch detection for those runners. Under `fail`, each cell on
a phone with a held touch fails in about 9 seconds. If the plan sets
`[retry] max_consecutive_failures`, that limit then stops the phone.

Filing a plan with the management server instead of driving devices directly
goes through a campaign: see [job generation](job-generation.md) for
`campaigns create`, and [submitter identity](submitter-identity.md) for the
`auth` verbs that give it an account to sign as.

## Reading the ladder

`status` prints a three-line header (the plan id, its state file, and
how its cells stand), then a section per state, worst first, each its
own ladder. Everything a group of cells shares becomes a branch, said
once; the models that share the whole path are the list at the end. The
descent is:

```text
benchmark → runtime → runtime_flags → model_flags → benchmark_flags → models
```

Every object appears as its own serialization, which is the form the
client is handed. What the plan *is* stays in the plan file; the header
does not repeat its targets or runtimes, and the ladder already draws
every runtime the plan expands to.

```text
failed cells (1):
decode_throughput_512_100  (1)
└── {"type":"llamacpp_cli_stock_tools","source":"github_release","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"23dff81c4","flavor":"android-arm64-v8a"}
    └── runtime_flags  {"runtime_type":"llamacpp_cli_stock_tools","model_type":"gguf_text","benchmark_type":"decode_throughput","threads":4}
        └── {"type":"gguf_text","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-GGUF","path":"LFM2.5-350M-Q4_K_M.gguf"}

missing cells (2):
decode_throughput_512_100  (2)
└── {"type":"llamacpp_cli_stock_tools","source":"github_release","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"23dff81c4","flavor":"android-arm64-v8a"}
    └── runtime_flags  {"runtime_type":"llamacpp_cli_stock_tools","model_type":"gguf_text","benchmark_type":"decode_throughput","threads":4}
        ├── {"type":"gguf_text","source":"huggingface","org":"unsloth","repo_name":"Qwen3.5-0.8B-GGUF","path":"Qwen3.5-0.8B-Q4_0.gguf"}
        └── {"type":"gguf_text","source":"huggingface","org":"unsloth","repo_name":"Qwen3.5-2B-GGUF","path":"Qwen3.5-2B-Q4_0.gguf"}
```

Reading it:

- The JSON is the object as serialized, not a summary: axis keys, empty
  fields skipped, the nested `readiness` block inline. What
  you read is what a cell will be sent, so the lines are long.
- Only what the plan authored appears. An unset knob is a default the
  engine picks at run time, not a value to show before anything has run.
- **A flag object is a branch, not a leaf.** A plan usually authors one
  entry per `(benchmark, runtime)`. The ladder states it once above the
  models, so the same long line does not repeat on each model. Cells carrying none group
  under `runtime_flags —`. Where a plan *does* author several for one
  `(benchmark, runtime)`, as the openvino `device` fan-out does, each is
  its own branch, which is what those cells are.
- **A flag level that no cell carries does not appear.** `model_flags`
  are eval-only, and many plans author no `benchmark_flags`. Such a plan
  shows neither level, so the drawing has no levels of dashes. The ladder
  decides each level across the whole drawing, so sibling branches stay
  the same depth.
- **State is a section, not a branch.** A cell does not share its state
  with the cells around it. As a branch, state would split each model list
  in two, to say what one heading says. Each section counts its cells, and each
  benchmark root counts its own.

## Device affinity

When the plan runs against multiple transports, each cell stays on
the device that first started it. The reason is the on-device sample
checkpoint ([eval-checkpoint.md](../pipette-cli/eval-checkpoint.md)): if
device A got 3,000/5,000 samples into a cell before the operator hit
Ctrl-C, restarting that cell on device B would throw away A's progress
and start from sample zero.

How it works:

- The first time a worker claims a cell it appends a `started` event
  to `state.jsonl` carrying the worker's transport `client_id`. The pin
  keys on `client_id`, not on the host or serial, so two transports
  that reach one device with different binaries keep separate queues.
- On the next `run` invocation, cells with a recorded pin go to the
  matching worker's queue. Cells without one go to a shared pool. Every
  worker that the variant's `clients` allows takes cells from that pool.
- Within one run, each worker takes from its pinned queue first. It
  takes from the shared pools only when its own queue is empty.
- Workers with no pinned cells also take from the shared pools, so a
  device that you add to the plan does not stay idle.

A stranded cell is pinned to a `client_id` that is not in
`[[transports]]`, or that the variant's `clients` excludes. By default the
runner skips stranded cells and reports them on stderr. The correct fix
is usually to bring the original device back. Two flags override this:

- `--reassign-stranded` sends stranded cells to the first free worker.
- `--ignore-pins` drops every recorded pin. Use it when the remaining
  cells all last ran on one device and should spread across every
  allowed worker.

With either flag, the runner leaves the on-device sample checkpoint on
the original device unused, and the cell restarts from sample zero.

`--limit` is applied before partitioning. If `--limit 10` is set and
4 of those 10 cells turn out to be stranded, only 6 actually run this
invocation. The limit caps total work considered, not work executed.
Re-run without `--limit`, or with `--reassign-stranded`, to run the
remaining cells.

A worker killed mid-cell leaves a `started`-without-terminal record;
the next `run` re-queues the cell to the same device. Missing cells
(never run, or interrupted mid-run) always re-run: only a `success`
attempt makes a cell `done` and takes it out of the queue, and only
`--include-failed` re-queues a cell whose latest terminal status is
`failed`. A `done` cell comes back only with `--include-done`.

Progress logs stream to stderr; summary tables (or JSON events with
`--json`) land on stdout.

## Re-running done cells

`--include-done` puts `done` cells back in the queue, the way
`--include-failed` does for failed ones. Passing both re-runs the entire
matrix. Use it when the plan's coordinates are unchanged but the thing
they measure is not: a new client build on the devices, a benchmark
definition edited server-side, or a result set you suspect.

Nothing recorded is discarded. Each re-run cell appends a fresh attempt
(numbered after its recorded ones) and submits its results as new rows
alongside the earlier ones, so `status` keeps counting the cell as done
throughout. Device affinity and `--shard` behave as they do on a normal
pass: the cell goes back to the device pinned in state unless
`--ignore-pins` is set.

`--limit` counts from the front of the matrix, and with `--include-done`
that front is mostly cells that already ran. Pair the two only when you
mean to re-run the first N cells; to cap outstanding work, leave
`--include-done` off.

What a re-run eval cell repeats depends on its on-device sample
checkpoint ([eval-checkpoint.md](../pipette-cli/eval-checkpoint.md)). A
cell whose samples all succeeded has no checkpoint left (the client
deletes it on finalize), so it re-runs the benchmark end to end. A cell
that finished with some samples marked failed left a failed-only
checkpoint behind: every other sample re-runs, but those ids keep
carrying over as failed. Delete the cell's file under the client's
`state/evals/` to make even those re-run.

## Sync passes

`pipette sync` pulls benchmark definitions as well as pushing results, so
a `run` with cells to execute syncs every transport (iOS excepted) at
three points:

- **initial**: before the first cell, so cells benchmark current
  definitions, not the ones the device last pulled. Every cell in the plan
  waits for this pass, so the runner probes each device first and skips
  a device that does not answer. `pipette storage gc` runs after this
  sync (see below).
- **rejoin**: when a device that did not answer comes back mid-run, before
  it resumes.
- **final**: after the last cell, to send the results that the run queued.

The initial and final passes run concurrently across transports, within
the `parallelism` budget of each device. A device that fails to sync logs
a warning, and the other devices continue.

The initial pass then cleans each device's artifact store (`pipette storage
gc`, see [storage-quota.md](../storage-quota.md)). The order matters:

1. A result queued for submission pins the artifacts it names, so a clean
   before the sync cannot reclaim them.
2. The sync sends those results, which unpins what the last run left.
3. The clean then frees space before the first cell. Without it, the device
   can reach its quota partway through the run, where a failed fetch costs
   a whole cell. A failed `gc` warns like a failed sync and does
not stop the run.

## Waiting on a device that drops

A worker probes its transport before it claims each cell, and probes again
after a transport error mid-cell. If either probe fails, the worker puts
the cell back in the queue and starts a **rejoin wait**: sleep, probe,
repeat until the device answers or the retries run out. A device that
comes back gets the rejoin sync above before it resumes, because it missed
the initial pass. If the device does not come back, its worker stops, and
the other workers continue with the queues.

The wait sleeps *before* its first probe. A probe failed just before the
wait started, so an immediate probe would only read the same failure. A
device that comes back is usually rebooting, which takes longer than one
probe.

`--rejoin-retry-delay-secs` and `--rejoin-max-retries` set how long the
wait lasts. The default, ten probes 30 s apart, is five minutes. That
suits a phone that reboots. For an ssh host that is gone for good, it
costs five minutes of an idle worker. On a fleet where a lost connection
means a lost host, lower both. `--rejoin-max-retries 0` stops at once,
with no probe: the worker stops, and every cell still queued for that
device stays `missing` for a later invocation.

Each retry writes a line to stderr, and a `device` event to the `--json`
stream (below).

## State

Each plan's progress lives at:

```
<work-dir>/.pipette-ctl/plans/<plan_id>/state.jsonl
```

The file is append-only JSONL (JSON Lines): one JSON event per line, for
each cell attempt. Re-running a plan
picks up where it left off. Moving or renaming the TOML is fine:
state is keyed by `plan_id`, not by file path.

## `--json` event stream

For automation. One JSON object per line on stdout:

```jsonc
{ "event": "start", "plan_id": "...", "total": 120, "transports": [...],
                    "stranded": {"absent:old-client": 4} }
{ "event": "cell",  "plan_id": "...", "transport": "adb:R5CY…", "attempt": 1,
                    "status": "success", "exit_code": 0,
                    "benchmark": "...", "model": "...", "runtime": "..." }
{ "event": "device","plan_id": "...", "transport": "adb:R5CY…",
                    "state": "waiting", "retry": 1, "max_retries": 10,
                    "retry_in_secs": 30, "exit_code": null, "error": null }
{ "event": "end",   "plan_id": "...", "done": 118, "failed": 1, "missing": 1 }
```

`start` and `end` are always emitted; a matrix with nothing left to run
gives `"total": 0`. Per-cell `started` records are written to the state
file (for affinity tracking) but intentionally not surfaced on the JSON
event stream; only terminal `success` / `failed` outcomes are emitted as
`"event": "cell"` lines.

`"event": "device"` is one line per turn of a rejoin wait (see
[Waiting on a device that drops](#waiting-on-a-device-that-drops)), so a
consumer can tell a fleet that is rebooting from a run that is stuck.
`state` is one of `waiting`, `refused` (the device answered, non-zero), `unreachable` (the probe could not run at all),
`online` (it is back, the worker resumes) or `gave_up` (retries spent, the
worker stops). `retry_in_secs`, `exit_code` and `error` are always present
and `null` where the state has nothing to put in them. The matching stderr
lines are written either way: `--json` adds these events, it does not
replace the progress log.

`stranded` on the `start` event counts the skipped cells by reason:
`absent:<client_id>` (pinned to a transport not in this run),
`excluded-from-allowed:<client_id>` (pinned to one the variant's
`clients` excludes) and `no-overlap` (none of the variant's `clients` is
in this run). Always present, possibly empty. `total` excludes them: when stranded > 0,
fewer cells will run than `state.jsonl` shows as missing.

## Secrets in the remote command line

**A plan's `auth_token` is visible to other users on the machines the
runner reaches. Treat a Hugging Face (HF) token in a plan as disclosed to
anyone with an account on those hosts.**

The runner hands one shell command to `ssh`, `adb shell` or `srun`. For
`local` and `slurm_local`, it hands that command to the local shell on the
driver host. The command carries environment values as `KEY=value`
prefixes, so `PIPETTE_HF_TOKEN` appears in a process's argv:

- on the remote host, for `ssh`, `adb`, `adb_over_ssh` and the SLURM
  transports;
- on the driver host, for `local` and `slurm_local`.

Anything that can list processes on that host can read it: `ps` for any
local user, and on SLURM also `squeue`/`sacct`, which show the submitted
command to other users of the cluster.

Every transport passes environment values this way. Sending the value over
stdin would need a different method for each of `ssh`, `adb` and `srun`.
The exposure is therefore the cost of carrying the token in the plan.

What follows from it:

- **Scope the token.** Use a read-only HF token limited to the gated
  repos a run needs, not an account-wide one.
- **Rotate it after a shared-host run.** Assume any run on a
  multi-tenant SLURM cluster or a shared lab host has disclosed it.
- **Prefer a public mirror** where one exists, and omit `auth_token`
  entirely.
- **Do not paste a plan containing `auth_token` into a ticket or a
  chat.** `pipette-ctl commands` prints the argv without the env wrapper
  for the same reason. The plan file on disk still holds it. What the
  management server does with a filed plan's token is in
  [job-generation.md](job-generation.md#gated-models).

The runner reads the token from the plan, not from the driver's
environment. A token that never enters a plan therefore never reaches a
command line.

## Troubleshooting

**`unknown field 'X'` at load.** Most common: a top-level key placed
after a `[section]` header. TOML parses top-level keys as members of
the preceding section. Move all top-level keys to the top of the
file, before any `[…]`.

**Runtime-specific field on the wrong runtime.** Load errors name the
unknown field. Move the field into the matching runtime entry, or use
the typed model/runtime variant that owns it.

**Gated/private HF models.** Set `auth_token = "hf_…"` on the model. The
plan carries one token: the runner takes the `auth_token` of the first
model that sets one, and forwards it to every non-iOS cell as
`PIPETTE_HF_TOKEN`. Give every gated model in a plan the same token. The
runner reads the token from the plan, not from the driver's environment. Omit `auth_token`
for public models. The token reaches the remote command line: see
[Secrets in the remote command line](#secrets-in-the-remote-command-line).

**One device hangs: others should keep going.** They do. Each worker
runs its own queue, and with `[retry] max_consecutive_failures = N` in
the plan a worker stops after `N` failed cells in a row (`0`, the
default, never stops it). Watch stderr; every line is prefixed with the
device label.

## See also

- [`docs/architecture.md`](../architecture.md): crate layout and
  workspace model.
- [`docs/pipette-cli/eval-checkpoint.md`](../pipette-cli/eval-checkpoint.md): how
  orchestrator-level state relates to per-sample state on the device.
