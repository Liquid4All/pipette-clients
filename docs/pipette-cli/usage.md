# Using the `pipette` CLI

**Scope:** operate the unified `pipette` client; workspace setup, registration,
installing runtimes, running benchmarks, and syncing results.

- **[Naming models, runtimes, and flags](models-and-runtimes.md)**: the URI and
  JSON notation for `--model` / `--runtime` / `--runtime-flags`, with recipes.
  Start there when the question is *how do I express the thing I want to run*.
- Backends: [llama.cpp](llamacpp.md) · [MLX](mlx.md) · [torch-oai](torch-oai.md) ·
  [OpenVINO](openvino.md) (also
  [IR](../openvino-ir.md) / [measurement](../openvino-measurement.md))
- [Eval checkpoint & resume](eval-checkpoint.md)

Batch orchestration across devices is a separate binary: see
[pipette-plan](../pipette-plan/plan-runner.md). Building binaries and mobile apps
is covered in the [top-level README](../../README.md).

## Your first benchmark

No server, no registration, no account. A `local/` benchmark is entirely
self-contained. On a Mac:

```bash
cargo build --release -p pipette-cli        # produces target/release/pipette
export PATH="$PWD/target/release:$PATH"

mkdir -p ~/pipette-workdir && cd ~/pipette-workdir
pipette init
pipette benchmarks init-local               # 34 standard local definitions

pipette benchmarks run \
  --benchmark local/prefill_throughput_smoke \
  --model 'gguf-text://repo=unsloth/gemma-3-270m-it-GGUF&path=gemma-3-270m-it-Q4_K_M.gguf' \
  --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'

pipette results list
```

The runtime and the model are fetched on first use; nothing needs pulling ahead
of time. Everything below adds to this: registration and `sync` only matter once
you want results in the management server.

If the run stops with `readiness wait timed out`, the machine was too warm or too
busy to measure on. See [§5b](#5b-the-readiness-gate).

## Prerequisites

You only need what the runtime(s) you use call for.

| Runtime | Where it runs | Prerequisites |
|---------|---------------|---------------|
| llama.cpp | Cross-platform | None: `pipette` downloads matching upstream release binaries |
| MLX | Apple Silicon Mac | [`uv`](https://docs.astral.sh/uv/) (`brew install uv`) |
| torch-oai (vLLM / SGLang) | Linux host | Docker Engine for `docker-vllm://` / `docker-sglang://`, **or** `uv` for `uv-vllm://` / `uv-sglang://`; `nvidia-container-toolkit` for NVIDIA GPUs under Docker |
| OpenVINO | Linux/Windows x86_64 | `uv` |

**Conditional:**

- **Management server**: only for *remote* benchmarks and score sync. Local
  benchmarks need no server.
- **`PIPETTE_HF_TOKEN`**: only for private/gated Hugging Face models.

## Workflow

Three modes share the same setup:

```text
# Local-only (no server, no registration)
init → benchmarks init-local → run benchmarks → results list

# Ad-hoc (operator-driven, results submitted)
init → auth register → (install runtime) → run benchmarks → sync

# Planner client (server-driven claim loop)
init → auth register → (install runtime) → worker
```

Only **install** and **run** vary by `--runtime` / `--model`. Everything else is
shared. The planner path is described in [§8](#8-planner-worker-pull-model).

## 1. Initialize

```bash
cd ~/pipette-workdir
pipette init

# or: pipette init --work-dir ~/pipette-workdir
# or: export PIPETTE_WORK_DIR=~/pipette-workdir && pipette init
```

Creates `.pipette/`:

```text
.pipette/
  manifest.toml          # workspace marker (legacy manifest.json migrated on open)
  identity/              # Ed25519 keypair, registration, settings.json
  runtimes/              # installed runtime builds
  models/                # cached model artifacts
  benchmarks/
    local/
    remote/
  results/
    local/
    remote/{pending,synced}/
  state/evals/           # resumable eval checkpoints
  cache/<runtime-key>/   # compiled artifacts, created on first compile
```

`cache/` is minted by the first engine that compiles something, not by `init`.
It sits outside the artifact stores so a storage sweep cannot delete what a run
depends on. It is therefore neither counted by `storage status` nor swept
by `storage gc`; `runtimes remove` is what reclaims it.

`models/` is the shared model store across backends (GGUF files, HF snapshots,
etc. handed to engines as local paths: no ambient HF cache required for normal
URI runs). `runtimes/` is the common root for installed builds; see
[§3](#3-install-a-runtime) for how each scheme lands there.

### Workspace resolution

| Priority | Source |
|----------|--------|
| 1 | `--work-dir <path>` |
| 2 | `PIPETTE_WORK_DIR` |
| 3 | current directory |

Commands other than `init`, `pipette --version`, `runtimes catalog <type>` and
`runtimes flavors` require an initialized workspace. Those two are discovery
commands, so they work anywhere: you can pick a runtime before creating the
workspace you will install it into.

## 2. Register

Skip this section entirely if you only run `local/` benchmarks: they need no
server, and their results never leave the machine.

```bash
pipette auth register \
  --organization LiquidAI \
  --contact-email user@example.com \
  --client-details "mac studio for benchmarks" \
  --preauth-key preauth_...   # optional
```

`--server-url` defaults to `https://collector.pipette.liquid.ai`.

This writes an Ed25519 keypair under `identity/` and registers the public key.
Status starts as `pending` unless a valid `--preauth-key` (or
`PIPETTE_PREAUTH_KEY`) is accepted: then the client can come up `approved`.
The preauth key is never written to disk. If a local identity already exists,
`auth register` refuses; use `pipette auth reset` to start over, or
`auth set-device` to relabel without rotating keys.

`auth reset` deletes the identity and everything derived from the server
relationship: private key, registration, device labels, pulled remote
benchmarks, remote results (pending *and* synced), and eval score-refresh state.
Local benchmarks and local results are server-independent and survive. It
refuses while unsubmitted remote results exist unless you pass `--force`.

```bash
pipette auth me    # client_id, organization, contact, status, tags,
                   # reindex_pending, and the advertised capabilities
pipette sync       # pull remote benchmarks assigned to this client
```

## 3. Install a runtime

Runtimes are named with a compact URI or a JSON `Runtime` object. The notation
is [models-and-runtimes.md](models-and-runtimes.md).

`benchmarks run` ensures both the runtime **and** the model through the same
store path `runtimes pull` / `models pull` use, so an explicit pull is only ever
a warm-up: it moves the download earlier, never changes what a run does. What
differs per scheme is where the build lands:

| Scheme | Installs as | Browse with |
|--------|-------------|-------------|
| `llamacpp-cli-stock-tools://` | upstream release archive under `runtimes/` | `runtimes catalog llamacpp_cli_stock_tools --flavor <flavor>` |
| `docker-vllm://` / `docker-sglang://` | image in the docker daemon + a list-visible workspace record | — |
| `mlx-macos-pipette://` | relocatable venv under `blobs/` (macOS) | `runtimes catalog mlx_macos_pipette` |
| `uv-vllm://` / `uv-sglang://` | relocatable venv under `blobs/` (Linux) | `runtimes catalog uv_vllm` / `uv_sglang` |
| `uv-openvino://` | relocatable venv under `blobs/` (Linux/Windows); one venv serves cpu/gpu/npu, since `device` is a per-cell choice | `runtimes catalog uv_openvino` |

An off-catalog uv/MLX requirements set has no URI form, so it needs an explicit
`runtimes pull` with a JSON `--runtime`.

```bash
pipette runtimes flavors                    # the --flavor vocabulary and the
                                            # upstream asset each resolves to
pipette runtimes list
pipette runtimes list --format uri          # rows paste back into `runtimes pull`
pipette runtimes remove --runtime '<uri>'   # drops the list-visible workspace record; docker images stay until docker rmi
```

`runtimes flavors` and `runtimes catalog` need neither a workspace nor (for
`flavors`) a network, so they are the right first stop on a fresh box.

### Referring to an installed runtime or model by digest

A uv/MLX runtime is defined by its whole `requirements.txt`, and a model by its
full source coordinates; neither is something to retype. Once one is
installed, `list` shows a `DIGEST` usable as the reference instead:

```bash
pipette runtimes list
#  RUNTIME | TYPE              | DIGEST       | PULLED
# ---------+-------------------+--------------+-----------------------------
#  0.31.3  | mlx_macos_pipette | c07a4fd35f0b | 2026-07-31T12:35:53.203406Z

pipette models list
#  MODEL                         | TYPE | DIGEST       | FETCHED
# -------------------------------+------+--------------+-----------------------------
#  LiquidAI/LFM2.5-350M-MLX-4bit | mlx  | d86cc299004a | 2026-07-31T12:53:14.522265Z

pipette benchmarks run \
  --runtime 'runtime://sha256=c07a4fd3' \
  --model   'model://sha256=d86cc299' \
  --benchmark local/decode_throughput_512_100
```

Both forms work anywhere a `--runtime` / `--model` reference is taken, including
`runtimes pull` / `runtimes remove`, `models pull` / `models delete`, and
`benchmarks run`.

Any prefix of 8 hex chars or more works; an ambiguous one is an error rather
than a guess. Each scheme resolves only its own store, so a `model://` digest is
never mistaken for a runtime. It resolves against **installed** artifacts only:
a catalog pin that has never been pulled has no entry to point at, and already
has a short form (`mlx-macos-pipette://version=0.31.3`).

The digest is the SHA-256 of the descriptor's canonical JSON, which is the same
`runtime_descriptor_sha256` / `model_descriptor_sha256` the management server
records on every result, so the prefix you type here also groups that
artifact's rows in the warehouse. Model digests are taken with the auth token
stripped, matching the submitted descriptor, so a token never changes the id.

It is a convenience for typing, not an identifier to write down: it names a
descriptor's exact shape, so it moves whenever the runtime or model schema gains
a field. Plans and job files carry the full object, never a digest.

## 4. Seed local benchmarks (optional)

Remote benchmarks come from `sync`. For a local-only set that is **never**
submitted:

```bash
pipette benchmarks init-local
pipette benchmarks list
pipette benchmarks show local/decode_throughput_512_100   # id, type, source, parameters
```

That writes 34 definitions: `prefill_throughput_<n>` and `max_memory_usage_<n>`
for n in 100/256/512/1024/2048/4096/8192, `decode_throughput_<n>_100` and
`end_to_end_latency_<n>_256` for the same n, plus the `*_smoke` definitions the
examples here use.

A benchmark reference is either a bare `<id>` (equivalently `remote/<id>`),
meaning the synced catalog, or `local/<id>` for a definition only this machine
has. A bare id on a workspace that has never synced fails with
`unknown benchmark reference`.

## 5. Run a benchmark

```bash
pipette benchmarks run \
  --benchmark local/prefill_throughput_smoke \
  --model '<model-uri-or-json>' \
  --runtime '<runtime-uri-or-json>'
```

Not every benchmark type runs on every runtime:

| Runtime | prefill | decode | end-to-end | max-memory | eval | vl |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| llama.cpp | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MLX (macOS only) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| vLLM / SGLang, docker | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| vLLM / SGLang, uv | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ |
| OpenVINO | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

An unsupported pair is rejected by the engine, which for torch-oai happens
*after* the image pull, the model download and a server launch, so check the
table before starting a long fetch.

Examples:

```bash
# llama.cpp
pipette benchmarks run \
  --benchmark local/prefill_throughput_smoke \
  --model 'gguf-text://repo=LiquidAI/LFM2-350M-GGUF&path=LFM2-350M-Q4_K_M.gguf' \
  --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'

# MLX
pipette benchmarks run \
  --benchmark local/prefill_throughput_smoke \
  --model 'mlx://repo=mlx-community/LFM2-350M-4bit' \
  --runtime 'mlx-macos-pipette://version=0.31.3&flavor=macos-arm64'

# torch-oai (Linux)
pipette benchmarks run \
  --benchmark local/end_to_end_latency_smoke \
  --model 'torch://repo=LiquidAI/LFM2-350M' \
  --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu'
```

Common options. The full notation, with the per-cell settings tables, is
[models-and-runtimes.md](models-and-runtimes.md#per-cell-flags):

- `--runtime-flags '{…}'`: one JSON **object** of settings for this cell, plus a
  `raw` array of passthrough argv tokens. Not an array, and it must not carry
  the `runtime_type` / `model_type` / `benchmark_type` keys a plan entry has.
  The CLI derives the cell from `--benchmark`, `--runtime` and `--model`.
- `--model-flags '{…}'` / `--model-enable-thinking <bool>`: chat-template
  thinking override, eval benchmarks only. Mutually exclusive. Unlike
  `--runtime-flags`, the JSON form *does* take its `model_type` /
  `benchmark_type` axes.
- `--http-timeout-seconds`: eval, end-to-end-latency and vl only.
- `--readiness-max-wait-secs` / `--readiness-skip-thermal`: see
  [§5b](#5b-the-readiness-gate).
- doom-loop overrides: 36 `--doomloop-<detector>-<setting>` flags across six
  detectors, eval only. They have their own `--help` heading; the detectors are
  documented in
  [doomloop-detection.md](../pipette-doomloop/doomloop-detection.md).
- `--sync`: submit immediately after a benchmark from the synced catalog. A
  `local/` run stays on disk regardless.
- `PIPETTE_HF_TOKEN`: gated model download (injected into the model definition).

A setting a cell does not accept is an error naming the cell, never a silent
no-op:

```text
knob `ctx_size` is not accepted by prefill_throughput × LlamacppCliStockTools × GgufText
```

Results land under `results/local/` or `results/remote/pending/` as a UUID
directory with `payload.json` (+ sidecars). Check one:

```bash
pipette results list
pipette results list --state local --type prefill-throughput
pipette results show <result-id>   # payload.json, then extras.json / metrics.json
```

The payload's `runtime_flags` is the cell as it ran (the plan's entry plus the
values the client derived (context size, mmap, the docker launcher settings))
as canonical JSON of the settings alone. The cell's `(runtime, model, benchmark)`
axes aren't repeated there: the payload already names the cell through
`runtime_descriptor`, `model_descriptor`, and `benchmark_id`. Flags a benchmark
fixes for every run (`--no-warmup`, `-r 1`) and wiring (`--model`, `--host`, …)
stay out; the spawned command line is in the extras sidecar.

## 5b. The readiness gate

Timing benchmarks refuse to measure a machine that is too warm or too busy,
because a throttled or contended run is not comparable to a clean one. Before
the measurement the client polls the host and waits for both a **thermal** and a
**load** criterion; on timeout the run fails rather than reporting a bad number:

```text
readiness wait timed out after 420s: last seen thermal=pressure:0(nominal) die:52C(hot) cpu=busy:1.8cores(busy)
```

This gates prefill, decode, end-to-end-latency and vl. Eval and max-memory carry
no readiness setting at all, so passing either flag there is an error.

| Control | Effect |
|---------|--------|
| `--readiness-max-wait-secs <n>` | raise or lower the patience. Default 300s, 420s on macOS |
| `PIPETTE_READINESS_MAX_WAIT_SECS` | same, as an environment default |
| `--readiness-skip-thermal` | waive the *thermal* criterion, keeping the load one |
| `PIPETTE_READINESS_SKIP_THERMAL` | same, fleet-wide |

Waiving thermal changes the criteria rather than the patience, so a cell run
that way **is not comparable** to a gated one. It is for smoke-testing on a
working laptop, not for numbers you intend to publish. On a shared or busy
development machine the load criterion alone can also keep a run waiting: close
what is competing for the CPU, or accept that the box is not a measurement host.

## 6. Sync

```bash
pipette sync
```

1. Pulls remote benchmark definitions  
2. Submits pending results from remote benchmarks  
3. Refreshes scores for eval jobs  

```bash
pipette results list
pipette results list --state scored
pipette results show <result-id>
```

```text
local run     → results/local/{id}/              stays local
remote run    → results/remote/pending/{id}/     awaits sync
sync          → results/remote/synced/{job}/     on server
score         → metrics.json                     scored
```

## 7. Plan orchestration (push model)

Plans drive many cells across devices. The plan runner is **`pipette-plan`**
(separate workspace `.pipette-plan/`). Each target still needs `pipette`
initialized, registered, and approved.

```bash
pipette-plan init
pipette-plan status --plan plans/my-plan.toml
pipette-plan run --plan plans/my-plan.toml
```

See [plan-runner.md](../pipette-plan/plan-runner.md) and
[`examples/plans/`](../../examples/plans/).

## 8. Planner worker (pull model)

`pipette worker` turns this machine into a long-running client of the
management server's planner queue. Instead of an operator picking the
benchmark/model/runtime, the client **claims** the next eligible job, runs it
while heartbeating, submits the result (or a failure), and loops.

Prerequisites: workspace initialized, identity registered **and approved**, and
at least one runtime installed (capabilities are advertised from the runtime
store at startup).

```bash
pipette worker

# optional knobs (--heartbeat-secs defaults to half the claim's time_window):
pipette worker --idle-secs 300 --idle-jitter-secs 60 --heartbeat-secs 300 --max-jobs 0
```

At startup the worker:

1. Detects the device profile and installed-runtime capabilities
   (`runtime:llama_cpp`, `runtime:llama_cpp:b9050`, …).
2. `PATCH /clients/me` so matching stays accurate.
3. If the server reports `reindex_pending`, waits for the gate to lift.
4. Enters the claim loop.

While running a job it heartbeats at half the lease `time_window` (or
`--heartbeat-secs`), reclaims on heartbeat `404`, and on lease abort (`409` /
failed reclaim) **skips submit** after the run finishes (runtimes have no cancel
seam in v1). Submissions echo the claim's `job_id` + `model_*`/`runtime_*`.
A `403` from claim means the client is not approved: the worker logs and exits
(restart after an operator approves). Plan-attached results are not kept on disk
if submit ultimately fails (use ad-hoc `benchmarks run` + `sync` for that).

See pipette-mgmt's `docs/client-integration.md` for the full protocol.

## 9. Storage quota

`runtimes/` and `models/` are capped. After each fetch publishes, the store is
swept back under the cap: garbage first (`.staging/` orphans, entries without a
manifest this build reads), then models least-recently-used, then runtimes.
Whatever the current run needs is never evicted, and every removal is reported.
Peak disk is therefore the quota plus the artifact just downloaded. Design:
[storage-quota.md](../storage-quota.md).

```bash
pipette storage status            # usage against the quota, in eviction order
pipette storage gc --dry-run      # what would be reclaimed
pipette storage gc                # reclaim it
```

`status` reads:

```text
quota: 200.0 GiB (built-in default)
used:  143.2 GiB (71%)
free:  56.8 GiB

 KIND    | ENTRY                        | SIZE      | LAST USED            | NOTE
---------+------------------------------+-----------+----------------------+-------------
 garbage | .staging/half-fetched        | 1.2 GiB   | -                    | orphaned staging dir
 model   | org/repo:Q4_K_M.gguf         | 4.1 GiB   | 2026-07-01T09:12:03Z |
 runtime | b9305:macos-arm64            | 312.0 MiB | 2026-07-20T18:00:00Z |
```

The cap resolves in this order:

| Priority | Source |
|----------|--------|
| 1 | `--storage-quota <size>` |
| 2 | `PIPETTE_STORAGE_QUOTA` |
| 3 | `identity/settings.json` → `{"storage_quota_bytes": 214748364800}` |
| 4 | built-in default, 200 GiB |

A size is plain bytes or an IEC suffix: `200GiB`, `512MiB`, `4KiB`, `100B`.

Notes:

- An artifact larger than the whole quota is refused **before** the download
  starts, naming both sizes; raise the quota to fetch it.
- Docker runtime entries are listed but never evicted: the image lives in the
  daemon, so dropping the entry frees nothing (`docker rmi` does).
- `gc` always reclaims garbage, then evicts live artifacts only as far as it
  takes to clear an overage. An under-quota store keeps every usable artifact
  and loses its garbage.
- `status` and `gc` stay usable when an entry's manifest is unreadable: they
  treat it as garbage. That is the recovery path when a manifest-version bump
  strands a store: `pipette storage gc`, never a hand-deleted directory.
