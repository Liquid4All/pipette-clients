# Using the `pipette` CLI

**Scope:** operate the unified `pipette` client; workspace setup, registration,
installing runtimes, running benchmarks, and syncing results.

- **[Naming models, runtimes, and flags](models-and-runtimes.md)**: the URI and
  JSON notation for `--model` / `--runtime` / `--runtime-flags`, with recipes.
  Start there when the question is *how do I express the thing I want to run*.
- Backends: [llama.cpp](llamacpp.md) · [MLX](mlx.md) · [Core AI](coreai.md) ·
  [torch-oai](torch-oai.md) ·
  [OpenVINO](openvino.md) (also
  [IR](../openvino-ir.md) / [measurement](../openvino-measurement.md)) ·
  [OVMS](../ovms.md) · [ONNX GenAI](onnx-genai.md)
- [How an eval runs](eval-runs.md) · [Eval checkpoint & resume](eval-checkpoint.md)

Batch orchestration across devices is a separate binary: see
[pipette-ctl](../pipette-ctl/plan-runner.md). Building binaries and mobile apps
is covered in the [top-level README](../../README.md).

## Your first benchmark

No server, no registration, no account. A `local/` benchmark is entirely
self-contained. On a Mac:

```bash
cargo build --release -p pipette-cli        # produces target/release/pipette
export PATH="$PWD/target/release:$PATH"

mkdir -p ~/pipette-workdir && cd ~/pipette-workdir
pipette init
pipette benchmarks init-local               # 38 standard local definitions

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
| llama.cpp CUDA (every `*-cuda-*` flavor, for example `linux-cuda-12-8`, `linux-x64-cuda-13`, `windows-x64-cuda-13`) | Linux or Windows with an NVIDIA GPU | An NVIDIA driver for the build's CUDA version. Also that CUDA's runtime libraries (`libcudart`, `libcublas`) on the loader path. Upstream ships them as a separate `cudart-` file, and `pipette` does not fetch it |
| MLX | Apple Silicon Mac | [`uv`](https://docs.astral.sh/uv/) (`brew install uv`) |
| Core AI | Apple Silicon Mac with macOS 27 or later | A Swift toolchain (Xcode or the Command Line Tools) to build the sidecar. See [Core AI](coreai.md) |
| torch-oai (vLLM / SGLang) | Linux host | Docker Engine for `docker-vllm://` / `docker-sglang://`, **or** `uv` for `uv-vllm://` / `uv-sglang://`; `nvidia-container-toolkit` for NVIDIA GPUs under Docker |
| OpenVINO | Linux/Windows x86_64; refused on any other OS before install | `uv` |
| OVMS | Linux/Windows x86_64; refused on any other OS before download | None: `pipette` downloads the upstream release archive |
| ONNX GenAI CPU | macOS, Linux or Windows with a compatible wheel (includes macOS arm64) | `uv` |
| ONNX GenAI CUDA | Linux x86_64 or Windows amd64 with an NVIDIA GPU | `uv`; NVIDIA driver 580 or newer, because the catalog row installs CUDA 13 libraries |

**Conditional:**

- **Management server**: only for *remote* benchmarks and score sync. Local
  benchmarks need no server.
- **`PIPETTE_HF_TOKEN`**: only for private/gated Hugging Face models. See
  [Gated models](#gated-models).

### Where the external programs come from

`pipette` shells out to `uv` (venv-backed runtimes), `docker`
(container-backed runtimes), `nvidia-smi` and `rocm-smi` (GPU preflight), and
`python3` / `python` (anything run outside a runtime's own venv). Each is looked
for on `PATH`, once, when the command starts. A global flag or env var names a
different copy (`--uv-path` / `PIPETTE_UV_PATH` and so on; see
`pipette --help`). The client uses a path you name as given. If the program is
not at that path, the run stops. It does not look for another copy on `PATH`.

A runtime's own Python is *not* on this list: it comes from that runtime's
install under `.pipette/runtimes/`, so `--python-path` never changes which
interpreter a benchmark runs.

A missing program is an error only when a command needs it. The first step
that needs the program reports the failure. So a host with no Docker still runs
`models pull`, `results list`, and every uv-backed benchmark. It is refused only
when a cell needs a container. The GPU preflight treats a host with neither GPU
CLI as a Mac, logs that, and continues. `pipette worker` behaves the same unless
`--strict` names a program; see [§8](#8-planner-worker-pull-model).

#### Checking them

`pipette check` answers whether this host's programs are *usable*, which is a
stronger question than whether they are installed: it locates each one and then
runs it. A `docker` whose daemon is stopped is on `PATH` and fails the first
pull; an `nvidia-smi` whose kernel module has drifted from its userspace library
is on `PATH` and fails every probe.

Each check gets 15 seconds. After that, the check kills the program and
everything it started, and reports the program as failed. A hung program is one
of the faults that a check looks for. Without the limit, a check on a hung
program would also hang the startup that it protects.

```bash
pipette check all           # every program
pipette check uv            # one of them
pipette check docker        # is a daemon reachable from here
pipette check python        # `python3`, then `python`
pipette check nvidia-smi
pipette check rocm-smi
```

| Subcommand | What it runs |
|------------|--------------|
| `uv` | `uv --version` |
| `docker` | `docker info --format {{.ServerVersion}}`, so the answer covers the daemon and not just the client |
| `python` | `python3 --version` |
| `nvidia-smi` | `nvidia-smi -L`, the same NVML probe the GPU preflight runs |
| `rocm-smi` | `rocm-smi --version` |
| `all` | every row above |

`pipette worker --strict <program>` gates on the same check at startup; see
[§8](#8-planner-worker-pull-model).

`check all` prints a row per program to stdout and the failures to stderr, and
exits non-zero unless every one of them passed:

```text
 PROGRAM        | STATUS | CHECK                            | PATH                  | REPORTED
----------------+--------+----------------------------------+-----------------------+---------------
 uv             | ok     | --version                        | /opt/homebrew/bin/uv  | uv 0.11.21
 docker         | failed | info --format {{.ServerVersion}} | /usr/local/bin/docker |
 python3/python | ok     | --version                        | /usr/bin/python3      | Python 3.12.13
 nvidia-smi     | failed | -L                               |                       |
 rocm-smi       | failed | --version                        |                       |
```

`check all` wants every program, both GPU CLIs included, so **a Mac does not
pass it** and is not meant to: check the programs a Mac uses one at a time. Any
of this works before `pipette init`, with no workspace, no server and no
registration, and each check honors the flag that names its binary, so
`pipette --docker-path /opt/bin/docker check docker` checks the docker a run
would actually spawn.

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
  manifest.toml          # workspace marker
  identity/              # Ed25519 keypair, registration, settings.json
  runtimes/              # installed runtime builds
  models/                # cached model artifacts
  benchmarks/
    local/
    remote/
      detail/            # definitions as the server sent them and their ETags,
                         # created by the first sync or claim (see §6)
      state/             # the server's list and the sync state, created by
                         # the first sync
  results/
    local/
    remote/{pending,synced}/
  state/evals/           # resumable eval checkpoints
  state/worker/          # the worker's in-flight job marker
  cache/<runtime-key>/   # compiled artifacts, created on first compile
  datasets/<benchmark-id>/  # a job's corpus, for one job only
```

The first engine that compiles something creates `cache/`; `init` does not.
`cache/` is outside the artifact stores, so a storage sweep cannot delete what a
run depends on. For the same reason, `storage status` does not count it and
`storage gc` does not sweep it. `runtimes remove` deletes a runtime's `cache/`
entry.

A corpus benchmark creates `datasets/`. It downloads its workload from the
management server for the one job that runs it:

- `vl_spec_decode` unpacks the corpus into `datasets/<benchmark-id>/` and
  deletes the tree when the job ends. A tree left behind belongs to a job that
  stopped between unpack and cleanup. The next run of that benchmark replaces
  it.
- `asr_eval` writes the corpus as one tar file in `datasets/<benchmark-id>/`,
  reads it in place, and deletes it when the job ends. The next run deletes a
  file that a killed run left behind.

Like `cache/`, `datasets/` is outside the quota.

`models/` is the shared model store across backends (GGUF files, Hugging Face (HF) snapshots,
etc. handed to engines as local paths: no ambient HF cache required for normal
URI runs). `runtimes/` is the common root for installed builds; see
[§3](#3-install-a-runtime) for how each scheme lands there.

The working directory resolves as described in
[Workspace](../architecture.md#workspace). Commands other than `init`,
`pipette --version`, `check`, `runtimes catalog <type>` and `runtimes flavors`
require an initialized workspace. `check`, `runtimes catalog` and
`runtimes flavors` ask about the host or the upstream catalog, so they work in
any directory. You can check that this host has a working `uv`, and pick a
runtime, before you create the workspace. `runtimes flavors` also needs no
network.

## 2. Register

Skip this section entirely if you only run `local/` benchmarks: they need no
server, and their results never leave the machine.

```bash
pipette auth register \
  --organization LiquidAI \
  --contact-email user@example.com \
  --client-details "mac studio for benchmarks" \
  --preauth-key client_preauth_...   # optional
```

`--server-url` defaults to `https://collector.pipette.liquid.ai`. This is the
only defaulted server URL in the CLI: registration is what records the answer.

This writes an Ed25519 keypair under `identity/` and registers the public key.
Status starts as `pending` unless a valid `--preauth-key` (or
`PIPETTE_PREAUTH_KEY`) is accepted: then the client can come up `approved`.
The pre-auth key is a device's credential. An account registers with
`pipette-ctl auth register --user-preauth-key`, against a different record on
the server. The key is never written to disk. If a local identity already exists,
`auth register` refuses; use `pipette auth reset` to start over, or
`auth set-device` to relabel without rotating keys.

The client detects the device name and form factor. Set them yourself only to
tell identical models apart, or to correct a wrong form factor. `auth register`
takes the same two flags.

```bash
pipette auth set-device --device-name "Mac Studio M3 Ultra 96GB" --device-form-factor desktop
pipette auth set-device --client-details "mac studio for benchmarks"
```

The form factor is one of `phone`, `tablet`, `laptop`, `desktop`, `server` or
`embedded`. The two device labels are local, and a `worker` sends them at
startup. `--client-details` is registration data on the server, so a change to
it needs a registered client and a reachable server.

`auth reset` deletes the identity and everything derived from the server
relationship: private key, registration, device labels, pulled remote
benchmarks, remote results (pending *and* synced), and eval score-refresh state.
Local benchmarks and local results are server-independent and survive. It
refuses while unsubmitted remote results exist unless you pass `--force`.

```bash
pipette auth me           # client_id, organization, contact, status, tags,
                          # reindex_pending, and the advertised capabilities
pipette auth me --local   # client_id, server_url, public_key, from disk
pipette sync              # pull the remote benchmark catalog (see §6)
```

`--local` makes no request. It answers with the three fields a machine can
answer about itself, which is what a provisioning step can assert on before the
server has approved the client. Everything else `auth me` prints is server-owned
and has no local copy.

### Moving an identity to another machine

The `client_id` is assigned once, in the registration response, and nothing can
recover it afterwards: it is a field inside the signed payload every
authenticated request carries, so `auth me` cannot be called without already
knowing it, and there is no lookup-by-public-key endpoint. Re-registering creates
a *different* id, which strands every result already submitted under the old
one. `auth export` is how that id, its server URL, and the key that signs for
it leave a machine together.

```bash
pipette auth export --out identity.bundle    # 0600; omit --out for stdout
pipette auth import --from identity.bundle   # or pipe it in on stdin
```

The bundle is validated in full before `auth import` writes anything, so a
token this build cannot read leaves the workspace untouched. Re-importing the
identity a workspace already holds is not a conflict, which lets a rerun finish
an import that failed partway. Replacing a *different* identity needs `--force`
and drops the caches derived from the outgoing one, for the reason `auth reset`
drops them: they are keyed to the `client_id` being replaced, and keeping them
would submit one client's runs under another's id on the next `sync`.

`auth import` reports whether it wrote:

```
client_id:  ev1_demo
server_url: https://collector.example.test
public_key: 40c76ce3…
changed:    no
```

`changed: no` means every file already held what the bundle carries, and nothing
was rewritten, not even to the same bytes. A configuration manager can run the
import on every pass and key its own change reporting on that line.

### Restoring an identity from a configuration manager

The bundle is a per-host secret, not a fleet-wide one. Two machines signing
under one `client_id` interleave their results under a single client, and each
one's `worker` startup overwrites the server's device profile with its own
hardware. Store one bundle per host and assert the restored `client_id` is the
one that host is supposed to have.

`auth import` reads stdin by default so the token never has to reach the target
filesystem or a command line. With Ansible that is the `stdin:` parameter of
`ansible.builtin.command`, and the identity task belongs after `pipette init`
(which import requires) and after a `pipette sync`. Draining unsubmitted
results first is what keeps a rebuild from ever needing `--force`.

Device labels and the storage quota stay behind deliberately. They describe the
machine, not the identity, so the importing machine detects its own hardware and
the first `worker` startup pushes that profile to the server under the restored
`client_id`.

The bundle holds the private key in plaintext, and so does the output of:

```bash
pipette auth secret    # the private key hex alone, nothing else
```

That verb exists for a prompt that asks you to prove ownership of the client by
pasting the key. Be sure it is one you trust with the key itself: a public key
is accepted at registration and nowhere else, so the server has no way to rotate
one, and a disclosed key can only be retired by re-registering into a new
`client_id`. Where you only need to *identify* the client, the public key that
`auth export` and `auth import` print discloses nothing.

## 3. Install a runtime

Runtimes are named with a compact URI or a JSON `Runtime` object. The notation
is [models-and-runtimes.md](models-and-runtimes.md).

```bash
pipette runtimes pull --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'
pipette models pull --model 'gguf-text://repo=unsloth/gemma-3-270m-it-GGUF&path=gemma-3-270m-it-Q4_K_M.gguf'
```

`benchmarks run` installs a missing runtime **and** a missing model through the
same store path that `runtimes pull` and `models pull` use. An explicit pull
only moves the download earlier. It does not change what a run does. The scheme
decides where the build goes. Every install in the store is under `runtimes/`:

| Scheme | Installs as | Browse with |
|--------|-------------|-------------|
| `llamacpp-cli-stock-tools://` | upstream release archive | `runtimes catalog llamacpp_cli_stock_tools --flavor <flavor>` |
| `docker-vllm://` / `docker-sglang://` | image in the docker daemon, plus a workspace record that `runtimes list` shows | — |
| `mlx-macos-pipette://` | relocatable venv (macOS) | `runtimes catalog mlx_macos_pipette` |
| `core-ai-macos-pipette://` | the Core AI engine that macOS includes, plus a cached Swift sidecar outside the store (macOS 27+) | [Core AI](coreai.md) |
| `uv-vllm://` / `uv-sglang://` | relocatable venv (Linux) | `runtimes catalog uv_vllm` / `uv_sglang` |
| `uv-openvino://` | relocatable venv (Linux/Windows). One venv serves cpu, gpu and npu, because `device` is a per-cell setting | `runtimes catalog uv_openvino` |
| `uv-onnx-genai://` | relocatable venv. One venv per execution provider, because each provider is a separate wheel | `runtimes catalog uv_onnx_genai` |
| `ovms://` | upstream release archive (Linux/Windows), checked against the sha256 that the bundled catalog pins for it. The flavor must match the host | `runtimes catalog ovms` |

A uv or MLX requirement set that is not a catalog row has no URI form. Pass it
as a JSON `--runtime` with a full `pip_requirements_text` source.

```bash
pipette runtimes flavors                    # the --flavor vocabulary and the
                                            # upstream asset each resolves to
pipette runtimes list
pipette runtimes list --format uri          # rows paste back into `runtimes pull`
pipette runtimes remove --runtime '<uri>'   # deletes the install and its cache/;
                                            # a docker image stays until `docker rmi`
pipette models list                         # cached models, with their digests
pipette models delete --model '<uri>'
```

`runtimes flavors` and `runtimes catalog` work without a workspace (see
[§1](#1-initialize)).

An installed runtime or model can also be named by the digest `list` prints
(`runtime://sha256=c07a4fd3`); see
[The digest form](models-and-runtimes.md#the-digest-form).

## 4. Seed local benchmarks (optional)

Remote benchmarks come from `sync`. For a local-only set that is **never**
submitted:

```bash
pipette benchmarks init-local
pipette benchmarks list
pipette benchmarks show local/decode_throughput_512_100   # id, type, source, parameters
```

That writes 38 definitions: `prefill_throughput_<n>` and `max_memory_usage_<n>`
for n in 100/256/512/1024/2048/4096/8192, `decode_throughput_<n>_100` and
`end_to_end_latency_<n>_256` for the same n, plus ten `*_smoke` definitions.
There is one smoke definition per benchmark type, except `vl_spec_decode` and
`asr_eval`. Their corpus comes from the server, so they have no local
definition.

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

| Runtime | prefill | decode | end-to-end | max-memory | eval | vl_throughput | vl_max_memory | vl_spec_decode | vlm_eval | asr_eval | asr_throughput | asr_max_memory |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| llama.cpp | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MLX (macOS only) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| vLLM / SGLang (docker or uv) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | SGLang only | uv vLLM only | uv vLLM only | ❌ | ❌ |
| OpenVINO | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| OVMS | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| ONNX GenAI | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Core AI (macOS 27+) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

This table is the summary for every backend. Each backend page gives the
conditions for its cells. The engine rejects an unsupported pair. For
torch-oai, that happens *after* the image pull, the model download and a server
start, so check the table before you start a long download.

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

`pipette benchmarks run --help` lists the options. The per-cell flag families
(`--runtime-flags`, `--model-flags`, `--benchmark-flags`) are described in
[models-and-runtimes.md](models-and-runtimes.md#per-cell-flags), the readiness
flags in [§5b](#5b-the-readiness-gate), and `--config` in
[§8a](#8a-client-configuration). `--sync` submits a result from the synced
catalog at once; a `local/` run stays on disk regardless.

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
pipette results list --benchmark local/prefill_throughput_smoke --limit 5
pipette results show <result-id>   # payload.json, then extras.json / metrics.json
pipette results delete <result-id>
```

The payload's `runtime_flags` is the cell as it ran, as canonical JSON of the
settings alone: the plan's entry plus the values the client derived (context
size, the weight-load setting, the docker launcher settings). The cell's
`(runtime, model, benchmark)` axes aren't repeated there: the payload already
names the cell through `runtime_descriptor`, `model_descriptor`, and
`benchmark_id`. Flags a benchmark fixes for every run (`--no-warmup`, `--repetitions 1`)
and wiring (`--model`, `--host`, …) stay out; the spawned command line is in the
extras sidecar.

**A run whose session ends.** Over adb or ssh (how `pipette-ctl` runs a cell),
the run's stderr pipe is the session. When the session ends, the run stops
every runtime it started and exits 141, so the device is idle for the next
cell. On Unix this happens at once; on Windows, at the run's next log line.

### Gated models

A private or gated Hugging Face model needs `PIPETTE_HF_TOKEN` in the
environment of `models pull`, `benchmarks run`, or a `worker` that may claim
it. The client folds the token into each Hugging Face source of the model that
carries none, and leaves it out of the submitted descriptor and the digest, so
a token never changes a model's identity. Filing a plan for a gated model is
covered in [job-generation.md](../pipette-ctl/job-generation.md#gated-models).

## 5b. The readiness gate

Timing benchmarks refuse to measure a machine that is too warm or too busy,
because a throttled or contended run is not comparable to a clean one. Before
the measurement the client polls the host and waits for both a **thermal** and a
**load** criterion; on timeout the run fails rather than reporting a bad number:

```text
readiness wait timed out after 420s: last seen thermal=pressure:0(nominal) die:52C(hot) cpu=busy:1.8cores(busy)
```

The gate applies to prefill, decode, end-to-end-latency, `vl_throughput`,
`asr_throughput` and `vl_spec_decode`. `eval` and the three peak-memory
benchmarks (`max_memory_usage`, `vl_max_memory`, `asr_max_memory`) have no
readiness setting, so a readiness flag on them is an error.

`vlm_eval` and `asr_eval` wait once, before the first sample. On timeout they
log a warning and run on a warm device; they do not fail. The wait protects a
device from thousands of back-to-back images or clips. The pinned decoding
gives the same answers on a hot or a cold device. What each platform's
gate waits for, and its default deadline, is in
[System Readiness Control](../methodology/end-to-end-latency.md#system-readiness-control).

From the command line, `--readiness-max-wait-secs` (or
`PIPETTE_READINESS_MAX_WAIT_SECS`) changes the deadline and
`--readiness-skip-thermal` (or `PIPETTE_READINESS_SKIP_THERMAL`) waives the
thermal criterion while keeping the load one. On the Android CLI,
`--readiness-skin-below-c 34` holds each rep until the hottest live HAL skin
sensor is below 34 °C. It overrides `[readiness.android] skin_below_c` for that
run; both forms accept whole degrees from 1 to 110. Other platforms reject the
skin flag. A Raspberry Pi 5's throttle bits and a Jetson Orin's throttling
cooling devices and over-current events still hold the gate when thermal is
waived. Android continues to require a quiet CPU load; its thermal status,
CPU-zone, and skin checks are waived together. A host's own numbers go in its
configuration file:
see [Tuning the readiness gate](#tuning-the-readiness-gate).

On the Android CLI, the gate also holds a rep while the touchscreen has a held
touch. A screen cooler can cause such a "phantom touch", and the touch
stays until an operator removes the cause. During the 1-second CPU sample, the
gate counts the interrupts (IRQs) of a recognized touch controller.

The gate reads `dumpsys input` to find out if a pointer is down when one of
these is true:

- the controller's rate is 20 or more interrupts per second
- the summed IRQ count decreases, as when a CPU goes offline
- a known controller's count is malformed

A held touch can pass the gate when the rate stays below 20 per second, or when
the gate does not recognize the controller's IRQ name. The 20/s threshold is
provisional: no contact rate has been measured.
`--readiness-held-touch` selects what the gate does about a held touch:

- `off` disables touch detection. The gate does not read `/proc/interrupts` or
  `dumpsys input`; its thermal and CPU checks still apply.
- `wait` (the default) holds the gate until the touch clears or the deadline
  passes, as for a busy CPU.
- `fail` fails the cell after three consecutive polls detect the touch. This
  takes about 9 seconds. A `pipette worker` does not submit that run, so the cell
  keeps its attempts. The worker then stops with an error.

To disable touch detection for one Android client, set its client configuration:

```toml
[readiness.android]
held_touch = "off"
```

The flag overrides `PIPETTE_READINESS_HELD_TOUCH`, and the environment
variable overrides `[readiness.android] held_touch`. Other platforms reject the
flag. `--readiness-skip-thermal` does not waive touch detection. The poll line
shows `touch=off` when detection is disabled, `touch=irq:<rate>/s(ok)` when the
rate stays below the trigger, and
`touch=irq:<rate>/s,pointers:<n>(held)` for a held touch. A phone with no
known touch interrupt line shows `touch=unchecked`, and the gate cannot check
it for a held touch.

Waiving thermal changes the criteria rather than the patience, so a cell run
that way **is not comparable** to a gated one. It is for smoke-testing on a
working laptop, not for numbers you intend to publish. On a shared or busy
development machine the load criterion alone can also keep a run waiting: close
the programs that compete for the CPU, or use a different host for measurement.

## 6. Sync

```bash
pipette sync
```

1. Pulls remote benchmark definitions  
2. Submits pending results from remote benchmarks  
3. Refreshes scores for `eval`, `vlm_eval` and `asr_eval` jobs  

A pending result another process is already submitting (a `benchmarks run
--sync`, or a second `sync`) is skipped, not counted as failed, so it is
posted once.

Step 1 needs no identity, since the management server serves the benchmark
catalog to anyone; a registered workspace signs the pull anyway, because the
catalog can hold benchmarks shown only to the principal that asks. Steps 2 and
3 are authenticated. The routes are in pipette-mgmt
[`docs/httpapi.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/httpapi.md).
`sync` therefore runs in one of two modes:

| Workspace | What `sync` does |
|---|---|
| Registered | all three steps, against the registered server; the pull is signed |
| No identity | step 1 only, unsigned, and only with `--server-url` |

An unregistered `sync` exits `0`, since the pull is the whole job it was able
to do, and `benchmarks list` then shows the `remote/` definitions. This is how a
machine sees what there is to run before committing to registering to run it.
A registration whose private key is missing counts as "no identity" for signing,
but still names the server the pull goes to.

The pull goes to the registered server, else `--server-url`, else nowhere:
`sync` refuses rather than falling back to an address nobody named. A
`--server-url` that disagrees with the registration is an error, because the
identity signs only for that one host, so there is no coherent run that pulls
from one server and submits to another. The cached catalog records which server
it came from, so a pull aimed at a different one starts cold.

```bash
pipette sync --server-url https://collector.pipette.liquid.ai   # unregistered
```

`pipette sync <result-id>` is an instruction to submit, so it is refused
outright without a registration rather than pulling and silently dropping the
argument.

**How definitions are cached.** `sync` keeps these files under
`benchmarks/remote/`:

- `state/index.json`: the `GET /benchmarks` list, every row as the server sent
  it.
- `state/sync.json`: the server that `sync` pulled from, and the list's ETag.
- `detail/<id>.json`: the definition of each row whose type this build knows
  and whose id it accepts, as the server sent it, every key included.
  `pulled N benchmark(s)` counts these rows. The client converts a definition
  only when it reads it. A definition that does not convert is logged and not
  cached.
- `detail/<id>.etag`: the definition's ETag, beside a SHA-256 of the body.
  `sync` sends the ETag only while the body still matches the hash. So a
  concurrent rewrite costs a full fetch, never a wrong `304`.

Because the cache keeps what the server sent, a parameter that this build does
not model is kept for a build that does model it. Also, a `304` on the list's
ETag cannot give one build a list that another build filtered. A row that this
build does not fetch (for example, a type it does not know) stays in the list.
The prune keeps every listed id, so a definition that a newer build cached for
that row stays. A pull from a different server sends no ETags. A reference is
`remote/<id>`, or a bare `<id>`.

A `worker` never runs `sync`. Instead, each claim revalidates its benchmark
with a signed `GET /benchmarks/{id}` that carries the cached ETag. The
management server's ETag hashes the whole body:

- A `304` means that the parameters, the corpus pin and the eval samples are
  current. The cached definition runs.
- A `200` replaces the cached definition, and the replacement runs. So the
  result declares the `benchmark_params` that the server checks it against.
- A cached definition without an ETag, or one that does not read, is fetched
  whole.
- A request that fails or is refused, or an answer that does not parse, runs
  the cached definition with a warning. If nothing is cached, the claim fails.
- A `404`, or a definition that this build cannot convert, fails the claim in
  every case. The server refuses a result for a definition that it does not
  serve.

The worker logs the outcome (`current`, `changed`, `fetched` when the cached
definition had no ETag, `unchecked`) with the `benchmark_id`. It never writes
`remote/state/sync.json`.

**Release 2026.08.1 and earlier in the same workspace.** Releases up to
2026.08.1 keep `<id>.json`, `index.json` and `sync.json` directly in
`benchmarks/remote/`. They store each definition as they parsed it, so they
lose a parameter that they do not model: 2026.08.1 reads an 80-image VL
benchmark back as one image. Later releases do not read those files. They
delete them only with the rest of `remote/` (`auth reset`, or an `auth import`
that replaces the identity). Releases up to 2026.08.1 do not read `detail/` or
`state/`, so both can use one workspace. In a workspace that only such a
release synced, the first `sync` of a later release is a full pull, and a `worker` fetches each benchmark on its first
claim. Until then, `show` or `run` of an id reports the id as unknown.

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

Plans drive many cells across devices. The plan runner is **`pipette-ctl`**
(separate workspace `.pipette-ctl/`). Each target still needs `pipette`
initialized, registered, and approved.

```bash
pipette-ctl init
pipette-ctl status --plan plans/my-plan.toml
pipette-ctl run --plan plans/my-plan.toml
```

See [plan-runner.md](../pipette-ctl/plan-runner.md) and
[`examples/plans/`](../../examples/plans/).

## 8. Planner worker (pull model)

`pipette worker` turns this machine into a long-running client of the
management server's planner queue. Instead of an operator picking the
benchmark/model/runtime, the client **claims** the next eligible job, runs it
while heartbeating, submits the result (or a failure), and loops.

Prerequisites: workspace initialized, and identity registered **and
approved**. No external program is required by default: a
job that needs a missing `uv` or `docker` fails when it would have spawned it,
like any other command.

`--strict <program>` adds a startup gate for the programs *you* name: each one
has to answer its own check ([above](#checking-them)), not merely be on `PATH`,
before the worker claims anything. Which programs a host should have is an
operator fact, so the worker asks rather than guesses. The gate reports every
program that failed, not the first, so you can fix a host in one pass:

```text
Error: every program named by --strict has to answer for itself before this worker takes a lease

Caused by:
    docker: `/usr/local/bin/docker info --format '{{.ServerVersion}}'` failed (exit status: 1): Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
    nvidia-smi: nvidia-smi not found on PATH; install the NVIDIA driver
```

`pipette worker --help` lists the settings: `--idle-secs` (default 300) and
`--idle-jitter-secs` (default 60) for the wait after an empty claim,
`--heartbeat-secs` (default: half the claim's `time_window`), `--max-jobs`
(default 0, which runs until stopped), `--strict`, and `--skip-profile-refresh`
(for debugging only). `--config` applies to every job the worker takes
([§8a](#8a-client-configuration)).

At startup the worker:

1. Reads the `auth set-device` labels, which name the device on the profile
   and on every result it submits. They are not read again, so relabel a
   running worker by restarting it, and a job whose workspace vanishes
   mid-run (a preempted SLURM job's `/tmp`) still reports them.
2. Detects the device profile, then the hardware, installed-runtime, and client
   build capabilities (`runtime:llama_cpp`, `runtime:llama_cpp:b9050`,
   `cpu_vendor:intel`, `accel:gpu`, `client_version:2026.08.11-0123abcd`, …).
   Runtime capabilities come from the runtime store, so install what this
   machine should be offered before starting. An installed runtime whose
   manifest does not parse is logged and not advertised; the next job that
   names it installs it again. A build that names no release (a local build
   reports `dev`) advertises no `client_version:` flag.
3. Pushes that profile to the server so matching stays accurate.
4. Waits out a server reindex if one is pending.
5. Enters the claim loop.

The worker follows the claim, heartbeat, reclaim and submit protocol in
pipette-mgmt
[`docs/client-integration.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/client-integration.md)
(queue semantics:
[`docs/queue.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/queue.md)). What that means on this side:

- A lost lease does not stop the run: the benchmark finishes and its result is
  not submitted.
- A claim refused because the client is not approved stops the worker; restart
  it after an operator approves.
- A claimed job's result is not kept on disk if its submit ultimately fails.
  Use `benchmarks run` + `sync` when a result must survive a failed submit.
- A job the device rebooted during is reported as a permanent failure when
  this worker claims it again, not run again, so a job that crashes its device
  cannot loop.
  The proof is a marker in `state/worker/` holding the kernel boot ID, so this
  works only on Linux and Android, and only with a workspace that survives the
  reboot. An orderly stop (Ctrl-C, or a reboot that sends SIGTERM) removes the
  marker and is never counted.

**Stopping a worker.** Ctrl-C or SIGTERM while the worker is *waiting* (idle
between jobs, backing off a failing claim, or backing off a failing submit)
stops the loop within a second and exits 0, the same exit `--max-jobs` takes.
A worker that was already backing off a submit drops that result, as an
exhausted backoff does. It also ends a readiness wait (the cooldown before a
measured rep, which can be five minutes), and the cell it was gating is dropped
rather than reported as a failure, so a stop costs the cell none of its
attempts. During a run with a server already up the signal still stops the
benchmark and exits 130.

**A worker whose session ends** stops its runtimes and exits 141, as
[a run does](#5-run-a-benchmark). Over a pty (`ssh -t`, an interactive
`adb shell`), the hangup signal can stop the worker first, and the worker then
exits 130. On Windows, the runtimes also stop if a hard kill
(`Stop-Process -Force`) ends the worker.

**A run that is killed.** A SIGKILL, an out-of-memory (OOM) kill or an abort
runs no teardown, so the programs that the run started keep running. Pipette
cleans them up later:

- Each program that pipette starts (a server, or a short command such as a
  program check) leads a session of its own.
- Pipette records that session in a directory that only this user can write.
- The program carries a token in its environment, and the programs that it
  starts inherit the token.
- The next server start, and a worker before its first claim, kill every
  session whose starter is gone.
- After the first process of a session exits, its session id can be reused by
  another session of the same user. So pipette kills a session only while one
  of its processes still carries the token.
- Some macOS releases do not let any user read the environment of their own
  system binaries, such as `sh` or `sleep`. On those releases, a session that
  holds only such binaries keeps running.

On Windows, a Job Object stops every child process when pipette exits.

### 8a. Client configuration

`--config <toml>` describes how *this client* operates. It holds facts about
the host, never about a cell: nothing in it enters the cell key, the request,
or the reported `runtime_flags`. It is read once at startup, validated, logged
by name, and then applies to every run the process makes, whether that is one
`benchmarks run` or every claim a `worker` takes.

A result does record which configuration it was run under, in its own
`client_configuration` field beside `runtime_flags`. See
[what a result records](#what-a-result-records-about-the-configuration) below.

Example files are in [`examples/client-configs/`](../../examples/client-configs/),
and a test parses each one.

The shape mirrors a plan's `runtime_flags`, split by runtime type alone: one
`[[runtimes]]` entry per runtime type this client spawns.

```toml
# One node, two cards, two clients: this is the client that owns card 1.
[[runtimes]]
runtime_type = "llamacpp_cli_stock_tools"
envs = ["CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=1"]

[[runtimes]]
runtime_type = "uv_vllm"
envs = ["CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=1", "VLLM_WORKER_MULTIPROC_METHOD=spawn"]
```

- `runtime_type`: the same vocabulary a plan's `runtime_type` uses. The types
  this client runs: `llamacpp_cli_stock_tools`, `mlx_macos_pipette`,
  `docker_vllm`, `docker_sglang`, `uv_vllm`, `uv_sglang`, `uv_openvino`,
  `uv_onnx_genai`, `ovms`.
- `envs`: `NAME=VALUE` entries, the same notation as a cell's `envs`, set on
  every process a run of that runtime spawns (server, bench tool, driver,
  tokenizer, calibration helper, crash-restarted server). They land *first*, so
  anything the client itself sets for that launch wins: a host file describes
  topology and must not be able to clobber, say, the venv `bin/` a uv server's
  `PATH` leads with. Against the cell the order never arises, since a
  cell naming the same variable is refused. Literal values only: the client's
  environment *is* the host's, so the bare inherit form would say nothing. A
  docker runtime receives them by name (`-e NAME`, value on the docker CLI's
  environment), so a value never reaches argv or the log.

Which card a run lands on is one of these entries, not a knob of its own:
`CUDA_VISIBLE_DEVICES` on every CUDA-reading type, with `CUDA_DEVICE_ORDER`
beside it so the index is the one `nvidia-smi` shows. A host with two cards is
two clients, each with its own workspace, identity and file; the warehouse
tells them apart by client, and the record names the variable without its
value, so it still says nothing about which card.

Rules, each refused at startup with the offending name:

- an unknown table or key is an error, so a typo cannot silently apply to
  nothing;
- a runtime type outside the vocabulary is an error, and so is a type named
  twice;
- a type that runs on a device rather than through this client
  (`llamacpp_apk_pipette`, `llamacpp_ios_pipette`, `mlx_ios_pipette`,
  `onnx_genai_ios_pipette`, `apple_foundation`) is an error;
- `core_ai_macos_pipette` is an error, because it has no configuration fields;
- an env name must be one a shell accepts, a value may not contain NUL, and
  the inherit form is refused.

And per cell, at run time: a cell whose `envs` forward a variable its runtime's
entry sets is refused. The operator's file and the plan author's cell are not
allowed to disagree silently. A cell that addresses devices on its own (a
docker `gpus` narrower than `all`, `tensor_parallel_size` above the visible
device count, a llama.cpp `-dev` in `raw`) is not checked against the file,
since the client cannot know what an env entry means; it fails at the
runtime's own startup instead.

#### Tuning the readiness gate

A `[readiness.<gate>]` table tunes the readiness gate this host runs, named as
the gate names itself in its log line: `windows`, `linux`, `pi5`, `orin`,
`macos` or `android`. Its numbers are measured per device type, not derived on the host.

| Gate | Key | Holds a rep until |
|------|-----|-------------------|
| `windows` | `ceilings` | each named sensor reads at or below its ceiling, °C |
| `windows` | `max_cpu_busy_pct` | the CPU busy share is at or below it |
| `linux` | `hottest_below_c` | the hottest thermal zone reads below it, °C |
| `linux` | `cpu_load_below_pct` | the CPU load, as a share of the cores the client may run on, is below it |
| `pi5` | `hottest_below_c` | the hottest thermal zone reads below it, °C |
| `pi5` | `cpu_load_below_pct` | as on `linux` |
| `orin` | `hottest_below_c` | the hottest thermal zone reads below it, °C |
| `orin` | `cpu_load_below_pct` | as on `linux` |
| `orin` | `gpu_load_below_pct` | the GPU busy share is below it; not gated where the kernel exposes no GPU load |
| `macos` | `die_below_c` | the hottest die sensor reads below it, °C |
| `macos` | `cpu_busy_below_cores` | fewer cores than this are busy |
| `android` | `hottest_below_c` | the hottest CPU-cluster zone reads below it, °C |
| `android` | `skin_below_c` | the hottest live HAL skin sensor (`mType=3`) reads below it, °C; unset means off |
| `android` | `cpu_busy_below_pct` | the CPU busy share is below it |
| `android` | `held_touch` | `off` disables touch detection; `wait` requires no detected held touch; `fail` ends the wait after three consecutive held polls. Default: `wait`. |

The keys and their ranges are defined in
[`configuration.rs`](../../crates/pipette-readiness/src/configuration.rs); each
gate's own numbers are in its module there and in
[System Readiness Control](../methodology/end-to-end-latency.md#system-readiness-control).
Every key is optional, and a gate keeps its own number for a key the table
leaves out.

##### Linux, Raspberry Pi 5, Jetson Orin, macOS and Android

A `*_below_*` key replaces one number the gate already compares against, in the
gate's own comparison: a reading equal to the number still holds the rep. So
on a host that reads the temperature, writing the gate's own number changes
nothing, and the gate's other criteria (the macOS thermal-pressure level, the
Android thermal status, the Pi 5 firmware's throttle bits, the Orin's
throttling cooling devices and over-current events) stay as they are.

```toml
# A Mac whose die idles near the 50 °C the gate was measured with on another
# machine type. 60 is a placeholder: write the number measured on that type.
[readiness.macos]
die_below_c = 60
```

- A key names the reading as that gate's log line prints it, so a key written
  for another gate's reading is refused: Linux reads a load average
  (`cpu=load:`), Android a busy share (`cpu=busy:N%`), macOS busy cores
  (`cpu=busy:Ncores`).
- On Linux, the Pi 5 and the Orin, `cpu_load_below_pct` holds either CPU reading the
  gate takes: the load average (`cpu=load:N%`), or the busy share of its own
  cores (`cpu=busy:N%`) for a client confined to some of the host's
  ([Linux](../methodology/end-to-end-latency.md#linux)).
- `--readiness-skip-thermal` waives the temperature keys with the rest of the
  thermal criterion. The CPU and GPU keys still apply.
- A temperature key needs a temperature to hold to it. A Linux host with no
  thermal zone fails the wait at once instead of gating on load alone, and a
  Mac whose die does not read is held (`die:unreadable(held)`) until the wait's
  deadline fails the cell, instead of being let through on the pressure level
  alone.
- The verdicts on the gate's line (`(ok)`, `(hot)`, `(busy)`) are judged against
  the table's numbers. The line rounds the reading (the macOS die to a whole
  degree, busy cores to a tenth, a percentage to a whole one) and the gate
  compares it unrounded, so a reading printed equal to the number can show
  either verdict.
- `cpu_busy_below_cores` above the machine's core count leaves the CPU
  criterion nothing to hold.
- A CPU key loosened to make room for other work on the host measures next to
  that work. That is a different measurement, not a quieter gate.
- `[readiness.android]` is read by the CLI on an Android device. The app runs
  its own gate and does not read the file. A configured `skin_below_c` is
  recorded in `benchmark_flags.readiness.skin_below_c` with each result,
  including a run that skips thermal gating. A complete live HAL block without
  a usable skin sensor on three consecutive polls fails the wait with a clear
  error. The poll line shows `skin:<value>°C(ok|hot)` when bounded and
  `skin:<value>°C(off)` when observed with no skin bound.

##### Windows

```toml
# Zenbook S 16: the firmware cuts power from 28 W toward 17 W once THZ0 reaches
# 68 °C, and gives it back only once THZ0 has cooled to about 43 °C.
[readiness.windows]
max_cpu_busy_pct = 15
ceilings = { "acpi:THZ0" = 43 }
```

- `ceilings`: sensor to ceiling, in whole °C. Before every timed rep the gate
  also waits until each named sensor reads at or below its ceiling. A sensor
  is `acpi:<zone>`, a Windows ACPI thermal zone by its last segment (`THZ0` for
  `\_TZ.THZ0`, any case), or `gpu`, the GPU's own temperature. The gate prints
  every sensor under these names on each poll. A poll that fails to read a
  sensor holds the gate like a hot one. A zone the host does not list fails the
  cell at once, naming the zones it does list. After 240 s, a sensor that has
  read but is still over its ceiling is let through with a warning, so a warm
  room costs time rather than failed cells; one that never read fails the
  cell.
- `max_cpu_busy_pct`: the CPU busy share, over all logical CPUs, that still
  counts as idle. Idle Windows hosts read 0 to 11 %.

##### Every gate

A file tunes one gate, and a client refuses a table for a gate it does not run.
Each value is range-checked at startup. The table only tunes the gate, and no
result carries it. A result records what the gate saw before each rep
(`readiness_gate_waits`) and the device's own readings around it: see
[Per-rep readings](../methodology/device-conditions.md#per-rep-readings).

#### What a result records about the configuration

A run under an entry submits that entry in `client_configuration`, canonical
JSON in a string like `runtime_flags` and `benchmark_flags` beside it. It is
recorded because it is a condition the measurement was made under, in the same
sense as the power and thermal fields: `GGML_CUDA_ENABLE_UNIFIED_MEMORY=0` in
one host's file moves that host's numbers while every other field of the payload
stays identical, and nothing else would tell the two apart.

What it carries follows the rule a cell's own `envs` already follow:

- env forwards by **name**, values dropped, since an operator authors the file
  and the record leaves the device;
- except in the `GGML_*` namespace, which keeps its values, because there they
  are the measurement rather than a secret;
- no `runtime_type`, since `runtime_descriptor` already names the runtime.

So the two-cards file above records as
`{"envs":["CUDA_DEVICE_ORDER","CUDA_VISIBLE_DEVICES"]}`. A client with no entry
for the runtime it ran omits the field.

The entry types and the recorded shape are defined in pipette-mgmt
[`crates/pipette-plan-types/src/extra_runtime_configuration.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/extra_runtime_configuration.rs).

## 9. Storage quota

`runtimes/` and `models/` are capped, and each fetch sweeps the store back
under the cap. The mechanism, the eviction order and how the cap resolves are
in [storage-quota.md](../storage-quota.md); `pipette --help` documents
`--storage-quota`.

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

`status` and `gc` treat an entry whose manifest they cannot read as garbage, so
a store stranded by a manifest-version bump is recovered with
`pipette storage gc`, never a hand-deleted directory.

## 10. Logs

Logs go to stderr; stdout stays parseable (`results show`, `auth export`,
every `list`). The default is `info` for every `pipette*` crate (the CLI, the
runtime crates that narrate server startup, the artifact stores) and `warn`
for dependencies.

`pipette --help` documents the level and format flags (`-v`, `--log-level`,
`--log-format json`) and `RUST_LOG`, which replaces the whole filter.

Every event is a fixed message plus named fields, so a journal filters on a
field rather than on wording. The same event in both formats:

```text
2026-09-28T17:30:59.689Z  INFO job{id=0f3c…}: pipette_artifacts::ensure: model ready model=hf://LiquidAI/LFM2-350M-GGUF elapsed=1.843712458s
```

```json
{"timestamp":"2026-09-28T17:30:59.689Z","level":"INFO","message":"model ready","model":"hf://LiquidAI/LFM2-350M-GGUF","elapsed":"1.843712458s","target":"pipette_artifacts::ensure","spans":[{"id":"0f3c…","name":"job"}]}
```

A field means the same thing in every crate:

| Field | Holds |
|-------|-------|
| `job_id`, `result_id`, `benchmark_id`, `client_id` | management-server ids |
| `model`, `runtime` | the cell's model and runtime references |
| `benchmark_type` | a benchmark's type, e.g. `prefill_throughput` |
| `cell` | the measurement a line belongs to: a benchmark type with `rep` and `reps` beside it, or a benchmark id with its phase (`<benchmark_id> rep 2/5`), depending on the engine |
| `sample`, `sample_id` | a sample's label (`eval sample 2/40: id=s1`, or `sample s1 (2/40)` in `vl_spec_decode`) and its id |
| `url`, `method`, `status` | an HTTP exchange; `status` is always the HTTP code |
| `elapsed`, `timeout`, `retry_in`, … | durations as text, in the unit that fits: `1.843712458s`, `71ms` |
| `error` | the error message |
| `path` | a file or directory |
| `*_ms`, `*_bytes`, `*_tokens` | readings, as numbers; `samples_ms` lists a metric's samples as text |

A field whose value is unknown is left out rather than written empty. In the
text format, free text such as a label, a prompt or a program's output is
quoted, so a space or `=` in it stays inside its field.

Under `worker`, every event a claimed job produces, from any crate and from
the heartbeat thread, sits inside a `job{id=…}` span, so a journal can be
filtered to one job without the event having to name it. In JSON that is the
`spans` array. A program the run starts prints inside its own
`process{program, pid}` span (`container{container, pid}` for docker), at
`debug`.

What the log says at `info` during a run, in order: `run` (the cell and the
client configuration), then each artifact phase (`runtime in store` / `runtime
not in store; installing`, `model …`) and its `… ready` with `elapsed`, then
the runtime's startup (`running` with the `command`, `ready`), then the
measurement (a `running` and a reading per rep, `reduced samples` with
`mean_ms` and `stddev_ms`), then the result. A corpus benchmark also logs
`downloading corpus` and `corpus ready` before its server starts. On Linux
and Android, `memory pressure` is logged at the cell's start and each rep's
start and end (`sample_point` is `before_cell`, `before_rep` or `after_rep`;
Android also logs `before_drop` and `after_drop` around its page-cache drop, as
[reading the logs](../methodology/galaxy-s26-ultra-memory-pressure.md#reading-the-logs)
describes):
`/proc/meminfo` and zram levels as `*_bytes` and, from the second event on,
the `/proc/vmstat` counters under the kernel's names (`pswpin`, `pswpout`,
`pgmajfault`, …) and PSI stall time (`psi_some_ms`, `psi_full_ms`) as what
moved over `window_ms` since the previous event. Each pid memory probe's stop
logs `memory probe stopped`, whose `first_swap_ms` and `peak_swap_ms` say when
the process's swap first rose and last peaked, in ms from the probe's start,
`window_ms` before the line. At `debug`,
each management-server and download request is one `sending request` and one
`response` (`status`, `elapsed`), so a stall is attributable to the exchange it
sits in.
