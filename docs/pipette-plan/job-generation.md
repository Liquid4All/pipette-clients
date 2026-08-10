# Generating jobs for the management server

`pipette-plan generate` expands a **scheduler-mode** plan into a directory of
job files, which `pipette-mgmt plans ingest` stages onto the job server. The
server then leases each job to a registered client whose capabilities match.

This is the second of `pipette-plan`'s two modes. The first (described in
[plan-runner.md](plan-runner.md)) drives devices directly over adb/ssh/ios and
keeps its own state. Scheduler mode hands the work off instead: it decides
*what* runs and *who is eligible*, and the server decides *when* and *on which
device*.

```
plan.toml ──generate──▶ dir/ of job bodies ──plans ingest──▶ todo/ queue ──claim──▶ client
   (you)                    (this doc)          (pipette-mgmt)
```

## The plan format

A scheduler-mode plan is a **distinct document** from a local-dispatch plan: no
`plan_id`, no `[[transports]]`, no `[retry]`. Because each format requires
top-level fields the other lacks, one authored for the wrong mode fails to
parse rather than being silently misread.

Start from [`examples/plans/scheduler/afm-mlx.toml`](../../examples/plans/scheduler/afm-mlx.toml).

Each `[[variants]]` block pairs a sub-matrix of benchmarks × models × runtimes
with the **eligibility** for those runs:

```toml
expires_at = "2026-09-01T00:00:00Z"          # optional; stamped on every job
benchmarks = ["decode_throughput_512_100", "end_to_end_latency_512_256"]

[[variants]]
requires   = ["os:ios"]                      # capability flags a client must have
clients    = ["ev1_9f2c1d3e4b5a6c7d"]        # …or be named outright
models     = [{ type = "apple_foundation_text" }]
runtimes   = [{ type = "apple_foundation" }]
benchmarks = ["decode_throughput_512_100"]   # overrides the top-level list

[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-350M-MLX-4bit" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.31.3", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.31.3\n" } }]
```

That expands to **3 jobs**: 1 benchmark × 1 model × 1 runtime from the first
variant, 2 from the second.

Every variant must declare at least one of `requires` / `clients`: a variant
with neither matches nobody. A client is eligible when it is named in `clients`
**or** its capability set satisfies the requirement; naming a client never
narrows, it only widens.

### Runtime versions belong in the descriptor, not in `requires`

Which build to run is part of the runtime descriptor. Eligibility says only
that a client *has* the runtime. A client that fetches builds on demand
(Android) advertises `runtime:<name>` and can run whatever version a job pins;
one with a compiled-in build (iOS) advertises both `runtime:<name>` and
`runtime:<name>:<build>`, and a plan that must pin to it puts the concrete flag
in `requires`. Platforms whose version semantics differ are authored as
separate variants. That is what variants are for.

## Hardware policy is injected, not authored

You write the eligibility you *care* about. The capability rules in
[`capability_rules.rs`](../../crates/pipette-plan/src/capability_rules.rs) add
what the hardware *demands*, resolved per cell at generation:

- **`requires` injection**: flags a runtime can't run without.
- **`any_of` groups**: device-family disjunctions, satisfied by one member.
  An iOS Apple-Foundation job lands on one of a curated list of supported
  iPhones; a plan author never types that list.
- **Contradiction rejection**: Apple Foundation with `requires = ["os:android"]`,
  or two flags from one single-valued namespace, fail the whole plan.

So `requires = ["os:ios"]` on an Apple-Foundation variant generates a job that
also carries a supported-iPhone `any_of` group. The rules are code, not config:
a new runtime kind doesn't compile until it states a policy, and a policy change
(a newly supported device, a raised floor) is an ordinary reviewed edit plus a
`pipette-plan` release.

## Running it

```bash
pipette-plan generate --plan examples/plans/scheduler/afm-mlx.toml --out ./jobs
```

```
 FILE          | BENCHMARK                  | MODEL                         | RUNTIME          | ELIGIBILITY
---------------+----------------------------+-------------------------------+------------------+------------------------------------------
 cell-000.json | decode_throughput_512_100  | apple/foundation-text         | apple_foundation | os:ios
               |                            |                               |                  | one of: device:iphone17, …, +2 more
               |                            |                               |                  | client: ev1_9f2c1d3e4b5a6c7d
 …

wrote 3 job file(s) from examples/plans/scheduler/afm-mlx.toml to ./jobs

next: pipette-mgmt plans ingest ./jobs
```

Nothing is written unless the whole plan validates: schema, model/runtime
compatibility, and hardware policy all run first, so a rejected plan leaves no
partial directory for a later ingest to find.

`--out` is created if absent and must hold no `.json` files of its own.
Ingestion stages **every** `.json` in the directory as a job of one plan, so a
leftover file from an earlier expansion would join this one silently.

## What lands in the directory

One `cell-NNN.json` per cell. Names index a deterministic order, so
re-generating an unchanged plan produces byte-identical files: two runs diff
cleanly.

```json
{
  "benchmark_id": "decode_throughput_512_100",
  "requires": ["os:macos"],
  "any_of": [["chip:applem1", "chip:applem1pro", "…"]],
  "clients": [],
  "expires_at": "2026-09-01T00:00:00Z",
  "model_descriptor": "{\"type\":\"mlx\",…}",
  "runtime_descriptor": "{\"type\":\"mlx_macos_pipette\",…}",
  "spec": {
    "benchmark": "decode_throughput_512_100",
    "model": { "type": "mlx", "…": "…" },
    "runtime": { "type": "mlx_macos_pipette", "…": "…" },
    "runtime_flags": null,
    "model_flags": null,
    "benchmark_flags": null
  }
}
```

A body has two readers. The **server** reads `benchmark_id`, the eligibility
fields, and `expires_at`; it copies the two descriptors verbatim into synthetic
failure records without interpreting them, and passes `spec` through untouched.
The **client** reads `spec` alone: a `ClientRunSpec`, the same shape the
desktop CLI runs directly, so a scheduled cell and a local one are one
contract.

There is **no `job_id` and no `plan_id`**: both are minted by the server at
ingestion, and a body that pre-set either would be rejected. There is no
manifest file either. The directory is the manifest.

### Gated models

A job body is stored by the server and served to every client that claims it,
so an `auth_token` inlined in a plan's model source is **stripped** during
generation. A client supplies its own from `HF_TOKEN` at run time. This differs
from local dispatch, where the token travels over your own transport env; in
scheduler mode, each device running a gated model needs its own token.

## Handing off

`generate` and `plans ingest` are separate commands run by an operator;
`generate` anywhere, `plans ingest` on the management server's host with direct
storage access. Copy the directory over and ingest it:

```bash
pipette-mgmt plans ingest ./jobs --plan-name afm-mlx-smoke
```

Ingestion mints the `plan_id` and every `job_id`, stages the jobs, and reports
the file → job-id map plus any warnings (a group of jobs matching no registered
client is a warning, not an error: queuing work ahead of the fleet is
legitimate). Track it afterwards with `pipette-mgmt plans status <plan_id>`.

The full contract, the lifecycle a plan follows once ingested, and the
capability-matching rules the server applies are specified in `pipette-mgmt`'s
`docs/plan-ingestion.md`.
