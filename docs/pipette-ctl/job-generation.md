# File plans with the management server

`pipette-ctl campaigns` sends a **scheduler-mode** plan to the management
server. The server expands the plan into jobs on its queue, and leases each job
to a registered client whose capabilities match.

This is the second of `pipette-ctl`'s two modes. The first
([plan-runner.md](plan-runner.md)) drives devices directly through its
transports (adb, ssh, local, SLURM, iOS) and keeps its own state. Scheduler mode hands the work off: the plan decides *what*
runs and *who is eligible*, and the server decides *when* and *on which
device*.

```
plan.toml ──campaigns create──▶ POST /api/v1/campaigns ──expand──▶ queue ──claim──▶ client
```

The server owns the plan schema, the capability rules, the expansion and the
queue. This document is a guide for plan authors. The rules are in pipette-mgmt
[`docs/plan-ingestion.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/plan-ingestion.md),
and the queue and claim protocol in pipette-mgmt
[`docs/queue.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/queue.md).

## The plan format

A scheduler-mode plan is a **distinct document** from a local-dispatch plan: no
`plan_id`, no `[[transports]]`, no `[retry]`. A plan for the wrong mode
fails to parse.

Start from [`examples/plans/scheduler/afm-mlx.toml`](../../examples/plans/scheduler/afm-mlx.toml).

Each `[[variants]]` block pairs a sub-matrix of benchmarks × models × runtimes
with the **eligibility** for those runs:

```toml
benchmarks = ["decode_throughput_512_100", "end_to_end_latency_512_256"]

[[variants]]
requires   = ["os:ios"]                      # capability flags a client must have
clients    = ["ev1_9f2c1d3e4b5a6c7d"]        # …or be named outright
models     = [{ type = "apple_foundation_text" }]
runtimes   = [{ type = "apple_foundation", private_thermal = true }]
benchmarks = ["decode_throughput_512_100"]   # overrides the top-level list

[[variants]]
requires        = ["os:macos"]
exclude_clients = ["ev1_4a1b2c3d4e5f6a7b"]  # never this client
excludes        = ["os_version:15.0"]       # never a client with this flag
not_all_of      = [["chip:applem1", "runtime:mlx:0.31.3"]]  # never both at once
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-350M-MLX-4bit" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.31.3", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.31.3\n" } }]
```

That expands to **3 jobs**: 1 from the first variant, and 2 from the second,
which runs both top-level benchmarks.

The envelope fields `expires_at` and `merge_duplicates`, the duplicate-job
merge, the job ceiling, and the warnings a filing returns are specified in
pipette-mgmt
[`docs/plan-ingestion.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/plan-ingestion.md)
(sections 4, 6 and 7).

### Eligibility: `requires` and `clients`

Every variant declares at least one of `requires` / `clients`. A client is
eligible when it is named in `clients` **or** its capability set satisfies
`requires`. A `clients` entry adds eligible clients; it never removes any.

So `clients` beside `requires` adds the named clients to whatever the flags
match, and `clients` **alone** pins the variant to the named clients. The
server injects its hardware policy (for example, the supported-iPhone list for
Apple Foundation) only into a variant with `requires`, so a roster-only variant
gains no capability path. To restrict a variant to a roster, drop `requires`.
[`llamacpp-vl-spec-decode.toml`](../../examples/plans/scheduler/llamacpp-vl-spec-decode.toml)
is roster-only for that reason: its cells pull a corpus from the server.

### Exclusions narrow, and a denial wins

`exclude_clients` (client ids), `excludes` (any one flag) and `not_all_of`
(every flag of one group) say who must *not* run a variant. Each defaults to
empty, which excludes nobody. A denial beats the `clients` roster, and an
exclusion alone does not count as eligibility: the variant still needs
`requires` or `clients`.

`not_all_of = [[]]`, one empty group, would deny every client, because every
capability set contains the empty group. The server rejects it.

### Runtime versions belong in the descriptor

Which build to run is part of the runtime descriptor. Eligibility says only
that a client *has* the runtime. The clients here advertise `runtime:<name>`
and, when they know the build, `runtime:<name>:<build>` as well (the Android
app for its compiled-in llama.cpp, the iOS app and a `pipette-cli` worker for
each runtime they hold). A plan that must pin a build puts that flag in
`requires`.

An iPhone that `pipette-ios-proxy` drives in version mode, as the device
farm's are, advertises no `runtime:<name>` or `runtime:<name>:<build>` flag.
It advertises one `runtime:descriptor:<digest>` build flag for each runtime
that its builds carry; see the next section. Author platforms whose version
semantics differ as separate variants (pipette-mgmt
[`docs/plan-ingestion.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/plan-ingestion.md#runtime-versions)
section 5, "Runtime versions").

### Pinning an iOS app build

The device farm is the set of iPhones that
[`pipette-ios-proxy`](../pipette-ios/build.md#on-the-farm-pipette-ios-proxy)
drives in version mode. The proxy reads app builds from a releases bucket:
an S3 bucket that holds the release archives. Each phone advertises one
`client_version:<version>` flag for each of the newest 20 app builds in that
bucket, plus the proxy's default. The proxy installs one of them per job.
A pin on an older build matches no phone. A variant pins the build with
`client_versions`, which the server writes as one more `any_of` group:

```toml
[[variants]]
requires        = ["os:ios"]
client_versions = ["2026.10.04-0123abc-internal"]   # the app's client_version
models          = [{ type = "apple_foundation_text" }]
runtimes        = [{ type = "apple_foundation", private_thermal = true }]
```

The proxy runs a job only on an admitted build whose manifest lists the job's
runtime field by field. The farm's app builds are private-API `-internal` builds
([private-API builds](../pipette-ios/private-thermal-release-build.md)), so an
iOS runtime needs `private_thermal = true`, or no build carries it.

The `clients` roster ignores `client_versions`. To pin a device and its
build, require the device's `tag:` flag.
[DESIGN.md section 9](../../tools/pipette-ios-proxy/DESIGN.md#9-version-mode-the-build-a-job-pins)
says which build a job gets, which flags a phone advertises, and how a miss
fails. [The iOS build guide](../pipette-ios/build.md#on-the-farm-pipette-ios-proxy)
says how a build reaches the bucket.

## File a plan

File a plan as a campaign. The command sends the plan file as it is on disk:

```bash
pipette-ctl campaigns create --name "AFM smoke" \
  --plan ./examples/plans/scheduler/afm-mlx.toml --note afm-mlx-smoke
```

`--dry-run` has the server expand and validate the plan and file nothing.
The server stages nothing unless the whole plan validates. The verbs that
file later revisions are in [management routes](management.md#areas);
the account they sign as is in [submitter identity](submitter-identity.md).

## Gated models

A model, or the draft of a speculative model, can carry an `auth_token` for a
gated HuggingFace repository. The campaign verbs send the plan file as it is,
so the token reaches the server in the signed request. The server imports the
model into the vault of the campaign's owner and stores no token; only that
owner's devices can fetch it. The import can take minutes, and `campaigns
create`, `file`, `revise` and `variants` show each model file's progress on
stderr. If the connection breaks before the answer, the server may still have
filed the plan: read the campaign (`campaigns list`, `campaigns plans`) before
filing again.

The server refuses a token anywhere else:

- a `runtime_flags` env forward that sets `HF_TOKEN`,
  `HUGGING_FACE_HUB_TOKEN` or `PIPETTE_HF_TOKEN` to a value (the bare name,
  which forwards the device's own value, is allowed);
- the name `auth_token` elsewhere in the document.

The rules are in pipette-mgmt
[`docs/plan-ingestion.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/plan-ingestion.md)
(section 6.1, item 2, and "The import of a gated model" in section 6.2) and
[`docs/api/campaigns.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/api/campaigns.md#the-plan-document)
("The plan document").

A gated model with no `auth_token` needs a token on each device that runs it.
The client reads only `PIPETTE_HF_TOKEN`, not the generic `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN` (see [usage](../pipette-cli/usage.md#gated-models)).
In local dispatch, the runner sends the plan's token over your transport. In
scheduler mode, each device needs its own token.
