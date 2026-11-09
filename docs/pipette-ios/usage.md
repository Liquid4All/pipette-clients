# Operating the iOS client headlessly

**Scope:** driving the installed iOS app from a terminal: the `headlessrun`
command surface, its output contract, and how each verb maps onto the
[`pipette` CLI](../pipette-cli/usage.md). Building the app is
[build.md](build.md); how it is put together is
[architecture.md](architecture.md).

The app is a benchmark client with a UI. A launch argument can start everything
that the UI can start, so you can script a phone the same way as a Linux host
that runs `pipette`.

## The grammar

```text
pipette   <group> <leaf> --key value
headlessrun <group> [<leaf>] key=value
```

One substitution: the separator. **Everything else takes the CLI's name**, so
the groups, the leaves, and the parameters all read as they do there. An
invocation copied out of `pipette --help` works here by rewriting `--key value`
as `key=value`, and the dashed spelling parses too, so `--model X` and
`model=X` are the same invocation.

`key=value` is the primary spelling because argv over a device console makes it
the cheaper token shape, and because `pipette-ios-proxy` and the [plan
runner](../pipette-ctl/plan-runner.md) emit it.

The four leaves where the CLI takes a positional take it here too:
`benchmarks show <ref>`, `results show <id>`, `results delete <id>` and
`sync <result-id>`. The bare word fills the key the table below names, so
`results show job-1/cell-2` is `results show result=job-1/cell-2`, and naming
both is refused. A value that starts with `-` or looks like `key=…` needs the
key form.

Launch it through `devicectl`:

```bash
xcrun devicectl device process launch --device <UDID> --console \
  ai.liquid.liquid-pipette headlessrun benchmarks run \
  benchmark=prefill_throughput_512 runtime=mlx \
  model=mlx://repo=LiquidAI/LFM2.5-350M-MLX-4bit sync=1
```

`-v` / `--verbose` is a global switch rather than any verb's parameter: it
mirrors debug diagnostics into the console capture.

`runtime=` takes one of four short tokens, `llama`, `mlx`, `onnx` or `afm`, or a
JSON `Runtime` object. A type tag such as `mlx_ios_pipette` is refused, because it
names an engine but not a build. The sweep forms default to `mlx` when `runtime=`
is absent.

### Refusals

Every refusal names the offending token, and a malformed invocation is
distinguishable from a failed run:

| Exit | Means |
|------|-------|
| `0` | the command succeeded |
| `1` | the command ran and failed |
| `2` | the invocation was not a command: unknown verb, unknown `key=`, missing or unusable value |

An unknown `key=` is an **error**, never ignored: a mistyped parameter would
otherwise be indistinguishable from one that had no effect. An unrecognized
`runtime=` is refused rather than defaulting to MLX.

### Output contract

One `[HEADLESS] <group> key=value…` line per record, with a `<group> count=N`
header before a list. Deliberately more machine-readable than the CLI's psql
tables, because a console capture is parsed far more often than it is read.

**Every key is `snake_case`**, and it is the crate's or the wire's spelling:
`client_id`, `reindex_pending`, `device_form_factor`, `used_bytes`,
`freed_bytes`, `size_bytes`. So a token greps the same here, in a server
response, in `pipette`'s own output, and in the Rust field it came from. Swift
spells those fields `camelCase` internally, as Swift does; the conversion
happens at the print, not in the model. That is the same boundary `CodingKeys`
draws for the wire.

The app's diagnostics go to stderr as `[HEADLESS] <level> <category> <message>`,
so stdout stays the contract, and `pipette-ios-proxy` logs each at the level it
carries.

The first line of every launch is `[HEADLESS] hello pid=<pid>` on stderr, printed
before the arguments are parsed, so it comes within a second even on a refusal.
It tells a console consumer the launch it asked for is running. On a farm phone
Single App Mode reopens the app with no arguments after every exit, and when
that reopen lands inside `devicectl`'s `--terminate-existing` launch, `--console`
attaches to the reopened UI copy, which prints nothing and never exits.
`pipette-ios-proxy` ends and relaunches, once, any launch that prints no
`[HEADLESS] ` line within its `first_line_timeout_secs` (60 s by default; see
its [DESIGN](../../tools/pipette-ios-proxy/DESIGN.md#4-one-job)).

Every terminal path ends with `BENCH_DONE <exit-code>`. `pipette-ctl` scans
stdout for it, and a consumer keyed on it hangs without one, so even the early
refusals emit it.

A sweep ends with its job's status, `BENCH_DONE 0 completed` once every cell
ran, whatever each cell did. A `claim=` run is one cell, so its cell's failure
is the run's: `BENCH_DONE 1 failed: <error>`, the cell's error on one line,
which `pipette-ios-proxy` files as the job's failure.

Non-log payloads (a result's JSON, an exported CSV) go to **stdout bare**, so
`results show … | jq` and `job export … > out.csv` work. Those two commands,
and only those, move every `[HEADLESS]` line *and* the sentinel to stderr;
everything else keeps them on stdout, because `pipette-ctl` scans stdout alone
for the sentinel and the transport never invokes either one.

That is the CLI's split: `pipette results show` prints the payload on stdout and
its section labels on stderr. The app's other diagnostics (`AppLog`, including
MLX's `footprint=… mlxActive=… mlxPeak=…` lines) go to `os.Logger`, which
`--console` does not carry. Stream them alongside:

```bash
log stream --predicate 'subsystem == "ai.liquid.liquid-pipette"' --level info
```

## The verbs

`*` marks a required parameter. Every group's `list` leaf is also its default,
so `headlessrun models` and `headlessrun models list` are the same command. The
CLI requires the leaf; the bare noun is this client's own shorthand.

### Identity: `auth`

| Verb | Parameters | `pipette` counterpart |
|---|---|---|
| `auth register` | `server-url` `organization`\* `contact-email`\* `client-details`\* `preauth-key` `device-name` `device-form-factor` | `auth register` |
| `auth me` | `local` | `auth me [--local]` |
| `auth set-device` | `device-name` `device-form-factor` `client-details` | `auth set-device` |
| `auth reset` | `force` | `auth reset --force` |
| `auth export` | | `auth export` |
| `auth import` | `bundle`\* `force` | `auth import --from --force` |

`organization`, `contact-email` and `client-details` are required, as they are
on the CLI. `server-url` keeps the CLI's one default, because there is one
production collector. Each is parsed to the
type the request carries as the command is read, so a `server-url` that is not
an http(s) base, an address with no domain, or an empty pre-auth key exits 2
before a keypair is minted or a key spent.

`auth register` refuses over an identity the device already holds, as the CLI
does: re-registering mints a new keypair, and every result submitted under the
old one stops being attributable. `auth reset` comes first.

Before that check, `register` handles a state that the CLI cannot reach. The
Keychain copy of the key outlives the app container, so a reinstall can leave a key
that no registration claims. The gate consults the record's own Keychain copy before
it concludes that. Nothing can sign as a client that this device has no record of,
so the key is useless:

- If nothing is awaiting submission, `register` discards the key and continues.
  Otherwise the device would stay on a refusal whose only remedy (`auth reset`) has
  no UI. `auth reset` gates on the same count, because pending results would
  otherwise be submitted under the new client id.
- If results are pending, the refusal stands. `auth reset force=1` is the
  deliberate way through.
- A register that fails while it writes its own record deletes the key too, so the
  retry is not refused.

`auth set-device` splits the way the CLI's does. The two device labels are
local, so an unregistered device can still be named and naming it reaches no
network; `client-details` is server-side registration metadata, so it needs a
registration and a `PATCH /clients/me`. Naming none of the three is refused.

`auth me local=1` answers from disk: `client_id`, `server_url`, and the public
key derived from the signing key. No request, no approval needed: every field
a device can assert about itself before the server has approved it.

`auth reset` deletes the identity (the signing key, the registration, the
device labels and the settings) and every cache derived from the server
relationship: the synced benchmarks, remote results pending and synced, and
eval state. Local benchmarks and local results are kept. It needs `force=1`
only when results are still awaiting submission, and the refusal counts them
first.

`auth export` prints the CLI's identity bundle
([moving an identity](../pipette-cli/usage.md#moving-an-identity-to-another-machine)),
so a token crosses between the two clients. This client adds the Hugging Face
tokens its Keychain holds as optional fields the CLI ignores. The summary goes
to stderr and the bare token to stdout, so `auth export > bundle.txt` captures
the token alone. It carries the signing key in plaintext, which is the point:
the Keychain exports it no other way, and erasing the device destroys it.

`auth import` takes the token as `bundle=`, where the CLI reads stdin or
`--from`: a `devicectl` launch has no stdin to pipe one through. Rerunning it
for the identity already held is not a conflict, so a rerun finishes an import
that failed partway, and reports `changed=no` when nothing is left to write. A
*different* identity needs `force=1` and drops the caches and per-model tokens
derived from the outgoing one. Device labels and the storage quota are not carried
either way: they describe the machine, not the identity.

### Models

| Verb | Parameters | `pipette` counterpart |
|---|---|---|
| `models list` | `format=name\|uri` | `models list --format` |
| `models pull` | `model`\* | `models pull --model` |
| `models delete` | `model`\* | `models delete --model` |
| `models rm` | `name` `repo` | — (iOS only) |

`model=` takes the two spellings `pipette --model` takes: a JSON `Model`
object, or the compact URI (`mlx://repo=org/r`,
`gguf-text://repo=org/r&path=Q4_K_M.gguf`,
`onnx-genai://repo=org/r&prefix=cpu-int4&rev=<sha>`). An ONNX Runtime GenAI repo
publishes several packages side by side, so `prefix=` is what selects one. It also takes
the [digest form](../pipette-cli/models-and-runtimes.md#the-digest-form)
`model://sha256=<prefix>`, as printed by `models list`. `spec=`, the older name
for `model=`, is taken only on the sweep forms below.

`models rm` selects by file name or repo slug instead of by coordinate; it has
no CLI counterpart because the CLI addresses a model only by its
self-contained reference. `name=` wins when both are given, and a repo match
deletes every quant under it.

`models list` adds `size_bytes=` and `path=` to the CLI's columns: neither is
derivable from the coordinate, and `storage status` reports only totals.

### Runtimes

| Verb | Parameters | `pipette` counterpart |
|---|---|---|
| `runtimes list` | `format=name\|uri` | `runtimes list --format` |

`list` is the only leaf. A phone installs no runtimes: the engines are compiled
into the app (llama.cpp, MLX and ONNX Runtime GenAI) or ship with the OS (Apple
Foundation). So `runtimes pull` / `remove` / `catalog` / `flavors` would act on a
store that does not exist here. `runtimes list` prints the engines in the app in
the CLI's columns (`runtime=` as the runtime's `cli_ref`, `type=`, `digest=`). It
adds one column of this client's, `descriptor=`: the canonical JSON that a
submitted result records and that `benchmarks run runtime=` accepts. The on-device runtimes have no URI form, so
`format=uri` renders the `cli_ref` too, as the CLI falls back for any runtime it
cannot write as a URI.

### Benchmarks

| Verb | Parameters | `pipette` counterpart |
|---|---|---|
| `benchmarks list` | `benchmark-type` | `benchmarks list --benchmark-type` |
| `benchmarks show` | `benchmark`\*, or `<ref>` | `benchmarks show <ref>` |
| `benchmarks init-local` | | `benchmarks init-local` |
| `benchmarks run` | `benchmark`\* `model`\* `runtime`\* `runtime-flags` `sync` `benchmark-flags` `readiness-max-wait-secs` `readiness-skip-thermal` | `benchmarks run` |

`benchmarks run` is the one-cell form and the primary one: every axis named,
one benchmark, one model, one runtime.

`benchmark=` takes a reference, as the CLI's does: a bare id or `remote/<id>`
names the synced catalog, and `local/<id>` a definition `benchmarks init-local`
generated. `benchmarks list` prints each row under that reference, and
`benchmarks show` prints the id, the type, the source and the definition's
parameters under their wire names. `benchmark-type=` takes either spelling of a
type (`prefill_throughput` or `prefill-throughput`).

A run resolves its reference from the half of the store it names, then from the id
itself (the ladder rungs spell their own parameters). How a claimed run (`claim=`,
which `pipette-ios-proxy` always passes) fetches and revalidates a remote definition
is in [architecture.md § 1.3](architecture.md#13-benchmark). The fetch goes to the
server this device reports to (its registration's, else the production collector),
before the model downloads. What the console shows:

- A reference that neither the store nor the id answers, a `404`, or a definition
  this build cannot read is skipped before any download
  (`skip <ref>: unrecognized benchmark id (...)`). A run left with nothing to
  measure ends `BENCH_DONE 1`.
- A fetch that fails in transit ends the run at once (`bench ERROR fetching a
  benchmark definition failed`, then `BENCH_DONE 1`).
- A `local/<id>` miss is never fetched: the server holds no copy of this device's
  own definitions.
- Without a claim nothing is fetched, as the CLI's `benchmarks run` fetches
  nothing: run `sync` first.

`runtime-flags=` is one JSON object of knobs (`{"threads":4}`), resolved
through the same path a claimed job takes, so a console experiment is
validated exactly as an orchestrated cell is. A knob the resolved cell does not
carry is refused by name rather than dropped. The resolved entry then rides on
the cell whole, so every knob it names reaches the run, `ctx_size` included:
the engine sizes the context from the benchmark only when the cell names none.
The prefill chunk comes from the entry's `n_ubatch`, else 512: `batch=` is a
sweep-form parameter and is refused here as an unknown key.

`readiness-max-wait-secs` and `readiness-skip-thermal` become the cell's
benchmark flags, resolved as the CLI's `build_benchmark_flags` resolves them, so
they are valid only where the cell's flags carry a readiness field: the timing
benchmarks, Apple Foundation's included. On eval and the memory cells the cell
has none, so passing one is an error rather than a value nothing reads. The deadline is
whole seconds, at least one as the CLI's `value_parser!(u64).range(1..)` makes
it, and is held to a one-day ceiling as the crate holds it.
`readiness-skip-thermal` waives every thermal signal (the thermal state and the
die temperature) and leaves the CPU-load check in force, as on every other
platform.

`benchmark-flags=` carries the same settings as one JSON object, as the CLI's
`--benchmark-flags` does: `{"readiness":{"max_wait_secs":900}}`. It is resolved
the way a claim's `benchmark_flags` is, without the axis keys, so a knob no iOS
cell carries (`http_timeout_seconds`) is refused by name. The two readiness keys
are its shorthand, and passing both forms is refused.

Where a readiness setting is absent from both forms, the app falls back to
`PIPETTE_READINESS_MAX_WAIT_SECS` or `PIPETTE_READINESS_SKIP_THERMAL` in its
launch environment, as the CLI does. `pipette-ctl` sets them through
`devicectl` when run with `--readiness-max-wait-secs` or
`--readiness-skip-thermal`. A job started from the UI or a deep link has
neither, so it keeps the default.

`--model-flags` / `--model-enable-thinking` have no counterpart.
`--http-timeout-seconds` does not apply: it bounds a request to a server the
cell starts, and the phone runs its engines in the app, so plan-types gives no
iOS cell the knob.

`--attempt-timeout-seconds` has no shorthand: `benchmark-flags=` carries it, as
`{"attempt_timeout_seconds":600}`. It is the limit on each attempt, timed on its
own: the model load and warm-up, each measured rep, the work after a rep, and
each peak-memory attempt. Unset, it is 300 s on the timing, throughput and
memory cells. `vl_spec_decode` has no limit and its iOS cells refuse the key;
eval takes no benchmark flags on iOS. Readiness waits and the VL and ASR
context calibration do not count. The app cannot stop llama.cpp, MLX, ONNX
Runtime GenAI or Apple Foundation mid-call, so past the limit a claimed launch
(`claim=`) saves the cell as failed, logs its failure (`error jobRun
<benchmark> / <model> failed: rep 2 ran 300.4s, past attempt_timeout_seconds
(300s), and was stopped`), prints `BENCH_DONE 1 failed: <the same error>` and
exits at once. Every other launch, and the resident app (the UI, `worker`),
fails the cell with that error when the engine next reaches a readiness gate,
a rep or a peak-memory attempt, or returns, and goes on to the next cell; the
worker reports the failure as not retriable. A result reports
`attempt_timeout_seconds` in `benchmark_flags` only when the cell set it.

### The sweep forms: `bench` and the bare invocation

| Verb | Parameters |
|---|---|
| `bench` | `model`\|`spec` `match` `quant` `runtime` `batch` `runtime-flags` `benchmark`\|`benchmarks` `metrics` `offsets` `submit` `claim` `benchmark-flags` `readiness-max-wait-secs` `readiness-skip-thermal` |
| (no verb) | the same set; without `model=` it refuses `runtime-flags`, `benchmark-flags`, the readiness keys and `claim` |

This client's own shape, kept for the fleet harness: `metrics=` × `offsets=`
expands to a set of catalog benchmark ids, so one launch measures a ladder.
`benchmarks=` names ids outright and overrides the expansion. `match=` selects
from the catalog by name substring. `model=` never does: it names one exact
model, so a coordinate from a plan cannot silently match nothing.

Both forms resolve and **download** a model named with `model=` (or `spec=`)
first: a bare invocation that names one runs as `bench`. Without `model=`, the
bare form runs against a matching model already on the device.

`runtime=afm` benchmarks Apple's on-device foundation model: no model, no
download.

`runtime=onnx` benchmarks an ONNX Runtime GenAI package. It measures
`prefill_throughput`, `decode_throughput`, `end_to_end_latency` and
`max_memory_usage`; `eval`, the `vl` cells and the `asr` cells raise an error
([onnx-genai-runtime.md](onnx-genai-runtime.md)).

`batch=` sets the prefill chunk (default 512) when `runtime-flags=` names no
`n_ubatch`; passing both is refused.

`metrics=promptseed|coherence|calibrate` selects a self-check rather than a
benchmark. It records nothing and submits nothing.

`benchmark-flags`, `readiness-max-wait-secs` and `readiness-skip-thermal` follow
`benchmarks run`'s rules and, like `runtime-flags=`, need the cell named exactly:
`model=` and one `benchmarks=` id, so a `match=` or a ladder is refused.
`pipette-ios-proxy` dispatches a claim through the bare form with its
`benchmark_flags` as `benchmark-flags=`, so this is where a plan's per-cell
`readiness` block and attempt limit arrive. The launch carries the claim, so the app fetches a missing
`benchmarks=` id from the server ([architecture.md § 1.3](architecture.md#13-benchmark)).

**`submit=` defaults to on**, where the CLI's `--sync` is opt-in. A phone that
measures for an hour and publishes nothing is the failure to prevent. The exception is a caller who named an exact model with `model=` (or
`spec=`) rather than selecting from the catalog: nothing sanctioned that model,
so an ad-hoc experiment stays opt-in. A
submitting run on an unregistered device is refused before the first cell
rather than after the last.

**`claim=<server job id>`** makes the run answer a job its caller claimed: the
run's own upload echoes the id as `job_id`, which closes that job on the server.
A result that upload missed is the caller's to send: neither a launch nor a
`sync` without `job=` picks it up, since the lease may be gone or handed out
again.
`pipette-ios-proxy` passes `submit=1 claim=<id>` on every run, then deletes what
went up and submits anything the app could not. A run that picks its own model
(`match=`, or neither `model=` nor `spec=`) refuses a claim; `runtime=afm` needs
no model.

A run given a JSON `Model` in `model=` reports that model as written, so the
`model_descriptor` carries the vault `custom_label` the argument had, or none,
whatever label the store's copy was first fetched under: that is how a job
`pipette-ios-proxy` claimed reports its claim's label. A `model://sha256=`
reference names the store's copy, and reports no label: the copy holds whichever
label it was first fetched under.

### Results

| Verb | Parameters | `pipette` counterpart |
|---|---|---|
| `results list` | `benchmark` `type` `state` `limit` | `results list` |
| `results show` | `result`\*, or `<id>` | `results show <id>` |
| `results delete` | `result`\*, or `<id>` | `results delete <id>` |

`benchmark=` takes a benchmark reference and matches the whole address, as the
CLI's `--benchmark` does: `local/<id>` and `remote/<id>` select different rows,
and a bare id means the synced half. `type=` takes either spelling of a
benchmark type, and `limit=` is a count, so a negative one is refused rather
than reaching the row slice.

`result=` is `<jobId>/<cellId>`. The store keys a result by its cell, and the job
manifest records the job that ran it, so the pair is the id. `results show` prints `payload.json` bare on stdout and labels its
sections on stderr, exactly as the CLI does, so it pipes into `jq`.

`results delete` removes the stored artifacts and leaves the cell's own record
alone. The job still ran that cell, and rewriting job state to hide it would
make the manifest disagree with what happened. A claimed job (`claim=`) is the
exception: the next launch deletes it once it has no results left, so a farm
phone doesn't keep a manifest for every job it ever ran.

### Jobs: no CLI counterpart

| Verb | Parameters |
|---|---|
| `jobs` | |
| `job run` | `id`\* `scope=failed\|cancelled` |
| `job export` | `id`\* |
| `job submit` | `id`\* |
| `job rm` | `id`\* |

The batch noun is this client's: a job is a manifest of cells, which is what
the Jobs tab lists and what the planner claims into. `job run` flips the scoped
terminal cells back to pending and re-runs them: `scope=failed` retries,
`scope=cancelled` (the default) resumes. `job export` prints the results CSV on
stdout, with its `BEGIN CSV` / `END CSV` markers on stderr, so redirecting
stdout captures the document alone.

### Storage

| Verb | Parameters | `pipette` counterpart |
|---|---|---|
| `storage status` | | `storage status` |
| `storage gc` | `dry-run` | `storage gc --dry-run` |

See [storage-quota.md](../storage-quota.md). `gc` pins what a resident app can
have in flight (every cell of a running or paused job, and every model a
transfer is still writing), which a CLI process, unable to be mid-run while it
sweeps, does not need to.

`storage status` prints the CLI's header (`quota_bytes=` with
`quota_source=settings|default`, `used_bytes=`, `used_percent=` and
`free_bytes=`), then one `storage entry` line per entry, in the order `gc` would
reclaim them: `kind=`, `entry=`, `size_bytes=`, `last_used=`, and `note=` for
garbage, which carries the reason it is garbage.

`storage gc` leads every line with its verb, as the CLI does, so no delete is
silent: `would_reclaim` / `would_evict` on a dry run, `reclaimed` / `evicted` on
a real one. The summary is the CLI's `freed X; using Y of Q` as `freed_bytes=`,
`used_bytes=` (what the sweep left) and `quota_bytes=`, with a dry run's
projection named apart as `would_free_bytes=` / `would_leave_bytes=`.
`still_over_bytes` appears only when the sweep could not get under the cap.

### Sync

| Verb | Parameters | `pipette` counterpart |
|---|---|---|
| `sync` | `job` `server-url` | `sync [<result-id>] [--server-url]` |

Pull the catalog, submit what is pending, then refresh scores, in that order,
one line per phase.

What runs depends on whether the device holds an identity, as it does for the
CLI. The catalog is a public read, so an unregistered device can still see what
there is to run before it registers to run any of it. It just has to name the
server with `server-url=`, since only registration records one. Submission is
authenticated, so that phase is skipped with a note rather than turning the
whole command into a refusal.

A registered device uses its registered server and refuses a `server-url=` that
disagrees with it: the signing key beside the registration signs only for that
host, so honoring an override would point the authenticated half at a server
that will refuse it.

`sync` narrows by **job**, not result: `result=`, and the positional
`<result-id>` that fills it, are refused by name rather than ignored.

Score refresh, the CLI's third phase, asks the server for the score of
every submitted eval that has none yet and saves it as `metrics.json`, then
prints `sync scored results=N already_scored=M pending=P` as `pipette sync`
prints its `scored N result(s)` line.

### Planner worker

| Verb | Parameters | `pipette` counterpart |
|---|---|---|
| `worker` | `idle-secs` `idle-jitter-secs` `heartbeat-secs` `max-jobs` `skip-profile-refresh` | `worker` |
| `settings run` | | — (the same loop, this client's spelling) |

The claim loop: claim → run → submit, against the management server. Every
knob is the CLI's, with the CLI's defaults: 300 s idle plus up to 60 s of
jitter, a heartbeat at half the claim's own `time_window`, and no job ceiling.

This is the one **resident** verb. It prints a `worker start …` line naming the
resolved knobs, then a status line every 30 s, and emits no `BENCH_DONE` until
the session ends. `max-jobs=` is what makes an end reachable: a bounded worker
is a command that finishes.

Every other verb is one-shot, and runs alone: the launch neither reports the
device profile nor drains stranded results, as a UI launch does.
`pipette-ios-proxy` drives a phone as its own client across such launches, so any of them would act beside it. A
`submit=1` run whose upload failed stays on the phone until `sync`, a `worker`
launch or the UI sends it; a `claim=` run's waits for its caller.

How it ended is on the final line as `outcome=`, and it drives the exit code:

| `outcome=` | Exit | Means |
|---|---|---|
| `completed jobs=N` | `0` | `max-jobs=` reached |
| `stopped` | `0` | the toggle went off, the app left the foreground, or the process was told to stop |
| `not_approved` | `1` | the server refused the claim, or the startup profile is not `approved`; an operator must approve this client |
| `not_started` | `1` | the loop never came up |
| `failed` | `1` | the loop hit an error it does not retry: a fatal claim error or a failed startup profile refresh |

`not_approved` exits non-zero deliberately, as `pipette worker` bails there: a
fleet script must not read "this device is not approved" as "it did its work".

### Settings, status, diagnostics

| Verb | Parameters | Notes |
|---|---|---|
| `status` | | registration, model count, job counts |
| `version` | | this build's `client_version` and the build flags that change what a measurement means |
| `diag memseq` | `models` `batch` `prefill` | sequential-load memory probe |
| `diag probe` | `kind=token\|generation\|cap\|enforcement` | the AFM diagnostics |

`diag memseq` runs `max_memory_usage_<prefill>` (default 512) for each `models=`
entry in order, in one process, which is where one cell's memory can reach the
next cell's peak.
`models=` takes the on-device artifact names; `diag memseq` with no `models=`
prints them. Each cell is built as a job builds it, so llama.cpp, MLX and ONNX
Runtime GenAI cells all run with production flags: llama's context is sized from
the benchmark, `batch=` (default 512) is llama's `n_ubatch` or MLX's prefill
chunk, and an ONNX cell carries no flags. Per model it logs `memseq PRE` with the
footprint carried in, then `memseq REPORTED` with the peak and the `ctx=` /
`n_ubatch=` the engine resolved. See
[peak-memory-ios.md](../methodology/peak-memory-ios.md#cross-cell-isolation-settle-to-floor),
and [memory carry-over](memory-carry-over.md) for the experiments run with it.

`version` is the counterpart of `pipette --version`, and prints the same
`client_version=` string a submitted result carries. It also prints
`private_thermal=on|off` and `private_memory=on|off`, with `thermal=` and
`memory=` describing what each build reads, because a run gated by a real SoC die temperature is not
comparable to one gated by `thermalState`. Nor is a memory cell that ran after
other cells in the process comparable across the two builds: a public build
cannot flush the reclaim ring, so it can still carry memory those cells freed.
So confirming which build is on a device is a prerequisite to trusting its
numbers. See [private-thermal-release-build.md](private-thermal-release-build.md)
and [private-API and public builds](../methodology/peak-memory-ios.md#private-api-and-public-builds).

## Deep links

The same grammar is reachable as a URL:

```text
pipette://run[/<verb>[/<leaf>]]?k=v&k=v
```

Path components become verbs, query items become parameters, so
`pipette://run/job/run?id=abc` is `headlessrun job run id=abc`. The empty path
is the bare sweep form.

The allow-list is deliberately narrow: run and observe work, never mutate
identity or stored data. A URL can be opened by any app, web page, or QR code.
`auth export` is the most important case: a link must never print a signing key.

Permitted: the bench forms, `job run`, `job submit`, `job export`, `status`.
Everything else is command-line only.

## What has no counterpart

Verbs `pipette` has and this client does not:

| CLI | Why not |
|---|---|
| `init` | the data root is fixed and created at launch; there is no workspace to choose |
| `runtimes pull` / `remove` / `catalog` / `flavors` | no runtime payload exists on a phone; the engines are compiled in |
| `check` | a phone runs no external programs; the engines are compiled in |
| `auth secret` | the key is a `0600` file in the app container and a `ThisDeviceOnly` Keychain item; `auth export` is how it leaves |

Verbs this client has and `pipette` does not: `jobs` / `job …`, `settings run`,
`status`, `diag memseq`, `diag probe`, `models rm`, and the `bench` / bare
sweep forms.

[client-parity.md](../client-parity.md) records where the Android client stands
on the same surface, and which of these differences are deliberate.
