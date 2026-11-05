# Eval Checkpoint & Resume

Applies to **eval** benchmarks on the `pipette` CLI's llama.cpp, MLX and
torch-oai backends. The other CLI backends (OpenVINO, OVMS, ONNX GenAI
and Core AI) do not implement eval, so they write no checkpoint.
`vlm_eval` and `asr_eval` keep their checkpoints in the same directory and
format, with the extra rules in
[vlm_eval and asr_eval](#vlm_eval-and-asr_eval).
Operator walkthrough: [usage.md](usage.md). How a run is carried out, from
the job to the finalized checkpoint: [eval-runs.md](eval-runs.md). Backend
notes: [llamacpp.md](llamacpp.md) · [mlx.md](mlx.md) ·
[torch-oai.md](torch-oai.md).

## Problem

An eval of thousands of samples on a slow device can take hours. If the device
reboots, the user hits Ctrl-C, the OS OOM-kills the process, or the
runtime crashes, every completion produced so far would be lost and the
next run would start at sample 0.

So each completed sample is written to disk as it is produced, and the
next run of the same cell skips finished samples and picks up where the
previous run stopped. Always on for eval benchmarks: no flag to enable.

## Operator reference

Active checkpoints live at
`<work-dir>/.pipette/state/evals/<digest16>.jsonl`: the first 16 hex
chars of the run digest (see **Identity**). Filenames do not carry the
benchmark id; the header's `meta` block holds it. The shell commands
below use paths relative to the work directory, so run them from there.

- **Audit what a checkpoint ran against**: `head -1 <file> | jq .meta`.
- **Progress after a kill**: `wc -l <file>` → rows recorded
  (completed plus failed) = line count − 1.
- **List failed entries**:
  `jq -c 'select(.failed == true) | {id, failed_reason}' <file>`
- **Un-skip failed samples after an upstream fix**: drop their lines,
  or `rm` the file to reset the cell entirely.
- **Force a fresh run for one cell**: inspect files with
  `head -1 <file> | jq .meta` to find the match, then `rm` it.
- **Force a fresh run for every cell of a benchmark**:
  ```
  for f in .pipette/state/evals/*.jsonl; do
    id=$(head -1 "$f" | jq -r .meta.benchmark_id)
    [ "$id" = "MY_BENCH" ] && rm "$f"
  done
  ```
- **Clean stale (rotated) files**: `rm .pipette/state/evals/*.stale-*`.
- **Debug "my run started from sample 0 again"**: a change to the
  cell's identity starts a new file and leaves the old one under its
  own name, so look for another `*.jsonl` with the same
  `meta.benchmark_id` and diff the two `meta` blocks. The differing
  field tells you what changed. If meta looks identical, the change
  was in the benchmark definition or the flags (meta is decorative;
  the full input set lives in the digest). If there is none, the
  previous run finished and its checkpoint was finalized, so there was
  nothing to resume.

`pipette auth reset`, and `pipette auth import` of a different
identity, delete `state/evals/` with the rest of the server-side state.

## State model

**Two layers of resumability** exist in the pipeline:

1. **Cell state** (owned by `pipette-ctl`). Records each
   `(benchmark, model, runtime, flags)` cell's attempts, from which a
   cell reads as Missing / Failed / Done. Lives on the **operator's
   machine** in the plan workspace. The plan runner decides which cells
   to execute or retry.
2. **Sample state** (this page). Tracks completed samples produced
   within a single cell's eval run. Lives on the **target device** in
   the CLI workspace.

The two layers are on different machines and do not share storage.
Deleting one has no effect on the other.

```
Operator machine (where you run `pipette-ctl run …`)
────────────────────────────────────────────────────────────
<plan-workspace>/.pipette-ctl/
  plans/<plan_id>/state.jsonl          ← Layer 1: cell state

Target device (where the CLI actually runs the benchmark)
────────────────────────────────────────────────────────────
<work-dir>/.pipette/
  state/evals/
    <digest16>.jsonl                    ← Layer 2: sample state
    <digest16>.jsonl.stale-<ts>         ← rotated sample state
```

**Identity.** Files are keyed by the run digest (the header's
`digest`): a SHA-256 over the request's portable inputs, never bound
host paths, so a cell resumes across machines. It covers the declared
runtime and model, the model flags, the runtime flags, the benchmark
flags and the benchmark definition, with these left out:

- **`parallel` and `ctx_size`** in the runtime flags. They change
  throughput, not what a completion is, so lowering `parallel` after
  an OOM resumes a dataset that may be hours in. A `ctx_size`
  truncation shows up per sample as `stop_reason`.
- **The benchmark flags' timeouts.** They bound how long the client
  waits for a completion, not what the completion is, so raising one
  after a run of timeouts resumes instead of starting over. An eval
  cell carries no other benchmark flag, so setting, changing or
  removing its timeout keeps the same file, and a new HTTP timeout
  sends the failed samples again (see **Skip set**).
- **A vault source's `custom_label`.** It names the record, not the
  model or runtime, so a labeled claim resumes the file an unlabeled
  run of the cell wrote.

A vLLM cell's digest also names its row format
(`reasoning_then_answer`): its rows carry the reasoning in front of
the answer, so a vLLM cell never resumes a checkpoint whose rows hold
the answer alone.

The first 16 hex chars appear in the filename; the full digest appears
in the header and alone decides whether the file resumes. Two runs
share state only if they produce the same digest.

**Repeated samples (`#k` ids).** An eval that repeats a prompt arrives
as distinct sample ids `<id>#0 … <id>#(N-1)`, which the client treats
as ordinary, independent samples (expansion: pipette-mgmt
[docs/scoring-service.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/scoring-service.md)).
Because the skip set is keyed on sample id, a resumed run skips the
attempts already completed and draws only the remaining `#k` ids.
Without a pinned seed a redrawn attempt is a new independent draw, not
a replay of the lost one; with one, each `#k`'s seed derives from it
and the id by the entry's rule, and the redraw replays it. Sampling,
seed rule included, is part of the benchmark definition, which the
digest covers, so re-pinning an entry starts a new file
instead of resuming half a cell under sampling it was not drawn at.

**File format.** Line 1 is a JSON header; lines 2+ are one completion
each.

```
{"digest":"…","meta":{"benchmark_id":"eval_smoke","runtime":"…","model":"…"},"http_timeout_seconds":600}
{"id":"s0001","completion":"…","stop_reason":"eos","completion_tokens":412}
{"id":"s0002","completion":"…","stop_reason":"truncated","completion_tokens":1024}
…
```

`meta` carries the benchmark id, runtime and model in plain text for
`jq` inspection. It is **not validated**; only `digest` decides
reusability. `http_timeout_seconds` is the timeout its failed rows
were written under, absent when the cell pins none. Timing info is
intentionally not stored: use `ls -l <file>` for last-write time and
read the `<ts>` (Unix seconds) from a stale sibling's filename for
rotation time. Rows are `BenchmarkEvalCompletion`
(pipette-mgmt
[`crates/pipette-plan-types/src/submission/mod.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/submission/mod.rs)).
A repeated id loads as its last row.

**Lifecycle.**

1. The first open of a digest writes the header. A file already at
   that path resumes when its header carries the same full digest.
2. One row is appended per sample; the run never deletes the file.
   A kill, a refused result or a lost lease leaves it in place, and
   the next run of the cell resumes every sample from it instead of
   generating them again.
3. On open, a header that is empty, unparseable or carries another
   full digest (a 16-char prefix collision) is renamed to
   `*.jsonl.stale-<ts>` and a fresh header is written. A digest change
   otherwise lands on a different filename, and the old file stays
   under its own name until you `rm` it.
4. Once the run's result is safe (`benchmarks run` after saving it,
   the worker after the server accepts it), the client finalizes the
   file: deleted when every sample completed, rewritten to the header
   plus the failed rows otherwise. Finalizing is best-effort: a file it
   fails to finalize is resumed, whole, by the next run of the cell.

Nothing locks the file, so two runs of the same cell in one workspace
at once would both append to it.

**Durability.** Each row is written to the OS before the next sample
starts: a clean kill loses at most the samples in flight. There is no
`fsync` (acceptable for a resumable cache; power-loss worst case is
losing a handful of OS-cached writes). A SIGKILL mid-write can leave a
truncated trailing line; on load it is cut from the file and that one
sample is redone, so the next append starts on a fresh line. An
unparseable header rotates the whole file as stale. Rewrites (finalize, a
changed timeout) go through a sibling `*.jsonl.tmp` and a rename, so a
kill cannot leave half a checkpoint.

> **Warning: local runtime patches.** If you replace the runtime
> binary in `<work-dir>/.pipette/runtimes/…/` with a locally patched
> build (for example, to debug an inference change), the declared runtime
> stays the same and any prior checkpoint will silently resume
> against the new binary. Delete the active checkpoint before
> re-running.

## Resume triggers

In the table, *rotate* means the run starts a new file from sample 0.

| Input change | Effect |
|---|---|
| Benchmark definition edited (incl. sampling) | rotate |
| Declared runtime (version, flavor, engine ref) | rotate |
| Declared model (`--model` ref, pinned `sha256`, vision mmproj ref) | rotate |
| `--model-flags` (e.g. `enable_thinking`) | rotate |
| `--runtime-flags` other than `parallel` / `ctx_size` (MLX eval takes none) | rotate |
| `parallel` or `ctx_size` | **no** |
| Benchmark flags' `http_timeout_seconds` | **no**, failed rows run again |
| A vault source's `custom_label` set, changed or dropped | **no** |
| Model file or runtime binary/venv replaced behind the same ref | **no** |
| Resolved host paths (model file, mmproj, runtime dir) | **no** |
| Header line corrupt or file unreadable | rotate (renamed `*.stale-<ts>`) |

## Failed-sample markers (llamacpp and MLX eval)

A sample that fails on its own (it crashed the server alone, timed out,
or was refused) gets a failed row in the checkpoint and the cell goes
on. The case that called for it (#103): on Windows, some prompts crash
`llama-server` with
`STATUS_STACK_OVERFLOW` from `std::regex` backtracking in its
tokenizer. When and how each engine decides that a sample failed, rather than
was lost with the others in flight, is in
[eval-runs.md](eval-runs.md#when-samples-are-lost) and
[What each engine adds](eval-runs.md#what-each-engine-adds). A failed
row has an empty completion and `stop_reason: "failure"` with the
reason in `stop_detail`, plus the legacy `failed: true` and
`failed_reason` (the same reason):

```
{"digest":"…","meta":{…}}
{"id":"s0001","completion":"…","stop_reason":"eos","completion_tokens":412}
{"id":"51cd2cdc9277","completion":"","failed":true,"failed_reason":"[2026-05-12T17:30:00Z] llama-server crashed mid-completion: exit signal: 11 (SIGSEGV); client saw: …","stop_reason":"failure","stop_detail":"[2026-05-12T17:30:00Z] llama-server crashed …"}
```

The `[<rfc3339>]` prefix on a llama.cpp crash reason is when the client
saw the exit.

**Finalize.** The result carries every persisted row, completed and
failed, so the submission carries both. What mgmt does with failed rows
(scoring, warehouse rows, `eval_metadata.samples_failed`) is in
pipette-mgmt
[docs/scoring-service.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/scoring-service.md)
and
[docs/storage.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/storage.md#per-run-metadata-eval_metadata).
On disk, a cell with failed rows keeps a failed-only file, so later
runs of the same digest skip the samples that failed. A `.jsonl` left
in `state/evals/` after a successful run is itself the operator signal.

**No cap on failures.** What stops a cell on each engine is in
[What each engine adds](eval-runs.md#what-each-engine-adds). The
operator decides whether to stop a cell with many failed samples, from
its `samples_failed`.

**Signalling.**

1. **Live log**: a warning per failed or lost sample, and `eval
   finished` with the totals and `failed_ids` at the end of the cell.
   The events are listed in
   [Reading a run's log](eval-runs.md#reading-a-runs-log).
2. **Checkpoint file**: the failed-only `.jsonl` survives finalize.
3. **mgmt**: the submitted rows and `samples_failed` (see the links
   above).

**Skip set.** The per-sample loop skips any id in the checkpoint,
completed or failed. A run under a different `http_timeout_seconds`
than the header records is the exception: it drops the failed rows on
open, rewrites the file under the new timeout, and sends those samples
again, since the old timeout may be all that failed them.

## vlm_eval and asr_eval

`vlm_eval` and `asr_eval` cells use the same directory, file format
and lifecycle as text `eval`, with these differences:

- **Attempts.** The next run of the cell sends a failed sample again
  while it has attempts left. Each sample gets three attempts, counted
  from its failed lines in the file, so the count holds across runs.
  Finalize keeps one failed line per attempt spent.
- **No partial result.** The client refuses to save or submit a cell
  without an answer to every sample, and the checkpoint stays for the
  next run. For `vlm_eval`, `PIPETTE_ALLOW_INCOMPLETE_EVAL=1` lets the
  cell finish so that you can inspect the failures; the management
  server still refuses the result. `asr_eval` ignores this variable.
- **Identity.** The digest hashes the corpus digest (or, for a local
  definition, a hash of its samples) in place of the samples. It
  leaves out the model's Hugging Face auth token, and also vLLM's
  `max_model_len` and `prefix_caching`. An `asr_eval` digest also
  names the version of the client's request recipe.
- **Timeouts.** An `asr_eval` digest leaves out the benchmark flags'
  timeouts, as text `eval` does. A `vlm_eval` digest includes them, so
  a changed `http_timeout_seconds` starts a new file.

The rules for each family are in
[vlm_eval: Robustness](../methodology/vlm-eval.md#robustness) and
[asr_eval: Resume identity](../methodology/asr-eval.md#resume-identity).

## Plan interaction

The plan runner invokes the CLI once per cell with the same inputs on
every attempt, so a retried cell produces the same digest and resumes
without the plan runner knowing about it. Each cell sticks to the
device that first started it, since its checkpoint lives there
([device affinity](../pipette-ctl/plan-runner.md#device-affinity)).
Different cells of the same benchmark get different digests and
therefore different files, so interrupting one cell does not touch
another cell's partial work.
