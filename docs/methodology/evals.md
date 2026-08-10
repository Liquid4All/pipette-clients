# Eval Methodology

## Measured Quantity

Eval benchmarks measure model-output correctness on a fixed prompt dataset.
Each sample has a stable ID, prompt messages for inference, and hidden
ground-truth data held by the score server. The client records the model's raw
completion for every sample; scoring happens later by joining completion IDs
back to the dataset.

In pipette, this metric is reported by the `eval` benchmark family.

The client-side result is not a numeric score. It is a payload of completions:

- `completions[].id`: sample ID returned by the score server's sample endpoint.
- `completions[].completion`: raw model output text for that sample.
- `completions[].failed`: optional marker for a sample the runtime could not
  complete.
- `completions[].failed_reason`: optional operator-facing reason for a failed
  sample.

The numeric score is produced after sync, when the management server submits
the completions to `pipette-scores`. The saved `metrics.json` contains the
server-side score metrics for that job.

The scope is quality of generated answers under the benchmark's prompt,
generation, and scoring rules. Eval benchmarks are not latency or throughput
measurements; any timing observed during generation is orchestration context,
not the submitted eval metric.

## Measurement Challenges and Controls

Eval results depend on prompt text, chat-template behavior, decoding settings,
model flags, and score-server dataset version. A result is comparable only when
the benchmark ID, eval ID, dataset name, model, runtime, quantization, and
prompt-shaping flags are the same.

The score server uses sample IDs as the join key. A completion whose ID is not
in the dataset is rejected, and duplicate completion IDs are rejected by the
score API. Clients also repair stale duplicate IDs before submission so older
checkpointed payloads do not fail a whole sync.

Long evals can run for hours on edge devices. The CLI and server runtimes
(llama.cpp, MLX, torch-oai) therefore checkpoint each completed sample locally:
a restarted run skips samples already present in the checkpoint and resumes from
the remaining set. The checkpoint identity is a digest over the benchmark
definition, model identity, runtime identity, and generation-affecting flags.
The iOS app does not checkpoint: an interrupted iOS eval restarts from the
first sample.

Free-text evals can produce degenerate repetition loops. The clients run a
doomloop detector while generating. On the streaming runtimes it runs over the
token stream and, when a detector fires, generation for that sample is aborted
early and the partial completion is kept for scoring. The iOS app runs the
built-in default detector pipeline (`DoomloopPipeline::default()`, not operator
configurable) and only begins checking once a sample has produced more than 256
bytes of output, so very short degenerate runs can slip under that floor.

## Measurement Protocol

The eval flow has two phases: local generation and server-side scoring.

The local generation phase is:

1. Resolve the eval benchmark definition.
2. Load prompt samples from the benchmark's embedded `samples` field.
3. Start or attach to the selected runtime.
4. Open the resumable eval checkpoint for the benchmark/model/runtime/flags
   identity.
5. For each sample not already checkpointed, generate one completion with the
   benchmark's generation settings.
6. Append each completion to the checkpoint before moving to the next sample.
7. Finalize the checkpoint into a result payload with a `completions` array.

The sync and scoring phase is:

1. Submit the pending result payload to the management server.
2. The management server forwards eval completions to the score service.
3. The score service validates unique IDs and dataset membership.
4. The eval-specific scorer computes per-sample correctness and aggregate
   context metrics.
5. Later `sync` refreshes the processed job and stores the scored metrics
   locally.

Sampling temperature is a **client-side policy keyed on the eval id**, not a
field the server sends. `BenchmarkDefinition::eval_temperature()` (in
[`pipette_plan_types::benchmark`](../../crates/pipette-plan-types/src/benchmark/mod.rs)) assigns it:
`ifbench` and `ifstruct` sample at `0.6`; every other eval (and every non-eval
benchmark) stays greedy at `0.0`. **No random seed is ever sent.** For the
greedy evals this means a re-run of the same sample produces the same completion
and scores do not move for sampling reasons. For the sampled evals it means each
repeated attempt (see [Repeats](#repeats-and-pass1) below) is an independent
draw. A pinned seed would make all repeats identical and collapse pass@1 to the
single-shot number.

The effective sampling config per runtime is:

| Runtime | How temperature is set | Notes |
| --- | --- | --- |
| llama.cpp CLI | `temperature` on `/completion`: `0.6` for ifbench/ifstruct, else `0.0` | `top_p`/`top_k`/`min_p` not set by pipette; inert at `0.0` (argmax) |
| MLX | `make_sampler(temp=…)` in the sidecar, fed `eval_temperature()` | none else set |
| torch-oai | `temperature` on `/v1/chat/completions`, fed `eval_temperature()` | `top_p`/`top_k`/`min_p` not set |
| iOS | **hard greedy argmax sampler (`ee_llama_sample_greedy`): no temperature knob** | does not apply `eval_temperature()`; always greedy (see caveat below) |

> **iOS caveat.** The iOS path has no temperature control, so it runs IFBench
> and IFStruct **greedy** regardless of the `0.6` policy. Because the repeated
> `#k` IFBench attempts are then deterministic, they produce identical
> completions and iOS pass@1 degenerates to the single-shot score. iOS
> instruction-following eval results are therefore not comparable to the
> sampled CLI/server runtimes; published numbers should state the runtime.

The exact chat-template boundary is still runtime-specific (see the per-runtime
sections below).

### Repeats and pass@1

Some datasets ask for each prompt to be completed more than once. This is driven
entirely by the **scoring service**, not the client: when a dataset declares
`metadata.repeats = N` (currently IFBench `2026.06.1`, `N = 5`), the samples
endpoint expands each base id `<id>` into salted ids `<id>#0 … <id>#(N-1)` and
serves them as ordinary distinct samples. The client has **no repeat-specific
code**. It completes each `#k` id like any other sample, one completion per id,
and submits a flat `completions` array. With temperature `0.6` and no seed, the
attempts differ; the scorer averages them into pass@1, and the per-id checkpoint
(below) resumes each attempt independently.

## Benchmark Shape and Reported Fields

Concrete benchmark definitions use these fields:

- `benchmark_id`: result and reporting identifier, such as
  `eval_ifbench_2026.06.1`.
- `parameter_eval_id`: score-server eval identifier, such as `ifbench`.
- `parameter_dataset_name`: score-server dataset name, such as `2026.06.1`.
- `parameter_max_tokens`: maximum generated tokens per sample.
- `parameter_mcq_choices`: optional finite choice set for MCQ-style evals.
- `samples`: prompt samples embedded in the benchmark definition.

The submitted result reports:

- `completions`: one completion object per sample, plus optional failed-sample
  markers on paths that support them.

The score server response contains:

- `scored_samples`: per-sample prompt, completion, and `is_correct` boolean.
- `context`: eval-specific aggregate metrics.
- `runtime_version`: score-server version string.

The client stores management-server job metrics in `metrics.json` after the
job reaches `processed`.

## Runtime Details

The current client implementations use these paths:

| Path | Generation boundary | Notes |
| --- | --- | --- |
| llama.cpp CLI | `llama-server` `/apply-template`, then `/completion` | Free-text streams through SSE; MCQ uses a grammar-constrained one-token request. |
| MLX CLI | local Python sidecar `/eval` endpoint | Sidecar applies the tokenizer chat template and streams JSONL events to Rust. |
| torch-oai | OpenAI-compatible `/v1/chat/completions` | Free-text uses streaming chat completions; MCQ prefers `guided_choice` and falls back to first-token `top_logprobs`. |
| iOS app | direct llama chat completion call | Clears context per sample. MCQ is unconstrained greedy with a silent first-choice fallback (not grammar-constrained), and the path does not checkpoint or emit `failed` markers. |

### llama.cpp

`pipette-llamacpp` starts `llama-server`, waits for readiness, and uses
`/apply-template` to render each sample's chat messages. Free-text samples use
streaming `/completion` with `temperature: 0.0`, `n_predict =
parameter_max_tokens`, and `stream: true`.

MCQ samples use `n_predict: 1` plus a grammar derived from
`parameter_mcq_choices`; a missing string `content` field is treated as a
runtime error instead of a wrong answer.

If `llama-server` exits during a sample, the runner records that sample with
`failed: true`, restarts the server, and continues. If the server is still
alive but the completion request failed, the runner records the failed sample
and continues against the same process. Failed entries are carried in the
payload so downstream systems can account for excluded samples.

### MLX

`pipette-mlx` starts a local Python HTTP sidecar. Rust sends the sample list,
max-token count, already-completed IDs, and optional `enable_thinking` flag to
`/eval`. The sidecar applies the tokenizer chat template, runs greedy
`mlx_lm.stream_generate`, and emits chunked JSONL events:

- `eval_sample_start`
- `eval_sample_chunk`
- `eval_sample_done`
- `eval_done`

Rust appends each `eval_sample_done` completion to the checkpoint. If doomloop
detection fires while chunks are streaming, Rust calls `/eval/abort` for that
sample and persists the stopped completion when the sidecar reports it done.

### torch-oai

`pipette-torch-oai` sends the sample `messages` array directly to an
OpenAI-compatible chat-completions server and lets the engine apply its chat
template. Free-text samples use `temperature: 0.0`, `max_tokens =
parameter_max_tokens`, and `stream: true`.

MCQ samples first try vLLM's top-level `guided_choice` field. If the server
rejects that extra field, the runner asks for one token with `logprobs` and
selects the listed choice with the highest first-token logprob.

### iOS

The iOS app path loads the local GGUF model, clears the context before each
sample, and calls the direct chat-completion binding. It returns the same
`benchmark_id` plus `completions` payload shape as the CLI runners, but its
generation path differs from them in two ways that affect comparability:

- **MCQ is not grammar-constrained.** Unlike the llama.cpp CLI (GBNF grammar)
  and torch-oai (`guided_choice`), the iOS path samples a single greedy token
  freely, converts it to a string, and exact-matches it against the choice set.
  On a match it returns that choice; on **no match it silently returns the first
  choice** (`choices[0]`). That fallback biases unmatched samples toward the
  first option rather than constraining the model's sampling, so iOS MCQ scores
  are not directly comparable to the grammar-constrained paths.
- **No checkpoint or `failed` markers.** The iOS path does not use
  `pipette_ops::eval_completions`, so an interrupted iOS eval restarts from the
  first sample, and it does not emit per-sample `failed: true` entries the way
  the CLI runners do.

## Score Server Details

Scoring is handled by the separate `pipette-scores` service (its own
repository). At the time of writing its score server accepts these eval IDs:

- `ifbench`
- `ifstruct`

The API endpoints are:

- `GET /evals/{eval_id}/datasets/{dataset_name}/samples`: returns prompt
  samples with no ground truth.
- `POST /score`: accepts `eval_id`, `dataset_name`, and completion objects.

The score endpoint rejects duplicate completion IDs and IDs that are absent
from the selected dataset before running the eval-specific scorer.

## Result Validity and Comparability

A run is invalid if the benchmark lacks samples, the runtime cannot produce a
completion payload, the checkpoint is incompatible with the current run
identity, completion IDs do not match the score-server dataset, or server-side
scoring fails.

To compare eval results:

- Use the same benchmark ID, eval ID, dataset name, model, runtime,
  quantization, and prompt-shaping flags.
- Keep `parameter_max_tokens` and MCQ settings fixed.
- Treat score-server dataset versions as part of the benchmark definition.
- Inspect failed-sample markers before interpreting aggregate scores.
- Compare each eval by its own metric semantics; `ifbench` and `ifstruct`
  both report correctness, but they test different abilities.

Eval scores are **runtime-specific**. Although the score server is
model-independent (it sees only completions, never the model or device) the
completions themselves are produced by a particular runtime, and the generation
paths differ in ways that move scores: MCQ is grammar-constrained on the
llama.cpp CLI and torch-oai but unconstrained-greedy with a first-choice
fallback on iOS (see the iOS note above), chat templates are applied by
different components, and doomloop handling differs. Treat a score as belonging
to the `(model, quantization, runtime, flags)` tuple it was produced under, not
as a property of the model alone, and do not compare scores across runtimes.
Published headline numbers should state which runtime produced them.

## Code References

The shared eval benchmark definition and result payload shape are implemented
in [`pipette_plan_types::benchmark`](../../crates/pipette-plan-types/src/benchmark/mod.rs).

Eval checkpointing is documented in
[Eval Checkpoint & Resume](../pipette-cli/eval-checkpoint.md) and implemented in
[`pipette_ops::eval_completions`](../../crates/pipette-ops/src/eval_completions.rs).

The runtime eval runners are:

- [`pipette-llamacpp eval`](../../crates/pipette-llamacpp/src/execute/eval.rs)
- [`pipette-mlx eval`](../../crates/pipette-mlx/src/execute/eval.rs)
- [`pipette-torch-oai eval`](../../crates/pipette-torch-oai/src/execute/eval.rs)
- [iOS native runtimes (Swift)](../../ios/Pipette/Pipette/Runtimes/)

The score-server API and scoring types are in
[`pipette-scores api`](https://github.com/Liquid4All/pipette-scores/blob/main/packages/pipette-scores/pipette_scores/api/app.py)
and
[`pipette-scores types`](https://github.com/Liquid4All/pipette-scores/blob/main/packages/pipette-scores/pipette_scores/types.py).
