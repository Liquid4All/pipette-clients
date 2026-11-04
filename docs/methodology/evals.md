# Eval Methodology

## Measured Quantity

Eval benchmarks measure model-output correctness on a fixed prompt dataset.
Each sample has a stable ID, prompt messages for inference, and hidden
ground-truth data held by the scoring service. The client records the model's
raw completion for every sample; scoring happens later, in the scoring
service mgmt forwards the completions to, by joining completion IDs back to
the dataset.

In pipette, this metric is reported by the `eval` benchmark family.

The client-side result is not a numeric score. It is a `completions` array,
one row per sample:

- `id`: the sample ID the benchmark served.
- `completion`: raw model output text; empty for a failed sample.
- `stop_reason`: how generation ended (`eos`, `truncated`, `doom_loop`,
  `failure`, `unknown`). Every row carries one, and `failure` is what marks a
  failed sample.
- `stop_detail`: optional human-readable detail behind the stop reason, such
  as the crash message.
- `completion_tokens`: optional generated-token count, where the engine
  reports one.

A failed row also carries `failed: true` and `failed_reason`, for consumers
that read those fields. The server takes a sample's failure from `stop_reason`
alone. The field contract, the `stop_reason` enum, and how the server
scores and stores the rows are owned by mgmt: pipette-mgmt
[docs/benchmarks.md §2.5](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#25-eval-accuracy)
and
[docs/scoring-service.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/scoring-service.md).

The scope is quality of generated answers under the benchmark's prompt,
generation, and scoring rules. Eval benchmarks are not latency or throughput
measurements; any timing observed during generation is orchestration context,
not the submitted eval metric.

## Measurement Challenges and Controls

Eval results depend on prompt text, chat-template behavior, decoding settings,
model flags, and the scoring dataset version. A result is comparable only when
the benchmark ID, eval ID, dataset name, model, runtime, quantization, and
prompt-shaping flags are the same.

Sample IDs are the join key, and the server refuses a submission with
duplicate IDs. Before the CLI submits a stored result (`sync`, `benchmarks run
--sync`), it removes duplicate IDs from it
([`score_validation`](../../crates/pipette-cli/src/score_validation.rs)), so a
`payload.json` from an older build does not fail a whole sync.

Long evals can run for hours on edge devices, so every runtime checkpoints
each completed sample locally and a restarted run resumes from the samples
not yet recorded. See [Eval checkpoint & resume](../pipette-cli/eval-checkpoint.md)
for the file, its identity, and what rotates it.

Free-text evals can produce degenerate repetition loops. No client detects
them itself: a sample runs to EOS or to the token cap, and a degenerate
completion is scored as it stands. The one exception is opt-in and vLLM's own:
a vLLM eval cell that sets the `repetition_detection` runtime flag asks vLLM to
stop a sample whose tail repeats (see
[Repetition stop](#repetition-stop-repetition_detection)).

## Measurement Protocol

The client's part of an eval is local generation:

1. Resolve the eval benchmark definition.
2. Load prompt samples from the benchmark's embedded `samples` field.
3. Start or attach to the selected runtime.
4. Open the resumable eval checkpoint for the benchmark/model/runtime/flags
   identity.
5. For each sample not already checkpointed, generate one completion with the
   benchmark's generation settings.
6. Append each completion to the checkpoint the moment it finishes.
7. Return every checkpointed row as the result payload's `completions` array.
   The checkpoint stays until the result is saved or accepted by the server,
   and is then deleted, or cut down to its failed rows.

The client then submits the payload, and a later `sync` stores the server's
scored metrics in `metrics.json` once the job is `processed`. Scoring itself
is mgmt's (see [Measured Quantity](#measured-quantity)), and
[How an eval runs](../pipette-cli/eval-runs.md) covers the client's path from a
claimed job to a submitted result.

Sampling is **the catalog's where the entry pins it, and a client-side policy
keyed on the eval id where it does not**. A benchmark entry may pin
`parameter_temperature`, `parameter_top_p`, `parameter_top_k`,
`parameter_min_p` and `parameter_seed`, each independently; the rules for
them, and the eval-id temperature fallback, are in pipette-mgmt
[docs/benchmarks.md §2.5](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#25-eval-accuracy)
and
[`BenchmarkDefinition::eval_temperature`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/benchmark/mod.rs).

On the CLI and iOS, the temperature is always sent, and a cut the entry leaves
unpinned is **not sent**: the request omits the key and the runtime's own
default decides. The resolved set travels as
[`eval_cell::Sampling`](../../crates/pipette-ops/src/eval_cell.rs), which writes
it into the request body. llama.cpp's `/completion` and the OpenAI-compatible
servers spell all five the same, so one writer serves every engine that builds
its body as JSON; MLX sets the same fields on its typed request. SGLang refuses
`top_k: 0`, so torch-oai sends a pinned `0` as `-1`, which vLLM accepts too.

**A seed is sent only when the entry pins one, and never the pinned value
itself.** Each repeated attempt (see [Repeats](#repeats-and-pass1) below) must
be an independent draw, so the `#k` repeat index offsets the base seed
(`EvalBenchmark::sample_seed`, wrapping inside `u32::MAX`, the width of the
narrowest runtime's seed). The base seed on every repeat would return one
completion `k` times and collapse pass@1 to the single-shot number.

The effective sampling config per runtime is:

| Runtime | How the sampling is set | Notes |
| --- | --- | --- |
| llama.cpp CLI | `temperature`, `top_p`, `top_k`, `min_p`, `seed` on `/completion` | Both arms, streaming and MCQ; an unpinned cut is omitted |
| MLX | the same five on `/v1/chat/completions` | Typed request fields, omitted when unpinned |
| torch-oai | the same five on `/v1/chat/completions` | `top_k`/`min_p` are the vLLM/SGLang extras; omitted when unpinned |
| iOS | the same five, in process | llama.cpp's sampler chain and MLX's sampler take all five; Apple Foundation refuses the pins it cannot apply (see caveat below) |

> **iOS caveat.** Apple Foundation takes one cut and no min-p, so an entry that
> pins `min_p`, or `top_k` with `top_p`, fails on it rather than drawing
> without them. Android sends no temperature at all and decodes every eval
> greedy. A published number must state the runtime that produced it.

The exact chat-template boundary is still runtime-specific (see the per-runtime
sections below).

### Repeats and pass@1

Some datasets ask for each prompt to be completed more than once. The scoring
service serves those as salted ids `<id>#0 … <id>#(N-1)`, and the scorer
averages the draws of one prompt into pass@1 (pipette-mgmt
[docs/scoring-service.md](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/scoring-service.md#get-evalseval_iddatasetsdataset_namesamples)).
The client has **no repeat-specific code**. It completes each `#k` id like any
other sample, one completion per id, and submits a flat `completions` array;
the per-id checkpoint resumes each attempt independently.

Sampled generation moves eval scores from run to run. The repeat count, not a
fixed seed, is what averages that out: a dataset with no repeats, such as
IFStruct, reports one draw per prompt, so its score carries the full sampling
noise.

Each eval's dataset, repeat count, sampling and scoring are on its mgmt
methodology page:
[gpqa_diamond](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/gpqa_diamond-2026.06.1.md),
[ifbench](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/ifbench-2026.06.1.md),
[ifstruct](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/ifstruct-release_v1_0.md)
and
[math_500](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/math_500-2026.06.1.md).

## Runtime Details

The current client implementations use these paths:

| Path | Generation boundary | Notes |
| --- | --- | --- |
| llama.cpp CLI | `llama-server` `/apply-template`, then `/completion` | Free-text streams through SSE; MCQ uses a grammar-constrained one-token request. |
| MLX CLI | `mlx_lm.server` `/v1/chat/completions` | Server applies the tokenizer chat template; Rust streams SSE deltas and drops the connection to stop early. |
| torch-oai | OpenAI-compatible `/v1/chat/completions` | Free-text uses streaming chat completions; MCQ prefers `guided_choice` and falls back to first-token `top_logprobs`. |
| iOS app | in-process chat completion: llama.cpp's C API, MLX, or Apple's on-device model | llama.cpp clears the context per sample, renders the template with its C renderer, or with llama.cpp's Jinja renderer when the C renderer rejects it, and answers MCQ with one grammar-constrained token; MLX and Apple's model answer MCQ free-form. |

How a failed or lost sample is handled, and what stops a cell, is the shared
runner's and differs per engine; see
[How an eval runs](../pipette-cli/eval-runs.md#what-each-engine-adds).

### Model flags

A cell's `model_flags` can set two knobs that change what the model generates.
Each is sent only when the cell sets it, and a row's `model_flags` string
names each one set (`enable_thinking=true,reasoning_effort=high`), so rows at
two settings are kept apart.

- `enable_thinking` (`true` or `false`) is the chat-template kwarg of that
  name. Unset, the template's own default applies, which is thinking on for
  most reasoning models.
- `reasoning_effort` (`low`, `medium` or `high`) is how much a reasoning model
  is asked to think, and only `gguf_text` and `torch` cells take it. gpt-oss
  renders it as the `Reasoning:` line of its prompt and defaults to `medium`;
  it has no thinking-off mode, so this is the knob that bounds its reasoning.
  `none` is not a level: vLLM's Harmony path refuses it, and `llama-server`
  turns thinking off instead.

| Runtime | `enable_thinking` | `reasoning_effort` |
| --- | --- | --- |
| llama.cpp CLI | `chat_template_kwargs` on `/apply-template` | `chat_template_kwargs` and top-level, on `/apply-template` |
| torch-oai | `chat_template_kwargs` on every chat request | top-level on every chat request |
| MLX CLI | `chat_template_kwargs` | not accepted |
| iOS app | refused at claim | refused at claim |

On llama.cpp the kwarg is what reaches the template: every build the client
runs hands `chat_template_kwargs` to it as variables. b11349 also reads the
top-level field and writes it into the same variable.

On vLLM, gpt-oss's Harmony renderer reads the effort from the request's own
field and nowhere else. On other models vLLM passes it to the template as a
kwarg, and, unless `chat_template_kwargs` names `enable_thinking`, sets that to
`true` with it. The row's `model_flags` names only what the cell set, so a
`torch` cell that sets the effort should set `enable_thinking` next to it.

A `vlm_eval` cell takes the same `enable_thinking` flag but applies it its own
way: it probes the chat template for a thinking switch, and refuses the raw
flags, variables and config keys that could change thinking or the model
outside the record. A text `eval` cell sends its model flags as set. See
[VLM eval](vlm-eval.md#reasoning-models).

### llama.cpp

`pipette-llamacpp` starts `llama-server`, waits for readiness, and uses
`/apply-template` to render each sample's chat messages. Free-text samples use
streaming `/completion` with the eval's temperature, `n_predict =
parameter_max_tokens`, and `stream: true`. The terminal chunk's `stop_type`
is the row's `stop_reason`: `eos` is `eos`, and `limit`, the token cap or a
full slot, is `truncated`.

MCQ samples carry the same temperature with `n_predict: 1` plus a grammar
derived from `parameter_mcq_choices`; a missing string `content` field is
treated as a runtime error instead of a wrong answer. Their `stop_reason` is
`unknown`, since one constrained token says nothing about how an answer ends.

Both arms send `preserved_tokens: ["<|START_THINKING|>", "<|END_THINKING|>",
"<|START_TEXT|>", "<|END_TEXT|>"]`. `llama-server` drops control tokens from
the text, and the GGUF converter makes every `<|…|>` token one, so without the
list North-Mini-Code's reasoning and answer arrive with no boundary between
them. The list makes the server render those tokens as text, as vLLM already
does, so the scorer can split on them on either runtime. It changes neither
sampling nor where generation stops, and the server skips a string that is
not one token of the model's vocabulary, so every model carries the list.

#### Concurrent samples (`parallel`)

By default the runner issues **one sample at a time**. The `parallel` runtime
flag keeps that many samples in flight instead. Each sample is still an
independent request with the same prompt, temperature and `n_predict`, and
completions are joined to dataset rows by id, so completion order does not
matter.

It does not follow that a completion is byte-identical to what the sequential
loop would have produced. On a continuous-batching engine it generally is not:
batched matmuls reduce in a different order than unbatched ones, so the logits
differ in the last bits, and under greedy decoding a single near-tie flips a
token and the texts diverge from there. Measured on Qwen3.5-9B at
`temperature = 0`, 17 of 30 completions matched the serial run exactly and the
other 13 shared a long prefix before diverging into an alternate phrasing of
the same derivation.

So `parallel` should be read as preserving the *distribution* a cell samples
from, not the individual strings. Aggregate scores are expected to be stable
within sampling noise, which is the right assumption for throughput work but
the wrong one for a reference run: a methodology comparison that has to
attribute a small score delta cannot tell batch nondeterminism apart from the
thing it is measuring, and should be run sequentially.

The flag is available on every eval cell, text `eval`, `vlm_eval` and
`asr_eval` alike, but it means different things either side of the engine
split:

- **llama.cpp** renders it as `--parallel`, so the server itself opens that
  many slots and divides its context across them. The slot arithmetic below is
  specific to this case. The llama.cpp eval runners share one sample pool
  (crash arbitration, serial retry, heartbeat); `vlm_eval` calibrates its
  per-slot context from the dataset's images instead of using a constant (see
  [VLM eval](vlm-eval.md#concurrent-samples-parallel)), and `asr_eval` sizes it
  from the clips' durations (see [ASR eval](asr-eval.md#context-sizing)).
- **vLLM and SGLang** get nothing new on the command line. Those engines batch
  continuously and size their own slots from `--max-num-seqs` and the KV
  budget. Here `parallel` is purely the number of concurrent HTTP requests the
  client keeps outstanding, and there is no context arithmetic to do.

`parallel` and `ctx_size` are throughput knobs, left out of the resume
identity on `eval` and `vlm_eval` alike, so a run that runs out of memory at
64 slots resumes at 32 without repeating its recorded samples (see
[Eval checkpoint & resume](../pipette-cli/eval-checkpoint.md)).

Two details of the llama.cpp case matter:

- **Context arrives two ways, and they behave differently.** `llama-server`
  divides `--ctx-size` across its slots. An explicit `ctx_size` is passed
  through **verbatim**, so at `parallel > 1` it is the *total* and each slot
  gets `ctx_size / parallel`: `589824` across 64 slots leaves 9216 each.
  Omit it and the per-cell default (`8192 + max_tokens`, one slot's budget) is
  **multiplied** by the slot count instead, which is what lets `parallel` be
  set on its own. The launch log names which source was used and what each
  slot ended up with; `llama-server` reports the same figure as `n_ctx_slot`.
  A multi-slot run whose slot would be under `max_tokens` is refused before it
  starts, since every generation would truncate.

  The default is generous on prompt headroom (8192 tokens, against real eval
  prompts of a few hundred), so it multiplies into a large total: at
  `parallel = 64` it asks for 1,048,576 tokens. Stating a `ctx_size` sized
  from the dataset (`parallel x (longest prompt + max_tokens)`) is worth real
  VRAM once the slot count is high.
- **Token tee-ing to stderr is off whenever `parallel > 1`**, because the tee
  holds the process-wide stderr lock for a whole generation and would
  serialize the pool. The per-sample prompt and completion log lines still
  carry the sample id.

### Truncation

A completion that hits `max_tokens` is recorded, submitted and scored like any
other; it is not a failure. So a run can report `completed=1500 failed=0` while
most answers stopped mid-sentence, and the score then reflects the token budget
as much as the model. On reasoning models that share is routinely the majority.
Each sample carries its own `stop_reason`, and the runner logs a summary line
whenever any completion was truncated. Treat a high rate as a caveat on the
score rather than a property of the model.

### MLX

`pipette-mlx` starts a local `mlx_lm.server` and sends each sample's `messages`
array to `/v1/chat/completions` with `stream: true`, letting the server apply
its chat template. The request carries the eval's temperature (`0.0` stays
greedy), `max_tokens = parameter_max_tokens`, and `chat_template_kwargs`
only when `enable_thinking` is set (absent otherwise, so the server keeps its
tokenizer-derived default).

Rust accumulates the streamed deltas and appends each finished completion to
the checkpoint. The request asks for the `usage` trailer
(`stream_options.include_usage`), so each row records the finish chunk's
`finish_reason` as its `stop_reason` (`stop` is `eos`, `length` is
`truncated`) and the trailer's `completion_tokens`, which counts the EOS token.

### torch-oai

`pipette-torch-oai` sends the sample `messages` array directly to an
OpenAI-compatible chat-completions server and lets the engine apply its chat
template. Free-text samples use the eval's temperature, `max_tokens =
parameter_max_tokens` (except under a [context bound](#context-bound)), and
`stream: true`. The row keeps the deltas' `content`, on vLLM with their
reasoning (below), and records the finish chunk's `finish_reason` as its
`stop_reason`: `stop` is `eos`, `length` is `truncated`, `repetition` (vLLM's
repetition stop, below) is `doom_loop`, and anything else, or a stream with
none, is `unknown` with the reason in `stop_detail`. The request does not ask
for the `usage` trailer, so rows carry no `completion_tokens`.

A vLLM server started with `--reasoning-parser`, and any vLLM serving gpt-oss
(vLLM gives it the `openai_gptoss` parser unless told otherwise), streams the
reasoning apart from the answer, in `delta.reasoning`. A vLLM row puts it
back as a chat template that opens the reasoning in the prompt would have the
model write it: `<reasoning></think>` followed by the answer, or the reasoning
alone, with no tag, when the stop came inside it (the cap, the repetition
stop, or EOS before any answer). A grader that cuts through the first
`</think>` reads the same answer as from the content alone, and grades a
sample cut in its reasoning on that text. Content
written before the reasoning stays in front of it, and reasoning streamed
after the answer began joins the rest before the `</think>`. A reply with no
reasoning is recorded as its content, unchanged, and an SGLang row keeps the
content alone. Stored vLLM rows whose `client_version` is dated 2026-10-14 or
earlier, and some dated 2026-10-15, kept the content alone. In those rows a
sample cut in its reasoning reads as an empty completion (see [eval checkpoint](../pipette-cli/eval-checkpoint.md)).

MCQ samples first try vLLM's top-level `guided_choice` field. If the server
rejects that extra field, the runner asks for one token with `logprobs` and
selects the listed choice with the highest first-token logprob. Their
`stop_reason` is `unknown`, as on llama.cpp.

#### Repetition stop (`repetition_detection`)

A vLLM eval cell (`docker_vllm` or `uv_vllm`) may set the
`repetition_detection` runtime flag, a table of the three fields of vLLM's
`RepetitionDetectionParams`:

```toml
runtime_flags = [
  { benchmark_type = "eval", runtime_type = "uv_vllm", model_type = "torch",
    repetition_detection = { min_pattern_size = 2, max_pattern_size = 128, min_count = 10 } },
]
```

The values in the example (`min_pattern_size = 2`, `max_pattern_size = 128`,
`min_count = 10`) are the settings that Liquid's reference eval runs use. Each
free-text request then carries the table as vLLM's own `repetition_detection`
field, and vLLM ends a sample once its last `n` tokens, for some `n` in the
pattern range, have come `min_count` times in a row. The MCQ arm never sends
it: a stop could cut a constrained choice short. The SGLang cells cannot set
the flag.

vLLM finishes such a sample with `finish_reason` `repetition`. The row keeps
the text generated up to the stop, and its `stop_reason` is `doom_loop` with
`stop_detail` `finish_reason=repetition`: a completed sample, neither a failure
nor `truncated`. The flag is recorded in the row's `runtime_flags`, the
request's command names it, and it is part of the resume identity, so turning
it on starts a checkpoint of its own.

It needs vLLM 0.17 or later. An older vLLM drops request fields it does not
know without an error, and would run the cell without the stop.

### Context bound

An entry that sets `parameter_max_context` stops each free-text sample once
its prompt plus its output reaches that bound, the way an engine stops at the
end of its context. The parameter, its catalog rules, and which runtimes and
cell pins plan ingestion accepts are mgmt's: pipette-mgmt
[docs/benchmarks.md, The context bound](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#the-context-bound).

The client never reads the bound off the entry to size a context. It runs the
context the cell pins, so before a run the CLI checks the cell against the
entry
([`EvalBenchmark::check_cell_context`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/context_bound.rs),
called from `run_cell` in [run.rs](../../crates/pipette-cli/src/run.rs)) and
refuses one that would run the bound unpinned or at a different size. mgmt
makes the same check at ingestion; the CLI's also covers a local run.

- **vLLM** (`docker_vllm`, `uv_vllm`): the cell pins `max_model_len` equal to
  `parameter_max_context`. A free-text request then carries no `max_tokens` at
  all, because vLLM refuses (HTTP 400, "maximum context length") a request
  whose prompt plus `max_tokens` exceeds `max_model_len` rather than clamp it.
  Without `max_tokens` vLLM generates up to `max_model_len` less the prompt,
  the same bound that an offline vLLM run applies. A model whose
  `generation_config.json` sets `max_new_tokens` is capped there too, since
  vLLM reads that as the default `max_tokens`. A prompt that alone fills
  `max_model_len` is still refused, and its row is failed, not `truncated`.
  The request's command names the bound in place of the cap:
  `max_tokens=unset (max_model_len=32768 bounds prompt + output)`.
- **llama.cpp** (`llamacpp_cli_stock_tools`): the cell pins `ctx_size` at
  `parameter_max_context` times its slots, and `llama-server` stops a sample
  (`truncated`) when its prompt plus its output fills a slot.

A sample the bound stops finishes with `length` or `limit`, so its row is
`truncated`, as under the token cap (see [Truncation](#truncation)). The MCQ
arm asks for its few tokens whatever the context.

On a free-text entry with no bound, a vLLM cell that pins `max_model_len` at
or under `parameter_max_tokens` is refused, since it would bound the context
the entry leaves open. Unpinned, a vLLM eval runs at the derived
`8192 + parameter_max_tokens`. A pin above the budget must leave room for the
longest prompt beside it: vLLM refuses every sample whose prompt plus
`parameter_max_tokens` exceeds `max_model_len`, so a pin just above the budget
fails all but the shortest prompts. The SGLang cells carry no `max_model_len`
(their context is `--context-length`, left to the operator), so they always
send `max_tokens`. `max_model_len` is part of the resume identity, so pinning
it starts a checkpoint of its own.

### iOS

The app runs evals in process on three engines: llama.cpp over a local GGUF
model, MLX, and Apple's on-device model. Each sends its samples through a Swift
port of the CLI runners' eval entry, one at a time, and returns the same
`benchmark_id` plus `completions` payload shape. The entry records each
completion under its sample id as it lands, and a sample that throws becomes a
failed row (`stop_reason` `failure`) instead of an aborted run. A stop ends
the run with the rows so far kept for the next one, and a sample that fails
while a stop is pending counts as stopped, not failed: backgrounding the app
can fail the sample in flight. The checkpoint is finalized once the result is
saved, or accepted by the server on the worker path, as on the CLI.

The llama.cpp path clears the context before each sample, samples under the
eval's temperature and pins, and answers MCQ with one token from a grammar that
admits only the choices, as the llama.cpp CLI does; an answer outside the set
fails the sample. A failed compute backend (`llama_decode` below -1) ends the
run with its rows kept, since every later sample would fail with it. It renders
the chat template with llama.cpp's C renderer first, so a model whose template
that renderer only approximates can score differently here. A template it
rejects, such as Gemma-4's, falls back to llama.cpp's Jinja renderer at the
server's default thinking setting, which is what `/apply-template` renders for
the CLI's request when the cell sets no `enable_thinking`. It renders no control
token as text, so North-Mini-Code's markers, which the CLI asks the
server to keep, do not reach an iOS row (gap 15 in the
[parity record](../client-parity.md)). MLX and
Apple's model answer MCQ free-form within the cell's token budget, as the MLX
CLI does.

## Result Validity and Comparability

A run is invalid if the benchmark lacks samples, the runtime cannot produce a
completion payload, the checkpoint is incompatible with the current run
identity, completion IDs do not match the scoring dataset, or server-side
scoring fails.

To compare eval results:

- Use the same benchmark ID, eval ID, dataset name, model, runtime,
  quantization, and prompt-shaping flags.
- Keep `parameter_max_tokens` and MCQ settings fixed.
- Treat scoring dataset versions as part of the benchmark definition.
- Inspect failed rows (`stop_reason` `failure`) before interpreting aggregate
  scores.
- Compare each eval by its own metric semantics; every eval reports
  correctness, but each one tests a different ability.

Eval scores are **runtime-specific**. The scorer never sees the model
identifier or the device: it sees the completions, how each one ended, whether
the job asked for thinking, and, for a `vlm_eval` or `asr_eval`, the prompt
profile.
The completions themselves are produced by a particular runtime, and the generation
paths differ in ways that move scores: MCQ is grammar-constrained on llama.cpp
(CLI and iOS), free-form on MLX (CLI and iOS) and Apple's on-device model, and
answered through `guided_choice` or first-token logprobs on torch-oai; chat
templates are applied by different components (see the iOS note above).
Treat a score as belonging
to the `(model, quantization, runtime, flags)` tuple it was produced under, not
as a property of the model alone, and do not compare scores across runtimes.
Published headline numbers should state which runtime produced them.

## Code References

The shared eval benchmark definition and result payload shape are implemented
in [`pipette_plan_types::benchmark`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/benchmark/mod.rs).

Eval checkpointing is documented in
[Eval Checkpoint & Resume](../pipette-cli/eval-checkpoint.md) and implemented in
[`pipette_ops::eval_completions`](../../crates/pipette-ops/src/eval_completions.rs).

The `parallel` and `repetition_detection` runtime flags, and the context
contract of `parallel`, are declared in
[`pipette_plan_types::RuntimeFlags`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/runtime_flags.rs)
and rendered to argv by
[`pipette_llamacpp::server`](../../crates/pipette-llamacpp/src/server.rs).

The model flags are declared in
[`pipette_plan_types::ModelFlags`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/model_flags.rs)
and carried to each runner on
[`pipette_ops::eval_cell::EvalCell`](../../crates/pipette-ops/src/eval_cell.rs).

Every CLI eval runs through
[`pipette_ops::measurement::eval`](../../crates/pipette-ops/src/measurement/eval.rs),
described in [How an eval runs](../pipette-cli/eval-runs.md). The runtime eval
runners are:

- [`pipette-llamacpp eval`](../../crates/pipette-llamacpp/src/execute/eval.rs)
- [`pipette-mlx eval`](../../crates/pipette-mlx/src/execute/eval.rs)
- [`pipette-torch-oai eval`](../../crates/pipette-torch-oai/src/execute/eval.rs)
- [iOS native runtimes (Swift)](../../ios/Pipette/Pipette/Runtimes/)
