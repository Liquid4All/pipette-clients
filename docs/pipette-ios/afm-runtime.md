# AFM runtime (Apple Foundation Models)

The **AFM** runtime benchmarks Apple's on-device system model (the
`FoundationModels` framework). It is one of the four iOS engines, beside llama.cpp,
MLX and ONNX Runtime GenAI. `RunCell.dispatchRun` routes `.appleFoundation` to
`AFMRuntime`, which plugs into the harness that every engine shares
([architecture.md](architecture.md#inference-engines)). Source: `Runtimes/AFM/`.

The model ships with the OS, so it has no on-disk path or download.
`availableModels()` appends `DiscoveredModel.appleFoundation` (size 0) whenever the
framework reports the model available, so AFM shows in the New Job picker beside
the on-disk models, and `RunCell.prepare` binds it without ensuring anything. The
headless form is `headlessrun runtime=afm …` ([usage.md](usage.md)).

## Why it's a partial runtime

The framework is high-level:

- It takes no token ids as input.
- It gives no access to prefill, decode or the KV cache.
- It runs out of process (on the Apple Neural Engine, ANE), so the app cannot see
  its memory.
- A session has a budget of about 4k tokens.

So AFM can only reproduce a harness benchmark where it measures the *same* quantity
the same way. **Policy: same metric, or raise an error**; never emit an approximation.
`isBenchmarkSupported` (`Benchmarks/BenchmarkCatalog.swift`) holds the supported
set, and both the New Job picker and `AFMRuntime` read it.

| Benchmark | AFM | Notes |
|-----------|-----|-------|
| `decode_throughput` | yes | `decode_time_ms` = streamed wall-time from the first snapshot to the last, the `N - 1` steps after the prefill's token, scaled to `N`; reads slightly low when the first snapshot holds several tokens |
| `end_to_end_latency` | yes | `total_time_ms` = the request's start to the stream's end |
| `eval` | yes | one chat completion per sample; server scores the text |
| `prefill_throughput` | error | only TTFT is observable (prefill + first-token + scheduling fused); can't isolate the prefill phase |
| `max_memory_usage` | error | out-of-process; the in-process footprint probe can't see ANE/system-service memory |
| `vl_throughput`, `vl_max_memory`, `vl_spec_decode`, `asr_throughput`, `asr_max_memory` | error | text-only model |

A batched entry is refused by name: the system model answers one request at a time
([batched entries](../methodology/end-to-end-latency.md#batched)).

The submitted `runtime_descriptor` is the bare `apple_foundation` tag plus the
private-API build flags; there is no package or version pin to embed.
`runtime_flags` is `nil`: plan-types declares no `RuntimeFlags` variant for this
runtime on any benchmark, so an AFM cell that carries runtime or benchmark flags is
refused rather than having them dropped.

## Throughput benches: forcing a fixed token count

`decode_throughput` / `end_to_end_latency` must decode a fixed
`parameter_decode_tokens` so their timing matches llama.cpp / MLX, which force
exactly N tokens. `GenerationOptions.maximumResponseTokens` is only a **ceiling**
(there is no min-tokens, ignore-EOS or logit-bias option), and the shipping SDK (iOS 26.5) exposes **no generated-token count**. Production
uses free generation with a re-tokenize-and-verify check (`AFMRuntime.swift`):

- The prompt is the shared corpus (`PromptSeed.buildPromptText`, sized under the
  model's own token counter to exactly `parameter_prefill_tokens`) followed by a
  **non-terminating counting seed** (`… 9, 10,` → generation keeps emitting
  `11, 12, …`), so an early EOS doesn't fire and the cap binds, at which point the
  framework has produced exactly N tokens. The body is the same prose the llama
  and MLX cells prefill, so only the eliciting tail is AFM's own.
- The rep confirms the cap bound by re-tokenizing the output (a proxy, near-exact
  for the clean counting output), and **errors if it fell short** of the floor
  `cap - max(2, cap / 20)`. The slack absorbs re-tokenization drift; a real early
  stop lands far below it. A number is only reported over a workload that matches
  the other runtimes.
- `decode_throughput` needs `parameter_decode_tokens` of at least 2, since the
  first token is the prefill's.
- The untimed warm-up is one token rather than the cell's shape (nothing here
  compiles per shape, and generation is the scarce resource), and a warm-up that
  fails fails the cell rather than being swallowed.
- Requires `tokenCount` (iOS 26.4+) for prompt sizing and the proxy count.

Guided generation (`@Guide`) would enforce N deterministically. An on-device
measurement found guided decode within noise of free decode (guided ÷ free
~0.96 to 0.97x), but production has not switched to it; see
[afm-token-enforcement.md](afm-token-enforcement.md).

## Eval

`eval` runs one dataset sample at a time and returns the `{id, completion}` shape
the payload builder ingests; the **server scores** the completions. There is no
token forcing (`maxTokens` is a ceiling; we want the model's natural answer) and no
readiness gate (it isn't a timing measurement). Sampling is the cell's `Sampling`,
as on the other engines: the entry's pinned temperature, else the one
`EvalId.samplingTemperature` names (0.6 for the known evals), greedy at zero. The
framework takes one cut, top-k or a probability threshold, plus a seed, so a pinned
min-p, or top-k with top-p, is refused before any sample runs rather than drawn
without. A sample that throws becomes a `.failed` completion rather than aborting
the run, and each is checkpointed as it lands; a stop ends the cell instead, and its
next run resumes ([eval checkpoint](../pipette-cli/eval-checkpoint.md)).

The framework exposes no stop signal, so `stop_reason` is inferred: `truncated`
when the re-tokenized completion reaches the cap, `eos` below it, `unknown` below
iOS 26.4 where nothing can be counted. A sample whose length lands near the cap can
be misclassified (PIP-266). Entry point: `AFMRuntime.eval`, through the shared eval
entry (`BenchmarkMeasurement.eval`).

### Chat-message → session mapping

Each `EvalSample.messages` entry is a `{role, content}` pair. The shipping
`FoundationModels` API has no manual chat template; the session applies its own, so
roles map to session constructs rather than to template tokens:

- **`system`** turns are joined (`\n\n`) into the session `instructions`
  (`LanguageModelSession(instructions:)`).
- **Remaining turns** become the prompt for `session.respond(to:options:)`:
  - a single user turn is used **verbatim**: the common case, since this suite
    (GPQA / MATH-500 / IFBench / IFStruct) is single-turn;
  - multiple turns are folded into one role-labeled prompt (`role: content`
    lines). This is an approximation: the shipping SDK only accepts prior turns via
    a `Transcript`, and few-shot assistant turns are not used in these datasets. A
    dataset that needs faithful multi-turn input needs a `Transcript` here.

`mcqChoices` is not used by AFM: the model answers in natural language and the
server maps and scores against the dataset. `evalId` is used only for the sampling
temperature above when the entry pins none.

### Verifying eval on device

Eval needs a benchmark whose payload carries `samples`, so it can't be driven by
the synthetic `metrics=` shortcuts. Run a real eval catalog entry through the
normal job path; `AFMRuntime.eval` logs `eval: <n> samples, maxTokens=…` under the
`afm` log category, and the completions flow to the standard submission pipeline.

## Diagnostics

`Runtimes/AFM/AFMRuntime+Diagnostics.swift` holds dev-only probes that record and
submit nothing, driven by `headlessrun runtime=afm metrics=<probe>` or
`headlessrun diag probe kind=<kind>` ([usage.md](usage.md)):

- `tokprobe`: `tokenCount` on short inputs (exposes the fixed template wrapper)
- `genprobe`: generated-vs-retokenized token counts under a small cap
- `capprobe`: whether short-answer prompts stop at EOS before the cap
- `enforceprobe`: guided vs free decode throughput
  ([afm-token-enforcement.md](afm-token-enforcement.md))
