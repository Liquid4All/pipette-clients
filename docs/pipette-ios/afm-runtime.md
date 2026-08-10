# AFM runtime (Apple Foundation Models)

The **AFM** runtime benchmarks Apple's on-device system model (the
`FoundationModels` framework) as a third iOS runtime alongside llama.cpp and MLX.
It plugs into the same harness: `Engine.run` routes the `(.appleFoundation, .afm)`
pair to `AFMRuntime`, which consumes the typed `BenchmarkDefinition`, averages over
`BenchmarkMeasurement`, gates on readiness, and returns a typed `BenchmarkResult`
with fields identical to the other runtimes, so the downstream payload → CSV →
submission pipeline is unchanged. Source: `Runtimes/AFM/`.

It is **headless-only** for now (`RuntimeChoice.afm` is not in the New Job picker);
drive it with `headlessrun runtime=afm …`. `Model.appleFoundation` has no on-disk
path (the model ships with the OS), so it is constructed directly at the headless
boundary rather than by `Model.detect`.

## Why it's a partial runtime

The framework is high-level: no token-id input, no prefill/decode/KV access, no
in-process memory, runs out-of-process (ANE), ~4k-token session budget. So AFM can
only reproduce a harness benchmark where it measures the *same* quantity the same
way. **Policy: same metric, or raise an error**; never emit an approximation.

| Benchmark | AFM | Notes |
|-----------|-----|-------|
| `decode_throughput` | ✅ | `decode_time_ms` = streamed wall-time (first→last token) |
| `end_to_end_latency` | ✅ | `total_time_ms` = TTFT + decode |
| `eval` | ✅ | one chat completion per sample; server scores the text |
| `prefill_throughput` | ❌ error | only TTFT is observable (prefill + first-token + scheduling fused); can't isolate the prefill phase |
| `max_memory_usage` | ❌ error | out-of-process; the in-process footprint probe can't see ANE/system-service memory |
| `vl_throughput` | ❌ error | text-only model |

The submission `runtime_name` is `apple_foundation`; `runtime_version` is the OS
version (there is no package pin to embed), `runtime_flags` is nil.

## Throughput benches: forcing a fixed token count

`decode_throughput` / `end_to_end_latency` must decode a fixed
`parameter_decode_tokens` so their timing matches llama.cpp / MLX, which force
exactly N tokens. `GenerationOptions.maximumResponseTokens` is only a **ceiling**,
and the shipping SDK (iOS 26.5) exposes **no generated-token count**, so:

- The prompt is a **non-terminating counting seed** (`… 9, 10,` → greedy keeps
  emitting `11, 12, …`), so an early EOS doesn't fire and the cap binds: at which
  point the framework has produced exactly N tokens.
- The rep confirms the cap bound by re-tokenizing the output (a proxy, near-exact
  for the clean counting output), and **errors if it fell short**. A number is only
  reported over a workload that matches the other runtimes.
- Requires `tokenCount` (iOS 26.4+) for prompt sizing and the proxy count.

Guided generation (`@Guide`) *would* enforce N deterministically; an on-device
experiment found it costs ~nothing (within noise of free generation), so it is a
viable enforcement path. See
[`afm-token-enforcement.md`](afm-token-enforcement.md) for the data and the
`metrics=enforceprobe` probe.

## Eval

`eval` runs one dataset sample at a time and returns the `{id, completion}` shape
the payload builder ingests; the **server scores** the completions. This is the
most natural fit (AFM is an instruction-tuned chat model), and the simplest
benchmark to support: no token forcing (`maxTokens` is a ceiling; we want the
model's natural answer), no thermal gate (it isn't a timing measurement), greedy
sampling for determinism. A sample that throws becomes a `.failed` completion
rather than aborting the run. Entry point: `AFMRuntime.evalCompletions`.

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
    a `Transcript`, and few-shot assistant turns are not used in these datasets. If
    a future dataset needs faithful multi-turn, build a `Transcript` here instead.

`mcqChoices` and `evalId` from the definition are not needed by AFM: the model
answers in natural language and the server maps/scores against the dataset.

### Verifying eval on device

Eval needs a benchmark whose payload carries `samples`, so it can't be driven by
the synthetic `metrics=` shortcuts. Sync a real eval catalog entry (or POST a job
with an `eval` cell that includes samples) and run it through the normal job path;
`AFMRuntime.evalCompletions` logs `[AFM] eval: <n> samples …` and the completions
flow to the standard submission pipeline.

## Diagnostics

`Runtimes/AFM/AFMRuntime+Diagnostics.swift` holds dev-only probes, driven by
`headlessrun runtime=afm metrics=<probe>`:

- `tokprobe`: `tokenCount` on short inputs (exposes the fixed template wrapper)
- `genprobe`: generated-vs-retokenized token counts under a small cap
- `capprobe`: whether short-answer prompts stop at EOS before the cap
- `enforceprobe`: guided vs free decode throughput (see the enforcement doc)
