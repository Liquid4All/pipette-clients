# Android Kotlin llama.cpp Engine Port: Design and Invariants

This document is the record of the Android engine port from the Rust kernel to a
Kotlin kernel. The port is complete: `kotlinEngine` is the default Gradle flavor,
and CI and the release workflow build it. The `rustEngine` flavor keeps the Rust
kernel. [architecture.md](architecture.md#engine-flavors) describes the two
flavors.

- **`rustEngine`**: Kotlin JNI stubs call the Rust crate `crates/pipette-android`,
  which runs the kernel in `android/native/benchmarks.rs`.
- **`kotlinEngine`**: the kernel is Kotlin (`LlamaBenchmark.kt`). It calls JNI
  primitives in `LlamaCpp.kt` and `android/native/jni.cpp`. This follows the iOS
  design in iOS #501 (`LlamaCpp.swift` and `LlamaBenchmark.swift`).

The document records the invariants that both kernels keep, the mapping between
their symbols, how the port handles parameter defaults, the off-ramp protocol,
and the A/B measurements that let `kotlinEngine` become the default. The port
used these phases:

- **Phase 0e**: a null A/B test (`rustEngine` against itself) that measures the
  noise floor.
- **Phase 4**: the A/B comparison of `rustEngine` and `kotlinEngine`.
- **Phase 5**: making `kotlinEngine` the default flavor, then a soak period.

Related: [Android architecture](architecture.md) · [iOS execution alignment](../pipette-ios/execution-alignment.md) · [model store design](../pipette-ios/model-store-design.md) · [storage quota](../storage-quota.md)

**Scope.** If this document disagrees with the code, the code is the reference:
`android/Pipette/app/src/kotlinEngine/java/ai/liquid/pipette/llama/LlamaBenchmark.kt`
for `kotlinEngine`, and `android/native/benchmarks.rs` for `rustEngine`.

---

## 1. Summary and Architecture Mapping

The `rustEngine` flavor runs a benchmark through these layers, which cross the
FFI (foreign function interface) boundary more than once:

1. Android UI / Headless Runner (Kotlin)
2. `EngineActor` / `LlamaEngine` (Kotlin JNI stubs)
3. `crates/pipette-android` (Rust JNI crate)
4. `crates/pipette-ops` (Rust support crate)
5. `android/native/benchmarks.rs` (Rust benchmark execution kernel)
6. `android/native/llama_shim.cpp` (C-ABI shim layer)
7. `vendor/llama.cpp` + `ggml` (C++ inference engine)

These layers add cost and complexity:
- Typed Kotlin benchmark definitions are serialized into flat JSON strings to cross JNI.
- Rust parses the JSON again into a loosely typed `BenchmarkDef` struct with 10 optional fields.
- Calls cross threads between the Android JVM actor and Rust.

The `kotlinEngine` flavor has fewer layers:
1. Android UI / Headless Runner (Kotlin)
2. `EngineActor` (Kotlin actor that calls `LlamaBenchmark`)
3. `LlamaBenchmark.kt` (Kotlin benchmark execution kernel)
4. `LlamaCpp.kt` (Kotlin model and sampler lifecycle wrapper)
5. `android/native/jni.cpp` (JNI primitives that call `llama_shim.cpp` and `vendor/llama.cpp`)
6. `vendor/llama.cpp` + `ggml` (C++ inference engine)

The readiness gate is the same in both flavors: `Readiness.kt` on Android
platform APIs (see [Section 6](#6-readiness-simplification-track-a)).

### Symbol and Layer Mapping

| Component | `rustEngine` | `kotlinEngine` |
|---|---|---|
| Model Handle | `crates/pipette-android/src/llama.rs:ModelHandle` | `ai.liquid.pipette.llama.LlamaModel` |
| Low-Level JNI | `crates/pipette-android/src/lib.rs` | `android/native/jni.cpp` + `LlamaCpp.kt` (`NativeLib.kt` for the build id) |
| Native Shim | `android/native/llama_shim.cpp` | `android/native/llama_shim.cpp` (shared) |
| Variant Loader | `android/native/native_loader.cpp` | `android/native/native_loader.cpp` (shared) |
| Crash Capture | `android/native/crash_reporter.cpp`, through `lib.rs` | `android/native/crash_reporter.cpp`, through `jni_crash_reporter.cpp`; `BenchmarkCrashReporter.kt` in both |
| Context Check | `android/native/benchmarks.rs:check_ctx_size` | `ai.liquid.pipette.llama.LlamaBenchmark.checkCtxSize` |
| Cell Measurement | `android/native/benchmarks.rs:measure_cell` | `ai.liquid.pipette.llama.LlamaBenchmark.measureCell` |
| Prefill Throughput | `android/native/benchmarks.rs:run_prefill_throughput_impl` | `ai.liquid.pipette.llama.LlamaBenchmark.runPrefillThroughputImpl` |
| Decode Throughput | `android/native/benchmarks.rs:run_decode_throughput_impl` | `ai.liquid.pipette.llama.LlamaBenchmark.runDecodeThroughputImpl` |
| End to End Latency | `android/native/benchmarks.rs:run_end_to_end_latency_impl` | `ai.liquid.pipette.llama.LlamaBenchmark.runEndToEndLatencyImpl` |
| Max Memory Usage | `android/native/benchmarks.rs:run_max_memory_usage` | `ai.liquid.pipette.llama.LlamaBenchmark.runMaxMemoryUsage` |
| Vision Throughput | `android/native/benchmarks.rs:run_vl_throughput` | `ai.liquid.pipette.llama.LlamaBenchmark.runVlThroughputImpl` |
| Eval Loop | `android/native/benchmarks.rs:run_eval` | `ai.liquid.pipette.llama.LlamaBenchmark.runEvalImpl` |
| Prompt Seed Growth | `crates/pipette-ops/src/prompt_seed.rs` | `ai.liquid.pipette.ops.PromptSeed` |
| Rep Observation | `crates/pipette-ops/src/measurement/mod.rs` | `ai.liquid.pipette.ops.Measurement` |
| Thermal Series | `crates/pipette-ops/src/measurement/mod.rs` | `ai.liquid.pipette.ops.ThermalSeries` |

---

## 2. Invariants Checklist

### Cross-client invariants

Every Pipette client (CLI, iOS, and Android) keeps the invariants below. A
violation makes results from different clients not comparable.

| Invariant | Specification | Rationale and Enforcement |
|---|---|---|
| Repetition Count | `REPS = 5` | Exactly 5 measured repetitions per cell. Never parameterized or shortened. |
| Warm-Up Rep | 1 untimed warm-up at measured shape | Runs before the 5 measured reps. Exercises KV allocation, caches, and thermal ramp without gating or timing. |
| Rep Gate | Gate before every measured rep | Calls the readiness gate (`LlamaBenchmark.readinessGate` in Kotlin: thermal headroom, with thermal status as the fallback) before each measured rep. Two measured reps never run back to back without the gate. |
| Thermal Pairing | `(before, after)` sensor pairing per rep | Captures thermal status and headroom immediately before and immediately after each measured rep to compute deltas. |
| Tokenization Window | Inside timed window for e2e only | `end_to_end_latency` measures in-window tokenization: tokenization runs inside the timed rep. Prefill and decode keep tokenization outside the measured window. |
| E2E Prompt Guard | Per-rep length assertion in `run_end_to_end_latency_impl` / `runEndToEndLatencyImpl` | Compares the in-window token count with the expected prompt length, and fails the cell if `rep.value != expected_prompt_tokens`. |
| Fixed Decode | Fixed-count decode that ignores EOG | `decode_throughput` and `end_to_end_latency` decode exactly `decode_tokens` and ignore EOG (end-of-generation) tokens, so throughput has no early stop. |
| Memory Bracketing | Brackets model load; refused on loaded model | `max_memory_usage` requires loading a fresh model to measure peak footprint delta. Refuses execution on an already-loaded model handle. |

### Android JNI rules (`kotlinEngine`)

| Rule | Specification | Rationale and Enforcement |
|---|---|---|
| Timed Decode Loop | Native execution without per-token JNI | `decodeNGreedyIgnoreEog` runs in C++ (`android/native/jni.cpp`), so the timed decode loop makes no JNI transition per token. |
| JNI Array Pinning | Region copies, not pointer pinning | Uses `GetIntArrayRegion` and `SetIntArrayRegion` for token buffers, so the JVM garbage collector heap is never pinned. |
| RAII Resource Safety | Cleanup on every scope exit | Wraps JNI strings and native handles in RAII (resource acquisition is initialization) guards, so `ReleaseStringUTFChars` and native frees run on every normal and exceptional exit. |

---

## 3. Defaults and the 24 `unwrap_or*` Sites Audit

The Rust kernel in `android/native/benchmarks.rs` takes an untyped `BenchmarkDef`
struct whose parameters are optional `Option<T>` fields, with an `unwrap_or*`
fallback at each use. The port audited 24 of these fallback sites. (The file also
has a saturating `u32` conversion and a test-only `unwrap_or_default`, which are
not parameter defaults.)

The Kotlin kernel uses its own `LlamaBenchmark.BenchmarkDef` data class. Its
parameters are also nullable (`Int?`), and the kernel applies the fallback with
the Elvis operator (`?:`). The Kotlin column shows the expression at each site.

### Audit Table of the 24 Sites

| # | Rust function | Rust expression | Rust fallback | Kotlin expression | Notes |
|---|---|---|---|---|---|
| 1 | `required_ctx_size` | `def.parameter_prefill_tokens.unwrap_or(0)` | 0 | `def.parameterPrefillTokens ?: 512` | The divergence site. See the analysis below. |
| 2 | `required_ctx_size` | `def.parameter_decode_tokens.unwrap_or(0)` | 0 | `def.parameterDecodeTokens ?: 128` | Kotlin uses the runner default. |
| 3 | `required_ctx_size` | `def.parameter_max_tokens.unwrap_or(0)` | 0 | `def.parameterMaxTokens ?: 256` | Kotlin uses the runner default. |
| 4 | `required_ctx_size` | `def.parameter_image_width.unwrap_or(0)` | 0 | `def.parameterImageWidth ?: 0` | VL patch calculation. |
| 5 | `required_ctx_size` | `def.parameter_image_height.unwrap_or(0)` | 0 | `def.parameterImageHeight ?: 0` | VL patch calculation. |
| 6 | `required_ctx_size` | `def.parameter_text_tokens.unwrap_or(0)` | 0 | `def.parameterTextTokens ?: 0` | VL text tokens. |
| 7 | `check_ctx_size` | `def.parameter_prefill_tokens.unwrap_or(0)` | 0 | `def.parameterPrefillTokens ?: 512` | Error message. |
| 8 | `check_ctx_size` | `def.parameter_decode_tokens.unwrap_or(0)` | 0 | `def.parameterDecodeTokens ?: 128` | Error message. |
| 9 | `check_ctx_size` | `def.parameter_max_tokens.unwrap_or(0)` | 0 | `def.parameterMaxTokens ?: 256` | Error message. |
| 10 | `to_pipette_error` | `err.downcast::<PipetteError>().unwrap_or_else(...)` | error map | Exception handling (`Throwable`) | Recovers a typed `PipetteError`. |
| 11 | `run_prefill_throughput_impl` | `def.parameter_prefill_tokens.unwrap_or(512)` | 512 | `def.parameterPrefillTokens ?: 512` | |
| 12 | `run_decode_throughput_impl` | `def.parameter_prefill_tokens.unwrap_or(512)` | 512 | `def.parameterPrefillTokens ?: 512` | |
| 13 | `run_decode_throughput_impl` | `def.parameter_decode_tokens.unwrap_or(128)` | 128 | `def.parameterDecodeTokens ?: 128` | |
| 14 | `run_end_to_end_latency_impl` | `def.parameter_prefill_tokens.unwrap_or(512)` | 512 | `def.parameterPrefillTokens ?: 512` | |
| 15 | `run_end_to_end_latency_impl` | `def.parameter_decode_tokens.unwrap_or(128)` | 128 | `def.parameterDecodeTokens ?: 128` | |
| 16 | `run_max_memory_usage` | `def.parameter_prefill_tokens.unwrap_or(512)` | 512 | `def.parameterPrefillTokens ?: 512` | |
| 17 | `run_vl_throughput` | `def.parameter_num_images.unwrap_or(1).max(1)` | 1 | `max(def.parameterNumImages ?: 1, 1)` | Mobile runs a single image only; more than 1 is refused. |
| 18 | `run_vl_throughput` | `prompt_tokens_observed.unwrap_or(0)` | 0 | `promptTokensObserved ?: 0` | Prompt tokens observed in the mtmd warm-up. |
| 19 | `run_vl_throughput` | `image_tokens_observed.unwrap_or(0)` | 0 | `imageTokensObserved ?: 0` | Image tokens observed in the mtmd warm-up. |
| 20 | `run_eval` | `def.parameter_max_tokens.unwrap_or(256)` | 256 | `def.parameterMaxTokens ?: 256` | |
| 21 | `run_eval` | `mcq_choices.map(...).unwrap_or_default()` | `""` | `def.parameterMcqChoices` (`List<String>?`) | Serialized only if present. |
| 22 | `run_eval` | `sample.get("id")...unwrap_or("")` | `""` | `sampleObj.optString("id", "")` | Sample id. |
| 23 | `run_eval` | `sample.get("messages")...unwrap_or(Value::Array(vec![]))` | `[]` | `sampleObj.optJSONArray("messages") ?: JSONArray()` | Messages array. |
| 24 | `run_eval` | `serde_json::to_string(&messages).unwrap_or_default()` | `""` | `messages.toString()` | Messages serialization. |

In the Kotlin VL runner, a missing image width, image height, or text-token count
fails the cell; only `requiredCtxSize` falls back to 0 for them.

### The `parameter_prefill_tokens` Divergence Analysis

The Rust kernel has a divergence between `required_ctx_size` and the runners:
- `required_ctx_size` uses
  `let prefill = def.parameter_prefill_tokens.unwrap_or(0);`
- `run_prefill_throughput_impl`, `run_decode_throughput_impl`,
  `run_end_to_end_latency_impl`, and `run_max_memory_usage` use
  `let prefill_tokens = def.parameter_prefill_tokens.unwrap_or(512);`

**Effect.** If a definition omits `parameter_prefill_tokens`, `required_ctx_size`
assumes that prefill needs 0 tokens, but the runner then runs with 512 tokens. If
`context_size < 512`, `check_ctx_size` succeeds and the run then fails with
context exhaustion.

In practice this rarely occurs, because the server catalog and
`BenchmarkDefinition.toJson()` always emit `parameter_prefill_tokens`. The Kotlin
kernel reads the definition directly, so the case is reachable there.

**Kotlin resolution.** `requiredCtxSize` and `checkCtxSize` use the same
fallbacks as the runners (`?: 512` prefill, `?: 128` decode, `?: 256` max
tokens), so the context check matches the values that the benchmark runs with.

---

## 4. Header Rule

**Transcribe from code, never from kernel comments.**

Comments in the Rust kernel can describe intended behavior, earlier assumptions,
or desktop-only behavior that differs from Android execution. Each Kotlin
implementation, constant, buffer allocation, and loop condition comes from the
code in `android/native/benchmarks.rs`, `crates/pipette-android/src/llama.rs`, and
`android/native/llama_shim.cpp`.

---

## 5. Named Off-Ramp Product Owner

If a Phase 4 A/B comparison shows a performance or behavior difference between
`rustEngine` and `kotlinEngine` that profiling and alignment cannot resolve, the
product owner who decides is:

**David Berrios (`@dberrios`), Pipette Engineering Lead.**

### Off-Ramp Decision Protocol
1. **Delta <= Noise Floor:** The difference is within procedure noise. Go to Phase 5 (make `kotlinEngine` the default and soak).
2. **Delta > Noise Floor (Performance Improvement):** Verify correctness with an eval run. If it is correct, accept the new baseline.
3. **Delta > Noise Floor (Performance Regression or Semantic Drift):**
   - Step 1: Profile with Simpleperf (the Android CPU profiler) and CheckJNI (the JNI checking mode) to find hot-path differences (for example JNI transition frequency, memory allocations, thread pinning).
   - Step 2: Hold a formal review with the product owner (`@dberrios`).
   - Step 3: The product owner chooses one of:
     - **Fix:** Find and remove the source of the overhead.
     - **Re-baseline:** If the delta comes from a deliberate fix (for example correct UTF-8 rune handling or non-locking memory sampling), re-baseline the fleet metrics.
     - **Abandon:** Set the release workflow default back to `rustEngine`, and keep the Rust backend as the shipping default.

---

## 6. Readiness Simplification (Track A)

Under the Android SELinux (Security-Enhanced Linux) sandbox policy, an app
process cannot read the Linux sysfs and thermal nodes that the native readiness
probe in `crates/pipette-readiness` used.

An audit on reference hardware found:
- Both a launcher start and an `adb shell am start` start produced `unavailable:` from `nativeProbe`.
- The probe then stayed latched at `nativeUnavailable = true`.

Track A therefore moves `Readiness.kt` onto Android platform APIs only:
- `PowerManager.getThermalHeadroom()`, with sample caching timed by the monotonic `SystemClock.elapsedRealtime()`.
- `PowerManager.getCurrentThermalStatus()` as the fallback.
- `crates/pipette-android/src/lib.rs` has no `nativeWaitUntilReady`, and the Android build does not depend on `crates/pipette-readiness`.
- The `ReadinessOutcome` contract and the `PIPETTE_READINESS_SKIP_THERMAL` debugging toggle stay the same.

Track A validation:
- Unit tests (`ReadinessTest.kt`) pass in both the `rustEngine` and `kotlinEngine` test tasks.
- Verified on a Google Pixel 10 Pro Fold: headless runs with `skip_thermal=false` gate reps correctly and report accurate `benchmark_flags.readiness` (`max_wait_secs` and `skip_thermal`) in exported payloads, with no JNI crashes and no SELinux errors.

---

## 7. Noise Floor and A/B Benchmarking Protocol

Phase 0e measured a baseline noise floor, to tell real performance differences
from noise in the test setup:
- **Reference Device:** Google Pixel 10 Pro Fold, Tensor G5, 16 GB RAM, Android 17 (Build 15828068). It is connected over USB, with an active thermoelectric (Peltier) cooler (TEC) attached, fixed screen brightness, "stay awake" on power enabled, and cpuset `/top-app`.
- **Procedure:** 10 alternating runs (5 passes of Arm A `rustEngine` and 5 passes of Arm B `rustEngine`) of identical workloads, with `scripts/android-engine-ab.sh --null-ab`.
- **Model:** `LiquidAI/LFM2.5-350M-GGUF` (`LFM2.5-350M-Q4_0.gguf`), 6 threads, CPU backend `arm_fma,dotprod,fp16_va,matmul_int8,neon,repack,sve`.
- **Measured Data (Phase 0e):**

| Benchmark | Unit | Arm A Mean +/- StdDev | Arm A Spread | Arm B Mean +/- StdDev | Arm B Spread | Between-Arm Delta | Detection Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `decode_throughput_512_128` | tok/s | 96.71 +/- 7.10 | 7.35% | 96.80 +/- 8.73 | 9.02% | **+0.09%** | +/- 1.0% |
| `end_to_end_latency_512_128` | ms | 2164.83 +/- 365.38 | 16.88% | 2065.43 +/- 140.24 | 6.79% | **-4.59%** | +/- 5.0% |
| `max_memory_usage_512` | bytes | 330.77 +/- 1.95 MB | 0.59% | 330.72 +/- 2.84 MB | 0.86% | **-0.02%** | +/- 0.2% |
| `prefill_throughput_512` | tok/s | 719.98 +/- 102.71 | 14.27% | 726.48 +/- 38.47 | 5.30% | **+0.90%** | +/- 1.5% |

The memory row is in MB (10^6 bytes); Section 8 gives memory in bytes.

- **Floor Determination:** For identical `rustEngine` binaries with active TEC cooling, the between-arm delta is under 1% for decode throughput, peak memory, and prefill throughput, and under 5% for end-to-end latency. The table gives two measures for each benchmark: the between-arm detection threshold, and the per-arm spread (the larger of the two arms is the noise floor spread that Section 8 uses).

---

## 8. Phase 4 A/B Comparison Results

Phase 4 used the same conditions as Phase 0e (Google Pixel 10 Pro Fold, Android 17 Build 15828068, active TEC cooling, airplane mode, fixed brightness, `/top-app` cpuset). It ran 5 alternating passes of Arm A (`rustEngine`) and Arm B (`kotlinEngine`) with `scripts/android-engine-ab.sh`.

### Measured Data (Phase 4)

| Benchmark | Unit | Arm A (`rustEngine`) Mean +/- StdDev | Arm A Spread | Arm B (`kotlinEngine`) Mean +/- StdDev | Arm B Spread | Between-Arm Delta | Phase 0e Noise Floor Spread | Evaluation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `decode_throughput_512_128` | tok/s | 89.30 +/- 17.21 | 19.28% | 93.81 +/- 3.58 | 3.82% | **+5.06%** | 9.02% | Within floor; tighter spread |
| `end_to_end_latency_512_128` | ms | 2027.45 +/- 135.31 | 6.67% | 2079.49 +/- 92.32 | 4.44% | **+2.57%** | 16.88% | Within floor (52 ms delta) |
| `max_memory_usage_512` | bytes | 329,512,550 +/- 244,915 | 0.07% | 331,420,467 +/- 1,864,997 | 0.56% | **+0.58%** | 0.86% | Within floor (1.8 MB delta) |
| `prefill_throughput_512` | tok/s | 682.70 +/- 18.52 | 2.71% | 705.35 +/- 82.43 | 11.69% | **+3.32%** | 14.27% | Within floor; slight speedup |

### Finding and Off-Ramp Decision

The gate compared each between-arm delta with the Phase 0e noise floor spread
(the larger per-arm spread in Section 7). Every delta is within that spread, so
under the Section 5 protocol:
- Delta <= Noise Floor: the difference is within procedure noise.
- Gate passed: go to Phase 5 (make `kotlinEngine` the default flavor and soak).

Three deltas are larger than the Section 7 between-arm detection thresholds:
decode (+5.06% against +/- 1.0%), memory (+0.58% against +/- 0.2%), and prefill
(+3.32% against +/- 1.5%). The gate did not use those thresholds.
