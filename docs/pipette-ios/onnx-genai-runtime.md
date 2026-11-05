# ONNX Runtime GenAI runtime

The **ONNX Runtime GenAI** runtime benchmarks ONNX GenAI packages. It is one of the
four iOS engines, and like llama.cpp and MLX it runs in the app's process.
`RunCell.dispatchRun` routes `.onnxGenaiIosPipette` to `OnnxGenaiExecute`, which
plugs into the harness that every engine shares
([architecture.md](architecture.md#inference-engines)). Source: `Runtimes/OnnxGenai/`.

It is **fully exposed**: the catalog's ONNX rows are visible, the New Job picker
offers the engine from those rows (`RunCell.prepare` downloads the package before
the cell runs), and the client advertises `runtime:onnx_genai` so a plan can target
it. Drive it headlessly with `headlessrun runtime=onnx …`.

The engine runs on the **CPU**. There is no Metal path and no GPU allocator, which
is why this runtime's memory breadcrumbs report the process footprint alone where
MLX's also report its allocator's figures.

The host engine's measurement and validation are in
[onnx-genai-measurement.md](../onnx-genai-measurement.md). This page covers what
differs on iOS.

## Support matrix

The policy is the one `afm-runtime.md` states: **same metric, or raise an error**;
never emit an approximation.

| Benchmark | ONNX GenAI | Notes |
|-----------|-----------|-------|
| `prefill_throughput` | ✅ | `prefill_time_ms` = time to first token |
| `decode_throughput` | ✅ | `decode_time_ms` = mean per-token interval × the cell's decode count |
| `end_to_end_latency` | ✅ | `total_time_ms` = TTFT + TPOT × (n − 1), reconstructed |
| `max_memory_usage` | ✅ | in-process, so `Runtimes/ProcessMemory.swift` sees the engine's arenas; mapped weights counted too |
| `eval` | ❌ error | `OGATokenizer` applies no chat template (see below) |
| `vl_throughput`, `vl_max_memory`, `vl_spec_decode`, `asr_throughput`, `asr_max_memory` | ❌ error | plan-types gives these kinds no ONNX variant |

The timing cells generate one sequence at a time, so they refuse a batched entry.

`max_memory_usage` is where this runtime differs from Apple Foundation. AFM cannot
report a peak: its model runs in a system process, so the in-process footprint probe
measures the app and not the model. This engine runs *in this process*, so the same
probe measures the thing being benchmarked.

It counts one thing the footprint alone would miss. ORT maps a package's
external-data file and reads the 4-bit scales straight from the mapping, so they are
resident but file-backed, and `phys_footprint` does not bill clean file-backed pages:
on an iPhone 17 Pro, 466 MiB of Phi-3 int4 sat outside it. The ONNX cell therefore polls
the footprint plus the file-backed pages that became resident after the bracket opened
(`OnnxGenaiRuntime.footprintWithMappedWeights`). llama.cpp avoids the question by
loading its weights anonymously.

## Why `eval` errors

Every other runtime applies the model's own chat template per sample:
`MlxExecute` through `Tokenizer.applyChatTemplate`, llama.cpp through
`llama_model_chat_template`. At the pinned commit, `OGATokenizer` exposes no
chat-template method: only `encode:`, `decode:length:` and the special-token
getters (`src/objectivec/include/ort_genai_objc.h`).

Borrowing a Jinja renderer would work mechanically, since swift-transformers is
already a dependency and Phi-3's package does carry a `tokenizer.json`. But it would
mean templating with one tokenizer and generating with another. The scored prompt would
then be one no other runtime ran, which is exactly the approximation the policy
above refuses. So `eval` raises an error, and this is the first thing to revisit on
a version bump.

## Measurement

The engine runs in process, so there is no server and no client clock: each cell
times the generator calls directly (`OnnxGenaiRuntime.generate`).

**Fixed generation length.** As on the host, `min_length` and `max_length` are both
`len(prompt_ids) + decode_tokens`, which is this engine's `ignore_eos`.

**TTFT brackets `appendTokens` *and* the first `generateNextToken`.** The runtime
may do the prefill in either, so timing `appendTokens` alone would report a prefill
of nearly nothing and charge the whole prompt to the decode. The engine offers no
prefill-only entry point, so every cell runs one generation and reads a different
interval off it.

**TPOT is the remaining interval divided by `n − 1`**, and is absent when only one
token was produced.

**End to end is reconstructed** as `ttft + tpot × (n − 1)`, never process wall
clock, which would include the tokenize and the generator construction. The host
instead times one streamed request from send to finish.

**The prompt comes from the shared corpus.** `PromptSeed.buildPromptText` builds it
and this model's own tokenizer encodes it, with no chat template. A prefill number is
comparable with llama.cpp's and MLX's only if all three ran the same text through
their own tokenizers.

## Package validation

`ResolvedOnnxGenaiModel` checks a bound directory before the engine sees it: a
`genai_config.json` is present, the decoder graph at `model.decoder.filename` is on
disk inside the package, the tokenizer assets ORT Extensions needs are there, and every
external-data file the graph references exists. The reason for the sidecar check is
the host's ([Model validation](../onnx-genai-measurement.md#model-validation)); the
iOS check differs in three ways:

- It parses the graph's protobuf and reads every `external_data` location, rather than
  scanning bytes for one sibling name.
- It has no provider check: this runtime has the CPU provider alone.
- A missing `genai_config.json` is refused by name, because most `LiquidAI/*-ONNX`
  repos are transformers.js exports: real `.onnx` graphs, no `genai_config.json`,
  and nothing this engine can load.

## Build pipeline

Two git submodules, both built from source by `ios/build-onnx-genai.sh`, which runs
as the "Build ONNX Runtime GenAI" phase ahead of `Sources`:

| Submodule | Pin | Role |
|---|---|---|
| `vendor/onnxruntime` | `v1.31.1` | the runtime that executes the graph |
| `vendor/onnxruntime-genai` | `533de8cc`, a commit on `main` whose `VERSION_INFO` reads `0.18.0-dev` | the generation loop and the Objective-C surface the app calls |

Two trees because onnxruntime-genai does not vendor the runtime it links against.
The script runs six steps:

1. Build onnxruntime for the slice.
2. Assemble an `ort_home` from it: `Headers/` plus a generated
   `onnxruntime.xcframework`, which is what genai's iOS branch requires.
3. Copy the genai submodule and apply the carried patch series.
4. Build genai against that `ort_home`.
5. Compile the Objective-C surface.
6. Combine everything into `libonnxruntime-genai.a` and `libonnxruntime.a` under
   `Generated/OnnxGenai/`. Swift reaches the engine through the bridging header, the
   same route `PipetteThermal.m`'s symbols take.

Both pins reach the submitted `runtime_descriptor`, as
`OnnxGenaiStack { onnxruntime, onnxruntime_genai }`. Both affect what a number
means, so neither alone identifies a build. `ios/gen-onnx-genai-build-info.sh`
writes `OnnxGenaiBuildInfo.swift` and `.json` from the submodule checkouts; both are
committed and CI fails if they are stale.

`vendor/onnxruntime.tag` records the onnxruntime release. The genai pin has no
release tag, so it has no file, and the descriptor names it by its commit.

The CI checkout clones each submodule without its tags, so for a pin that is a
release, the generator reads the tag from `vendor/<submodule>.tag`. The file holds
`<tag> <full-sha>`, the same form as `vendor/llama.cpp.tag`. If the sha does not
match the submodule commit, the generator ignores the file and reports the build
as untagged.

`runtime_flags` is `nil`: plan-types declares no `RuntimeFlags` variant for this
runtime on any benchmark
([pipette-mgmt crates/pipette-plan-types/src/cell/runtime_flags.rs](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/runtime_flags.rs)),
which is also why an authored entry for an ONNX cell is refused rather than dropped.
The ONNX graph fixes its own batching at export, so there is no load setting to carry.

### Three things the build needs that are not defaults

- **`ENABLE_TELEMETRY=OFF`.** `build.py` defaults it on, which links Microsoft's 1DS
  client and drags in UIKit and Network. Pipette reports its own measurements to its
  own warehouse.
- **A static library.** Upstream declares the engine `SHARED` unconditionally;
  `ios/patches/onnx-genai/002-static-library.patch` changes that, so the app stays a
  single binary with no framework to embed or codesign.
- **`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`.** CMake 4 dropped compatibility with
  `cmake_minimum_required` below 3.5, which psimd (an XNNPACK dependency) still
  declares.

### The carried patch series

`ios/patches/onnx-genai/` holds patches applied to a build copy of the genai
submodule, the way `ios/patches/` works for llama.cpp. The submodule itself is never
modified, and a patch that stops applying fails the build loudly.

`001-objc-layer-compile-fixes.patch` fixes four separate defects that stop
`src/objectivec` compiling for iOS: duplicate Objective-C selectors, an
undeclared identifier, a `std::span`/`std::vector` mismatch against the C++ core,
and three `NS_SWIFT_NAME` attributes missing a parameter. Upstream CI does not
appear to build this layer. Retire the patch on a version bump that fixes them.

`ort_genai_objc.h` also uses `NS_ENUM` and `NSInteger` without importing anything
that defines them. It assumes a CocoaPods umbrella header has already pulled
Foundation in. The bridging header imports Foundation first rather than patching
that.

### Bumping the pins

1. Move each submodule to its new pin: a release tag, or a commit on a release
   branch or `main`. Keep the two in step: pin onnxruntime to the version the
   genai wheel requires on PyPI, which is the pair the desktop catalog resolves
   (`onnxruntime==1.31.1` beside genai 0.18.0 in
   `crates/pipette-artifacts-catalog/bundled-catalog/onnx_genai/catalog.toml`).
   Do not use `ort_build_version` in genai's `.pipelines/packaging.yml`: it is the
   oldest ONNX Runtime the binaries are built against (`1.26.0` at the current pin).
2. For a pin that is a release tag, write `<tag> <full-sha>` into
   `vendor/<submodule>.tag`. For a pin with no tag, delete the file.
3. Run `./ios/build-onnx-genai.sh sim`. If a patch no longer applies, check whether
   upstream fixed the defect. If so, drop that hunk; if not, rebase it.
4. Commit the regenerated `OnnxGenaiBuildInfo.swift` and `OnnxGenaiBuildInfo.json`.
5. Re-check `eval`: if `OGATokenizer` gains a chat-template method, the support
   matrix above changes.

The script keys a `.slice` stamp on the SDK, both submodule commits, the patch series
and the script itself, so an unchanged tree rebuilds nothing and a bumped pin
rebuilds everything.

## Known limitations

**No mid-file resume for the sidecar.** Phi-3's external-data file is 2.72 GB in one
piece. A user on cellular who backgrounds the app loses that file's progress; the
files that completed are kept and the transfer resumes at whole-file granularity.

**The CPU execution provider only.** CoreML and XNNPACK are compiled in but the
engine is not configured to use them. `OnnxGenaiRuntime.loadModel` goes through
`OGAConfig` rather than `OGAModel(path:)` so that setting provider options is a line
there rather than a restructure.

## Device validation

Measured on an iPhone 16 Pro Max (`iPhone17,2`), against the
catalog's Phi-3 row: `microsoft/Phi-3-mini-4k-instruct-onnx`,
`cpu_and_mobile/cpu-int4-rtn-block-32`, commit `5f5f794c`, 2.73 GB over 10 files.

| Cell | Figure | Derived |
|---|---|---|
| `prefill_throughput_512` | `prefill_time_ms` 22304.3 ± 1737.0 | 23.0 tok/s |
| `decode_throughput_512_100` | `decode_time_ms` 33124.5 ± 55.1 | 3.0 tok/s, TPOT 331 ms |
| `end_to_end_latency_512_256` | `total_time_ms` 111993.7 ± 182.6 | |
| `max_memory_usage_512` | `max_host_bytes` 2 561 658 968 | 2.56 GB, footprint alone; GPU and NPU absent |

With the mapped weights counted, the same memory cell on an iPhone 17 Pro reads
3 051 738 120 (2910 MiB), 466 MiB above the footprint alone.

The end-to-end figure is the one to read against the other two. `ttft + tpot × 255`
over the numbers above is 106 760 ms, and the measured 111 994 ms is 4.7% above
that. The direction is the expected one: TPOT is measured over 100 tokens and this
cell decodes 256, so its later tokens attend over a longer cache. A number *below*
the reconstruction would be the one to investigate.

`max_host_bytes` reports and the two accelerator figures do not, which is the
matrix above holding on real hardware: the engine runs in this process on the CPU,
so `Runtimes/ProcessMemory.swift` measures the model, and there is no allocator to
ask for a GPU figure.

These are not compared against host numbers: the host figures for the same package
and cells are not in this repository.

The storage path was exercised at the same time:

- the transfer froze to one commit before any byte moved, staged under an attempt
  id, published atomically, and released its byte budget;
- killing the app through the 2.72 GB sidecar left **no** entry in the store and
  **no** reservation: the next launch reconciled the dead transfer's grant away,
  and `storage status` reported the one model that was there before;
- GGUF and MLX rows still download, publish and release alongside it.
