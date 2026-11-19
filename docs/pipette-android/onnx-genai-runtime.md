# ONNX Runtime GenAI runtime

The **ONNX Runtime GenAI** engine benchmarks ONNX GenAI packages in the Android app's
`:benchmark` process, beside llama.cpp, in the `kotlinEngine` flavor. `EngineActor`
routes a model path that is a package (a directory holding a `genai_config.json`) to
`OnnxGenaiBenchmark`, and everything else to `LlamaBenchmark`
([architecture.md](architecture.md#engine-flavors)). Source:
`app/src/kotlinEngine/java/ai/liquid/pipette/onnx/` and `android/native/onnx_genai_jni.cpp`.

It is **exposed to a run, and by default its results stay on the device**. A run
can name a package and the app fetches it: headlessly with
`model=onnx-genai://repo=<org/name>[&prefix=<dir>][&rev=<ref>]`, and from the New Job
picker, which offers the catalog's ONNX rows. The last step waits on plan-types: a
result is submitted under the runtime `onnx_genai_apk_pipette`, which plan-types
does not define yet. A build says whether to submit anyway
(`pipette.onnxGenaiSubmit`, off by default); without it an ONNX cell's result is
written to its job as any other and never uploaded, and the client advertises no
`runtime:onnx_genai` capability ([Known limitations](#known-limitations)).

The engine runs on the **CPU**, with no execution-provider options set.

The iOS engine's page is [../pipette-ios/onnx-genai-runtime.md](../pipette-ios/onnx-genai-runtime.md),
and the host engine's measurement is in
[../onnx-genai-measurement.md](../onnx-genai-measurement.md). This page covers what
differs on Android.

## Support matrix

The policy is the iOS engine's: **same metric, or raise an error**; never emit an
approximation.

| Benchmark | ONNX GenAI | Notes |
|-----------|-----------|-------|
| `prefill_throughput` | ✅ | `prefill_time_ms` = time to first token of a one-token generation |
| `decode_throughput` | ✅ | `decode_time_ms` = mean per-token interval × the cell's decode count |
| `end_to_end_latency` | ✅ | `total_time_ms` = TTFT + TPOT × (n − 1), reconstructed |
| `max_memory_usage` | ✅ | `max_ram_bytes`, the RSS rise over a fresh load, one prefill and one decode |
| `eval` | ❌ error | the tokenizer applies no chat template, as on iOS |
| `vl_throughput`, `vlm_eval` and every other kind | ❌ error | no ONNX variant in plan-types |

The timing cells generate one sequence at a time, so they refuse a batched entry.
The `rustEngine` flavor does not have this engine.

## Measurement

The definitions are the iOS engine's, ported from `OnnxGenaiRuntime+Generate.swift`,
so an Android number and an iOS number measure the same thing. The engine offers no
prefill-only entry point, so every timing cell runs one generation and reads a
different interval off it (`GenerationTiming`).

- **Fixed generation length.** `min_length` and `max_length` are both
  `len(prompt_ids) + decode_tokens`, which is this engine's `ignore_eos`. The kernel
  refuses a generation whose length is not the cell's.
- **TTFT brackets `AppendTokens` and the first `GenerateNextToken`**, since the
  runtime may do the prefill in either.
- **TPOT is the remaining interval divided by `n − 1`**, absent for one token.
- **End to end is reconstructed** as `ttft + tpot × (n − 1)`, never wall clock.
- **The prompt comes from the shared corpus** (`PromptSeed`) under the package's own
  tokenizer, with no chat template. It is cut to the cell's prefill count, as
  `LlamaBenchmark.seedPromptTokens` cuts, and refused when it falls short.

Each cell goes through the harness llama.cpp's cells use (`LlamaBenchmark.measureCell`):
one warm-up generation at the measured shape, then five repetitions, each behind the
readiness gate and paired with its thermal readings
([engine-port.md](engine-port.md#2-invariants-checklist)).

### Where Android differs from iOS

**The timed generation is one native call.** The whole generation, and the three
clock readings around it, run in `nativeGenerate`, so the timed loop makes no JNI
call per token (the Android JNI rule in engine-port.md). That is why the app calls
genai's C API through its own JNI file and does not use genai's Java bindings, whose
`generateNextToken` is a JNI call each time.

**No C++ exception reaches JNI.** The C API returns an `OgaResult` from every call;
the JNI file turns an error into an `IllegalStateException`.

**The memory cell needs no mapped-weights term.** iOS adds the file-backed pages to
its footprint, because `phys_footprint` does not bill them. Android's cell polls RSS,
which counts resident file-backed pages already: on an emulator, a generation over
Phi-3 int4 left 1 828 MiB anonymous and 467 MiB file-backed in RSS, the same mapped
part iOS measures at 466 MiB.

**A device short of memory under-reports the peak**, as it does for llama.cpp's cell:
clean file-backed pages are dropped under pressure and the rest is swapped. On a
3.7 GB phone the cell read 2.0 GB at a 16-token prompt, where an 8 GB emulator read
2.4 GB at 64 tokens and the phone's own high-water mark could not pass what it had free.

## Models

A package is a directory, so it is a [`Model.OnnxGenai`](../../android/Pipette/app/src/main/java/ai/liquid/pipette/ModelRef.kt):
a repo, the subdirectory of it that holds the package, and optionally the revision it
is pinned to. It mirrors plan-types' `onnx_genai` model, and its `model_descriptor` is
that shape.

**The catalog** has the row iOS has, `microsoft/Phi-3-mini-4k-instruct-onnx` at
`cpu_and_mobile/cpu-int4-rtn-block-32`, pinned to the same commit
(`ModelTemplateCatalog.onnxDefaults`). `ci/checks/model-catalog-parity.py` compares the
two catalogs' ONNX rows on `(repo, prefix, revision)`, as it compares their GGUF rows.
**A build without the engine refuses an ONNX model before fetching it.** That is the
`rustEngine` flavor, and `kotlinEngine` built with `PIPETTE_ENABLE_ONNX_GENAI=0`
(`BuildConfig.ONNX_GENAI_ENGINE`). There the catalog row is not offered, a package
on disk is listed but not offered to a run, a headless `model=onnx-genai://…` is a
usage error, and no fetcher is wired, so nothing downloads 2.7 GB to learn that no
engine loads it.

**Fetching** is `OnnxPackageFetcher`, called by `ModelProvisioner` before the cell
runs, as a GGUF file is fetched. A package is a model only when every file is there,
so it follows the iOS store's order:

1. Freeze the repo to one commit before any byte moves: the coordinate's revision,
   or `main` as it is then.
2. Stage the files in a sibling directory named for that commit. Each file resumes
   from where a previous attempt stopped, and a staging directory of another commit
   is discarded.
3. Check each file against the size the listing declares, and an LFS file against
   its SHA-256.
4. Validate the staged directory as a package.
5. Rename it into place, with the commit recorded inside it (`.pipette-revision`).

So the models directory never holds half a package, and a transfer cut short
publishes nothing. A device without room for the missing bytes is told before the
download starts.

**A pinned coordinate is not taken at a directory's word.** A package on disk
satisfies a pin only if it was fetched at that commit; one from another commit, or
one put there by hand with no record, is refused with what to do about it. An
unpinned coordinate takes what is there.

**Validation** is `OnnxGenaiPackage.validate`, a port of iOS's
`OnnxGenaiModels.validatePackage`: a `genai_config.json` that names a decoder graph,
the graph on disk inside the package, every external-data file the graph references
(read from the graph's protobuf, node attributes and subgraphs included), and the
tokenizer's config and vocabulary. A missing `genai_config.json` is refused by name,
because most of LiquidAI's `-ONNX` repos are transformers.js exports with real `.onnx`
graphs and nothing this engine can load. Paths inside the package's metadata are
untrusted and may not leave it.

**The model store** lists a package as one model, its path the directory and its size
its files', beside the `.gguf` files (`ModelStore`). A package being fetched is not
listed. `models-delete` and the Models tab remove a package whole.

### Where this differs from a GGUF model

- **Fetched by the run, not by the Models tab.** The app's download layer
  (`DownloadCoordinator`, WorkManager) names a transfer by one `.gguf` file. The
  fetcher runs on the job's own thread, inside the foreground hold a run keeps. The
  Models tab cannot start a package download or show its progress, and a transfer
  that dies with the process resumes when the cell is run again.
- **No mid-file resume across a changed listing.** A file resumes by HTTP range
  within one commit; a new commit starts the package over.
- **No storage quota**, as for every Android model (client-parity gap 2).

## Build pipeline

The two git submodules iOS builds from, at the same pins, built from source by
`android/Pipette/build-onnx-genai-android.sh`. `build-native-android.sh` runs it
first (`PIPETTE_ENABLE_ONNX_GENAI=0` skips it and leaves the engine's JNI refusing),
then hands its output to CMake as `PIPETTE_ONNX_GENAI_DIR`.

| Submodule | Pin | Role |
|---|---|---|
| `vendor/onnxruntime` | `v1.31.1` | the runtime that executes the graph |
| `vendor/onnxruntime-genai` | `533de8cc` | the generation loop and the C API the app calls |

The script runs four steps:

1. Build onnxruntime as a shared library for `arm64-v8a`.
2. Assemble an `ort_home` in the layout of the onnxruntime AAR (`headers/` and
   `jni/<abi>/`), which is what genai's Android branch reads.
3. Build genai against that `ort_home`, with CMake directly as on iOS.
4. Publish `libonnxruntime.so`, `libonnxruntime-genai.so` and `ort_genai_c.h` under
   `target/onnx-genai-android-arm64-v8a/out/`, and fail if a load segment of either
   library is not 16 KB aligned.

`libpipette_android.so` links `libonnxruntime-genai.so`, which loads
`libonnxruntime.so` by name at run time, so `cmake/stage_android_jnilibs.cmake`
stages both beside the engine.

**Both pins come from one committed stamp**,
`ios/Pipette/Pipette/Generated/OnnxGenai/OnnxGenaiBuildInfo.json`, which
`ios/gen-onnx-genai-build-info.sh` writes from the same two submodules: a release
tag where the pin is one (`v1.31.1`), else the short commit (`533de8cc9`). The
engine reads it at build time (`OnnxGenai.onnxRuntimeVersion()`,
`OnnxGenai.genaiCommit()`) and so does the app's `BuildConfig`, which is what a
payload's `runtime_version` is written from in the main process. Not the submodules
themselves: Gradle configures before they are checked out, on a CI runner always.
The build script fails, before building anything, when the stamp does not match
the pinned commits; `OnnxGenaiBenchmarkInstrumentedTest` checks the engine and
`BuildConfig` agree.

The script keys a stamp on the NDK, the API level, both submodule commits and
itself, so an unchanged tree rebuilds nothing, and reads that stamp before it clones
either tree. CI caches the published libraries on the same inputs (`android-app` in
`build.yml`, and the release workflow), so a run at unchanged pins neither clones
nor compiles them; `uv` is installed for the run that misses. Both libraries'
licenses are in the app's attribution file (`ci/android-attribution.py`). On an Apple Silicon Mac the cold
build takes about 13.5 minutes, nearly all of it onnxruntime, and the stamped one
half a second.

### What differs from the iOS build

- **No carried patches.** iOS's two are for the Objective-C surface and a static
  library; Android has neither, and both trees build unpatched.
- **Shared libraries, not static.** The app already ships its native libraries
  extracted, and genai keeps the runtime as a separate library on Android.
- **`ENABLE_TELEMETRY=OFF` and `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`** are needed
  here for the reasons they are on iOS.
- **No page-size flag.** Both trees add `-Wl,-z,max-page-size=16384` for Android
  themselves; step 4 checks the result.
- **libc++ is linked statically into both libraries**, while the rest of the app
  uses `libc++_shared.so`. Only C crosses the boundary.

The two libraries add 24.1 MB (runtime) and 6.1 MB (engine) to the unpacked native
libraries.

## Known limitations

**Results are not submitted by default.** A result goes up under the runtime
`onnx_genai_apk_pipette`: iOS's `onnx_genai_ios_pipette` with the flavor
`android-arm64-v8` and no `private_apis`. plan-types does not define it yet, and the
planner has no capability rule for it (`os:android`); both are a pipette-mgmt
change. A collector takes the result without them, a descriptor being opaque to
it: an unchanged pipette-mgmt accepted and scored these cells from a phone. What it
files is a row whose runtime nothing there can name, and no plan can ask for the
runtime. So submission is a build switch until the name is plan-types':

```
# android/Pipette/local.properties, or PIPETTE_ONNX_GENAI_SUBMIT=1
pipette.onnxGenaiSubmit=true
```

It sets `BuildConfig.ONNX_GENAI_SUBMIT`. An ONNX cell's payload is the same either
way: `runtime_name` `onnx_genai_apk_pipette`, both pins as its `runtime_version`,
the descriptor with both source trees (`SubmissionRef.onnxGenaiRuntime`), and no
`runtime_flags` or `runtime_cpu_variant`, which describe llama.cpp. The switch
decides two things:

| | off (default) | on |
|---|---|---|
| the cell's result | kept on the device, not counted as unsent (`JobCell.isSubmittable`) | submitted as any other |
| capability flags | none for this engine | `runtime:onnx_genai`, `runtime:onnx_genai:<genai>`, `runtime:onnx_genai:onnxruntime-genai=<genai>`, `runtime:onnx_genai:onnxruntime=<ort>`, as iOS |

The flags follow submission and not the engine alone: a client that advertises a
runtime whose results it cannot file invites work it would lose. The Android
client claims no planned job today, so the flags describe the fleet and match
nothing yet. [client-parity.md](../client-parity.md) gap 11 tracks the rest.

**No package download from the Models tab**, as above.

**The CPU execution provider only**, and the thread count is the runtime's default:
the cells do not report `runtime_thread_count`.

**No physical-device numbers yet.** See below.

## Validation

All through the instrumented tests in `app/src/androidTestKotlinEngine/.../onnx/`,
which call the kernel without a readiness gate or thermal callbacks. So the figures
below show that the cells run, agree with each other and with iOS, and are not
benchmark numbers: no repetition was gated, and no device was cooled.

### A phone: Moto G40 Fusion

Snapdragon 732G, 3.7 GB of RAM, Android 12, 4 KB pages, on mains power. All nine
tests of `OnnxGenaiBenchmarkInstrumentedTest` and `OnnxGenaiCheckJniTest` pass, under
CheckJNI, against `Arm/qwen3-0-6b-onnx-genai-int4-kquantlast-emb-int4` at commit
`e56a5fe7` (495 MB, a package at the root of its repo):

| Cell | Figure | Derived |
|---|---|---|
| `prefill_throughput_512` | `prefill_time_ms` 9910.5 ± 188.8 | 51.7 tok/s |
| `decode_throughput_512_100` | `decode_time_ms` 8039.0 ± 257.7 | 12.4 tok/s, TPOT 80 ms |
| `end_to_end_latency_512_100` | `total_time_ms` 18846.3 ± 736.8 | |
| `max_memory_usage_512` | `max_ram_bytes` 894 095 360 | 853 MiB, over a 483 MB weights file |

`ttft + tpot × 99` over the first two is 17 869 ms, and the measured end to end is
5.5% above it, the direction and size of the gap iOS reports.

The fetch ran on the same phone against HuggingFace
(`OnnxPackageFetchInstrumentedTest`), pinned to a commit both times: that package in
118 s, and the catalog's Phi-3 row (10 files, 2 725 538 838 bytes) in 602 s with the
weights' SHA-256 checked. Each recorded its commit, was refused for another pin
without a download, was listed by the model store at its package size, and ran a
cell through `EngineActor`. A second run found each on disk and fetched nothing.

**Phi-3 cannot be timed on this phone.** It loads and generates, which shows the
libraries and their loading work on a physical device, but the phone swaps (9 to 11
seconds a token in a bare run) and the low-memory killer ended the app during the
third timing cell. A reference phone with 8 GB or more is still to run the catalog
row.

### An emulator: Android 14 arm64, 8 GB of RAM

All ten ONNX tests pass, under CheckJNI, against the catalog's Phi-3 row
(`cpu_and_mobile/cpu-int4-rtn-block-32`, 2.73 GB over 10 files). An emulator does not
throttle and shares its host:

| Cell | Figure |
|---|---|
| `prefill_throughput_512` | `prefill_time_ms` 15341.1 ± 5132.2 (four repetitions between 12.2 and 13.5 s, one of 24.5 s while the host was busy) |
| `decode_throughput_512_100` | `decode_time_ms` 21670.1 ± 1108.8 |
| `end_to_end_latency_512_100` | `total_time_ms` 33265.5 ± 236.9 |
| `max_memory_usage_512` | `max_ram_bytes` 2 974 384 128 |

The end-to-end figure agrees with the other two to within 3%. The memory figure is
2 837 MiB, beside the 2 910 MiB the iOS cell reads on an iPhone 17 Pro with the mapped
weights counted.

### Not validated

- The catalog's own row on a phone with the memory for it.
- A whole headless job, and the New Job picker's ONNX row: both need a registered
  device with a synced benchmark catalog. The path was run in its parts instead:
  the coordinate's parse, the fetch, the listing and the cell.
- The fetch's resume after an interruption, on a device. It is covered against a
  scripted network (`OnnxPackageFetcherTest`).
- The two workflows' ONNX steps, which have not run on a CI runner.
