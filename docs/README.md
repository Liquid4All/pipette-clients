# Documentation

Start at the [top-level README](../README.md) for an orientation to the
repository and the build entry point for each component. This directory holds
the detailed docs, and this page is their index.

Each component documents itself in `architecture.md` (structure and
boundaries), `build.md` (how to compile it) and `usage.md` (how to operate it)
as needed; other files keep descriptive names.

## Two kinds of client

- **Host-native CLI** (`pipette`) installs and invokes external runtimes:
  llama.cpp, MLX, Core AI, OpenVINO, OVMS, ONNX GenAI, TensorRT Edge-LLM, vLLM
  and SGLang. One workflow covers all of them (init → register → install a
  runtime → run → sync); see
  [pipette-cli usage](pipette-cli/usage.md).
- **Native mobile apps** (`pipette-ios`, `pipette-android`) compile their
  inference engines into the app and run inference in-process, so there is no
  separate runtime to install. iOS links four engines: llama.cpp, MLX (via the
  `mlx-swift` packages), ONNX Runtime GenAI and Apple Foundation Models (whose
  engine and weights ship with the OS). Android's default `kotlinEngine` flavor
  drives llama.cpp over Kotlin/JNI (`android/native/jni.cpp`); the Rust core is
  the non-default `rustEngine` flavor. The Apple-only engines are a standing
  reason the two apps cannot reach parity; see [client parity](client-parity.md).

## Index

### Repo-wide

- [Architecture](architecture.md): crates, dependency graph, workspace, traits
- [Client parity](client-parity.md): where the iOS and Android apps differ, which differences are deliberate, and why the gaps recur
- [Storage quota](storage-quota.md): the disk cap over downloaded artifacts, and how eviction picks
- [Third-party licensing](licensing.md): what may enter each client's dependency graph, and the gates that enforce it
- [OpenVINO IR](openvino-ir.md): IR layout, precision identification, and NPU constraints
- [OpenVINO measurement](openvino-measurement.md): timing, compile, and cache behavior
- [OVMS](ovms.md): OpenVINO Model Server cells, and how their memory is measured
- [TensorRT Edge-LLM](pipette-cli/trt-edgellm.md): the Jetson runtime, its engine build step and its VL cells
- [ONNX GenAI measurement](onnx-genai-measurement.md): provider validation, token timing, and host memory
- [Fleet perf troubleshooting](fleet-perf-troubleshooting/gmktec-evo-x2.md): diagnosing speed differences between identical benchmark boxes, per box type
- [GMKtec GPU idle residency](fleet-perf-troubleshooting/gmktec-evo-x2-gpu-idle.md): why the first request after a few idle seconds is slow on that box type, and the keepalive that removes it
- [CI checks](../ci/README.md): the convention checks, `lint.sh`, and how to add a check

### Methodology

- [Benchmark methodology](methodology/README.md): published results, the pipeline, and the goals the articles share
- [Device conditions](methodology/device-conditions.md): the readiness gate, cooling and power, and the run environment recorded with each result
- [End-to-end latency](methodology/end-to-end-latency.md)
- [Prefill throughput](methodology/prefill-throughput.md)
- [Decode throughput](methodology/decode-throughput.md)
- [Peak memory usage](methodology/peak-memory-usage.md), per platform: [Android](methodology/peak-memory-android.md) · [iOS](methodology/peak-memory-ios.md) · [Linux](methodology/peak-memory-linux.md) · [macOS](methodology/peak-memory-macos.md) · [Windows](methodology/peak-memory-windows.md)
- [Galaxy S26 Ultra memory pressure](methodology/galaxy-s26-ultra-memory-pressure.md): where the swap in the S26 Ultra rows comes from, measured
- [MacBook Neo thermal behavior](methodology/macbook-neo-thermal-behavior.md) · [MacBook Pro (M5 Max) thermal behavior](methodology/macbook-m5-thermal-behavior.md): what the macOS thermal signals mean, measured
- [Pixel 11 Pro thermal behavior](methodology/pixel-11-pro-thermal-behavior.md) · [Galaxy S26 Ultra thermal behavior](methodology/galaxy-s26-ultra-thermal-behavior.md) · [iPhone 18 Pro thermal behavior](methodology/iphone-18-pro-thermal-behavior.md): the phone thermal defaults, coolers, phantom touches, thread layouts and the skin criterion, measured
- [VL throughput and latency](methodology/vl-throughput.md) · [VL peak memory](methodology/vl-max-memory.md): one image-plus-text prompt through a vision-language model, timed and its peak memory read
- [ASR throughput and latency](methodology/asr-throughput.md) · [ASR peak memory](methodology/asr-max-memory.md): the audio counterparts of the two VL benchmarks, with their own audio-token counting
- [VL speculative decoding](methodology/vl-spec-decode.md): a regular model and a spec-decoding model each run on the same corpus, one row per run, compared outside the service
- [VL image tokens](methodology/vl-image-tokens.md) · [VL token calibration](methodology/vl-calibration.md): how image tokens are counted and cached
- [Evals](methodology/evals.md): how the client runs eval benchmarks; each eval's methodology is in pipette-mgmt: [GPQA Diamond](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/gpqa_diamond-2026.06.1.md) · [IFBench](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/ifbench-2026.06.1.md) · [IFStruct](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/ifstruct-release_v1_0.md) · [MATH-500](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/math_500-2026.06.1.md)
- [VL evals](methodology/vlm-eval.md): eval benchmarks over image and text prompts, scored outside the client like text evals
- [ASR evals](methodology/asr-eval.md): word error rate on audio clips from the entry's pinned corpus, on llama.cpp or vLLM, scored outside the client like text evals
- [Selection policies](methodology/selection-policies.md): which models, weights and runtime builds are benchmarked
- [Core AI specialization cache](methodology/coreai-specialization.md): how Core AI caches and reuses specialized model artifacts on the device
- Energy: [landscape](methodology/energy/01-landscape.md) · [device capabilities](methodology/energy/02-device-capabilities.md) · [proposal](methodology/energy/03-proposal.md)

### Host-native CLIs

- [pipette-cli usage](pipette-cli/usage.md): unified `pipette` client; start here
- [Naming models, runtimes, and flags](pipette-cli/models-and-runtimes.md): the URI and JSON notation for `--model` / `--runtime` / `--runtime-flags`, with recipes
- [llama.cpp](pipette-cli/llamacpp.md) · [MLX](pipette-cli/mlx.md) · [Core AI](pipette-cli/coreai.md) · [torch-oai](pipette-cli/torch-oai.md) · [OpenVINO](pipette-cli/openvino.md) · [ONNX GenAI](pipette-cli/onnx-genai.md): runtime backends
- [How an eval runs](pipette-cli/eval-runs.md): from a job to a scored result, the runner every engine shares, and what decides an eval's speed
- [Eval checkpoint & resume](pipette-cli/eval-checkpoint.md)
- [pipette-ctl](pipette-ctl/plan-runner.md): the operator CLI (separate binary) that drives plans across devices and calls the management server
- [File plans](pipette-ctl/job-generation.md): the guide to a scheduler-mode plan, which the management server expands into jobs when you file it as a campaign
- [Management routes](pipette-ctl/management.md): the management server's account routes as `pipette-ctl` CLI verbs and an MCP server
- [Submitter identity](pipette-ctl/submitter-identity.md): the user account the management verbs sign as, and the `auth` verbs that manage it

### Native mobile apps

- [pipette-ios](pipette-ios/architecture.md) · [build](pipette-ios/build.md) · [usage](pipette-ios/usage.md) · [model store design](pipette-ios/model-store-design.md) · [execution alignment](pipette-ios/execution-alignment.md) · [private-API builds](pipette-ios/private-thermal-release-build.md) · [memory carry-over](pipette-ios/memory-carry-over.md) · [AFM runtime](pipette-ios/afm-runtime.md) · [ONNX GenAI runtime](pipette-ios/onnx-genai-runtime.md) · [AFM token enforcement](pipette-ios/afm-token-enforcement.md)
- [pipette-android](pipette-android/architecture.md) · [build](pipette-android/build.md) · [Kotlin/JNI engine port](pipette-android/engine-port.md) · [implementation ledger](pipette-android/implementation.md) · [native engine source](../android/native/README.md)
- [pipette-ios-proxy](../tools/pipette-ios-proxy/README.md) · [design](../tools/pipette-ios-proxy/DESIGN.md): runs jobs on tethered iPhones from a Mac, one server client per phone ([how builds reach the phones](pipette-ios/build.md#on-the-farm-pipette-ios-proxy))

### Shared libraries

Reusable crates the clients build on. The full crate map and dependency graph
are in [architecture.md](architecture.md#crates); the ones with their own docs:

- [pipette-memprobe](../crates/pipette-memprobe/README.md): reads a benchmarked program's memory from the parent
- [pipette-torch-oai](../crates/pipette-torch-oai/README.md): the vLLM and SGLang backend (Docker or uv)
- Bundled runtime catalogs: [mlx](../crates/pipette-artifacts-catalog/bundled-catalog/mlx/README.md) · [onnx_genai](../crates/pipette-artifacts-catalog/bundled-catalog/onnx_genai/README.md) · [openvino](../crates/pipette-artifacts-catalog/bundled-catalog/openvino/README.md) · [ovms](../crates/pipette-artifacts-catalog/bundled-catalog/ovms/README.md) · [torch_oai](../crates/pipette-artifacts-catalog/bundled-catalog/torch_oai/README.md)

### Tools

- [adb-connector-apk](../tools/adb-connector-apk/README.md): re-enables wireless debugging on fleet phones after a reboot
- [energy-probes](../tools/energy-probes/README.md): the probes behind [energy device capabilities](methodology/energy/02-device-capabilities.md)
- [gpu-keepalive](../tools/gpu-keepalive/README.md): Windows service that keeps the GMKtec GPU from idling
- [macos-thermal-probe](../tools/macos-thermal-probe/README.md): samples the macOS thermal signals the readiness gate reads
- [pdh-multiproc-exp](../tools/pdh-multiproc-exp/README.md): reproduces the Windows multi-process GPU memory attribution finding
- [SLURM helpers](../scripts/slurm/README.md): scripts for running `pipette-ctl` work on a SLURM cluster
