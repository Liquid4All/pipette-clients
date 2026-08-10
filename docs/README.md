# Documentation

Start at the [top-level README](../README.md) for an orientation to the
repository and the build entry point for each component. This directory holds
the detailed docs.

## Convention

Docs use a small, consistent vocabulary so a reader always knows what a file is
for. When a topic is large enough to stand alone it gets its own file; when it
is small it lives as a `##` section under the component's `overview.md`. Either
way the *names* are the same:

| Name              | Answers                                              |
|-------------------|------------------------------------------------------|
| `architecture.md` | What is it, how is it structured, where are the boundaries? (no commands) |
| `build.md`        | How do I compile it / produce its artifacts? (prerequisites + commands)   |
| `usage.md`        | How do I operate it once it's built?                 |
| `overview.md`     | A consolidated stand-in for the above when a component is small: same section headings, one file. |

Files that don't fit the vocabulary keep descriptive names (e.g.
`implementation.md` for a historical ledger, `design.md` for a design record,
`workflow.md` for a worked example).

### Two kinds of client

The split above is applied *by need*, and need differs by the two client
families described in the [top-level README](../README.md):

- **Host-native CLI** (`pipette`) installs and invokes external runtimes:
  `llama.cpp`, MLX, OpenVINO, vLLM, and SGLang. One workflow covers all of them
  (init → register → install runtime → run benchmark → sync); see
  [pipette-cli/](pipette-cli/usage.md), the
  [notation reference](pipette-cli/models-and-runtimes.md), backend notes, and
  the OpenVINO [IR](openvino-ir.md) / [measurement](openvino-measurement.md) docs.
- **Native mobile apps** (`pipette-ios`, `pipette-android`) compile their
  inference engines *into* the app and run inference in-process. There is no
  separate runtime to install. Which engines are linked is platform-specific:
  iOS links both `llama.cpp` (native Swift over the in-app llama.cpp build) and
  MLX (via the `mlx-swift` packages), while Android currently links `llama.cpp`
  (GGUF) only via its Rust core.
  Expect this set to grow with device-category-specific runtimes. They are
  otherwise self-contained, buildable clients with their own build pipelines and
  architecture, so they get a full `architecture.md` + `build.md` split.

## Index

### Repo-wide

- [Architecture](architecture.md): crates, dependency graph, workspace, traits
- [Storage quota](storage-quota.md): the disk cap over downloaded artifacts, and how eviction picks
- [Third-party licensing](licensing.md): what may enter each client's dependency graph, and the gates that enforce it
- [OpenVINO IR](openvino-ir.md): IR layout, precision identification, and NPU constraints
- [OpenVINO measurement](openvino-measurement.md): timing, compile, and cache behavior
- [Benchmark methodology](methodology/README.md): benchmark definitions and measurement rules
- [Fleet perf troubleshooting](fleet-perf-troubleshooting/gmktec-evo-x2.md): diagnosing speed differences between identical benchmark boxes, per box type

### Host-native CLIs

- [pipette-cli usage](pipette-cli/usage.md): unified `pipette` client; start here
- [Naming models, runtimes, and flags](pipette-cli/models-and-runtimes.md): the URI and JSON notation for `--model` / `--runtime` / `--runtime-flags`, with recipes
- [llama.cpp](pipette-cli/llamacpp.md) · [MLX](pipette-cli/mlx.md) · [torch-oai](pipette-cli/torch-oai.md) · [OpenVINO](pipette-cli/openvino.md): runtime backends
- [Eval checkpoint & resume](pipette-cli/eval-checkpoint.md)
- [pipette-plan](pipette-plan/plan-runner.md): plan orchestrator (separate binary)
- [Job generation](pipette-plan/job-generation.md): expanding a plan into jobs for the `pipette-mgmt` server

### Native mobile apps

- [pipette-ios](pipette-ios/architecture.md) · [build](pipette-ios/build.md) · [model store design](pipette-ios/model-store-design.md) · [execution alignment](pipette-ios/execution-alignment.md) · [private-thermal builds](pipette-ios/private-thermal-release-build.md) · [AFM runtime](pipette-ios/afm-runtime.md) · [AFM token enforcement](pipette-ios/afm-token-enforcement.md)
- [pipette-android](pipette-android/architecture.md) · [build](pipette-android/build.md) · [implementation ledger](pipette-android/implementation.md)

### Shared libraries

Reusable crates the clients build on, not standalone tools. The full crate
map and dependency graph live in [architecture.md](architecture.md#crates);
the crates with their own dedicated docs are:

- [pipette-doomloop](pipette-doomloop/doomloop-detection.md): repetition
  ("doom loop") detection, used by every client
