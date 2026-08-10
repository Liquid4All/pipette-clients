# Selection Policies

This article states how pipette chooses what it benchmarks: which models, which
weights, and which runtime build. Benchmark numbers are only fair to compare
when these choices follow a consistent, vendor-neutral policy and are recorded
with the result. It complements the per-benchmark methodology articles, which
describe how each metric is measured once the model, weights, and runtime are
fixed.

## Principles

- **No model preference.** We benchmark what is available on the market, and we
  sweep every model across the same device-fit quant ladder rather than
  hand-picking a flattering configuration for any one. The selection rules below
  are applied identically to our own models and everyone else's, and because
  every run records its full selection (model, weight source, quantization,
  runtime, version, flags), any result can be checked against this policy rather
  than taken on trust.
- **Upstream, not hand-built.** Runtimes and weights are consumed from their
  upstream or community sources, at published versions, rather than produced by
  a private build step. This keeps a result reproducible from public artifacts.
- **Recorded, then comparable.** Whatever is chosen is recorded in the result
  artifact: model, quantization, weight source, runtime, runtime version, and
  effective flags. A comparison is valid only across runs whose recorded
  selection matches.
- **Providers can recommend the best setup.** Model and runtime providers are
  welcome to point us at the implementation, weight repository, and especially
  the quantization that best represents their work; we adopt any recommendation
  that is reproducible from public artifacts and record it. See
  [Provider Recommendations](#provider-recommendations).

## Runtime Backends

The unified `pipette` client drives four runtime backends (the iOS app is a
fifth surface). Each consumes a specific weight format and sources its runtime
from a specific upstream.

| Runtime | Platforms | Weight format | Runtime source |
| --- | --- | --- | --- |
| `pipette-llamacpp` | macOS, Windows, Linux, Android | GGUF | `llama.cpp` official upstream release (`ggml-org/llama.cpp`), tag-pinned |
| `pipette-ios` | iOS | GGUF | `llama.cpp` pinned in the app; re-pinned when a new model needs it |
| `pipette-mlx` | macOS (Apple Silicon) | MLX | `mlx_lm` (PyPI), models loaded from the HuggingFace Hub |
| `pipette-torch-oai` | Linux (Docker / uv) | HuggingFace safetensors | vLLM or SGLang, pinned in the bundled runtime catalog |

## Weights Selection

The rule is the same across clients: prefer the model authors' published weights
in the format the runtime consumes, and fall back to a well-known community
conversion only when the authors do not publish that format.

- **GGUF (`pipette-llamacpp`, `pipette-ios`).** Use the official upstream GGUF
  when the model authors publish one. When no official GGUF exists, the default
  source is the `unsloth` community conversions.
- **MLX (`pipette-mlx`).** Use the `mlx-community` conversions from the
  HuggingFace Hub.
- **HuggingFace safetensors (`pipette-torch-oai`).** vLLM and SGLang load the
  model's upstream weights directly, so this path runs the published
  safetensors. Any quantization used follows the same rule (authors' published
  quant first, a community quant only as a fallback), and is recorded with the
  result.

### Quantization

The quantization is chosen to fit the target device rather than fixed to a
single level: we benchmark the common quants that fit a given device class.

- **GGUF (`pipette-llamacpp`, `pipette-ios`).** The common set is `q4_0`,
  `q4_k_m`, and `q5_k_m`. On laptops, which have more memory headroom, `q8_0`
  is also considered.
- **MLX and HuggingFace safetensors (`pipette-mlx`, `pipette-torch-oai`).** The
  quants are likewise chosen to fit the common devices for that runtime:
  usually 4- to 5-bit.

A model is deliberately benchmarked at more than one quantization. The goal is
not to report only a model's best configuration but to characterize how size
trades off against speed and quality across the common quant levels: the
scaling relationship, not a single headline number. Because quantization is
part of the recorded selection, each point in that sweep stays reproducible, and
a comparison between two models is only fair at the same quantization.

The selected weight repository and quantization are part of the recorded
selection and must match for two results to be compared.

## Provider Recommendations

Model and runtime providers are welcome to tell us the best way to run their
work, and we will adopt it. A model author can point us at the implementation,
the weight repository, and (especially) the quantization that best represents
their model; a runtime maintainer can point us at the configuration or version
that best represents their engine. We would rather measure a model or runtime at
the setup its own authors consider optimal than at a default we picked.

A recommended setup is held to the same rules as any other selection. It must be
consumable from a public upstream or community source at a pinned version, and
the resulting model, weight source, quantization, runtime, version, and flags
are recorded with the result so the run stays reproducible and comparable.
Recommendations that cannot be reproduced from public artifacts, or that would
change what a benchmark measures rather than how well it runs, are not adopted.

This invites a fair question: do providers who engage get their optimal setup
while everyone else gets whatever we picked? The quant sweep above is what keeps
that from being an asymmetry. We do not run unengaged models at a single
arbitrary default. Every model is swept across the same device-fit quant ladder
(named above), so the common configurations are covered for engaged and
unengaged models alike. A provider recommendation adds a specific point to that
sweep; it does not grant a configuration that competitors were denied. And
because the recommended quant comes from the same public ladder and is recorded
in the selection like any other, a reader can see exactly which configuration
produced a given number rather than having to infer it.

## Workflow: New Model

1. Acquire whatever the market makes available; apply no model preference.
2. Pick weights per client using the weights-selection rule above (authors'
   published format first, community fallback otherwise).
3. If the model uses an architecture or feature the currently pinned runtime
   does not yet support, re-pin that runtime to a version that does (see
   [Workflow: New Runtime](#workflow-new-runtime)). This is most common on
   `pipette-ios`, where the pinned `llama.cpp` is re-pinned when a new model
   needs it.
4. Record the model identifier, weight source, and quantization with the result.

## Workflow: New Runtime

Runtimes are adopted from upstream by moving a pin, never by forking behavior.

- **`pipette-llamacpp`.** The runtime binary is downloaded from the official
  `ggml-org/llama.cpp` GitHub releases at a specific tag. Adopting a newer
  upstream means selecting the newer release tag.
- **`pipette-ios`.** The app pins a `llama.cpp` version. The pin is advanced to
  the latest when convenient and re-pinned earlier when a new model requires a
  feature only a newer `llama.cpp` provides.
- **`pipette-torch-oai`.** vLLM and SGLang versions are pinned in the bundled
  runtime catalog (the single source of truth for the bundled runtimes), one
  entry per hardware build (CUDA, ROCm, CPU). Adopting a new upstream version
  means adding or updating a catalog entry.
- **`pipette-mlx`.** The `mlx_lm` version is sourced from PyPI.

In every case the runtime name and resolved version are recorded with the
result, so a runtime change is visible to anyone comparing numbers.

## Recording and Comparability

Each result artifact preserves the selection it ran under: model, weight source
and quantization, runtime name, runtime version, and the effective runtime
flags. Two results are comparable only when those recorded fields match. A
difference in any of them (a different GGUF conversion, a newer vLLM build, a
re-pinned `llama.cpp`) is a difference in the experiment, not a difference in
the model, and must be read that way.

Each result also records the volatile run environment it was measured in: the
device's power state and the thermal/cooling conditions it ran under. Those are
covered in [Device conditions](device-conditions.md), which states the expected
device state and the per-platform specifics.

## Code References

The bundled vLLM and SGLang versions are pinned in
[`bundled-catalog/catalog.toml`](../../crates/pipette-torch-oai/bundled-catalog/catalog.toml).

The `llama.cpp` runtime is resolved from `ggml-org/llama.cpp` GitHub releases in
[the llama.cpp runtime resolver](../../crates/pipette-llamacpp/src/runtimes.rs).

MLX models are resolved from the HuggingFace Hub by the model store in
[`pipette-ops`](../../crates/pipette-artifacts/src/model/fetch.rs).
