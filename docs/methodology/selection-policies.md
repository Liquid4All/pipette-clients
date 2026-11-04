# Selection Policies

This article states how pipette chooses what it benchmarks: which models, which
weights, and which runtime build. Benchmark numbers are only fair to compare
when these choices follow a consistent, vendor-neutral policy and are recorded
with the result. It complements the per-benchmark methodology articles, which
describe how each metric is measured once the model, weights, and runtime are
fixed.

## Principles

- **No model preference.** We benchmark models whose weights are published in
  a format that a supported runtime can load, and we sweep every model across
  the same device-fit quant ladder rather than
  hand-picking a flattering configuration for any one. The selection rules below
  are applied identically to our own models and everyone else's, and because
  every run records its full selection (model, weight source, quantization,
  runtime, version, flags), any result can be checked against this policy rather
  than taken on trust.
- **Upstream first.** Runtimes and weights come from their upstream or
  community sources at published versions. Where upstream cannot run a model,
  pipette runs a named public fork or a patched build, and the result records
  its repository and version. This keeps a result reproducible from public
  artifacts.
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

The `pipette` CLI drives eight runtime backends: llama.cpp, MLX, Core AI,
torch-oai (vLLM or SGLang), OpenVINO GenAI, ONNX Runtime GenAI, OVMS, and
TensorRT Edge-LLM. The iOS app runs llama.cpp, MLX, ONNX Runtime GenAI, and
Apple Foundation; the Android app runs llama.cpp. Each backend consumes one
weight format and takes its runtime from a public upstream at a pinned version,
with these exceptions:

- Apple Foundation: the model and the runtime ship with the OS.
- Core AI: the engine ships with macOS. Pipette builds its Swift sidecar
  locally against a pinned `apple/coreai-models` commit
  ([Core AI](../pipette-cli/coreai.md)).
- TensorRT Edge-LLM: upstream publishes no server archive, so a runtime names
  an archive by URL ([TensorRT Edge-LLM](../pipette-cli/trt-edgellm.md)).

Published iOS results come from a build of this app with the private-API flags
on (`PIPETTE_PRIVATE_THERMAL`, `PIPETTE_PRIVATE_MEMORY`). These flags read the
exact SoC die temperature and the kernel's footprint peak, which the public iOS
API does not expose. App Review rejects such a build, so it runs only
side-loaded on registered development devices, and anyone can make the same
build from this repository
([iOS results and the private-API build](README.md#ios-results-and-the-private-api-build)). How to name each runtime and model is in
[Naming models, runtimes, and per-cell flags](../pipette-cli/models-and-runtimes.md).

## Weights Selection

The rule is the same across clients: prefer the model authors' published weights
in the format the runtime consumes, and fall back to a well-known community
conversion only when the authors do not publish that format.

- **GGUF (llama.cpp).** Use the official upstream GGUF
  when the model authors publish one. When no official GGUF exists, the default
  source is the `unsloth` community conversions.
- **MLX.** Use the `mlx-community` conversions from the
  HuggingFace Hub.
- **HuggingFace safetensors (torch-oai).** vLLM and SGLang load the
  model's upstream weights directly, so this path runs the published
  safetensors. Any quantization used follows the same rule (authors' published
  quant first, a community quant only as a fallback), and is recorded with the
  result.

### Quantization

The quantization is chosen to fit the target device rather than fixed to a
single level: we benchmark the common quants that fit a given device class.

- **GGUF (llama.cpp).** The common set is `Q4_0`,
  `Q4_K_M`, and `Q5_K_M`. On laptops, which have more memory headroom, `Q8_0`
  is also considered.
- **MLX and HuggingFace safetensors (MLX, torch-oai).** The
  quants are likewise chosen to fit the common devices for that runtime:
  usually 4- to 5-bit.

A model is deliberately benchmarked at more than one quantization. The goal is
not to report only a model's best configuration but to characterize how size
trades off against speed and quality across the common quant levels: the
scaling relationship, not a single headline number. Some families carry a single
point anyway, which is a gap in coverage rather than an exception to the policy:
a newly onboarded model enters at whatever quant its benchmark run measured, and
some repos offer nothing else worth cataloging. The published results are the
inventory of which. The mobile apps offer a GGUF preset row only when the results
dashboard publishes a reading for it on that handset, so a family shows one
quant there exactly when one quant was published for that device. Because
quantization is part of the recorded selection, each point in that sweep stays
reproducible, and a comparison between two models is
only fair at the same quantization.

The selected weight repository and quantization are part of the recorded
selection and must match for two results to be compared.

## Swap Exclusion

A result describes a model on a device only when the device holds the whole run
in physical memory. When the operating system moves part of a run to swap, the
measurement describes a memory-starved device instead of the model. We hold that
configuration back from the published results.

The exclusion is a display rule, not a collection rule. We still run the
benchmark, and we still store the result with its full recorded selection and run
environment. The result stays available for investigation and comparison against a
later rerun. What the exclusion removes is the public presentation of that number,
because a swapping run does not answer the question a reader asks of it.

The unit of exclusion is one quantization of one model on one device. A single
swapping run excludes that quantization on that device in every configuration and
for every benchmark, not only for peak memory. The context length, benchmark, and
runtime flags of the run that swapped do not limit the exclusion.

The moment of the swap does not matter. Weights that the kernel compresses during
model load and pages that it evicts during inference mean the same thing: the
device lacks the memory for that quantization. Both cases are excluded on the
same terms.

"Swap" means whatever the platform provides: zram on Android, the Windows page
file, macOS memory compression and swap files, or a Linux swap partition or file.
Detection is per platform, and we refine the mechanism as necessary for any
operating system we support. The exclusion follows from the observation rather than
from the signal that produced it.

An exclusion is provisional, and it is not a verdict on the model. A swapping run
raises a question about the cause: the device's memory ceiling, a background
process that consumed memory, a runtime flag that loads the weights into
anonymous memory, or a defect in the runtime. We investigate the cause. If the
same quantization then runs on the same device with no swap, it appears again in a
later update of the published results. The rerun carries the published number on
its own; we do not publish the swapping run with a caveat attached.

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

A provider that engages with us does not get an advantage over one that does
not. The quant sweep above prevents that asymmetry. We do not run unengaged
models at a single arbitrary default. Every model is swept across the same device-fit quant ladder
(named above), so the common configurations are covered for engaged and
unengaged models alike. A provider recommendation adds a specific point to that
sweep; it does not grant a configuration that competitors were denied. The
recommended quant comes from the same public ladder, and the selection records
it like any other. A reader can therefore see exactly which configuration
produced a given number.

## Workflow: New Model

1. Take the model's published weights; apply no model preference.
2. Pick weights per client using the weights-selection rule above (authors'
   published format first, community fallback otherwise).
3. If the model uses an architecture or feature the currently pinned runtime
   does not yet support, re-pin that runtime to a version that does (see
   [Workflow: New Runtime](#workflow-new-runtime)). This is most common for
   the apps' `llama.cpp` pin.
4. Record the model identifier, weight source, and quantization with the result.

## Workflow: New Runtime

Runtimes are adopted from upstream by moving a pin. Where upstream cannot run a
model, the pin names a public fork or a patched build, and the result records
its repository and version.

- **llama.cpp on the CLI.** The runtime binary is downloaded from the official
  `ggml-org/llama.cpp` GitHub releases at a specific tag. Adopting a newer
  upstream means selecting the newer release tag.
- **llama.cpp in the apps.** The apps compile the `vendor/llama.cpp` submodule.
  The pin is advanced when convenient, and sooner when a new model needs a
  feature only a newer `llama.cpp` provides. The iOS app also compiles the
  `vendor/llamacpp-<name>` variant pins, and it applies a patch series to each
  pin ([iOS build](../pipette-ios/build.md)).
- **Catalog runtimes (MLX, torch-oai, OpenVINO, ONNX GenAI, OVMS).** The bundled
  catalog pins each version; adopting a new upstream version means adding a
  catalog entry. Each catalog's rules are in its README:
  [mlx](../../crates/pipette-artifacts-catalog/bundled-catalog/mlx/README.md),
  [torch_oai](../../crates/pipette-artifacts-catalog/bundled-catalog/torch_oai/README.md),
  [openvino](../../crates/pipette-artifacts-catalog/bundled-catalog/openvino/README.md),
  [onnx_genai](../../crates/pipette-artifacts-catalog/bundled-catalog/onnx_genai/README.md),
  [ovms](../../crates/pipette-artifacts-catalog/bundled-catalog/ovms/README.md).

In every case the runtime name and resolved version are recorded with the
result, so a runtime change is visible to anyone comparing numbers.

## Recording and Comparability

Each result artifact preserves the selection it ran under: model, weight source
and quantization, runtime name, runtime version, and the effective runtime
flags. Pipette records those coordinates to identify the experiment, but they do
not universally identify the resolved model artifact byte-for-byte. Two results
are comparable only when those recorded fields match. A difference in any of
them (a different GGUF conversion, a newer vLLM build, a re-pinned `llama.cpp`)
is a difference in the experiment, not a difference in the model, and must be
read that way. Exact reproduction across time therefore also requires resolving
the same underlying artifact or snapshot.

For Hugging Face sources, a commit SHA in `revision` gives the recorded
selection a stable repository coordinate across time. A tag or a branch
resolves to one immutable commit at the moment it is read. The same ref can
resolve to a different commit later, and an omitted revision resolves to
`main`. A GGUF model specification carries an optional per-file SHA-256 field,
and pipette checks the downloaded bytes against it. Directory-backed MLX,
Torch, OpenVINO, and ONNX GenAI model specifications carry no equivalent per-file content
digest.

Each result also records the volatile run environment it was measured in: the
device's power state and the thermal/cooling conditions it ran under. Those are
covered in [Device conditions](device-conditions.md), which states the expected
device state and the per-platform specifics.

## Code References

The `llama.cpp` runtime is installed from `ggml-org/llama.cpp` GitHub releases by
[the llama.cpp runtime installer](../../crates/pipette-artifacts/src/runtime/llamacpp.rs).

Hugging Face models are fetched, and GGUF `sha256` digests checked, by
[the model fetch in pipette-artifacts](../../crates/pipette-artifacts/src/model/fetch.rs).
