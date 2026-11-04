# pipette-torch-oai

Library backend for the unified `pipette` client: Docker-hosted or uv-managed
OpenAI-compatible inference servers (vLLM, SGLang) that serve
PyTorch/HuggingFace models. `pipette benchmarks run` owns the whole
container or venv lifecycle; the crate has no CLI of its own.

Operator docs live outside this crate:

- [torch-oai runtime](../../docs/pipette-cli/torch-oai.md): install, launch
  defaults, supported benchmarks, caveats
- [Usage guide](../../docs/pipette-cli/usage.md): the CLI workflow
- [`--runtime-flags`](../../docs/pipette-cli/models-and-runtimes.md#--runtime-flags):
  per-cell flag notation
