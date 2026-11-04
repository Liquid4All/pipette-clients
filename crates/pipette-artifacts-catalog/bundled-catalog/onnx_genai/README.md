# Bundled onnxruntime-genai runtime catalog

This directory ships in the `pipette-artifacts-catalog` crate via
`include_str!` (see `crates/pipette-artifacts-catalog/src/runtime/onnx_genai.rs`).
`catalog.toml` is the single source of truth for bundled ONNX Runtime GenAI
runtimes.

Each `[[onnx_genai]]` table is a complete catalog entry; `version` + `ep` +
`requirements`:

```toml
[[onnx_genai]]
version = "0.16.0"
ep = "cpu"
requirements = '''
flatbuffers==25.12.19
numpy==2.4.6
onnxruntime==1.30.0
onnxruntime-genai==0.16.0
packaging==26.3
protobuf==7.36.1
'''
```

Rules:

- The catalog key is the `version` + `ep` pair. An
  `uv-onnx-genai://version=<v>&ep=<ep>` ref selects one row.
- `version` is the onnxruntime-genai version.
- `ep` is the ONNX Runtime execution provider the row installs the wheel
  for. It names a plan-types `OnnxExecutionProvider`.
- `requirements` is the exact locked text materialized into
  `requirements.txt` during install. Pin every package, including the
  `onnxruntime` or `onnxruntime-gpu` engine. The engine decides the
  measurement, and the storage key is a digest of this body, so an unpinned
  engine puts different builds under one key.
- A provider with no row here reaches no cell through the catalog. A structural
  JSON runtime can still name such a provider, and the capability rules in
  pipette-mgmt's `pipette-planner` crate give it an empty policy.
  DirectML has no row. `onnxruntime-genai-directml` stops at 0.14.1, which is
  older than every row. Its newest release that installs from PyPI is 0.13.1,
  because 0.13.2 and later need an `onnxruntime-directml` newer than PyPI's
  1.24.4. The `version` + `ep` key already expresses such a row, so adding one
  is one more `[[onnx_genai]]` table and one more sentence here.

## Bumping / adding a runtime

1. Resolve the pin set with
   `uv pip install --dry-run 'onnxruntime-genai==<version>'`.
   For a CUDA row, resolve the wrapper together with the engine and the
   engine's CUDA extras. The extras add the `nvidia-*` libraries that the CUDA
   execution provider loads with the model. The cross-resolve needs no CUDA
   host.

   ```bash
   uv pip install --dry-run --python-platform x86_64-unknown-linux-gnu \
     'onnxruntime-genai-cuda==<version>' 'onnxruntime-gpu[cuda,cudnn]==<engine>'
   ```

2. Append an `[[onnx_genai]]` block in `catalog.toml`. Never edit a shipped
   row: the storage key is a digest of its requirements body, so new pins
   strand the installed environment and leave past results citing a body the
   catalog no longer produces.
3. Fill in `version` and `ep`, then paste the resolved set as `requirements`.
4. Add one block per execution provider. Each provider is a separate
   distribution, so one version needs one row per provider pipette offers.
5. Commit the catalog edit. The catalog uses no per-version requirement files.
