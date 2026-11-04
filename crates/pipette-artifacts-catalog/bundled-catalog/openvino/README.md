# Bundled openvino-genai runtime catalog

This directory ships in the `pipette-artifacts-catalog` crate via
`include_str!` (see `crates/pipette-artifacts-catalog/src/runtime/openvino.rs`).
`catalog.toml` is the single source of truth for bundled OpenVINO runtimes.

Each `[[openvino]]` table is a complete catalog entry; `version` +
`requirements`:

```toml
[[openvino]]
version = "2026.4.1"
requirements = '''
openvino-genai==2026.4.1.0
...
'''
```

Rules:

- `version` is the openvino-genai version and the catalog key; it is what
  an `uv-openvino://version=<v>` ref selects.
- `requirements` is the exact locked text materialized into
  `requirements.txt` during install.
- There is no flavor column. One wheel serves CPU, GPU and NPU: the
  device is a runtime flag on the cell.

## Bumping / adding a runtime

1. Append an `[[openvino]]` block in `catalog.toml`. Never edit a shipped
   row: the storage key is a digest of its requirements body, so new pins
   strand the installed environment and leave past results citing a body the
   catalog no longer produces.
2. Fill in `version` and the inline `requirements` text.
3. Keep `openvino`, `openvino-genai` and `openvino-tokenizers` on the same
   release. A mismatched tokenizers wheel fails to load the compiled
   tokenizer IR.
4. Commit the catalog edit. No per-version requirement files are used.
