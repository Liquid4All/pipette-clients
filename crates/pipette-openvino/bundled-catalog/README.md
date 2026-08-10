# Bundled openvino-genai runtime catalog

This directory ships in the `pipette-openvino` binary via `include_str!`
(see `crates/pipette-openvino/src/catalog.rs`). `catalog.toml` is the
single source of truth for bundled OpenVINO runtimes.

Each `[[openvino]]` table is a complete catalog entry; `version` +
`requirements`:

```toml
[[openvino]]
version = "2026.2.1"
requirements = '''
openvino-genai==2026.2.1.0
...
'''
```

Rules:

- `version` is the openvino-genai version and the catalog key; it is what
  an `uv-openvino://version=<v>` ref selects.
- `requirements` is the exact locked text materialized into
  `requirements.txt` during install.
- There is no flavor column. One wheel serves CPU, GPU and NPU: the
  device is a field on the runtime, chosen per cell.

## Bumping / adding a runtime

1. Edit or append an `[[openvino]]` block in `catalog.toml`.
2. Fill in `version` and the inline `requirements` text.
3. Keep `openvino`, `openvino-genai` and `openvino-tokenizers` on the same
   release. A mismatched tokenizers wheel fails to load the compiled
   tokenizer IR.
4. Commit the catalog edit. No per-version requirement files are used.
