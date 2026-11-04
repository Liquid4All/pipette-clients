# Bundled torch-oai (uv vLLM and SGLang) runtime catalog

This directory ships in the `pipette-artifacts-catalog` crate (and thus the
unified `pipette` client) via `include_str!` (see
`crates/pipette-artifacts-catalog/src/runtime/torch_oai/mod.rs`).
`catalog.toml` is the single source of truth for the bundled `uv-vllm` and
`uv-sglang` runtimes.

Each table is a complete catalog entry:

```toml
[[uv_vllm]]
server_version = "0.21.0"
build = "cu121"
python_version = "3.12"
requirements = '''
--extra-index-url https://download.pytorch.org/whl/cu121
vllm==0.21.0
'''
```

The slug and runtime version are computed at load time:

```text
<server>@<server_version>+<build>.py<python_version>
```

For example, the table above resolves to:

```text
vllm@0.21.0+cu121.py3.12
```

Rules:

- Server identity comes from the table name: `uv_vllm` or `uv_sglang`.
- `build` must identify the wheel flavor: `cu*`, `rocm*`, or `cpu`.
- `python_version` is uv's raw selector, for example `3.12`; the slug
  adds the `py` prefix when composing the runtime version.
- `requirements` is the exact text materialized into `requirements.txt`
  during install.
- `install_flags` are extra `uv pip install` arguments. Omitted, they default
  to `--index-strategy unsafe-best-match`, so PyTorch's `--extra-index-url`
  can combine with PyPI transitive deps (uv's default first-index strategy
  refuses).

## Adding or bumping a runtime

1. Append a `[[uv_vllm]]` or `[[uv_sglang]]` block to `catalog.toml`. Never
   edit a shipped row: its slug is what plans and past results cite, and the
   installed environment is keyed by a digest of its requirements body, so new
   pins under an old slug strand that environment and make the slug name a
   body it did not run.
2. Fill in `server_version`, `build`, `python_version` and `requirements`.
3. If the row adds a server/build combination `install_smoke` in
   `crates/pipette-artifacts/tests/uv_catalog_smoke.rs` does not cover, add a
   `#[case]` there.
