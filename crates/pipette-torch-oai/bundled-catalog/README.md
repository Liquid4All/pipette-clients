# Bundled uv runtime catalog

This directory ships in the `pipette-torch-oai` library (and thus the
unified `pipette` client) via `include_str!` (see
`crates/pipette-torch-oai/src/catalog.rs`). `catalog.toml` is the
single source of truth for bundled uv runtimes.

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

## Bumping A Runtime

1. Update the relevant component fields in `catalog.toml`.
2. Update the inline `requirements` text.
3. Commit the catalog edit. No per-slug requirement files are used.

## Adding A New Entry

1. Append a `[[uv_vllm]]` or `[[uv_sglang]]` block to `catalog.toml`.
2. Fill in `server_version`, `build`, `python_version`,
   and `requirements`.
3. Add a smoke-test stanza in
   `crates/pipette-torch-oai/tests/uv_catalog_smoke.rs` if the entry
   adds a vendor/server combination not already covered.
