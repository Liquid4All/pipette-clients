# Bundled mlx-lm runtime catalog

This directory ships in the `pipette-mlx` binary via `include_str!`
(see `crates/pipette-mlx/src/runtimes.rs`). `catalog.toml` is the
single source of truth for bundled mlx-lm runtimes.

Each `[[mlx]]` table is a complete catalog entry; `version` + `flavor`
+ `requirements`:

```toml
[[mlx]]
version = "0.31.3"
flavor = "macos-arm64"
requirements = '''
mlx-lm==0.31.3
...
'''
```

Rules:

- `version` is the mlx-lm version and the catalog key; it is what a
  `mlx-macos-pipette://version=<v>` ref selects.
- `flavor` is the hardware build target (`macos-arm64`); it is written
  into the runtime manifest.
- `requirements` is the exact locked text materialized into
  `requirements.txt` during install.

## Bumping / adding a runtime

1. Edit or append a `[[mlx]]` block in `catalog.toml`.
2. Fill in `version`, `flavor`, and the inline `requirements` text.
3. Commit the catalog edit. No per-version requirement files are used.
