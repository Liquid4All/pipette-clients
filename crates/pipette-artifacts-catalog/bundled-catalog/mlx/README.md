# Bundled mlx-lm runtime catalog

This directory ships in the `pipette-artifacts-catalog` crate via
`include_str!` (see `crates/pipette-artifacts-catalog/src/runtime/mlx.rs`).
`catalog.toml` is the single source of truth for bundled mlx-lm runtimes.

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

- `version` is the catalog key: what a `mlx-macos-pipette://version=<v>`
  ref selects. It is the version of the package that the row moved:
  - mlx-lm moved: mlx-lm's version (the usual case).
  - Only the mlx core moved: the mlx version. The `0.32.2` row holds mlx-lm
    at 0.31.3 and moves the core.
  - The row ships mlx-vlm: mlx-vlm's version.
  - The key is already taken: the new mlx core version, as the `0.32.4` row
    does.

  Keys from different packages do not sort by age: `0.32.0` (mlx-lm 0.32.0
  on the 0.32.3 core) is newer than `0.32.2`.
- `flavor` is the hardware build target (`macos-arm64`); it is written
  into the runtime manifest.
- `requirements` is the exact locked text materialized into
  `requirements.txt` during install.

## Bumping / adding a runtime

1. Append a `[[mlx]]` block in `catalog.toml`; a shipped row is not
   edited in place. A pulled runtime is keyed by its requirements body,
   so rewriting a row's pins strands the installed environment under a
   label that no longer describes it, and past results cite a body the
   catalog can no longer produce.
2. Fill in `version`, `flavor`, and the inline `requirements` text. Choose
   `version` as the `version` rule above says.
3. Commit the catalog edit. No per-version requirement files are used.

## Rows that ship mlx-vlm

The VL cells load a VL bundle (and, for `vl_spec_decode`, a drafter) through
`mlx_vlm`, which `mlx-lm` cannot do, so they need a row that pins both packages.
`0.7.11` is the current one, on the same mlx-lm and core as the `0.32.4` row.
The older `0.7.1+lfm2-vl-speculative` row pins an mlx-vlm commit on `main`,
because LFM2-VL targets could not pair with a drafter until Blaizzy/mlx-vlm#2280,
which first shipped in 0.7.2. Its PEP 440 local label marks it as 0.7.1 plus
that change. `0.7.4` pins a release under a plain key, as a new row per step 1.
