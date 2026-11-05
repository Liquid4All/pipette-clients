# Bundled OVMS release catalog

This directory ships in the `pipette-artifacts-catalog` crate via
`include_str!` (see `crates/pipette-artifacts-catalog/src/runtime/ovms/mod.rs`).
`catalog.toml` pins the sha256 of each upstream OpenVINO Model Server release
asset pipette offers.

Each `[[ovms]]` table is one asset, keyed by `tag` + `flavor`:

```toml
[[ovms]]
tag = "v2026.4.1"
flavor = "windows-x64"
sha256 = "1bc2372d81bb4767829a7bf513745bd481ccbfec9550e278804ebe2051e3b678"
```

Rules:

- `tag` is the `openvinotoolkit/model_server` release tag, verbatim. An
  `ovms://version=<tag>&flavor=<flavor>` ref selects one row.
- `flavor` is a named `OvmsFlavor`. The asset is
  `OvmsFlavor::release_asset_name(tag)`, so a row whose tag names no asset is
  refused when the catalog is parsed.
- `sha256` is the published digest of that asset. A pull verifies the download
  against it and fails on a mismatch without retrying.
- A `github_release` from another repository, a `remote_archive` and a tag with
  no row install unpinned. A `vault_remote_archive` is verified against its
  vault record instead.

## Adding a release

1. Read the digests upstream publishes beside each `python_off` asset:

   ```bash
   tag=v2026.4.1; v=${tag#v}
   for os in windows ubuntu22 ubuntu24 redhat; do
     ext=tar.gz; [ "$os" = windows ] && ext=zip
     curl -sL "https://github.com/openvinotoolkit/model_server/releases/download/$tag/ovms_${os}_${v}_python_off.$ext.sha256"
   done
   ```

2. Check each digest against the one GitHub reports for the asset:
   `gh api repos/openvinotoolkit/model_server/releases/tags/<tag> --jq '.assets[] | "\(.name) \(.digest)"'`.
3. Append one `[[ovms]]` block per flavor.
