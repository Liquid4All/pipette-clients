import Foundation

// Directory-entry constants for one store entry. Mirrors
// `pipette-artifacts/src/entry.rs`, which the crate keeps module-private for the same
// reason this stays small: where an artifact lands is the store's answer to give, so
// callers read a path off a manifest or the store rather than rebuilding the layout.

nonisolated enum Entry {
    /// `entry.rs`'s `BLOBS_DIR_NAME` — the payload directory inside an entry.
    static let blobsDirName = "blobs"

    /// `entry.rs`'s `MANIFEST_NAME`. The crate writes TOML; this side writes JSON,
    /// because the serialization stack differs and the field names already match.
    static let manifestName = "manifest.json"

    /// `entry.rs`'s `STAGING_DIR_NAME`. The crate stages a fetch here and publishes by
    /// rename. iOS installs into the entry directly, so nothing writes this yet — the
    /// name exists because the quota sweep has to recognise an orphan under it as
    /// garbage, which it did by prose before.
    static let stagingDirName = ".staging"
}
