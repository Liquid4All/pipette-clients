import Foundation

// Store-entry layout constants and on-disk size, after `crates/pipette-artifacts/src/entry.rs`.
// Callers read paths off a manifest or the store rather than rebuilding the layout.

nonisolated enum Entry {
    static let blobsDirName = "blobs"

    /// The crate writes `manifest.toml`; this side writes JSON with the same field names.
    static let manifestName = "manifest.json"

    static let stagingDirName = ".staging"
}

/// Bytes the item at `url` occupies on disk (the crate's `entry_size_bytes`): allocated
/// blocks, as `du` reports, recursive including hidden children, `0` if missing.
/// Symlinks count as the link, never followed, so a link into another entry is not
/// billed twice and a link out of the store is not billed at all.
///
/// Run once per publish to stamp `blobsBytes`, so the quota survey walks only garbage
/// and entries whose manifest lacks that stamp.
nonisolated func entrySizeBytes(at url: URL) -> Int64 {
    var info = stat()
    guard lstat(url.path, &info) == 0 else { return 0 }
    var total = allocatedBytes(info)
    guard info.st_mode & S_IFMT == S_IFDIR else { return total }
    // Lists hidden children, and does not descend a symlinked subdirectory.
    for child in (try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? [] {
        let (sum, overflow) = total.addingReportingOverflow(
            entrySizeBytes(at: url.appendingPathComponent(child)))
        total = overflow ? .max : sum
    }
    return total
}

private nonisolated func allocatedBytes(_ info: stat) -> Int64 {
    let (bytes, overflow) = Int64(info.st_blocks).multipliedReportingOverflow(by: 512)
    return overflow ? .max : bytes
}
