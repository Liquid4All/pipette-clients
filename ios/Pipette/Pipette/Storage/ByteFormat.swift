import Foundation

/// Shared byte formatting (file-style and memory-style), deduplicated from several
/// call sites that had copy-pasted formatters.
nonisolated enum ByteFormat {
    /// GB/MB, file count style — model and download sizes.
    static func fileSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB]
        formatter.countStyle = .file
        return formatter.string(fromByteCount: bytes)
    }

    /// Binary count style — the storage limit, which is authored in GiB. `.file` would
    /// echo a 16 GiB pick back as "17.18 GB", so the user would not recognize the number
    /// they chose. Measured sizes stay on `fileSize`.
    static func storageLimit(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: bytes)
    }

    /// Memory count style, automatic units — RAM usage.
    static func memory(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory)
    }
}
