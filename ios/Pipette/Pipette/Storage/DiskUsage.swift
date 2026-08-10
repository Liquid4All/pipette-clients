import Foundation

/// Allocation-accurate on-disk sizing for the model store and its quota, as opposed
/// to `MemoryGate`'s estimate of what a model will cost in RAM. Recursive, hidden children included
/// (a `.staging` orphan occupies real bytes), and symlinks are counted as the link
/// itself rather than followed — otherwise a link into another entry would bill its
/// target twice and a link out of the store would bill bytes the store can't reclaim.
///
/// Never persisted: a recorded size drifts from the bytes on disk, and the walk is
/// cheap at store sizes of tens of entries.
nonisolated enum DiskUsage {
    /// Bytes the item at `url` occupies — `st_blocks * 512` (what `du` reports),
    /// falling back to the file length when the filesystem reports no blocks.
    /// `0` for a path that doesn't exist.
    static func bytes(at url: URL) -> Int64 {
        var info = stat()
        // `lstat`, not `stat`: a symlink is measured as the link, never traversed.
        guard lstat(url.path, &info) == 0 else { return 0 }
        var total = allocated(info)
        guard info.st_mode & S_IFMT == S_IFDIR else { return total }
        // `contentsOfDirectory(atPath:)` lists hidden children too, and returns the
        // link entries of a symlinked subdirectory without descending into it.
        for child in (try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? [] {
            total += bytes(at: url.appendingPathComponent(child))
        }
        return total
    }

    private static func allocated(_ info: stat) -> Int64 {
        let blocks = Int64(info.st_blocks) * 512
        return blocks > 0 ? blocks : Int64(info.st_size)
    }
}
