import Foundation
import os

/// Memory pre-flight for model loads: a model that doesn't fit gets the process killed by
/// jetsam, never a graceful failure. It warns but never blocks, since barely fitting runs
/// are legitimate benchmarks, and its numbers let a crash report say how tight the fit was.
enum MemoryGate {
    /// KV cache, compute buffers and the app itself: ~150 MB for a small model at short ctx
    /// to well past this for 7B-class at long ctx. A middle value suits a warning.
    nonisolated static let headroomBytes: Int64 = 600 * 1024 * 1024

    struct Snapshot: nonisolated Sendable {
        let modelBytes: Int64
        let availableBytes: Int64
    }

    /// nil when either side is unreadable; the simulator reports a 0 budget.
    nonisolated static func snapshot(modelPath: String) -> Snapshot? {
        let available = Int64(os_proc_available_memory())
        guard available > 0 else { return nil }
        let size = sizeOfFileOrDirectory(atPath: modelPath)
        guard size > 0 else { return nil }
        return Snapshot(modelBytes: size, availableBytes: available)
    }

    nonisolated static func warning(
        modelName: String,
        modelBytes: Int64,
        availableBytes: Int64
    ) -> String? {
        guard modelBytes > 0, availableBytes > 0 else { return nil }
        guard availableBytes < modelBytes + headroomBytes else { return nil }
        return "Low memory for \(modelName): \(ByteFormat.memory(availableBytes)) free for a "
            + "\(ByteFormat.memory(modelBytes)) model. iOS may kill the app mid-benchmark. "
            + "Closing other apps frees memory."
    }

    nonisolated static func sizeOfFileOrDirectory(atPath path: String) -> Int64 {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: path, isDirectory: &isDirectory) else { return 0 }

        let url = URL(fileURLWithPath: path)
        if !isDirectory.boolValue {
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return Int64(size)
        }

        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
        return total
    }
}
