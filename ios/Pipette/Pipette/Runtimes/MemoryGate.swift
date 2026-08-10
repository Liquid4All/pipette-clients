import Foundation
import os

/// Memory pre-flight for model loads. A model that doesn't fit the app's
/// remaining jetsam budget never fails gracefully — the load (or first
/// decode) pushes the footprint over the limit and iOS kills the process
/// outright. The pre-flight deliberately does NOT block the load: on a
/// benchmarking app, barely-fitting runs are legitimate work (max_memory
/// benchmarks probe exactly that edge), and the active-cell sentinel already
/// turns a kill into an explained cell failure. Instead it warns up front
/// and feeds the measured numbers to the sentinel so a crash report can say
/// how tight the fit actually was.
enum MemoryGate {
    /// Allowance on top of the weights for everything else a benchmark keeps
    /// resident: KV cache, compute buffers, the app + UI itself. True
    /// overhead ranges from ~150 MB (small model, short ctx) to well past
    /// this for 7B-class models at long ctx — a middle value is fine for a
    /// warning, which can afford to fire eagerly because it blocks nothing.
    nonisolated static let headroomBytes: Int64 = 600 * 1024 * 1024

    /// What the pre-flight measured at load time.
    struct Snapshot: nonisolated Sendable {
        let modelBytes: Int64
        let availableBytes: Int64
    }

    /// Measure the model (GGUF file or MLX model directory) against the
    /// process's remaining jetsam budget. nil when either side is
    /// unreadable — the simulator reports a 0 budget, and a missing file has
    /// no size — in which case there is nothing to warn about or record.
    nonisolated static func snapshot(modelPath: String) -> Snapshot? {
        let available = Int64(os_proc_available_memory())
        guard available > 0 else { return nil }
        let size = sizeOfFileOrDirectory(atPath: modelPath)
        guard size > 0 else { return nil }
        return Snapshot(modelBytes: size, availableBytes: available)
    }

    /// Warning when the fit looks tight (weights + headroom exceed the
    /// remaining budget), nil when comfortable. Pure core so tests can
    /// drive both sides.
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

    /// Size of a GGUF file, or the summed contents of an MLX model directory.
    /// 0 when nothing readable is at `path`.
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
