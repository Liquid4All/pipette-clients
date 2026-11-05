import Foundation

/// ONNX Runtime GenAI model lifecycle and memory breadcrumbs; the benchmarks are
/// `OnnxGenaiExecute`. CPU only, so the breadcrumbs carry the footprint alone.
nonisolated enum OnnxGenaiRuntime {
    // MARK: - Milestone logging

    /// For the breadcrumbs; `max_memory_usage` polls `footprintWithMappedWeights`.
    static func footprintMB() -> Double {
        Double(ProcessMemory.physFootprintBytes()) / 1_048_576
    }

    static func logMem(_ label: String) {
        AppLog.jobRun.info(String(format: "onnx %@ footprint=%.0fMB", label, footprintMB()))
    }

    /// The footprint plus file-backed pages made resident since the bracket opened: ORT
    /// reads its 4-bit scales from the mapped external-data file (456 MiB on Phi-3 int4),
    /// and the footprint does not bill clean file-backed pages.
    static func footprintWithMappedWeights() -> () -> UInt64 {
        let base = vmCounters().fileBacked
        return {
            let now = vmCounters()
            return withMappedWeights(footprint: now.footprint, fileBacked: now.fileBacked,
                                     base: base)
        }
    }

    static func withMappedWeights(footprint: UInt64, fileBacked: UInt64, base: UInt64) -> UInt64 {
        footprint + (fileBacked > base ? fileBacked - base : 0)
    }

    /// Read together so the sum describes one instant; zeros if the query fails.
    private static func vmCounters() -> (footprint: UInt64, fileBacked: UInt64) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return (0, 0) }
        return (UInt64(info.phys_footprint), UInt64(info.external))
    }

    // MARK: - Load / unload

    /// Together because the tokenizer is built from the model; building it separately
    /// would load the package twice.
    struct Loaded {
        let model: OGAModel
        let tokenizer: OGATokenizer
    }

    /// Through `OGAConfig` rather than `OGAModel(path:)`: the config is where
    /// execution-provider options would be set.
    static func loadModel(path: String) throws -> Loaded {
        logMem("load.start \(URL(fileURLWithPath: path).lastPathComponent)")
        do {
            let config = try OGAConfig(path: path)
            let model = try OGAModel(config: config)
            logMem("load.afterModel")
            let tokenizer = try OGATokenizer(model: model)
            logMem("load.afterTokenizer")
            return Loaded(model: model, tokenizer: tokenizer)
        } catch {
            // Memoized by `JobExecutor` for the model's later cells.
            throw ModelLoadFailure(
                cause: .engine("ONNX Runtime GenAI model load failed: \(error.localizedDescription)"))
        }
    }

    /// Release is the ObjC objects going out of scope: ORT frees its arenas with the
    /// session, so unlike MLX there is no cache to clear. The `release` breadcrumb shows
    /// whether that held.
    static func withFreshModel<T>(path: String,
                                  _ body: (Loaded) async throws -> T) async throws -> T {
        let loaded = try loadModel(path: path)
        defer { logMem("release") }
        return try await body(loaded)
    }
}
