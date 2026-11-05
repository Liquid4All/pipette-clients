import Foundation
import MLX

/// `max_memory_usage` is the process physical footprint, the counter jetsam kills on,
/// rather than an allocator's high-water mark, so the figure compares across engines.
///
/// Every cell runs in one long-lived process, so a prior model's unreclaimed pages would
/// count toward the next peak. `settleToFloor()` narrows that; only an internal build
/// closes it (`flushReclaimRing`), and on a public build only a process's first cell is
/// exact (`hasRunEngineWork`).
nonisolated enum ProcessMemory {
    private static let engineWorkLock = NSLock()
    nonisolated(unsafe) private static var engineWorkRan = false

    /// Only a relaunch clears it. Why it matters: `MemoryFreshness`.
    static var hasRunEngineWork: Bool {
        engineWorkLock.withLock { engineWorkRan }
    }

    static func markEngineWork() {
        engineWorkLock.withLock { engineWorkRan = true }
    }

    /// 0 if the kernel query fails.
    static func physFootprintBytes() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
    }

    /// A dropped MLX model's buffers go back to MLX's cache, not the OS, so every runtime
    /// drains it: a prior MLX cell would inflate a following llama measurement.
    static func drainCaches() {
        MLX.Memory.clearCache()
        flushReclaimRing()
    }

    /// Drains caches and waits for the footprint to plateau, returning the floor. The peak
    /// is not reduced by it (jetsam counts the absolute figure); it is logged so a run that
    /// never settled back is visible.
    @discardableResult
    static func settleToFloor(pollIntervalMs: UInt32 = 50,
                              stableSamples: Int = 3,
                              timeoutMs: UInt32 = 4000) -> UInt64 {
        settleToFloor(pollIntervalMs: pollIntervalMs, stableSamples: stableSamples,
                      timeoutMs: timeoutMs, drain: drainCaches, sample: physFootprintBytes,
                      sleepMs: { usleep($0 * 1000) })
    }

    /// Injectable for tests.
    static func settleToFloor(pollIntervalMs: UInt32, stableSamples: Int, timeoutMs: UInt32,
                              drain: () -> Void, sample: () -> UInt64,
                              sleepMs: (UInt32) -> Void) -> UInt64 {
        drain()
        // A smaller drop counts as a plateau, so slow trailing reclamation isn't waited out.
        let epsilon: UInt64 = 1 << 20  // 1 MiB
        var last = sample()
        var stable = 0
        var elapsed: UInt32 = 0
        while elapsed < timeoutMs, stable < stableSamples {
            sleepMs(pollIntervalMs)
            elapsed += pollIntervalMs
            let now = sample()
            if last >= now, last - now < epsilon {
                stable += 1
            } else {
                stable = 0
            }
            last = now
        }
        return last
    }

    /// Samples across the whole load and drive, since a blocking C load can only be
    /// observed from another thread. An internal build (`PIPETTE_PRIVATE_MEMORY`) also
    /// reads the kernel's interval high-water mark, which catches a spike shorter than the
    /// poll (MLX's prefill).
    ///
    /// `counter` is the footprint unless a runtime holds memory it does not bill (ONNX's
    /// mapped weights).
    static func maxMemoryBracket(
        label: String, counter: () -> () -> UInt64 = { physFootprintBytes },
        work: () async throws -> Void
    ) async throws -> MaxMemoryMeasurementReading {
        let enter = physFootprintBytes()
        let fresh = !hasRunEngineWork
        let floor = settleToFloor()
        let resetRc = resetKernelFootprintPeak()
        let sampler = MemoryPeakSampler(source: counter())
        sampler.start()
        do {
            try await work()
        } catch {
            // Or the poll thread leaks for the process's life.
            _ = sampler.stop()
            throw error
        }
        let sampled = sampler.stop()
        let kernel = kernelFootprintPeak()
        let peak = reportedPeak(sampled: sampled, resetRc: resetRc, kernel: kernel)
        func mb(_ bytes: UInt64) -> String { String(format: "%.0fMB", Double(bytes) / 1_048_576) }
        // `off`: a public build, which has only the sampler.
        let kernelField = resetRc.map { "\(kernel.map(mb) ?? "na") reset_rc=\($0)" } ?? "off"
        AppLog.memory.info(
            "\(label) max_memory: enter=\(mb(enter)) floor=\(mb(floor)) peak=\(mb(peak)) "
                + "sampled=\(mb(sampled)) kernel=\(kernelField) fresh=\(fresh ? "yes" : "no")")
        if !fresh, !BuildFlavor.hasPrivateMemory {
            AppLog.memory.warning(
                "\(label) max_memory ran after other work in this app process; a build without "
                    + "PIPETTE_PRIVATE_MEMORY cannot flush what that work freed, so the peak can "
                    + "read high. Run each memory benchmark first in a fresh launch.")
        }
        return MaxMemoryMeasurementReading(maxHostBytes: peak, maxGpuBytes: nil, maxNpuBytes: nil)
    }

    /// The kernel peak counts only if its interval was reset for this bracket; otherwise
    /// an earlier bracket's peak would be charged here.
    static func reportedPeak(sampled: UInt64, resetRc: Int32?, kernel: UInt64?) -> UInt64 {
        guard resetRc == 0, let kernel else { return sampled }
        return max(sampled, kernel)
    }

    // Private libproc and mach_vm_reclaim calls (no iOS header), resolved with dlsym and
    // compiled in only on an internal build so a public binary carries no trace of their names.
    // `bitPattern: -2` is RTLD_DEFAULT, a cast macro Swift cannot import.

    /// `proc_reset_footprint_interval(getpid())`: its return code (-1 with errno on a
    /// refused call), -2 if the symbol is missing, nil on a public build.
    private static func resetKernelFootprintPeak() -> Int32? {
        #if PIPETTE_PRIVATE_MEMORY
        typealias Reset = @convention(c) (pid_t) -> Int32
        guard let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2),
                              "proc_reset_footprint_interval") else { return -2 }
        return unsafeBitCast(sym, to: Reset.self)(getpid())
        #else
        return nil
        #endif
    }

    /// `ri_interval_max_phys_footprint` from `proc_pid_rusage(RUSAGE_INFO_V4)`: the
    /// footprint's exact high-water mark since the last reset, nil if unavailable.
    private static func kernelFootprintPeak() -> UInt64? {
        #if PIPETTE_PRIVATE_MEMORY
        typealias PidRusage = @convention(c) (pid_t, Int32, UnsafeMutableRawPointer) -> Int32
        guard let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2),
                              "proc_pid_rusage") else { return nil }
        var info = rusage_info_v4()
        let rc = withUnsafeMutableBytes(of: &info) {
            unsafeBitCast(sym, to: PidRusage.self)(getpid(), RUSAGE_INFO_V4, $0.baseAddress!)
        }
        return rc == 0 ? info.ri_interval_max_phys_footprint : nil
        #else
        return nil
        #endif
    }

    /// Empty libmalloc's deferred-reclaim ring. iOS 27 parks freed blocks there, still
    /// counted in `phys_footprint`, until the kernel trims it, so a cell would otherwise
    /// inherit what earlier work freed (about 300 MB after an ONNX Runtime cell). The
    /// flush's code, -2 if the calls are missing, -3 if the task has no ring, nil on a
    /// public build.
    @discardableResult
    static func flushReclaimRing() -> Int32? {
        #if PIPETTE_PRIVATE_MEMORY
        typealias GetRings = @convention(c) (
            mach_port_t, UnsafeMutableRawPointer, UnsafeMutablePointer<UInt32>) -> kern_return_t
        typealias Capacity = @convention(c) (
            UnsafeMutableRawPointer, UnsafeMutablePointer<UInt32>) -> kern_return_t
        typealias Flush = @convention(c) (UnsafeMutableRawPointer, UInt32) -> kern_return_t
        let rc = { () -> Int32 in
            let rtld = UnsafeMutableRawPointer(bitPattern: -2)
            guard let getRings = dlsym(rtld, "mach_vm_reclaim_get_rings_for_task"),
                  let capacity = dlsym(rtld, "mach_vm_reclaim_ring_capacity"),
                  let flush = dlsym(rtld, "mach_vm_reclaim_ring_flush") else { return -2 }
            var ref: (addr: UInt64, size: UInt64) = (0, 0)
            var count: UInt32 = 1
            let kr = withUnsafeMutableBytes(of: &ref) {
                unsafeBitCast(getRings, to: GetRings.self)(mach_task_self_, $0.baseAddress!, &count)
            }
            guard kr == KERN_SUCCESS else { return kr }
            guard count > 0, let ring = UnsafeMutableRawPointer(bitPattern: UInt(ref.addr)) else {
                return -3
            }
            var entries: UInt32 = 0
            let capacityKr = unsafeBitCast(capacity, to: Capacity.self)(ring, &entries)
            guard capacityKr == KERN_SUCCESS else { return capacityKr }
            return unsafeBitCast(flush, to: Flush.self)(ring, entries)
        }()
        if rc != KERN_SUCCESS { AppLog.memory.warning("reclaim ring flush failed: rc=\(rc)") }
        return rc
        #else
        return nil
        #endif
    }
}

/// On a background thread because the measured work is a blocking C call. Peer of
/// `pipette_memprobe::Sampler`.
nonisolated final class MemoryPeakSampler: @unchecked Sendable {
    private let source: () -> UInt64
    private let intervalMs: UInt32
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private var running = false
    private var thread: Thread?
    /// So `stop()` reads the peak only after the last `source()` call. The wait is
    /// bounded, so a thread that never started falls through on timeout.
    private let finished = DispatchSemaphore(value: 0)

    init(source: @escaping () -> UInt64, intervalMs: UInt32 = 20) {
        self.source = source
        self.intervalMs = intervalMs
    }

    func start() {
        let first = source()
        lock.lock(); peak = first; running = true; lock.unlock()
        let t = Thread { [weak self] in
            guard let self else { return }
            while self.isRunning() {
                self.observe(self.source())
                usleep(self.intervalMs * 1000)
            }
            self.finished.signal()
        }
        t.start()
        thread = t
    }

    func stop() -> UInt64 {
        lock.lock(); running = false; lock.unlock()
        _ = finished.wait(timeout: .now() + .milliseconds(Int(intervalMs) * 4 + 200))
        observe(source())
        lock.lock(); defer { lock.unlock() }
        return peak
    }

    private func observe(_ value: UInt64) {
        lock.lock(); if value > peak { peak = value }; lock.unlock()
    }

    private func isRunning() -> Bool {
        lock.lock(); defer { lock.unlock() }; return running
    }
}
