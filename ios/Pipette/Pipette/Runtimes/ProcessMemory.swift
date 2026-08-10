import Foundation
import MLX

/// Process-wide memory introspection shared by both benchmark runtimes.
///
/// `max_memory_usage` reports the process **physical footprint**
/// (`task_vm_info.phys_footprint`) — the same counter iOS jetsam kills on — rather
/// than an allocator-specific high-water mark (Metal device alloc for llama.cpp,
/// MLX's own buffer cache for MLX). One runtime-agnostic figure that's both
/// comparable across engines and meaningful for OOM analysis. Each runtime brackets
/// its fresh-load + work with a `MemoryPeakSampler` over `physFootprintBytes` and
/// reports the high-water mark.
///
/// Because iOS runs every benchmark cell in **one long-lived process**, the
/// footprint carries over between cells: a prior (large) model's un-reclaimed pages
/// would be counted as the next model's peak. `settleToFloor()` closes that gap —
/// it drains caches and waits for the footprint to fall back to a clean floor
/// before a measurement starts.
nonisolated enum ProcessMemory {
    /// Current process physical footprint in bytes, or 0 if the kernel query fails.
    /// `task_vm_info.phys_footprint` is the resident + compressed memory iOS
    /// attributes to the process and jetsam-kills on — not an allocator counter.
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

    /// Return freed weight/activation buffers to the OS. Dropping a model — llama
    /// *or* MLX — hands its buffers back to MLX's Metal buffer cache, **not** to the
    /// OS, so `phys_footprint` stays elevated until the cache is drained. Called
    /// runtime-agnostically (the llama path drains it too) because a prior MLX cell
    /// can leave a cache behind that inflates a following llama measurement.
    static func drainCaches() {
        MLX.Memory.clearCache()
    }

    /// Settle the process footprint to a clean floor before a `max_memory_usage`
    /// sample, and return the floor reached.
    ///
    /// iOS runs every cell in one process, so a prior model's not-yet-reclaimed
    /// pages would otherwise be counted as this model's peak (a small model right
    /// after a large one inherits the large one's footprint). This drains caches,
    /// then polls `phys_footprint` until it stops falling (plateaus for
    /// `stableSamples` consecutive polls) or `timeoutMs` elapses.
    ///
    /// The reported figure is **not** reduced by this floor — the absolute peak is
    /// still what jetsam counts. The floor is returned so the caller can log it and
    /// flag a run whose floor never fell back to the harness level (contamination
    /// the platform couldn't reclaim in time).
    @discardableResult
    static func settleToFloor(pollIntervalMs: UInt32 = 50,
                              stableSamples: Int = 3,
                              timeoutMs: UInt32 = 4000) -> UInt64 {
        settleToFloor(pollIntervalMs: pollIntervalMs, stableSamples: stableSamples,
                      timeoutMs: timeoutMs, drain: drainCaches, sample: physFootprintBytes,
                      sleepMs: { usleep($0 * 1000) })
    }

    /// Injectable core of `settleToFloor` — the plateau loop with the cache drain,
    /// footprint source, and sleep passed in so the settling logic is unit-testable
    /// with scripted footprints (no device, no real timing). `sample` is read once
    /// after `drain`, then after every `sleepMs(pollIntervalMs)`; the loop returns
    /// the last sample once it stops falling for `stableSamples` polls or `timeoutMs`
    /// elapses.
    static func settleToFloor(pollIntervalMs: UInt32, stableSamples: Int, timeoutMs: UInt32,
                              drain: () -> Void, sample: () -> UInt64,
                              sleepMs: (UInt32) -> Void) -> UInt64 {
        drain()
        // "Not falling" tolerance: a poll that drops less than this counts as a
        // plateau. Small enough to ignore rounding, large enough not to wait out
        // slow trailing reclamation forever.
        let epsilon: UInt64 = 1 << 20  // 1 MiB
        var last = sample()
        var stable = 0
        var elapsed: UInt32 = 0
        while elapsed < timeoutMs, stable < stableSamples {
            sleepMs(pollIntervalMs)
            elapsed += pollIntervalMs
            let now = sample()
            // Plateau = footprint neither rose nor fell meaningfully since the last
            // poll. A still-falling footprint (reclamation in progress) or a rise
            // resets the streak so we keep waiting for it to settle.
            if last >= now, last - now < epsilon {
                stable += 1
            } else {
                stable = 0
            }
            last = now
        }
        return last
    }

    /// Bracket a fresh model load + drive with the process-footprint peak sampler
    /// and report `max_memory_usage`. Both runtimes call this with their own
    /// load-and-drive closure and a log `label` (`"llama"` / `"mlx"`), so the
    /// sampling, floor settling, logging, and result packaging live in one place —
    /// the two engines stay structural twins here by construction.
    ///
    /// Sampling starts *before* the load so a blocking C load is observed off the
    /// poll thread, and brackets the whole load + drive so the peak captures
    /// resident weights + activations — the true jetsam-relevant high-water mark.
    /// `settleToFloor()` runs first so a small model right after a large one doesn't
    /// inherit un-reclaimed pages as its peak; the floor is logged, not subtracted
    /// (jetsam counts the absolute footprint), so a run that never settled back is
    /// detectable. `gpu`/`npu` bytes are nil: the footprint is one host figure.
    static func maxMemoryBracket(
        label: String, work: () async throws -> Void
    ) async throws -> BenchmarkResult {
        let enter = physFootprintBytes()
        let floor = settleToFloor()
        let sampler = MemoryPeakSampler(source: physFootprintBytes)
        sampler.start()
        do {
            try await work()
        } catch {
            // Stop the poll thread before propagating; a load that throws (e.g. the
            // OOM this benchmark courts) would otherwise leak it for the process life.
            _ = sampler.stop()
            throw error
        }
        let peak = sampler.stop()
        AppLog.memory.info(String(
            format: "%@ max_memory: enter=%.0fMB floor=%.0fMB peak=%.0fMB",
            label, Double(enter) / 1_048_576, Double(floor) / 1_048_576, Double(peak) / 1_048_576))
        return .maxMemoryUsage(hostBytes: peak, gpuBytes: nil, npuBytes: nil)
    }
}

/// Samples a byte counter on a background thread to capture a high-water mark, for
/// `max_memory_usage` (the work between `start()`/`stop()` is a blocking C call, so
/// the peak can't be observed inline). Both runtimes bracket their fresh-load +
/// drive with one of these over `ProcessMemory.physFootprintBytes`. Peer of the Rust
/// `PhysFootprintPoller`.
nonisolated final class MemoryPeakSampler: @unchecked Sendable {
    private let source: () -> UInt64
    private let intervalMs: UInt32
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private var running = false
    private var thread: Thread?
    /// Signalled by the poll loop on exit so `stop()` can wait on it before reading
    /// the peak. On the signalled path this rules out a `source()` call racing past
    /// the return; the wait is bounded, so a thread that never started (no signal)
    /// falls through on timeout instead.
    private let finished = DispatchSemaphore(value: 0)

    init(source: @escaping () -> UInt64, intervalMs: UInt32 = 20) {
        self.source = source
        self.intervalMs = intervalMs
    }

    /// Capture the first sample as the initial peak and begin polling.
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

    /// Stop polling, best-effort wait for the poll thread to exit, take a final
    /// sample, and return the captured peak. On the normal (signalled) path the wait
    /// returns once the background loop has exited (no more `source()` calls) before
    /// we read the high-water mark; the wait is bounded, so on timeout `stop()`
    /// proceeds without that guarantee. A timeout is only expected when the thread
    /// never started (e.g. `stop()` without `start()`).
    func stop() -> UInt64 {
        lock.lock(); running = false; lock.unlock()
        // Bounded wait for the poll loop to exit. The timeout keeps `stop()` from
        // hanging if the thread never started (e.g. `stop()` without `start()`).
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
