import Foundation
import Testing

@testable import Pipette

/// `ProcessMemory`: the `settleToFloor` cross-cell contamination gate (driven with scripted
/// footprints), the `MemoryPeakSampler` high-water mark every `max_memory_usage` cell
/// reports, and `reportedPeak`'s choice between it and the kernel's.
struct ProcessMemoryTests {

    private func settle(_ footprints: [UInt64], stableSamples: Int = 3,
                        timeoutMs: UInt32 = 1000)
        -> (floor: UInt64, drains: Int, samples: Int) {
        var seq = footprints
        var drains = 0
        var samples = 0
        let floor = ProcessMemory.settleToFloor(
            pollIntervalMs: 10, stableSamples: stableSamples, timeoutMs: timeoutMs,
            drain: { drains += 1 },
            sample: {
                samples += 1
                return seq.isEmpty ? (footprints.last ?? 0) : seq.removeFirst()
            },
            sleepMs: { _ in })
        return (floor, drains, samples)
    }

    @Test func settleWaitsForFootprintToStopFalling() {
        // A large model's pages drain down, then plateau at 300 MB. The gate must
        // return the plateau (the clean floor), not the initial contaminated 5 GB.
        let mb: UInt64 = 1_048_576
        let r = settle([5000 * mb, 3000 * mb, 800 * mb, 300 * mb, 300 * mb, 300 * mb, 300 * mb])
        #expect(r.floor == 300 * mb)
        #expect(r.drains == 1)   // caches drained exactly once, up front
    }

    @Test func settleReturnsImmediatelyWhenAlreadyFlat() {
        // Already at a clean floor (footprint never moves): three flat polls settle it.
        let mb: UInt64 = 1_048_576
        let r = settle([250 * mb, 250 * mb, 250 * mb, 250 * mb])
        #expect(r.floor == 250 * mb)
    }

    @Test func settleGivesUpAtTimeoutWhenNeverStable() {
        // Footprint keeps oscillating (never plateaus): the gate must not hang, it
        // returns the last sample once the timeout budget is spent.
        let mb: UInt64 = 1_048_576
        var osc: [UInt64] = []
        for i in 0..<200 { osc.append((i % 2 == 0 ? 900 : 400) * mb) }
        let r = settle(osc, stableSamples: 3, timeoutMs: 1000)
        #expect(r.samples > 3)                 // it kept polling rather than settling early
        #expect(r.floor == 400 * mb || r.floor == 900 * mb)
    }

    // MARK: - Memory peak sampler

    @Test func memoryPeakSamplerCapturesHighWaterMark() {
        let lock = NSLock()
        var scripted: [UInt64] = [100, 500, 300, 900, 200]   // 900 is the high-water mark
        var callCount = 0
        let sampler = MemoryPeakSampler(source: {
            lock.lock(); defer { lock.unlock() }
            callCount += 1
            return scripted.isEmpty ? 200 : scripted.removeFirst()
        }, intervalMs: 1)
        sampler.start()
        // Wait until the poller has actually drained every scripted value, not
        // a fixed interval: a 1ms timer coalesces to much coarser ticks on a
        // loaded CI runner, so a fixed sleep races the high-water value (900).
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            lock.lock(); let drained = scripted.isEmpty; lock.unlock()
            if drained { break }
            Thread.sleep(forTimeInterval: 0.005)
        }
        let peak = sampler.stop()
        #expect(peak >= 900)                  // captured the high-water mark

        // stop() joins the poll thread, so no source() call may happen after it
        // returns. Snapshot the count, wait well past the poll interval, re-read.
        lock.lock(); let afterStop = callCount; lock.unlock()
        Thread.sleep(forTimeInterval: 0.1)
        lock.lock(); let settled = callCount; lock.unlock()
        #expect(settled == afterStop)         // the polling thread was joined
    }

    // MARK: - Reported peak

    @Test func reportedPeakIsSampledOnAPublicBuild() {
        #expect(ProcessMemory.reportedPeak(sampled: 900, resetRc: nil, kernel: nil) == 900)
    }

    @Test func reportedPeakIgnoresTheKernelWhenTheResetFailed() {
        // The interval still runs from an earlier bracket: 5000 is some other model's peak.
        #expect(ProcessMemory.reportedPeak(sampled: 900, resetRc: -1, kernel: 5000) == 900)
    }

    @Test func reportedPeakIsSampledWhenTheKernelReadFailed() {
        #expect(ProcessMemory.reportedPeak(sampled: 900, resetRc: 0, kernel: nil) == 900)
    }

    @Test func reportedPeakKeepsTheSampledPeakWhenTheKernelIsBelowIt() {
        #expect(ProcessMemory.reportedPeak(sampled: 900, resetRc: 0, kernel: 880) == 900)
    }

    @Test func reportedPeakTakesTheKernelPeakWhenAboveTheSampledOne() {
        // A spike shorter than the 20 ms poll, as MLX's prefill is.
        #expect(ProcessMemory.reportedPeak(sampled: 894, resetRc: 0, kernel: 919) == 919)
    }
}
