import Foundation
import Testing

@testable import Pipette

/// Covers the `settleToFloor` cross-cell contamination gate — the injectable core is
/// driven with scripted footprints (no device, no real timing) so the plateau logic
/// is exercised deterministically. The `MemoryPeakSampler` high-water mark is covered
/// in `LlamaBenchmarkTests`.
struct ProcessMemoryTests {

    /// Scripted-source helper for the injectable `settleToFloor` core.
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
        // Footprint keeps oscillating (never plateaus): the gate must not hang — it
        // returns the last sample once the timeout budget is spent.
        let mb: UInt64 = 1_048_576
        var osc: [UInt64] = []
        for i in 0..<200 { osc.append((i % 2 == 0 ? 900 : 400) * mb) }
        let r = settle(osc, stableSamples: 3, timeoutMs: 1000)
        #expect(r.samples > 3)                 // it kept polling rather than settling early
        #expect(r.floor == 400 * mb || r.floor == 900 * mb)
    }
}
