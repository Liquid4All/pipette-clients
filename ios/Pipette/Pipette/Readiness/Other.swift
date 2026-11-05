import Foundation

// Port of `crates/pipette-readiness/src/other.rs`, the gate for a target with no probes: the
// simulator, which has no die sensor and whose CPU is the host's.

#if targetEnvironment(simulator)
/// Proceeds at once, but still reports: a caller draws one frame per wait, and this is the
/// only record that the gate ran and checked nothing.
nonisolated func waitUntilReady(maxWait: Duration, thermal: ThermalGate,
                                callbacks: ReadinessCallbacks) -> ReadinessOutcome {
    callbacks.report(ReadinessStatus(phase: .proceeding, platform: .other,
                                     observed: "no readiness probes on this platform",
                                     elapsed: .zero, maxWait: maxWait))
    return .ready
}
#endif
