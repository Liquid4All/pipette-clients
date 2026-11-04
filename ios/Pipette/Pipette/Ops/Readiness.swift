import Foundation

// The readiness gate: `crates/pipette-ops/src/readiness.rs`. The wait behind it is
// `Readiness/`.

/// The readiness wait a run passes before it loads a model and before each measured rep,
/// injected by the caller: the crate's `ReadinessGate`. Engines decide when to wait; how long
/// and by what criteria is the run's policy, which needs the cell's `benchmark_flags`.
typealias ReadinessGate = () throws -> Void
