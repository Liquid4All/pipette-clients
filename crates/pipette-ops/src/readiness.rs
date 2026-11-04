//! The readiness wait a runtime seam is handed ([`ReadinessGate`]).
//!
//! See `docs/architecture.md` (“Benchmark vs run vs execute”).

/// The readiness wait a run must pass before it starts a server and before
/// each measured rep, injected by the caller.
///
/// Injected because when to wait is the engine's call, but how long and by
/// what criteria is the run's policy, resolved from the plan's
/// `benchmark_flags`. It also keeps `pipette-readiness` off the engines' and
/// `pipette-ops`' dependency lists.
pub type ReadinessGate<'a> = &'a dyn Fn() -> anyhow::Result<()>;
