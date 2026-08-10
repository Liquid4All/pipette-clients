//! The readiness wait a runtime seam is handed ([`ReadinessGate`]).
//!
//! The signature it belongs to —
//! `run(&RunRequest, …, ReadinessGate) -> Result<RunResponse>` — is
//! [`pipette_plan_types::run`], which owns both shapes. What stays here is the
//! injected wait: a function type over the engines, not a shape.
//!
//! See `docs/architecture.md` (“Benchmark vs run vs execute”).

/// The readiness wait a run must pass before it starts a server and before
/// each measured rep, injected by the caller.
///
/// Engines receive it rather than calling the readiness crate themselves: when
/// to wait is theirs to decide, but how long and by what criteria is a policy
/// of the run, and resolving it needs the plan's `benchmark_flags`. Injection
/// also keeps `pipette-readiness` off the engines' dependency lists — and off
/// `pipette-ops`', since this is only a function type.
pub type ReadinessGate<'a> = &'a dyn Fn() -> anyhow::Result<()>;

/// Notified at each end of a measured repetition. The engine reports the
/// event; what the caller makes of it — sampling sensors, timing, nothing — is
/// the caller's business, so an engine needs no telemetry vocabulary and no
/// device dependency of its own.
///
/// An `Err` from either hook is unrecoverable: the caller is saying the run can
/// no longer be recorded, so the engine abandons the cell rather than finishing
/// a measurement the caller has disowned. An observer with nothing to fail on
/// returns `Ok(())`.
///
/// The two reports are positional: the caller pairs them by order, so the
/// *n*-th start belongs to the *n*-th end. A cell that measures no repetitions
/// never calls either, which is how a runner without per-rep bracketing
/// reports no series at all.
pub struct RepObserver<'a> {
    started: Box<dyn Fn() -> anyhow::Result<()> + 'a>,
    finished: Box<dyn Fn() -> anyhow::Result<()> + 'a>,
}

impl<'a> RepObserver<'a> {
    /// Takes the hooks by value so a caller can write them inline; the pair is
    /// built once per run, so the boxing is not on any measured path.
    pub fn new(
        started: impl Fn() -> anyhow::Result<()> + 'a,
        finished: impl Fn() -> anyhow::Result<()> + 'a,
    ) -> Self {
        Self {
            started: Box::new(started),
            finished: Box::new(finished),
        }
    }

    /// A repetition is about to begin. Report it **the moment the readiness
    /// gate clears**, before any of the rep's timed work: a caller sampling
    /// here is recording the rep's entry condition, and a report placed before
    /// the gate — or after the workload starts — records something else. Model
    /// load and warmup happen earlier, outside this point.
    ///
    /// (`pipette-mgmt` `docs/methodology/thermal-telemetry.md`, capture
    /// contract.)
    pub fn rep_started(&self) -> anyhow::Result<()> {
        (self.started)()
    }

    /// The repetition's timed work has completed. Report it immediately, so
    /// the pair brackets the timed region and nothing else.
    pub fn rep_finished(&self) -> anyhow::Result<()> {
        (self.finished)()
    }
}
