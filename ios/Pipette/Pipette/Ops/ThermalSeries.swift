import Foundation

/// The per-repetition thermal series a run collects — the crate's `ThermalSeries`
/// (`pipette-ops/src/thermal_series.rs`): one reading as each rep starts, one as it
/// finishes, paired positionally.
///
/// Owned by `RunCell.dispatch`, which hands the engines a `RepObserver` closing over it;
/// nothing below that sees this type, so an engine carries no telemetry vocabulary.
///
/// Where upstream takes each reading as an argument (`start(reading)`), this samples its
/// own: the probe is two Apple-specific reads that no other caller assembles, so passing
/// them in would only move the same code up a layer. The `sample` closure stays injectable
/// so tests can feed a deterministic sequence instead of the live device state.
///
/// `@unchecked Sendable`: an instance is written only from the observer hooks, which fire
/// inside `measure`'s sequential single-task rep loop, and read only after that run
/// completes (the owner `await`s the run before reading), so there is no concurrent access.
nonisolated final class ThermalSeries: @unchecked Sendable {
    private let sample: () -> AppleThermalState?
    private(set) var deviceAppleThermalStateBefore: [AppleThermalState] = []
    private(set) var deviceAppleThermalStateAfter: [AppleThermalState] = []
    /// SoC die temperature in whole °C, one reading per rep in iteration order.
    /// Collected only on a `PIPETTE_PRIVATE_THERMAL` build (the appends below are
    /// compiled out otherwise, leaving these empty); a `nil` marks a rep whose
    /// read failed. The wire fold and the not-aligned-with-the-state-series
    /// contract live on `BenchmarkSubmissionPayload.deviceAppleSocTempC*`.
    private(set) var deviceAppleSocTempCBefore: [Float?] = []
    private(set) var deviceAppleSocTempCAfter: [Float?] = []

    init(sample: @escaping () -> AppleThermalState? = {
        AppleThermalState(ProcessInfo.processInfo.thermalState)
    }) {
        self.sample = sample
    }

    /// The collected series as the shape a run answers with. The collector keeps parallel
    /// arrays because the SoC one is compiled out on a public build; this pairs them.
    var runThermal: RunThermal {
        RunThermal(
            before: RunThermal.series(states: deviceAppleThermalStateBefore,
                                      socTemps: deviceAppleSocTempCBefore),
            after: RunThermal.series(states: deviceAppleThermalStateAfter,
                                     socTemps: deviceAppleSocTempCAfter))
    }

    /// A repetition has begun — record its entry condition.
    func start() {
        if let value = sample() { deviceAppleThermalStateBefore.append(value) }
        appendSocTemp(to: &deviceAppleSocTempCBefore)
    }

    /// The repetition's timed work is done — record the condition it ended in.
    func finish() {
        if let value = sample() { deviceAppleThermalStateAfter.append(value) }
        appendSocTemp(to: &deviceAppleSocTempCAfter)
    }

    /// Appends the current SoC die temp (°C) to `series`, or nothing on a build
    /// without the private read. A failed read (`socTemp() <= 0`) maps to `nil`,
    /// keeping the `-1` sentinel off the wire.
    ///
    /// Rounded to whole °C, matching the Rust reader's `macos_thermal` — the two
    /// populate one column and must round alike. `socTemp()` itself stays raw, so
    /// the readiness gate keeps its sub-degree reading.
    private func appendSocTemp(to series: inout [Float?]) {
        #if PIPETTE_PRIVATE_THERMAL
        let t = socTemp()
        series.append(t > 0 ? Float(t.rounded()) : nil)
        #endif
    }
}
