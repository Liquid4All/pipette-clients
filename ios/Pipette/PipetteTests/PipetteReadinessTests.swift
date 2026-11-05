import Foundation
import Testing

@testable import Pipette

/// `pipette-readiness`'s `lib.rs` tests, against `PipetteReadiness.swift`.
struct PipetteReadinessTests {

    /// A literal, not `defaultMaxWait`, so retuning the shared deadline has to fail here and
    /// be looked at. The number rides the wire as `benchmark_flags.readiness.max_wait_secs`,
    /// so a change here is a change to what a submitted result claims it waited for.
    @Test func sharedDefaultDeadlineIsFiveMinutes() {
        #expect(defaultMaxWait == .seconds(300))
        #expect(ResolvedReadiness().maxWaitSecs() == 300)
    }

    @Test(arguments: [
        ("", false), ("   ", false), ("0", false), ("  0  ", false), ("false", false),
        ("False", false), ("FALSE", false), ("f", false), ("no", false), ("NO", false),
        ("n", false), ("off", false), ("OFF", false),
        ("1", true), ("true", true), ("TRUE", true), ("t", true), ("yes", true), ("y", true),
        ("on", true),
    ] as [(String, Bool)])
    func onlyExplicitValuesSkipTheThermalGate(raw: String, wantSkip: Bool) {
        #expect(skipThermalFromStr(raw) == wantSkip)
    }

    /// Every channel in takes an arbitrary `u64` of seconds, and the progress UI rounds the
    /// deadline into an `Int`, so it has to be clamped.
    @Test func theCeilingKeepsDeadlineArithmeticRepresentable() {
        // `Duration.seconds(UInt64.max)` itself traps; `fromSecs` is the crate's
        // non-trapping `Duration::from_secs`.
        #expect(Duration.fromSecs(.max) > maxWaitCeiling)
        let absurd = resolveReadiness(maxWait: .fromSecs(.max), thermal: .unset)
        #expect(absurd.maxWait == maxWaitCeiling)
        #expect(absurd.maxWaitSecs() == 24 * 60 * 60)
        let cooling = JobCoolingState(since: Date(timeIntervalSinceReferenceDate: 0),
                                      deadline: Double(absurd.maxWaitSecs()), targetC: 36)
        #expect(cooling.caption(at: Date(timeIntervalSinceReferenceDate: 0))
            == "Cooling 0:00 / 1440:00 max")
    }

    /// A fanless host genuinely needs minutes, and the ceiling itself is inclusive.
    @Test(arguments: [1, 300, 420, 3600, 24 * 60 * 60] as [UInt64])
    func clampLeavesReasonableDeadlinesAlone(secs: UInt64) {
        #expect(clampMaxWait(.seconds(secs)) == .seconds(secs))
    }

    /// A cell's `max_wait_secs` can be any `u64`, and it meets the same ceiling.
    @Test func aCellOverrideMeetsTheCeilingToo() {
        let gate = resolveReadiness(maxWait: .fromSecs(.max), thermal: .skip)
        #expect(gate.maxWait == maxWaitCeiling)
        #expect(gate.skipThermal())
    }

    /// Swift's enum has no `Default`; the gate with nothing requested is `ResolvedReadiness()`,
    /// built from `unset`.
    @Test func thermalGateDefaultsToNoOpinionThatResolvesToEnforcing() {
        #expect(!ThermalGate.unset.skipped())
        #expect(ThermalGate.skip.skipped())
        #expect(ThermalGate.unset.resolve(envAsksSkip: false) == .enforce)
        #expect(ResolvedReadiness().thermal == .enforce)
    }

    @Test(arguments: [
        (ThermalGate.skip, false, ThermalGate.skip),
        (.skip, true, .skip),
        (.enforce, false, .enforce),
        (.enforce, true, .enforce),
        (.unset, false, .enforce),
        (.unset, true, .skip),
    ] as [(ThermalGate, Bool, ThermalGate)])
    func resolveLetsAnExplicitArgumentBeatTheEnvironment(
        requested: ThermalGate, envAsksSkip: Bool, want: ThermalGate
    ) {
        #expect(requested.resolve(envAsksSkip: envAsksSkip) == want)
    }

    /// Every platform treats "not skip" as enforcing, so an unresolved value would work by
    /// accident.
    @Test(arguments: [ThermalGate.unset, .enforce, .skip])
    func resolveAlwaysProducesADecidedGate(requested: ThermalGate) {
        #expect(requested.resolve(envAsksSkip: false) != .unset)
        #expect(requested.resolve(envAsksSkip: true) != .unset)
    }
}
