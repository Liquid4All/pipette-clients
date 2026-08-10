import Testing
import Foundation
@testable import Pipette

/// Unit tests for the engine-agnostic `BenchmarkMeasurement` core, shared by the
/// llama and MLX runtimes — the rep loop, the population stats, and the timer.
/// Tested once here so the methodology can't silently drift between runtimes.
@Suite struct BenchmarkMeasurementTests {

    /// This used to pin the population form, freezing the divergence it should have caught.
    @Test func statsComputesMeanAndSampleStddev() {
        let (mean, stddev) = BenchmarkMeasurement.stats([10, 20, 30, 40, 50])
        #expect(mean == 30)
        // Σ(x−30)² = 1000, over four degrees of freedom.
        #expect(abs(stddev - 250.0.squareRoot()) < 1e-9)
    }

    /// A single rep has no spread to estimate.
    @Test func statsOfOneSampleReportsNoSpread() {
        let (mean, stddev) = BenchmarkMeasurement.stats([42])
        #expect(mean == 42)
        #expect(stddev == 0)
    }

    @Test func statsEmptyIsZero() {
        let (mean, stddev) = BenchmarkMeasurement.stats([])
        #expect(mean == 0)
        #expect(stddev == 0)
    }

    @Test func measureDiscardsWarmupThenAveragesReps() async throws {
        var calls = 0
        let (mean, _) = try await BenchmarkMeasurement.measure(
            label: "test", runs: 3,
            warmup: { calls += 1 },
            gate: {},
            observer: .ignore,
            body: { calls += 1; return Double(calls) })   // reps observe 2, 3, 4
        #expect(calls == 4)        // 1 warm-up + 3 reps
        #expect(mean == 3.0)       // (2 + 3 + 4) / 3
    }

    // Pins the measurement discipline — warm-up first, then a gate before every
    // measured rep — the invariant that keeps llama and MLX numbers comparable.
    @Test func measureOrdersWarmupThenGatedReps() async throws {
        var order: [String] = []
        _ = try await BenchmarkMeasurement.measure(
            label: "test", runs: 2,
            warmup: { order.append("warmup") },
            gate: { order.append("gate") },
            observer: .ignore,
            body: { order.append("body"); return 1 })
        #expect(order == ["warmup", "gate", "body", "gate", "body"])
    }

    @Test func measureGateAbortsBeforeBody() async {
        struct Stop: Error {}
        var bodyCalls = 0
        await #expect(throws: Stop.self) {
            try await BenchmarkMeasurement.measure(
                label: "test", runs: 5, warmup: {}, gate: { throw Stop() }, observer: .ignore,
                body: { bodyCalls += 1; return 1 })
        }
        #expect(bodyCalls == 0)    // the gate threw before the first measured body
    }

    // Eval runs flat-out (no thermal gate), but stays cancellable: a throwing
    // `beforeSample` (the cancel-only checkpoint the llama/MLX/AFM eval paths pass)
    // must abort the whole run — not turn into a `.failed` completion — and stop
    // before the next sample. This is the guarantee that keeps long evals interruptible.
    @Test func evalSamplesAbortsWhenBeforeSampleThrows() {
        struct Cancel: Error {}
        var completedCount = 0
        #expect(throws: Cancel.self) {
            _ = try evalSamples(
                [EvalSample(id: "a", messages: []), EvalSample(id: "b", messages: [])],
                beforeSample: { index in if index == 1 { throw Cancel() } },
                progress: { _ in },
                complete: { _ in
                    completedCount += 1
                    return EvalGeneration(text: "ok", stopReason: .eos, stopDetail: nil, completionTokens: nil)
                })
        }
        #expect(completedCount == 1)   // first sample ran; cancel aborted before the second
    }

    @Test func timedReturnsNonNegativeDuration() throws {
        let elapsed = try BenchmarkMeasurement.timed { _ = (0 ..< 1000).reduce(0, +) }
        #expect(elapsed >= .zero)
        #expect(elapsed.milliseconds >= 0)
    }

    // The observer reports one start/finish per rep, fired after the gate and after the
    // body respectively — the per-iteration thermal series. The fake sampler reads a phase
    // the gate/body flip, so the captured tokens prove the two hooks land where they should.
    @Test func measureReportsEachRepToTheObserver() async throws {
        let phase = ThermalPhaseBox()
        let series = ThermalSeries(sample: { AppleThermalState(rawValue: phase.current) })
        _ = try await BenchmarkMeasurement.measure(
            label: "test", runs: 2,
            warmup: {},
            gate: { phase.current = "nominal" },
            observer: RepObserver(started: { series.start() }, finished: { series.finish() }),
            body: { phase.current = "fair"; return 1 })
        #expect(series.deviceAppleThermalStateBefore == [.nominal, .nominal])  // after each gate
        #expect(series.deviceAppleThermalStateAfter == [.fair, .fair])         // after each body
    }

    @Test func measureWithoutAnObserverIsANoOp() async throws {
        // `.ignore` → reporting is a silent no-op, so a caller that records no series
        // neither pays for nor crashes on thermal capture.
        let (mean, _) = try await BenchmarkMeasurement.measure(
            label: "test", runs: 2, warmup: {}, gate: {}, observer: .ignore, body: { 1 })
        #expect(mean == 1.0)
    }
}

/// Mutable phase a test's gate/body flip so the injected thermal sampler returns
/// a value that reveals *when* it was read. Reference type: the measurement
/// closures mutate it through the shared reference.
private final class ThermalPhaseBox: @unchecked Sendable {
    var current = "nominal"
}
