import Foundation
import Testing

@testable import Pipette

/// The attempt limit, its clock and its watchdog: the crate's `attempt.rs` and
/// `attempt_watchdog.rs` tests.
struct AttemptTests {
    @Test func anAttemptOverrunsOnlyPastTheLimit() {
        let start = ContinuousClock.now
        let clock = AttemptClock(limit: .seconds(300), at: start)
        #expect(clock.overrun(at: start + .seconds(300)) == nil)
        #expect(clock.overrun(at: start + .seconds(301)) == Attempt.Overrun(
            what: Attempt.loadAndWarmUp, limit: .seconds(300), elapsed: .seconds(301)))
    }

    @Test func aNewAttemptGetsTheWholeLimitAgain() throws {
        let start = ContinuousClock.now
        let clock = AttemptClock(limit: .seconds(300), at: start)
        try clock.begin("rep 1", at: start + .seconds(290))
        #expect(clock.overrun(at: start + .seconds(580)) == nil)
        #expect(clock.overrun(at: start + .seconds(591))?.what == "rep 1")
    }

    @Test func aHeldClockNeverOverruns() {
        let start = ContinuousClock.now
        let clock = AttemptClock(limit: .seconds(300), at: start)
        clock.hold()
        #expect(clock.overrun(at: start + .seconds(10_000)) == nil)
    }

    /// A caught overrun fails every boundary after it, though a `begin` would restart the
    /// attempt that ran over; the first one caught is the one reported, with how long its
    /// attempt has run by the boundary.
    @Test func aCaughtOverrunFailsTheNextBoundary() throws {
        let start = ContinuousClock.now
        let clock = AttemptClock(limit: .seconds(300), at: start)
        try clock.begin("rep 1", at: start)
        clock.keep(Attempt.Overrun(what: "rep 1", limit: .seconds(300), elapsed: .seconds(301)),
                   at: start + .seconds(301))
        clock.keep(Attempt.Overrun(what: "rep 2", limit: .seconds(300), elapsed: .seconds(1)),
                   at: start + .seconds(302))

        let failed = AttemptTimeoutError(
            overrun: Attempt.Overrun(what: "rep 1", limit: .seconds(300), elapsed: .seconds(900)),
            engineError: nil)
        #expect(throws: failed) { try clock.check(at: start + .seconds(900)) }
        #expect(throws: failed) {
            try clock.begin(BenchmarkMeasurement.peakMemoryMeasured, at: start + .seconds(900))
        }
        let afterRep = #expect(throws: AttemptTimeoutError.self) { try clock.beginAfterRep() }
        #expect(afterRep?.overrun.what == "rep 1")
    }

    /// The crate's wording, so the line `pipette-ios-proxy` files reads as the CLI's. The
    /// proxy finds an overrun by it: its `ATTEMPT_TIMED_OUT` fixture holds the same message.
    @Test func theFailureNamesTheAttemptAndTheLimit() {
        let overrun = Attempt.Overrun(what: "rep 2", limit: .seconds(300),
                                      elapsed: .milliseconds(300_400))
        #expect(AttemptTimeoutError(overrun: overrun, engineError: "decode failed")
            .localizedDescription == "rep 2 ran 300.4s, past attempt_timeout_seconds (300s), and "
            + "was stopped; the engine then reported: decode failed")
        #expect(AttemptTimeoutError(overrun: overrun, engineError: nil).localizedDescription
            == "rep 2 ran 300.4s, past attempt_timeout_seconds (300s), and was stopped")
    }

    @Test func onlyTheTimingAndMemoryKindsDefault() {
        #expect(Attempt.defaultLimit(for: .decodeThroughput) == .seconds(300))
        #expect(Attempt.defaultLimit(for: .vlMaxMemory) == .seconds(300))
        #expect(Attempt.defaultLimit(for: .vlSpecDecode) == nil)
        #expect(Attempt.defaultLimit(for: .eval) == nil)
    }

    /// The cell's flags, else the default for its kind.
    @Test(arguments: [
        (nil, Duration.seconds(300)),
        (.prefillThroughputLlamacppIosPipetteGgufText(readiness: nil), .seconds(300)),
        (.prefillThroughputLlamacppIosPipetteGgufText(attemptTimeoutSeconds: 900, readiness: nil),
         .seconds(900)),
    ] as [(BenchmarkFlags?, Duration)])
    func theLimitComesFromTheCellsFlags(flags: BenchmarkFlags?, expected: Duration) throws {
        let model = try ggufTextSpec("org/g-GGUF", "g-Q4_0.gguf")
        let request = RunRequest(
            runtime: .alreadyBound(Runtime.thisBuild(for: model)), model: .alreadyBound(model),
            runtimeFlags: nil, benchmarkFlags: flags,
            benchmark: .prefillThroughput(PrefillThroughputBenchmark(
                benchmarkId: BenchmarkId(validated: "prefill_throughput_256"),
                parameterPrefillTokens: 256)))
        #expect(Attempt.limit(request) == expected)
    }

    /// Final for every client: another device would run the cell under the same limit.
    @Test func anOverrunIsNotRetriable() {
        let overrun = Attempt.Overrun(what: "rep 1", limit: .seconds(300), elapsed: .seconds(301))
        #expect(!classifyRunError(AttemptTimeoutError(overrun: overrun, engineError: nil)))
    }

    /// The resident app cannot stop the engine: the watchdog keeps the overrun, the next
    /// boundary fails the cell, and the run fails over whatever the engine said.
    @Test func anOverrunFailsTheNextBoundaryAndTheRun() async throws {
        let clock = AttemptClock(limit: .milliseconds(100))
        let watchdog = AttemptWatchdog(watching: clock, abandon: nil)
        try clock.beginRep()
        try await Task.sleep(for: .seconds(1))

        let boundary = #expect(throws: AttemptTimeoutError.self) { try clock.beginAfterRep() }
        #expect(boundary?.overrun.what == "rep 1")
        #expect(boundary?.overrun.limit == .milliseconds(100))
        // The rep's whole second, not the poll's reading at about 100 ms.
        #expect(boundary.map { $0.overrun.elapsed >= .seconds(1) } == true)
        let run = #expect(throws: AttemptTimeoutError.self) {
            try watchdog.finish(after: RuntimeError.engine("decode failed"))
        }
        #expect(run?.overrun.what == "rep 1")
        #expect(run?.engineError == RuntimeError.engine("decode failed").localizedDescription)
    }

    /// A boundary's own failure comes back from the engine as is, not as the engine's error.
    @Test func aBoundarysFailureIsTheRunsAsIs() async throws {
        let clock = AttemptClock(limit: .milliseconds(100))
        let watchdog = AttemptWatchdog(watching: clock, abandon: nil)
        try await Task.sleep(for: .seconds(1))
        let caught = #expect(throws: AttemptTimeoutError.self) { try clock.beginRep() }
        let boundary = try #require(caught)

        #expect(throws: boundary) { try watchdog.finish(after: boundary) }
    }

    /// An attempt past the limit when the engine returns counts, though no poll caught it.
    @Test func anAttemptPastTheLimitAtTheEndFailsTheRun() {
        let clock = AttemptClock(limit: .seconds(300), at: .now - .seconds(301))
        let watchdog = AttemptWatchdog(watching: clock, abandon: nil)
        let run = #expect(throws: AttemptTimeoutError.self) { try watchdog.finish(after: nil) }
        #expect(run?.overrun.what == Attempt.loadAndWarmUp)
    }

    @Test func aRunInsideTheLimitReportsNoOverrun() throws {
        let clock = AttemptClock(limit: .seconds(300))
        let watchdog = AttemptWatchdog(watching: clock, abandon: nil)
        try clock.beginRep()
        try watchdog.finish(after: nil)
    }
}
