import Foundation
import Testing

@testable import Pipette

/// `pipette-readiness`'s `wait.rs` tests, against `Readiness/Wait.swift`. The crate's
/// `a_probe_failure_is_returned_unchanged` has no counterpart: this client's probes cannot
/// fail.
struct ReadinessWaitTests {

    /// Reports `verdicts` in order and records every status and sleep instead of spending it.
    private final class Scripted {
        private var verdicts: [Bool]
        private(set) var polls = 0
        private(set) var statuses: [ReadinessStatus] = []
        private(set) var slept: [Duration] = []
        private var cancelAfterPolls: Int?
        private var cancelAfterChunks: Int?

        init(_ verdicts: [Bool]) {
            self.verdicts = verdicts
        }

        /// Raise the cancel flag once `polls` probes have run. `0` is a caller who cancelled
        /// before the wait started.
        func cancellingAfterPolls(_ polls: Int) -> Scripted {
            cancelAfterPolls = polls
            return self
        }

        /// Raise the cancel flag part-way through a poll's sleep, once `chunks` of it have
        /// been spent.
        func cancellingAfterChunks(_ chunks: Int) -> Scripted {
            cancelAfterChunks = chunks
            return self
        }

        private func probe() -> Probe {
            polls += 1
            // A loop that polls past its script is the bug this reports.
            let verdict = verdicts.isEmpty ? false : verdicts.removeFirst()
            return Probe(ready: verdict, summary: "poll \(polls)")
        }

        private func cancelled() -> Bool {
            (cancelAfterPolls.map { polls >= $0 } ?? false)
                || (cancelAfterChunks.map { slept.count >= $0 } ?? false)
        }

        /// The phase a wait ended in, never `waiting`.
        func finalPhase() -> ReadinessPhase? {
            statuses.last?.phase
        }

        /// Every status but the last, which is what a held poll reports.
        func heldPolls() -> [ReadinessStatus] {
            Array(statuses.dropLast())
        }

        func run(maxWait: Duration) -> ReadinessOutcome {
            let callbacks = ReadinessCallbacks()
                .withStatus { self.statuses.append($0) }
                .withCancel { self.cancelled() }
                .withSleep { self.slept.append($0) }
            return waitLoop(platform: .other, maxWait: maxWait, callbacks: callbacks) {
                self.probe()
            }
        }
    }

    @Test func aReadyHostProceedsOnTheFirstPoll() {
        let script = Scripted([true])
        #expect(script.run(maxWait: .seconds(300)) == .ready)

        #expect(script.polls == 1)
        #expect(script.finalPhase() == .proceeding)
        #expect(script.heldPolls().isEmpty)
        #expect(script.slept.isEmpty)
    }

    /// One `waiting` per held poll and a terminal phase, spent in cancel-checkable pieces.
    @Test func aCoolingHostReportsEveryHeldPollThenProceeds() {
        let script = Scripted([false, false, true])
        #expect(script.run(maxWait: .seconds(300)) == .ready)

        let held = script.heldPolls()
        #expect(held.count == 2)
        #expect(held.allSatisfy { $0.isWaiting() })
        #expect(script.finalPhase() == .proceeding)
        #expect(script.slept.reduce(.zero, +) <= sensorPollInterval * 2)
        #expect(script.slept.allSatisfy { $0 <= cancelCheckInterval })
        let chunksPerInterval = Int((sensorPollInterval / cancelCheckInterval).rounded(.up))
        #expect(script.slept.count == 2 * chunksPerInterval)
    }

    @Test func aHostThatNeverClearsTimesOutWithItsLastReading() {
        let script = Scripted([false])
        // Zero budget: the first poll is also the last.
        #expect(script.run(maxWait: .zero) == .timedOut(maxWait: .zero, observed: "poll 1"))
        #expect(script.finalPhase() == .timedOut)
    }

    /// A caller who has already given up pays for nothing, not even a 1 s CPU sample.
    @Test func aCancelBeforeTheFirstPollNeverProbes() {
        let script = Scripted([true]).cancellingAfterPolls(0)
        #expect(script.run(maxWait: .seconds(300)) == .cancelled)
        #expect(script.polls == 0)
        #expect(script.statuses.isEmpty)
    }

    @Test func aCancelMidSleepIsHonoredWithinOneChunk() {
        let script = Scripted([false, true]).cancellingAfterChunks(1)
        #expect(script.run(maxWait: .seconds(300)) == .cancelled)
        #expect(script.polls == 1)
        #expect(script.slept.count == 1)
    }

    /// The held poll reports the remainder it actually slept.
    @Test func aSlowProbeShortensTheSleepInsteadOfExtendingTheInterval() throws {
        let spent = Duration.milliseconds(200)
        var slept: [Duration] = []
        var statuses: [ReadinessStatus] = []
        var polls = 0
        let callbacks = ReadinessCallbacks()
            .withStatus { statuses.append($0) }
            .withSleep { slept.append($0) }

        let outcome = waitLoop(platform: .other, maxWait: .seconds(300), callbacks: callbacks) {
            polls += 1
            Thread.sleep(forTimeInterval: spent / .seconds(1))
            return Probe(ready: polls > 1, summary: "slow")
        }

        #expect(outcome == .ready)
        let total = slept.reduce(.zero, +)
        #expect(total <= sensorPollInterval - spent)
        #expect(try #require(statuses.first).phase == .waiting(pollIn: total))
    }
}
