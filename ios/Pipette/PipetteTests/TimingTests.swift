import Foundation
import Testing

@testable import Pipette

/// `Client/Worker/Timing.swift`, mirroring `pipette-cli`'s `client/worker/timing.rs` tests.
struct TimingTests {
    @Test(arguments: [
        ("PT10M", 600.0),
        ("PT5M", 300),
        ("PT1H2M3S", 3723),
        ("PT30S", 30),
        ("pt10m", 600),
    ])
    func parseIso8601DurationOk(_ raw: String, _ seconds: TimeInterval) {
        #expect(parseIso8601Duration(raw) == seconds)
    }

    /// The crate's refusals, plus a mixed-case prefix and non-ASCII digits that
    /// `Character.isNumber` accepts.
    @Test(arguments: ["", "10M", "PT10", "PTfoo", "PT", "Pt10M", "pT10M", "PT\u{663}M", "PT\u{BD}M"])
    func parseIso8601DurationErr(_ raw: String) {
        #expect(parseIso8601Duration(raw) == nil)
    }

    @Test func resolveHeartbeatIntervalDefaultAndOverride() {
        #expect(resolveHeartbeatInterval(timeWindow: "PT10M") == 300)
        // An unparseable window takes the 10-minute default, halved.
        #expect(resolveHeartbeatInterval(timeWindow: "bogus") == 300)
        #expect(resolveHeartbeatInterval(timeWindow: "PT10M", overrideSecs: 60) == 60)
        // Floor at 1 s.
        #expect(resolveHeartbeatInterval(timeWindow: "PT10M", overrideSecs: 0) == 1)
        // An override longer than half the window is still accepted; the worker warns.
        #expect(resolveHeartbeatInterval(timeWindow: "PT10M", overrideSecs: 900) == 900)
    }

    /// The protocol default is floored only when the half is zero, as the crate's
    /// `Duration` is: a one-second window heartbeats every half second.
    @Test func theHalfWindowIsFlooredOnlyWhenItIsZero() {
        #expect(resolveHeartbeatInterval(timeWindow: "PT1S") == 0.5)
        #expect(resolveHeartbeatInterval(timeWindow: "PT0S") == 1)
    }

    /// The driver stops at the first answer, having slept the schedule once per try that
    /// gave none, and counts tries from one so the first wait is `first`.
    @Test func retryWithBackoffStopsAtTheFirstAnswer() async {
        var tries: [Int] = []
        var sleeps: [TimeInterval] = []
        let answer = await retryWithBackoff(
            attempts: 1...32, first: 1, cap: 30,
            attempt: { n in
                tries.append(n)
                return n < 3 ? nil : "accepted"
            },
            sleep: { sleeps.append($0) })
        #expect(answer == "accepted")
        #expect(tries == [1, 2, 3])
        #expect(sleeps == [1, 2])
    }

    /// Running out of attempts is `nil`, not an answer: the caller decides what giving up
    /// means.
    @Test func retryWithBackoffReportsExhaustionAsNil() async {
        var sleeps = 0
        let answer: String? = await retryWithBackoff(
            attempts: 1...3, first: 1, cap: 30,
            attempt: { _ in nil },
            sleep: { _ in sleeps += 1 })
        #expect(answer == nil)
        #expect(sleeps == 3, "one wait per try that gave no answer")
    }

    @Test func idleWaitStaysWithinJitterBounds() {
        for _ in 0..<20 {
            let wait = idleWaitWithJitter(base: 300, jitter: 60)
            #expect(wait >= 300 && wait <= 360)
        }
        #expect(idleWaitWithJitter(base: 300, jitter: 0) == 300)
    }
}
