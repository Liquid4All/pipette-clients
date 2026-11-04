import Foundation
import Testing

@testable import Pipette

/// `pipette-readiness`'s `callbacks.rs` tests, against `Readiness/Callbacks.swift`.
struct ReadinessCallbacksTests {

    /// The line the gate writes, character for character, with the closing phrase derived
    /// from the phase and durations as Rust prints them.
    @Test(arguments: [
        (ReadinessPhase.proceeding, ReadinessPlatform.ios,
         "thermal=state:nominal die:31C(ok) cpu=busy:0.4cores(ok)",
         "readiness: ios thermal=state:nominal die:31C(ok) cpu=busy:0.4cores(ok) → proceeding"),
        (.waiting(pollIn: .milliseconds(1_800)), .ios,
         "thermal=state:fair die:41C(hot) cpu=busy:0.3cores(ok)",
         "readiness: ios thermal=state:fair die:41C(hot) cpu=busy:0.3cores(ok) → waiting 1.8s"),
        (.timedOut, .ios,
         "thermal=state:serious die:45C(hot) cpu=busy:1.2cores(busy)",
         "readiness: ios thermal=state:serious die:45C(hot) cpu=busy:1.2cores(busy) "
             + "→ timed out after 300s"),
        (.proceeding, .other, "no readiness probes on this platform",
         "readiness: other no readiness probes on this platform → proceeding"),
    ] as [(ReadinessPhase, ReadinessPlatform, String, String)])
    func theStatusRendersTheGatesLogLine(phase: ReadinessPhase, platform: ReadinessPlatform,
                                         observed: String, expected: String) {
        let status = ReadinessStatus(phase: phase, platform: platform, observed: observed,
                                     elapsed: .seconds(12), maxWait: .seconds(300))
        #expect(status.description == expected)
    }

    /// Only a held rep is worth a live line; the other two phases are that line's last frame.
    @Test(arguments: [
        (ReadinessPhase.waiting(pollIn: .seconds(3)), true),
        (.proceeding, false),
        (.timedOut, false),
    ] as [(ReadinessPhase, Bool)])
    func onlyWaitingIsWorthDrawing(phase: ReadinessPhase, want: Bool) {
        let status = ReadinessStatus(phase: phase, platform: .ios, observed: "",
                                     elapsed: .zero, maxWait: .seconds(300))
        #expect(status.isWaiting() == want)
    }

    @Test(arguments: [(ReadinessPlatform.ios, "ios"), (.other, "other")]
          as [(ReadinessPlatform, String)])
    func eachGateKeepsTheWordItsLogLineUsed(platform: ReadinessPlatform, expected: String) {
        #expect(platform.asStr() == expected)
        #expect(platform.description == expected)
    }

    /// Replacing one callback leaves the others at their defaults: the untouched cancel still
    /// never cancels.
    @Test func aReplacedCallbackLeavesTheOthersAlone() {
        var slept: [Duration] = []
        let callbacks = ReadinessCallbacks().withSleep { slept.append($0) }

        let flow = callbacks.sleepCancellably(.milliseconds(7))

        #expect(slept == [.milliseconds(7)])
        #expect(flow == .continue)
    }
}
