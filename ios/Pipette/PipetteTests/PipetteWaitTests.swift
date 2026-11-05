import Foundation
import Testing

@testable import Pipette

struct PipetteWaitTests {

    /// Stands in for the log line a real status writes, so the default narration is
    /// observable.
    private nonisolated final class Spoken: WaitStatus {
        var said = 0

        func reportDefault() {
            said += 1
        }
    }

    @Test func aStatusNarratesItselfWhenNoReporterIsInstalled() {
        let status = Spoken()
        WaitCallbacks<Spoken>().report(status)
        #expect(status.said == 1)
    }

    /// A reporter replaces the default rather than joining it: two narrators would fight for
    /// the line.
    @Test func anInstalledReporterTakesOverTheNarration() {
        let status = Spoken()
        var seen = 0
        WaitCallbacks<Spoken>().withStatus { _ in seen += 1 }.report(status)
        #expect(seen == 1)
        #expect(status.said == 0)
    }

    /// The last piece is the remainder rather than a whole interval.
    @Test(arguments: [
        (Duration.milliseconds(250), [Duration.milliseconds(250)]),
        (cancelCheckInterval, [cancelCheckInterval]),
        (.milliseconds(2_500), [cancelCheckInterval, cancelCheckInterval, .milliseconds(500)]),
    ] as [(Duration, [Duration])])
    func aWaitIsSpentInPiecesNoLongerThanTheCheckInterval(total: Duration, want: [Duration]) {
        var slept: [Duration] = []
        let flow = sleepCancellably(total, cancelled: { false }, sleep: { slept.append($0) })
        #expect(flow == .continue)
        #expect(slept == want)
    }

    @Test func aCancelMidWaitLeavesTheRemainderUnspent() {
        var slept = 0
        let flow = sleepCancellably(cancelCheckInterval * 10, cancelled: { slept >= 2 },
                                    sleep: { _ in slept += 1 })
        #expect(flow == .break)
        #expect(slept == 2)
    }

    @Test func aWaitThatIsAlreadyCancelledSpendsNothing() {
        var slept = false
        let flow = sleepCancellably(.seconds(3_600), cancelled: { true }, sleep: { _ in slept = true })
        #expect(flow == .break)
        #expect(!slept)
    }

    @Test(arguments: [
        (Duration.seconds(3), Duration.seconds(60), Duration.seconds(3)),
        (.seconds(3), .seconds(1), .seconds(1)),
        (.seconds(3), .zero, .zero),
    ] as [(Duration, Duration, Duration)])
    func aGapIsClampedToWhatIsLeftOfTheBudget(interval: Duration, left: Duration,
                                              want: Duration) {
        let clamped = clampToDeadline(interval, ContinuousClock.now.advanced(by: left))
        // The clock moves between reads: a hair under `want` is fine, over is not.
        #expect(clamped <= want)
        #expect(want - clamped < .milliseconds(50))
    }

    /// A poll whose probe outran its interval computes a zero top-up, and must not be the one
    /// poll in a wait that cannot be cancelled.
    @Test(arguments: [(false, ControlFlow.continue), (true, .break)] as [(Bool, ControlFlow)])
    func aZeroWaitStillAnswersTheCancel(cancelled: Bool, want: ControlFlow) {
        var slept = false
        let flow = sleepCancellably(.zero, cancelled: { cancelled }, sleep: { _ in slept = true })
        #expect(flow == want)
        #expect(!slept)
    }
}
