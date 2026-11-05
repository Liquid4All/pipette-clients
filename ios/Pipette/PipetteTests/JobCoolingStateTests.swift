import Foundation
import Testing

@testable import Pipette

struct JobCoolingStateTests {

    @Test func coolingCaptionFormatsElapsedAndDeadline() {
        let start = Date(timeIntervalSinceReferenceDate: 1_000)
        let cooling = JobCoolingState(since: start, deadline: 300, targetC: 36)
        #expect(cooling.caption(at: start.addingTimeInterval(20)) == "Cooling 0:20 / 5:00 max")
        #expect(cooling.caption(at: start.addingTimeInterval(125)) == "Cooling 2:05 / 5:00 max")
    }

    @Test func coolingCaptionClampsNegativeElapsed() {
        let start = Date(timeIntervalSinceReferenceDate: 1_000)
        let cooling = JobCoolingState(since: start, deadline: 300, targetC: 36)
        #expect(cooling.caption(at: start.addingTimeInterval(-5)) == "Cooling 0:00 / 5:00 max")
    }

    /// A progress caption built from any deadline must not trap, clamped or not.
    @Test func theCoolingCaptionSurvivesANonFiniteDeadline() {
        let start = Date(timeIntervalSinceReferenceDate: 0)
        for deadline in [Double.infinity, -.infinity, .nan, 1e300] {
            _ = JobCoolingState(since: start, deadline: deadline, targetC: 36).caption(at: start)
        }
    }
}
