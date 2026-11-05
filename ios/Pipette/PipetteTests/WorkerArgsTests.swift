import Foundation
import Testing

@testable import Pipette

/// The claim loop's pacing, from the knobs `pipette worker` names.
struct WorkerArgsTests {

    @Test func theDefaultsAreTheCratesDefaults() {
        let options = WorkerArgs()
        #expect(options.idleSecs == 300)
        #expect(options.idleJitterSecs == 60)
        #expect(options.heartbeatSecs == nil)
        #expect(options.maxJobs == 0)
        #expect(options.skipProfileRefresh == false)
        for _ in 0..<20 {
            let wait = options.idleWaitWithJitter()
            #expect(wait >= 300 && wait <= 360)
        }
    }

    /// Zero jitter is a real setting (a fixture wanting a deterministic cadence).
    @Test func zeroJitterMeansTheBaseWaitExactly() {
        let options = WorkerArgs(idleSecs: 12, idleJitterSecs: 0)
        #expect(options.idleWaitWithJitter() == 12)
    }
}
