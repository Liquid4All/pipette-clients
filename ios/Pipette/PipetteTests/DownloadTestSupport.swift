import Foundation

@testable import Pipette

/// A wait that ran out before the transfer reached the state its test meant to act on.
struct DownloadWaitTimedOut: Error, CustomStringConvertible {
    let key: String
    let timeoutMs: Int
    let state: DownloadCoordinator.Download.State?

    var description: String {
        let reached = state.map { "\($0)" } ?? "no row"
        return "waited \(timeoutMs)ms for transfer \(key), which was still \(reached)"
    }
}

/// Poll `coordinator.downloads[key]` until `done` holds, so a test acts on a transfer that
/// has reached the state it means to act on rather than racing its start. Awaiting yields
/// the main actor each turn, which is what lets the coordinator's own completion hops run.
///
/// Throws once `timeoutMs` passes with `done` still false. The wait is a precondition, not
/// an assertion, and a caller handed a transfer in the wrong state reports the wrong thing:
/// a pause raised on a row that never started is a pause that does nothing, and the test
/// then waits out the fetch's own deadline to fail minutes later on something else.
@MainActor
func waitUntil(
    _ coordinator: DownloadCoordinator, _ key: String, timeoutMs: Int = 5000,
    _ done: (DownloadCoordinator.Download?) -> Bool
) async throws {
    var waited = 0
    while !done(coordinator.downloads[key]) {
        guard waited < timeoutMs else {
            throw DownloadWaitTimedOut(
                key: key, timeoutMs: timeoutMs, state: coordinator.downloads[key]?.state)
        }
        try await Task.sleep(for: .milliseconds(20))
        waited += 20
    }
}
