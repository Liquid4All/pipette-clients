import Foundation
import Testing

@testable import Pipette

/// The result ladder follows the result's directory, as in `pipette-cli`'s
/// `results/store.rs`: `local` and `remote/pending` are the `local` rung, `remote/synced` is
/// `submitted`, and only `scored` reads an artifact.
struct BenchmarkResultStateTests {
    private let cellId = CellId("cell-1")

    private func store() -> BenchmarkStoreFixture { BenchmarkStoreFixture() }

    /// A pending or failed cell is job state, not an empty result.
    @Test func aCellWithNoPayloadIsNotAResult() {
        let fixture = store()
        defer { fixture.remove() }
        #expect(fixture.results.state(of: cellId) == nil)
    }

    /// The crate's `the_listed_state_follows_location_and_metrics`. Both unsubmitted
    /// locations are one rung: whether a result will go up is the benchmark's provenance, not
    /// how far the result has travelled.
    @Test(arguments: [
        (BenchmarkResultLocation.local, false, BenchmarkResultState.local),
        (.remotePending, true, .local),
        (.remoteSynced, false, .submitted),
        (.remoteSynced, true, .scored),
    ])
    func theStateFollowsLocationAndScore(
        location: BenchmarkResultLocation, scored: Bool, expected: BenchmarkResultState
    ) throws {
        let fixture = store()
        defer { fixture.remove() }
        try fixture.record(cellId, at: location)
        if scored { try fixture.writeMetrics(cellId) }
        #expect(fixture.results.state(of: cellId) == expected)
    }

    /// The crate's `move_result_dir`: the payload has to arrive at a location never written
    /// before, not just the state label change.
    @Test func acceptingAResultMovesIt() throws {
        let fixture = store()
        defer { fixture.remove() }
        try fixture.record(cellId, at: .remotePending)

        try fixture.results.move(cellId, to: .remoteSynced)

        #expect(fixture.results.location(of: cellId) == .remoteSynced)
        #expect(fixture.results.state(of: cellId) == .submitted)
        #expect(fixture.results.loadPayload(cellId) != nil)
    }

    /// Moving something already there, or with no result at all, is a no-op rather than a
    /// failure, since the sweep can re-run over an accepted result.
    @Test func movingToTheCurrentLocationIsANoOp() throws {
        let fixture = store()
        defer { fixture.remove() }
        try fixture.record(cellId, at: .remoteSynced)
        try fixture.results.move(cellId, to: .remoteSynced)
        #expect(fixture.results.location(of: cellId) == .remoteSynced)

        let absent = CellId("cell-2")
        try fixture.results.move(absent, to: .remoteSynced)
        #expect(fixture.results.location(of: absent) == nil)
    }

    /// A local result is never submittable, wherever the sweep looks.
    @Test func onlyANonLocalResultIsSubmittable() throws {
        let fixture = store()
        defer { fixture.remove() }
        try fixture.record(cellId, at: .local)
        #expect(fixture.results.submittableDir(cellId) == nil)

        let pending = CellId("cell-2")
        try fixture.record(pending, at: .remotePending)
        #expect(fixture.results.submittableDir(pending) != nil)
    }

    @Test func theLadderIsOrdered() {
        #expect(BenchmarkResultState.local < .submitted)
        #expect(BenchmarkResultState.submitted < .scored)
    }
}

/// A `ResultsStore` on scratch space, with the two writes these tests need.
private struct BenchmarkStoreFixture {
    let results: ResultsStore

    init() {
        results = ResultsStore(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("PipetteResults-\(UUID().uuidString)", isDirectory: true))
    }

    func record(_ id: CellId, at location: BenchmarkResultLocation) throws {
        try results.saveResult(
            location, id, payload: Data("{}".utf8), extras: Data("{}".utf8))
    }

    func writeMetrics(_ id: CellId) throws {
        guard let path = results.metricsPath(of: id) else { return }
        try Data("{}".utf8).write(to: path)
    }

    func remove() { try? FileManager.default.removeItem(at: results.root) }
}
