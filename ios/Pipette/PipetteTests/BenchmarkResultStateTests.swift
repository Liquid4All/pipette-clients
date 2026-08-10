import Foundation
import Testing

@testable import Pipette

/// The result ladder, which is now what a result's *directory* says — the crate's
/// `results/store.rs:168`, where `local` and `remote/pending` are both the first rung and
/// `remote/synced` is `submitted`.
///
/// It used to be inferred from which artifacts sat beside each other, so a submission
/// record was what made a result "submitted". The location is the record now; these pin
/// the mapping and the one case that still reads an artifact (a score).
struct BenchmarkResultStateTests {
    private let cellId = CellId("cell-1")

    private func store() -> BenchmarkStoreFixture { BenchmarkStoreFixture() }

    /// No payload is not a result — a pending or failed cell is job state, not a result
    /// that happens to be empty.
    @Test func aCellWithNoPayloadIsNotAResult() {
        let fixture = store()
        defer { fixture.remove() }
        #expect(fixture.results.state(of: cellId) == nil)
    }

    /// Both un-submitted locations are the same rung. "Generated here" and "waiting its
    /// turn" differ in whether they *will* go up, which is the benchmark's provenance,
    /// not how far the result has travelled.
    @Test(arguments: [BenchmarkResultLocation.local, .remotePending])
    func anUnsubmittedResultIsRecorded(location: BenchmarkResultLocation) throws {
        let fixture = store()
        defer { fixture.remove() }
        try fixture.record(cellId, at: location)
        #expect(fixture.results.state(of: cellId) == .recorded)
    }

    @Test func aSyncedResultIsSubmitted() throws {
        let fixture = store()
        defer { fixture.remove() }
        try fixture.record(cellId, at: .remoteSynced)
        #expect(fixture.results.state(of: cellId) == .submitted)
    }

    /// A score outranks the location: it can only have come back for something the
    /// collector took, so the metrics file is the more truthful answer.
    @Test func aScoreOutranksTheLocation() throws {
        let fixture = store()
        defer { fixture.remove() }
        try fixture.record(cellId, at: .remoteSynced)
        try fixture.writeMetrics(cellId)
        #expect(fixture.results.state(of: cellId) == .scored)
    }

    /// Accepting a result advances it — the crate's `move_result_dir`. The payload has to
    /// arrive at the new location, not just the state label change.
    @Test func acceptingAResultMovesIt() throws {
        let fixture = store()
        defer { fixture.remove() }
        try fixture.record(cellId, at: .remotePending)

        try fixture.results.move(cellId, to: .remoteSynced)

        #expect(fixture.results.location(of: cellId) == .remoteSynced)
        #expect(fixture.results.state(of: cellId) == .submitted)
        #expect(fixture.results.loadPayload(cellId) != nil)
    }

    /// Moving something that is already there is a no-op, not a failure — the sweep can
    /// re-run over an accepted result.
    @Test func movingToTheCurrentLocationIsANoOp() throws {
        let fixture = store()
        defer { fixture.remove() }
        try fixture.record(cellId, at: .remoteSynced)
        try fixture.results.move(cellId, to: .remoteSynced)
        #expect(fixture.results.location(of: cellId) == .remoteSynced)
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
        #expect(BenchmarkResultState.recorded < .submitted)
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
