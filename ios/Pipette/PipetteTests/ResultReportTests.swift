import Foundation
import Testing

@testable import Pipette

/// What a finished headless run says about where its results went — the counterpart of the
/// CLI's `print_record_done`, whose absence made "never offered to the collector" and
/// "submitted" read identically.
@Suite struct ResultReportTests {

    private func report(_ id: String, _ location: BenchmarkResultLocation?,
                        payloadPath: String? = nil) -> ResultReport {
        ResultReport(cellId: CellId(id), benchmarkId: "prefill_throughput_256",
                     location: location, payloadPath: payloadPath)
    }

    /// The location is read from the store, not re-derived: the directory a result sits in
    /// *is* how far it travelled, which is the rule `ResultsStore` is built on.
    @Test func theLocationComesFromWhereTheStoreFiledIt() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let cell = JobCell(cellId: CellId("c-1"), benchmarkId: "prefill_throughput_256",
                           benchmarkType: .prefillThroughput, runStatus: .completed,
                           serverJobId: nil, errorMessage: nil,
                           source: ggufTextFixture("org/a-GGUF", "a-Q4_0.gguf"))
        try storage.results.saveResult(.remotePending, cell.cellId,
                                       payload: Data("{}".utf8), extras: Data("{}".utf8))

        let before = ResultReport(cell: cell, store: storage.results)
        try storage.results.move(cell.cellId, to: .remoteSynced)
        let after = ResultReport(cell: cell, store: storage.results)

        #expect(before.location == .remotePending)
        #expect(before.payloadPath?.contains("remote/pending") == true)
        #expect(after.location == .remoteSynced)
    }

    /// A cell that produced no payload reports no location rather than guessing one.
    @Test func aCellWithNoResultReportsNone() {
        #expect(report("1", nil).line
            == "result cell=1 benchmark=prefill_throughput_256 location=none")
    }

    @Test func theLineCarriesTheCellAndItsPayload() {
        #expect(report("c-1", .remotePending, payloadPath: "/data/results/c-1").line
            == "result cell=c-1 benchmark=prefill_throughput_256 "
            + "location=remotePending payload=/data/results/c-1")
    }

    /// A run that never asked to submit says only where each result was filed.
    @Test func aRunThatDidNotAskToSubmitReportsNoOutcome() {
        let lines = ResultReporter.lines(reports: [report("1", .remotePending)],
                                         submitRequested: false, blocker: nil, errors: [])

        #expect(lines.count == 1)
        #expect(lines[0].hasPrefix("result cell=1"))
    }

    /// The case that cost an hour: a submitting run whose results were never offered,
    /// because the gate the client keeps — registration, then connectivity — stopped it.
    @Test func aBlockedSubmitNamesTheTermThatStoppedIt() {
        let lines = ResultReporter.lines(reports: [report("1", .remotePending)],
                                         submitRequested: true, blocker: "offline", errors: [])

        #expect(lines.last == "result submit SKIPPED 1 pending: offline")
    }

    @Test func blockerNamesRegistrationBeforeConnectivity() {
        #expect(ResultReporter.submitBlocker(registered: false, online: false)?
            .contains("not registered") == true)
        #expect(ResultReporter.submitBlocker(registered: true, online: false) == "offline")
        #expect(ResultReporter.submitBlocker(registered: true, online: true) == nil)
    }

    /// A local-catalog result is not a failed submission — the crate files it under
    /// `local/` and never offers it either, so a submitting run reports no outcome for it.
    @Test func aLocalOnlyResultIsNotReportedAsAFailedSubmit() {
        let lines = ResultReporter.lines(reports: [report("1", .local)],
                                         submitRequested: true, blocker: nil, errors: [])

        #expect(lines.count == 1)
        #expect(!lines.contains { $0.contains("submit") })
    }

    @Test func acceptedAndRefusedCellsAreBothCounted() {
        let lines = ResultReporter.lines(
            reports: [report("1", .remoteSynced), report("2", .remotePending)],
            submitRequested: true, blocker: nil, errors: ["collector said 503"])

        #expect(lines.contains("result submitted cells=1"))
        #expect(lines.last == "result submit FAILED 1 pending: collector said 503")
    }
}
