import Foundation
import Testing

@testable import Pipette

/// What a finished headless run says about where its results went, the counterpart of the
/// CLI's `write_record_done`, so "never offered to the collector" and "submitted" read apart.
@Suite struct ResultReportTests {

    private func report(_ id: String, _ location: BenchmarkResultLocation?,
                        payloadPath: String? = nil) -> ResultReport {
        ResultReport(cellId: CellId(id), benchmarkId: "prefill_throughput_256",
                     location: location, payloadPath: payloadPath)
    }

    /// The location is read from the store, not re-derived: the directory a result sits in
    /// *is* how far it travelled, which is the rule `ResultsStore` is built on.
    @Test func theLocationComesFromWhereTheStoreFiledIt() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let cell = JobCell(cellId: CellId("c-1"), benchmarkId: "prefill_throughput_256",
                           benchmarkType: .prefillThroughput, runStatus: .completed,
                           serverJobId: nil, errorMessage: nil,
                           source: ggufTextFixture("org/a-GGUF", "a-Q4_0.gguf"))
        try ws.results.saveResult(.remotePending, cell.cellId,
                                       payload: Data("{}".utf8), extras: Data("{}".utf8))

        let before = ResultReport(cell: cell, store: ws.results)
        try ws.results.move(cell.cellId, to: .remoteSynced)
        let after = ResultReport(cell: cell, store: ws.results)

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
            + "location=pending payload=/data/results/c-1")
    }

    /// A run that never asked to submit says only where each result was filed.
    @Test func aRunThatDidNotAskToSubmitReportsNoOutcome() {
        let lines = ResultReporter.lines(reports: [report("1", .remotePending)],
                                         submitRequested: false, blocker: nil, errors: [])

        #expect(lines.count == 1)
        #expect(lines[0].hasPrefix("result cell=1"))
    }

    /// A submitting run whose results were never offered, because the client's gate
    /// (registration, then connectivity) stopped it.
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

    /// A local-catalog result is not a failed submission: the crate files it under
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

    /// The crate's `submitted as job <id>`, read from the submission record.
    @Test func aSyncedResultNamesTheJobItWasSubmittedAs() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let cell = JobCell(cellId: CellId("c-1"), benchmarkId: "prefill_throughput_256",
                           benchmarkType: .prefillThroughput, runStatus: .completed,
                           serverJobId: nil, errorMessage: nil,
                           source: ggufTextFixture("org/a-GGUF", "a-Q4_0.gguf"))
        try ws.results.saveResult(.remotePending, cell.cellId,
                                       payload: Data("{}".utf8), extras: Data("{}".utf8))
        #expect(ResultReport(cell: cell, store: ws.results).serverJobId == nil)

        try ws.results.saveSubmission(.submitted(serverJobId: "job-9"), cell.cellId)
        try ws.results.move(cell.cellId, to: .remoteSynced)
        let line = ResultReport(cell: cell, store: ws.results).line

        #expect(line.contains("location=synced"))
        #expect(line.hasSuffix("submitted_as_job=job-9"))
    }

    // MARK: - run_summary_lines

    /// The crate's `run_summary_lines`, field for field: identity first, then the result's
    /// own fields under their wire names at the crate's precision.
    @Test func aVisionSummaryReadsAsTheCratePrintsIt() {
        let lines = runSummaryLines(
            benchmarkId: "vl_throughput_1024x1024_text0_decode64_img1",
            benchmarkType: .vlThroughput,
            data: .vlThroughput(promptTokens: 268, imageTokens: 256,
                                promptMs: 150.905425, promptMsStddev: 7.0410111,
                                predictedMs: 266.4449248, predictedMsStddev: 3.2245632,
                                totalTimeMs: 430.1, totalTimeMsStddev: 8.25))
        #expect(lines == [
            "benchmark: vl_throughput_1024x1024_text0_decode64_img1",
            "type: vl_throughput",
            "prompt_tokens: 268",
            "image_tokens: 256",
            "prompt_ms: 150.905",
            "prompt_ms_stddev: 7.041",
            "predicted_ms: 266.445",
            "predicted_ms_stddev: 3.225",
            "total_time_ms: 430.100",
            "total_time_ms_stddev: 8.250",
        ])
    }

    /// Not printed as null or as `Optional(…)`: the reason this exists rather than
    /// interpolating the enum.
    @Test func anAbsentImageTokenCountIsOmitted() {
        let lines = runSummaryLines(
            benchmarkId: "vl_throughput_smoke", benchmarkType: .vlThroughput,
            data: .vlThroughput(promptTokens: 83, imageTokens: nil, promptMs: 69.3,
                                promptMsStddev: nil, predictedMs: 41, predictedMsStddev: nil))
        #expect(!lines.contains { $0.contains("image_tokens") })
        #expect(!lines.contains { $0.contains("Optional") })
        #expect(!lines.contains { $0.contains("total_time_ms") })
        #expect(lines == ["benchmark: vl_throughput_smoke", "type: vl_throughput",
                          "prompt_tokens: 83", "prompt_ms: 69.300", "predicted_ms: 41.000"])
    }

    /// The type is the definition's, not the payload's. `vl_max_memory` reports through
    /// the variant it shares with the text cell, so a summary taken from the data would
    /// label a vision memory run `max_memory_usage`.
    @Test func aVisionMemoryRunKeepsItsOwnType() {
        let lines = runSummaryLines(
            benchmarkId: "vl_max_memory_512x512_text0_img1", benchmarkType: .vlMaxMemory,
            data: .maxMemoryUsage(maxHostBytes: 1_073_741_824, maxGpuBytes: nil,
                                  maxNpuBytes: nil))
        #expect(lines[1] == "type: vl_max_memory")
        #expect(lines.last == "max_host_bytes: 1073741824")
    }

    /// The crate's `run_summary_prints_what_the_engine_measured`, row for row.
    @Test func theSummaryPrintsWhatTheEngineMeasured() {
        let rows: [(BenchmarkType, BenchmarkResultData, [String])] = [
            (.decodeThroughput, .decodeThroughput(decodeTimeMs: 3, decodeTimeMsStddev: 0.5),
             ["decode_time_ms: 3.000000", "decode_time_ms_stddev: 0.500000"]),
            (.endToEndLatency, .endToEndLatency(totalTimeMs: 940, totalTimeMsStddev: 11),
             ["total_time_ms: 940.000000", "total_time_ms_stddev: 11.000000"]),
            (.endToEndLatency, .endToEndLatency(totalTimeMs: 940, totalTimeMsStddev: nil),
             ["total_time_ms: 940.000000"]),
            (.maxMemoryUsage, .maxMemoryUsage(maxHostBytes: 8, maxGpuBytes: nil, maxNpuBytes: nil),
             ["max_host_bytes: 8"]),
            (.maxMemoryUsage, .maxMemoryUsage(maxHostBytes: 8, maxGpuBytes: 4, maxNpuBytes: 2),
             ["max_host_bytes: 8", "max_gpu_bytes: 4", "max_npu_bytes: 2"]),
            (.asrThroughput, .asrThroughput(promptTokens: 137, audioTokens: 118, promptMs: 83.3,
                                            promptMsStddev: 0.9, predictedMs: 129.4,
                                            predictedMsStddev: nil, totalTimeMs: 250,
                                            totalTimeMsStddev: nil),
             ["asr_prompt_tokens: 137", "asr_audio_tokens: 118", "asr_prompt_ms: 83.300",
              "asr_prompt_ms_stddev: 0.900", "asr_predicted_ms: 129.400",
              "total_time_ms: 250.000"]),
        ]
        for (type, data, metrics) in rows {
            let lines = runSummaryLines(benchmarkId: "b", benchmarkType: type, data: data)
            #expect(Array(lines.dropFirst(2)) == metrics, "\(data)")
        }
    }

    /// The timing kinds print at the crate's six places, and an eval reports its count.
    @Test func theTextKindsPrintTheirOwnFields() {
        #expect(runSummaryLines(benchmarkId: "p", benchmarkType: .prefillThroughput,
                                 data: .prefillThroughput(prefillTimeMs: 12.5,
                                                          prefillTimeMsStddev: nil)).last
            == "prefill_time_ms: 12.500000")
        #expect(runSummaryLines(benchmarkId: "e", benchmarkType: .eval,
                                 data: .eval(completions: [])).last == "completions: 0")
    }

    /// A run prints what each cell measured before where it was filed, as upstream orders
    /// the two.
    @Test func theSummaryPrecedesTheFilingLine() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let cell = JobCell(cellId: CellId("c-2"), benchmarkId: "vl_throughput_smoke",
                           benchmarkType: .vlThroughput, runStatus: .completed,
                           serverJobId: nil, errorMessage: nil,
                           source: ggufTextFixture("org/a-GGUF", "a-Q4_0.gguf"))
        let payload = """
        {"benchmark_id":"vl_throughput_smoke","prompt_tokens":83,"image_tokens":64,\
        "prompt_ms":69.324,"predicted_ms":40.988}
        """.replacingOccurrences(of: "\\\n", with: "")
        try ws.results.saveResult(.local, cell.cellId,
                                       payload: Data(payload.utf8), extras: Data("{}".utf8))

        let lines = ResultReporter.lines(
            reports: [ResultReport(cell: cell, store: ws.results)],
            submitRequested: false, blocker: nil, errors: [])

        #expect(lines.first == "benchmark: vl_throughput_smoke")
        #expect(lines.contains("image_tokens: 64"))
        #expect(try #require(lines.last).hasPrefix("result cell=c-2"))
    }

    /// A payload that cannot be read back still files: the run happened, and the console
    /// is the only place this is going.
    @Test func anUnreadablePayloadStillReportsItsFiling() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let cell = JobCell(cellId: CellId("c-3"), benchmarkId: "prefill_throughput_256",
                           benchmarkType: .prefillThroughput, runStatus: .completed,
                           serverJobId: nil, errorMessage: nil,
                           source: ggufTextFixture("org/a-GGUF", "a-Q4_0.gguf"))
        try ws.results.saveResult(.local, cell.cellId,
                                       payload: Data("not json".utf8), extras: Data("{}".utf8))

        let report = ResultReport(cell: cell, store: ws.results)
        #expect(report.summaryLines.isEmpty)
        #expect(report.line.hasPrefix("result cell=c-3"))
    }
}
