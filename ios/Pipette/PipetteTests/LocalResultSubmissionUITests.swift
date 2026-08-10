import Foundation
import Testing

@testable import Pipette

/// The two counters that answer "is there anything to submit?" independently of the
/// submit sweep's own gate.
///
/// They exist because the question is asked in three places — the sweep, the job page's
/// "Submit N Results" affordance, and the uploader's stranded check. The sweep already
/// skips a result from the generated catalog half; these pin the other two, since a
/// disagreement is what produced a button that counted work it would never do.
@Suite(.serialized)
struct LocalResultSubmissionUITests {

    private func cell(_ id: String, source: BenchmarkSource?) -> JobCell {
        JobCell(
            cellId: CellId(id), benchmarkId: "prefill_throughput_512",
            benchmarkType: .prefillThroughput,
            runStatus: .completed, serverJobId: nil, errorMessage: nil,
            source: ggufTextFixture("test/model-GGUF", "model.gguf"),
            benchmarkSource: source)
    }

    private func manifest(_ cells: [JobCell], contribute: Bool = true) -> JobManifest {
        JobManifest(
            jobId: "job-1", createdAt: "2026-06-24T00:00:00Z",
            nGpuLayers: 99, contextSize: 4096, cells: cells, status: .completed,
            contributeResults: contribute)
    }

    /// A local result is not stranded — it is finished. Counting it woke the uploader on
    /// every launch and foreground to submit nothing.
    @Test func aLocalResultIsNotStranded() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let local = cell("local-1", source: .local)
        try writePayload(storage: storage, cellId: local.cellId, benchmarkId: local.benchmarkId)

        let uploader = ResultUploader(storage: storage)
        #expect(await uploader.hasStrandedResults(manifest([local])) == false)
    }

    /// A remote one still is — the fix must not silence the case the uploader exists for.
    @Test func aRemoteResultIsStillStranded() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let remote = cell("remote-1", source: .remote)
        try writePayload(storage: storage, cellId: remote.cellId, benchmarkId: remote.benchmarkId)

        let uploader = ResultUploader(storage: storage)
        #expect(await uploader.hasStrandedResults(manifest([remote])))
    }

    /// A cell written before the local half existed carries no source and came from the
    /// synced catalog, so it must keep counting — otherwise an upgrade silently strands
    /// every result already waiting to go up.
    @Test func aCellWithNoRecordedSourceIsStillStranded() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let old = cell("old-1", source: nil)
        try writePayload(storage: storage, cellId: old.cellId, benchmarkId: old.benchmarkId)

        let uploader = ResultUploader(storage: storage)
        #expect(await uploader.hasStrandedResults(manifest([old])))
    }

    /// A mixed job counts only the submittable half — the number the job page puts on
    /// the button has to match what the sweep will actually send.
    @Test func onlySubmittableCellsAreStranded() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let cells = [cell("local-1", source: .local), cell("remote-1", source: .remote)]
        for c in cells {
            try writePayload(storage: storage, cellId: c.cellId, benchmarkId: c.benchmarkId)
        }

        let uploader = ResultUploader(storage: storage)
        #expect(await uploader.hasStrandedResults(manifest(cells)))
        // And the local one alone is not enough to make the job look stranded.
        #expect(await uploader.hasStrandedResults(manifest([cells[0]])) == false)
    }

    // MARK: - Provenance through the headless / deep-link front-ends

    /// A reference carries its own source, as the CLI's `--benchmark` does. Without this
    /// every front-end but the New Job screen built cells that read as submittable, so
    /// `benchmarks run benchmark=<a local ladder id>` bypassed the gate entirely.
    @Test func aSourcedReferenceBuildsACellThatCarriesIt() {
        let model = DiscoveredModel.appleFoundation

        let local = JobCell.pending(
            benchmarkIds: ["local/prefill_throughput_8192"], for: model)
        #expect(local.count == 1)
        #expect(local.first?.benchmarkId == "prefill_throughput_8192")
        #expect(local.first?.isSubmittable == false)

        let explicit = JobCell.pending(
            benchmarkIds: ["remote/prefill_throughput_512"], for: model)
        #expect(explicit.first?.benchmarkId == "prefill_throughput_512")
        #expect(explicit.first?.isSubmittable == true)
    }

    /// A bare id means the synced catalog — the form plans and claims carry, and what
    /// every existing caller passes. Reading it as local would strand real results.
    @Test func aBareReferenceIsRemote() {
        let cells = JobCell.pending(
            benchmarkIds: ["prefill_throughput_512"], for: .appleFoundation)
        #expect(cells.first?.isSubmittable == true)
    }

    /// A reference that names no benchmark is skipped and reported, as before.
    @Test func anUnparseableReferenceIsSkipped() {
        var skipped: [String] = []
        let cells = JobCell.pending(
            benchmarkIds: ["elsewhere/foo", "not_a_benchmark"], for: .appleFoundation
        ) { skipped.append($0) }
        #expect(cells.isEmpty)
        #expect(skipped == ["elsewhere/foo", "not_a_benchmark"])
    }
}
