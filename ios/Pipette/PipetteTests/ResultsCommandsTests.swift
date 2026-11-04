import Foundation
import Testing

@testable import Pipette

/// `results list|show|delete`, mirroring `pipette-cli`'s `commands/results.rs`.
@MainActor
struct ResultsCommandsTests {
    private func seed(
        _ ws: PipetteWorkspace, _ id: CellId, at location: BenchmarkResultLocation,
        runtimeDescriptor: String? = nil, submittedAt: String = "2026-03-26T00:00:00Z"
    ) throws {
        var payload: [String: Any] = [
            "benchmark_id": "prefill_throughput_smoke", "submitted_at": submittedAt,
        ]
        if let runtimeDescriptor { payload["runtime_descriptor"] = runtimeDescriptor }
        try ws.results.saveResult(
            location, id, payload: JSONSerialization.data(withJSONObject: payload),
            extras: Data(#"{"command":[],"stderr":"","stdout":""}"#.utf8))
    }

    private func saveManifest(_ ws: PipetteWorkspace, cells ids: [CellId]) {
        ws.saveJobManifest(JobManifest(
            jobId: "job-1", createdAt: "2026-03-26T00:00:00Z", nGpuLayers: 99, contextSize: 4096,
            cells: ids.map { cell(id: $0, benchmarkId: "prefill_throughput_smoke") },
            status: .completed))
    }

    private func descriptor(_ type: RuntimeType) throws -> String {
        String(decoding: try JSONEncoder().encode(try #require(Runtime.thisBuild(for: type))),
               as: UTF8.self)
    }

    // MARK: - list

    /// The RUNTIME column is the descriptor's `cli_ref`, not the engine's friendly name.
    @Test func listRendersRuntimeAsACompactRef() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seed(ws, "c-1", at: .local, runtimeDescriptor: try descriptor(.llamacppIosPipette))
        saveManifest(ws, cells: ["c-1"])

        let row = try #require(ResultsCommands.allRows(ws: ws).first)
        #expect(row.resultId.description == "job-1/c-1")
        #expect(row.runtime == "\(LlamaPins.main.submissionVersion):ios-arm64")
        #expect(row.benchmark == "prefill_throughput_smoke")
        #expect(row.createdAt == "2026-03-26T00:00:00Z")
    }

    @Test func runtimeColumnFallsBackToTheRawDescriptor() {
        #expect(ResultsCommands.runtimeColumn("{not a runtime}") == "{not a runtime}")
        #expect(ResultsCommands.runtimeColumn(nil) == "")
    }

    /// MLX spells its build as the pinned package stack, and Apple Foundation as its type,
    /// as `Runtime::cli_ref` does for both.
    @Test func theOtherRuntimesRenderAsTheCrateSpellsThem() throws {
        let packages = "mlx-swift=\(MLXBuildInfo.mlxSwiftRepositoryUrl)@\(MLXBuildInfo.mlxSwift.submissionVersion) "
            + "mlx-swift-lm=\(MLXBuildInfo.mlxSwiftLMRepositoryUrl)@\(MLXBuildInfo.mlxSwiftLM.submissionVersion) "
            + "swift-transformers=\(MLXBuildInfo.swiftTransformersRepositoryUrl)@\(MLXBuildInfo.swiftTransformers.submissionVersion)"
        #expect(ResultsCommands.runtimeColumn(try descriptor(.mlxIosPipette)) == "\(packages):ios-arm64")
        #expect(ResultsCommands.runtimeColumn(try descriptor(.appleFoundation)) == "apple_foundation")
    }

    /// A result that never went up lists as `local`, the word `pipette` prints, and a
    /// `state=scored` filter lists nothing for it.
    @Test func listFiltersByState() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seed(ws, "c-1", at: .remotePending)
        saveManifest(ws, cells: ["c-1"])

        #expect(ResultsCommands.rows(benchmark: nil, benchmarkType: nil, state: .scored, limit: nil,
                                         ws: ws).isEmpty)
        #expect(ResultsCommands.allRows(ws: ws).map(\.state.rawValue) == ["local"])
    }

    /// A payload from before the descriptor and timestamp fields still lists: no
    /// `runtime_descriptor`, or an empty one, is a blank column, and a missing
    /// `submitted_at` sorts as the epoch.
    @Test(arguments: [nil, ""])
    func aLegacyPayloadListsWithDefaults(runtimeDescriptor: String?) throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        var payload: [String: Any] = ["benchmark_id": "prefill_throughput_smoke"]
        if let runtimeDescriptor { payload["runtime_descriptor"] = runtimeDescriptor }
        try ws.results.saveResult(
            .local, "c-1", payload: JSONSerialization.data(withJSONObject: payload),
            extras: Data("{}".utf8))
        saveManifest(ws, cells: ["c-1"])

        let row = try #require(ResultsCommands.allRows(ws: ws).first)
        #expect(row.runtime == "")
        #expect(row.createdAt == "1970-01-01T00:00:00Z")
    }

    /// `--benchmark` compares the whole `SourcedBenchmarkId`, so one id on both catalog
    /// sides is two filters: only the remote row is submittable.
    @Test func theBenchmarkFilterHonoursTheCatalogSide() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seed(ws, "c-local", at: .local)
        try seed(ws, "c-remote", at: .local)
        ws.saveJobManifest(JobManifest(
            jobId: "job-1", createdAt: "2026-03-26T00:00:00Z", nGpuLayers: 99, contextSize: 4096,
            cells: [
                cell(id: "c-local", benchmarkId: "prefill_throughput_smoke",
                     benchmarkSource: .local),
                cell(id: "c-remote", benchmarkId: "prefill_throughput_smoke",
                     benchmarkSource: .remote),
            ],
            status: .completed))

        func ids(_ filter: SourcedBenchmarkId?) -> [CellId] {
            ResultsCommands.rows(benchmark: filter, benchmarkType: nil, state: nil, limit: nil,
                                     ws: ws)
                .map(\.resultId.cellId)
        }

        #expect(ids(.local("prefill_throughput_smoke")) == ["c-local"])
        #expect(ids(.remote("prefill_throughput_smoke")) == ["c-remote"])
        #expect(ids(nil).sorted() == ["c-local", "c-remote"])
        // A row whose side is absent reads as `remote`, as `isSubmittable` reads it.
        #expect(ids(.remote("no_such_benchmark")).isEmpty)
    }

    /// `limit=` truncates after the filters, as the crate truncates after its own.
    @Test func theLimitAppliesToTheFilteredRows() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seed(ws, "old", at: .local, submittedAt: "2026-01-01T00:00:00Z")
        try seed(ws, "new", at: .local, submittedAt: "2026-02-01T00:00:00Z")
        saveManifest(ws, cells: ["old", "new"])

        #expect(ResultsCommands.rows(benchmark: nil, benchmarkType: nil, state: nil, limit: 1,
                                         ws: ws).map(\.resultId.cellId) == ["new"])
        #expect(ResultsCommands.rows(benchmark: nil, benchmarkType: nil, state: nil, limit: 0,
                                         ws: ws).isEmpty)
    }

    @Test func rowsAreMostRecentFirst() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seed(ws, "old", at: .local, submittedAt: "2026-01-01T00:00:00Z")
        try seed(ws, "new", at: .local, submittedAt: "2026-02-01T00:00:00Z")
        saveManifest(ws, cells: ["old", "new"])

        #expect(ResultsCommands.allRows(ws: ws).map(\.resultId.cellId) == ["new", "old"])
    }

    // MARK: - show

    /// Each file after its stderr label, in the crate's order.
    @Test func showPrintsThePayloadThenExtrasThenMetrics() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seed(ws, "c-1", at: .remoteSynced)
        try ws.results.saveMetrics(
            .remoteSynced, "c-1", BenchmarkScoredResult(scoredAt: "t", metrics: []))

        let files = try #require(ResultsCommands.storedFiles(
            resultId: ResultId(jobId: "job-1", cellId: "c-1"), ws: ws))
        #expect(files.map(\.label) == [nil, "--- extras ---", "--- metrics ---"])
        #expect(files.first?.contents.contains("prefill_throughput_smoke") == true)
    }

    @Test func showOmitsTheSectionsAResultDoesNotHave() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.results.saveResult(.local, "c-1", payload: Data("{}".utf8), extras: Data("{}".utf8))
        try FileManager.default.removeItem(at: try #require(ws.results.extrasPath(.local, "c-1")))

        let files = try #require(ResultsCommands.storedFiles(
            resultId: ResultId(jobId: "job-1", cellId: "c-1"), ws: ws))
        #expect(files.map(\.label) == [nil])
    }

    // MARK: - delete

    /// The crate's `delete_removes_result_dir_and_says_so`.
    @Test func deleteRemovesTheResult() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try seed(ws, "c-1", at: .local)

        #expect(ResultsCommands.delete(resultId: ResultId(jobId: "job-1", cellId: "c-1"), ws: ws))
        #expect(ws.results.location(of: "c-1") == nil)
        #expect(!ResultsCommands.delete(resultId: ResultId(jobId: "job-1", cellId: "c-1"), ws: ws))
    }
}
