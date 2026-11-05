import Testing
import Foundation
@testable import Pipette

/// `hasRegistration`, not an `IdentityRegistration?`, keeps the case `Sendable` for
/// `@Test(arguments:)`.
private struct GateCase: Sendable, CustomTestStringConvertible {
    let name: String
    let autoSubmit: Bool
    let contribute: Bool?
    let online: Bool
    let hasRegistration: Bool
    let expected: Bool
    var testDescription: String { name }
}

/// The auto-submit gate is shared by the per-cell upload and the job-end sweep.
@Suite @MainActor struct JobExecutorTests {
    // MARK: - The auto-submit gate

    private func manifest(contributeResults: Bool?) -> JobManifest {
        var manifest = JobManifest(
            jobId: "job-1",
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [],
            status: .completed
        )
        manifest.contributeResults = contributeResults
        return manifest
    }

    @Test(arguments: [
        GateCase(name: "all conditions met", autoSubmit: true, contribute: true, online: true, hasRegistration: true, expected: true),
        GateCase(name: "offline skips", autoSubmit: true, contribute: true, online: false, hasRegistration: true, expected: false),
        GateCase(name: "not opted in", autoSubmit: true, contribute: false, online: true, hasRegistration: true, expected: false),
        GateCase(name: "opt-in unset", autoSubmit: true, contribute: nil, online: true, hasRegistration: true, expected: false),
        GateCase(name: "autoSubmit off (headless)", autoSubmit: false, contribute: true, online: true, hasRegistration: true, expected: false),
        GateCase(name: "no registration", autoSubmit: true, contribute: true, online: true, hasRegistration: false, expected: false),
    ])
    fileprivate func gateRequiresEveryCondition(_ testCase: GateCase) throws {
        let got = JobExecutor.shouldAutoSubmit(
            manifest(contributeResults: testCase.contribute),
            autoSubmit: testCase.autoSubmit,
            online: testCase.online,
            registration: try testCase.hasRegistration ? registrationData() : nil
        )
        #expect(got == testCase.expected)
    }

    // MARK: - The live cell label

    /// `modelName` names the repo, since the label derives it from the cell's source.
    private func cell(benchmarkId: String, type: BenchmarkType?, modelName: String) -> JobCell {
        JobCell(
            cellId: "c",
            benchmarkId: benchmarkId,
            benchmarkType: type,
            runStatus: .running,
            serverJobId: nil,
            errorMessage: nil,
            source: ggufTextFixture(modelName, "weights-Q4_0.gguf")
        )
    }

    @Test func decodeLabelShowsPrefillAndDecodeTokens() {
        let c = cell(benchmarkId: "decode_throughput_512_100", type: .decodeThroughput,
                     modelName: "LiquidAI/LFM2.5-350M-GGUF")
        let label = JobExecutor.liveCellLabel(c, params: [
            "parameter_prefill_tokens": 512,
            "parameter_decode_tokens": 100
        ])
        #expect(label == "Decode Throughput · 512→100 tok · LFM2.5-350M-GGUF")
    }

    @Test func prefillLabelShowsSingleTokenCount() {
        let c = cell(benchmarkId: "prefill_throughput_512", type: .prefillThroughput, modelName: "org/Foo")
        #expect(JobExecutor.liveCellLabel(c, params: ["parameter_prefill_tokens": 512]) == "Prefill Throughput · 512 tok · Foo")
    }

    @Test func labelWithoutParamsOmitsSummary() {
        let c = cell(benchmarkId: "decode_throughput", type: .decodeThroughput, modelName: "org/Foo")
        #expect(JobExecutor.liveCellLabel(c, params: nil) == "Decode Throughput · Foo")
    }

    @Test func evalLabelUsesDatasetId() {
        // Eval benchmarks differ by dataset, not token params, so the id is shown.
        let c = cell(benchmarkId: "ifbench", type: .eval, modelName: "org/Foo")
        #expect(JobExecutor.liveCellLabel(c, params: ["parameter_max_tokens": 512]) == "ifbench · Foo")
    }

    @Test func vlLabelShowsImageDimsAndTokens() {
        let c = cell(benchmarkId: "vl_throughput_336_336", type: .vlThroughput, modelName: "org/Foo")
        let label = JobExecutor.liveCellLabel(c, params: [
            "parameter_image_width": 336,
            "parameter_image_height": 336,
            "parameter_text_tokens": 64
        ])
        #expect(label == "Vision-Language Throughput · 336×336px · 64 tok · Foo")
    }

    @Test func decodeLabelFallsBackToDecodeTokensWhenPrefillMissing() {
        let c = cell(benchmarkId: "decode_throughput_100", type: .decodeThroughput, modelName: "org/Foo")
        #expect(JobExecutor.liveCellLabel(c, params: ["parameter_decode_tokens": 100]) == "Decode Throughput · 100 tok · Foo")
    }

    // MARK: - The kept fetch failure

    /// The worker classifies the run's first vault failure by its type, so neither a
    /// non-vault error nor a later vault failure may replace it.
    @Test func keptFetchFailureHoldsTheFirstVaultFailure() {
        let kept = KeptFetchFailure()
        kept.keep(URLError(.timedOut))
        #expect(kept.error == nil)

        kept.keep(VaultFetchError.refused(
            resourceId: VaultResourceId(validated: "vault_1"), status: 403, detail: "refused"))
        kept.keep(VaultFetchError.mgmt(.httpStatus(
            method: "GET", url: "https://mgmt.test/vault/resources/vault_1/content",
            status: 503, body: "")))
        kept.keep(URLError(.timedOut))

        guard case .refused(_, 403, _) = kept.error else {
            Issue.record("kept \(String(describing: kept.error)), not the first vault refusal")
            return
        }
        #expect(kept.error?.isPermanent == true)
    }

    // MARK: - The benchmark a cell runs

    private func prefillWire(_ id: String, _ tokens: Int) -> String {
        #"{"benchmark_id":"\#(id)","benchmark_type":"prefill_throughput","parameter_prefill_tokens":\#(tokens)}"#
    }

    private func prefillTokens(_ item: BenchmarkItem?) -> Int32? {
        guard case let .prefillThroughput(b) = item?.definition else { return nil }
        return b.parameterPrefillTokens.value
    }

    /// Not the synced list's row, which a worker never refreshes.
    @Test func aClaimedCellRunsWhatTheClaimsRevalidationReturned() async throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let id = "prefill_throughput_x"
        try store.putRemoteIndex(Coding.decoder.decode(
            [PipetteMgmtClient.BenchmarkSummary].self,
            from: Data("[\(prefillWire(id, 256))]".utf8)))
        try store.cacheRemote(json: prefillWire(id, 256), etag: #""v256""#)
        let served = prefillWire(id, 512)
        let (http, server, seen) = try RoutedURLProtocol.serve { request, _ in
            request.value(forHTTPHeaderField: "If-None-Match") == #""v512""#
                ? .reply(304) : .reply(served, etag: #""v512""#)
        }

        let resolved = try await JobCell.resolvable(
            benchmarkIds: [id], store: store, server: server, http: http, auth: nil)

        #expect(seen.methodsAndPaths == ["GET /benchmarks/\(id)"])
        #expect(seen.requests.first?.value(forHTTPHeaderField: "If-None-Match") == #""v256""#)
        let claimed = JobExecutor.pendingBenchmarks(ids: [id], store: store, claimed: resolved.claimed)
        #expect(claimed.map(prefillTokens) == [512])
        let unclaimed = JobExecutor.pendingBenchmarks(ids: [id], store: store, claimed: [])
        #expect(unclaimed.map(prefillTokens) == [256], "unclaimed, the synced row runs")
    }

    /// Unclaimed, the order is the synced row, then a synced detail the list does not name,
    /// then the structured id; a claimed definition goes ahead of all three.
    @Test func aCellsBenchmarkResolvesInOrder() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.putRemoteIndex(Coding.decoder.decode(
            [PipetteMgmtClient.BenchmarkSummary].self,
            from: Data("[\(prefillWire("prefill_throughput_listed", 256))]".utf8)))
        try store.cacheRemote(json: prefillWire("prefill_throughput_listed", 128))
        try store.cacheRemote(json: prefillWire("prefill_throughput_detail", 64))
        let claimed = try Coding.decoder.decode(
            BenchmarkDefinition.self, from: Data(prefillWire("prefill_throughput_512", 32).utf8))
        let ids: Set<String> = [
            "prefill_throughput_listed", "prefill_throughput_detail", "prefill_throughput_512",
        ]

        let tokens = { (claimed: [BenchmarkDefinition]) in
            Dictionary(uniqueKeysWithValues: JobExecutor.pendingBenchmarks(
                ids: ids, store: store, claimed: claimed).map { ($0.benchmarkId, self.prefillTokens($0)) })
        }

        #expect(tokens([]) == [
            "prefill_throughput_listed": 256, "prefill_throughput_detail": 64,
            "prefill_throughput_512": 512,
        ])
        #expect(tokens([claimed])["prefill_throughput_512"] == 32)
    }
}
