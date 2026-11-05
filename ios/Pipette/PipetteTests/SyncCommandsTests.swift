import Foundation
import Testing

@testable import Pipette

/// Where `sync` sends its catalog pull, mirroring the crate's `SyncArgs::resolve_server_url`
/// case for case. It decides whether an unregistered device can see the catalog at all.
struct SyncCommandsTests {

    /// The registration wins. The signing key beside it signs only for that server, so
    /// honoring an override would point the authenticated half at a host that refuses it.
    @Test func aRegisteredDeviceUsesItsRegisteredServer() throws {
        let registered = try ServerUrl("https://collector.example.com")
        #expect(SyncCommands.resolveServerUrl(registered: registered, requested: nil)
            == registered)
        #expect(SyncCommands.resolveServerUrl(registered: registered,
                                              requested: registered.value) == registered)
    }

    /// An override that disagrees is an error, not a silent no-op: someone who typed it
    /// meant it, and what they want is `auth reset`.
    @Test func aDisagreeingServerUrlIsRefused() throws {
        #expect(SyncCommands.resolveServerUrl(
            registered: try ServerUrl("https://collector.example.com"),
            requested: "https://staging.example.com") == nil)
    }

    /// A `server-url=` that is not a URL at all cannot name the registered
    /// server, and is refused rather than compared as text.
    @Test func anUnparseableServerUrlIsRefused() throws {
        #expect(SyncCommands.resolveServerUrl(
            registered: try ServerUrl("https://collector.example.com"),
            requested: "collector.example.com") == nil)
        #expect(SyncCommands.resolveServerUrl(registered: nil,
                                              requested: "collector.example.com") == nil)
    }

    /// ``ServerUrl`` normalizes on construction, so the comparison is between two values
    /// already in one form.
    @Test func aTrailingSlashIsNotADisagreement() throws {
        let registered = try ServerUrl("https://collector.example.com")
        #expect(SyncCommands.resolveServerUrl(registered: registered,
                                              requested: "https://collector.example.com/")
            == registered)
        #expect(SyncCommands.resolveServerUrl(registered: try ServerUrl("https://h/"),
                                              requested: "https://h")
            == (try ServerUrl("https://h")))
        // Case too, but only where case is insignificant: scheme and host.
        #expect(SyncCommands.resolveServerUrl(registered: registered,
                                              requested: "HTTPS://Collector.Example.com")
            == registered)
    }

    /// A path *is* case-sensitive, so two endpoints that differ only in path case are
    /// two servers. Folding the whole URL would have called them one.
    @Test func aPathsCaseIsNotFoldedAway() throws {
        #expect(SyncCommands.resolveServerUrl(
            registered: try ServerUrl("https://h.example.com/API"),
            requested: "https://h.example.com/api") == nil)
    }

    /// The point of the parameter: the catalog is a public read, so a device with no
    /// identity can still see what there is to run.
    @Test func anUnregisteredDeviceReadsFromTheNamedServer() throws {
        #expect(SyncCommands.resolveServerUrl(registered: nil,
                                              requested: "https://mgmt.example.com")
            == (try ServerUrl("https://mgmt.example.com")))
    }

    /// With neither, the answer is a refusal rather than the production collector.
    /// `auth register` is the one command that defaults a server, because it records
    /// what it picked.
    @Test func withNoRegistrationAndNoParameterNothingNamesAServer() {
        #expect(SyncCommands.resolveServerUrl(registered: nil, requested: nil) == nil)
    }
}

extension SyncCommandsTests {

    // MARK: - The command

    /// The pull runs and lands in the store, the two authenticated phases are skipped, and
    /// the command succeeds. The pending result is the decoy that tells "skipped because
    /// unregistered" from "nothing to submit".
    @MainActor @Test func anUnregisteredSyncPullsTheCatalogAndSkipsTheRest() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let pending = cell(id: "r-1", benchmarkId: "prefill_throughput_256")
        try writePayload(ws: ws, cellId: pending.cellId, benchmarkId: pending.benchmarkId)
        ws.saveJobManifest(JobManifest(
            jobId: "job-1", createdAt: "2026-03-26T00:00:00Z", nGpuLayers: 99, contextSize: 4096,
            cells: [pending], status: .completed, contributeResults: true))
        let body = #"{"benchmark_id":"b1","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#
        let submits = CallCount()

        let ok = await SyncCommands.run(
            jobId: nil, serverUrl: "https://mgmt.example.test", ws: ws,
            pullCatalog: { server in
                try await pullRemoteBenchmarks(
                    store: ws.benchmarks,
                    server: server,
                    fetchList: { _, _ in
                        .modified(value: try Coding.decoder.decode(
                            [PipetteMgmtClient.BenchmarkSummary].self, from: Data("[\(body)]".utf8)),
                                  etag: nil)
                    },
                    fetchDetail: { _, _, _ in
                        .modified(value: try Coding.decoder.decode(
                            RemoteBenchmark.self, from: Data(body.utf8)), etag: nil)
                    })
            },
            uploader: ResultUploader(
                submitResultBatch: { _, _, _ in
                    await submits.hit()
                    throw URLError(.cannotConnectToHost)
                },
                ws: ws))

        #expect(ok)
        #expect(try ws.benchmarks.listRemoteIndex().map(\.benchmarkId) == ["b1"])
        #expect(await submits.count == 0)
        #expect(ws.results.listIds(.remotePending) == ["r-1"],
                "the pending result is untouched, not submitted")
    }

    /// The score half of the crate's `a_registered_sync_pulls_submits_and_scores`.
    @MainActor @Test func aRegisteredSyncRefreshesTheScoreOfASubmittedEval() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        try ws.registration.put(try registrationData())
        let id = CellId("c-1")
        try ws.results.saveResult(
            .remoteSynced, id,
            payload: JSONSerialization.data(withJSONObject: ["benchmark_id": "eval_smoke"]),
            extras: Data("{}".utf8))
        try ws.results.saveSubmission(.submitted(serverJobId: "job-1"), id)
        let reply = try Coding.decoder.decode(JobResponse.self, from: Data(
            #"{"status":"processed","metrics":[{"metric":"accuracy","value":0.5,"unit":"ratio"}]}"#.utf8))
        let asked = CallCount()

        let ok = await SyncCommands.run(
            jobId: nil, ws: ws,
            pullCatalog: { _ in 0 },
            uploader: ResultUploader(
                submitResultBatch: { _, _, _ in throw URLError(.cannotConnectToHost) },
                ws: ws),
            getJob: { _, _, jobId in
                #expect(jobId == "job-1")
                await asked.hit()
                return reply
            })

        #expect(ok)
        #expect(await asked.count == 1)
        #expect(ws.results.state(of: id) == .scored)
    }
}

/// How many times a fake endpoint was reached.
private actor CallCount {
    private(set) var count = 0
    func hit() { count += 1 }
}
