import Foundation
import Testing

@testable import Pipette

/// The two catalog halves and the address that names them — against
/// `pipette-cli/src/benchmarks/{reference,standard,store}.rs`.
struct BenchmarkCatalogHalvesTests {

    // MARK: - SourcedBenchmarkId

    /// A bare id is the distributed form and means the *synced* catalog; a prefix is an
    /// explicit override; anything else names no benchmark. Mirrors the crate's
    /// `parse_accepts_bare_ids_and_explicit_sides`.
    @Test(arguments: [
        ("foo", SourcedBenchmarkId.remote("foo")),
        ("local/foo", .local("foo")),
        ("remote/foo", .remote("foo")),
    ])
    func parseAcceptsBareIdsAndExplicitSides(reference: String, expected: SourcedBenchmarkId) {
        #expect(SourcedBenchmarkId(reference: reference) == expected)
    }

    @Test(arguments: ["local/remote/foo", "elsewhere/foo", "local/", "remote/", "", "foo bar"])
    func parseRefusesAnythingElse(reference: String) {
        #expect(SourcedBenchmarkId(reference: reference) == nil)
    }

    @Test func descriptionRoundTripsThroughParse() {
        for reference in [SourcedBenchmarkId.local("foo"), .remote("foo")] {
            #expect(SourcedBenchmarkId(reference: reference.description) == reference)
        }
    }

    // MARK: - The standard local set

    /// The crate's `torch_oai_kind_set_is_seventeen`, on the same kind set: 7×2 ladder
    /// (end-to-end + max-memory) plus 3 smoke entries.
    @Test func theTorchOaiKindSetIsSeventeen() {
        #expect(StandardBenchmarks.all(kinds: [.endToEndLatency, .maxMemoryUsage, .eval]).count == 17)
    }

    /// `eval` and `vl_throughput` have no ladder form, so they contribute a smoke entry
    /// only — the crate returns `None` from `ladder_entry` for exactly those two.
    @Test func kindsWithoutALadderContributeOnlySmoke() {
        #expect(StandardBenchmarks.all(kinds: [.eval, .vlThroughput]).map(\.benchmarkId).sorted()
            == ["eval_smoke", "vl_throughput_smoke"])
    }

    /// Duplicates and ordering in `kinds` must not change the output — the crate drives
    /// from `BenchmarkType::ALL` for the same reason.
    @Test func outputIsIndependentOfTheKindsOrder() {
        let a = StandardBenchmarks.all(kinds: [.decodeThroughput, .prefillThroughput])
        let b = StandardBenchmarks.all(kinds: [.prefillThroughput, .decodeThroughput, .prefillThroughput])
        #expect(a.map(\.benchmarkId) == b.map(\.benchmarkId))
    }

    /// The ladder ids carry the crate's format strings verbatim — these are the ids the
    /// warehouse groups on, so a divergence would split one benchmark into two rows.
    @Test func ladderIdsMatchTheCratesFormatStrings() {
        let ids = Set(StandardBenchmarks.all(kinds: BenchmarkType.allCases).map(\.benchmarkId))
        for expected in ["prefill_throughput_512", "decode_throughput_512_100",
                         "end_to_end_latency_512_256", "max_memory_usage_512",
                         "eval_smoke", "vl_throughput_smoke"] {
            #expect(ids.contains(expected), "missing \(expected)")
        }
    }

    // MARK: - Seeding

    /// Seeding is idempotent: a second run updates rather than double-creating, as the
    /// crate's `seed_standard_then_get_local` asserts.
    @Test func seedingTwiceUpdatesRatherThanDuplicating() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        let first = try StandardBenchmarks.seedLocal(into: store, kinds: [.prefillThroughput])
        #expect(first.created > 0)
        #expect(first.updated == 0)

        let again = try StandardBenchmarks.seedLocal(into: store, kinds: [.prefillThroughput])
        #expect(again.created == 0)
        #expect(again.updated == first.created)
        #expect(store.list(.local).count == first.created)
    }

    /// A seeded definition round-trips through the store — the encoder added for the
    /// local half has to produce what the decoder reads.
    @Test func aSeededDefinitionRoundTrips() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        try StandardBenchmarks.seedLocal(into: store, kinds: [.decodeThroughput, .eval])

        #expect(store.get(.local("decode_throughput_512_100"))
            == .decodeThroughput(
                benchmarkId: "decode_throughput_512_100", prefillTokens: 512, decodeTokens: 100))
        // The eval smoke entry keeps its sample through the round trip.
        guard case let .eval(_, evalId, dataset, maxTokens, _, samples) =
            store.get(.local("eval_smoke"))
        else {
            Issue.record("eval_smoke did not round-trip as an eval definition")
            return
        }
        #expect(evalId.rawValue == "eval_smoke")
        #expect(dataset == "local")
        #expect(maxTokens == 4)
        #expect(samples?.count == 1)
    }

    /// The halves are independent, and a benchmark defined on both is the submittable
    /// one — preferring `local/` would silently make its results unsubmittable.
    @Test func remoteWinsADuplicateId() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        try StandardBenchmarks.seedLocal(into: store, kinds: [.prefillThroughput])
        let row = """
        [{"benchmark_id":"prefill_throughput_512","benchmark_type":"prefill_throughput",\
        "parameter_prefill_tokens":512,"server_only_key":"kept"}]
        """
        try store.putRemoteIndex(Data(row.utf8))

        let all = BenchmarkCatalog.all(store: store)
        let duplicated = all.filter { $0.benchmarkId == "prefill_throughput_512" }
        #expect(duplicated.count == 1)
        // The remote row wins, and its unmodelled key survives for the raw readers.
        #expect(duplicated.first?.rawJson["server_only_key"] as? String == "kept")
        // The rest of the seeded ladder is still listed.
        #expect(all.count == StandardBenchmarks.all(kinds: [.prefillThroughput]).count)
    }

    // MARK: - Submittability

    /// The rule the local half exists for: a locally-generated benchmark was never
    /// sanctioned by the server, so its result is not published. The crate routes such a
    /// result to `results/local/`, which `sync` never walks; here the cell carries the
    /// source and the submit sweep skips it.
    @Test func aLocalCellIsNotSubmittableAndARemoteOneIs() {
        let local = cell(id: "local-1", benchmarkId: "prefill_throughput_8", source: .local)
        let remote = cell(id: "remote-1", benchmarkId: "prefill_throughput_512", source: .remote)
        #expect(!local.isSubmittable)
        #expect(remote.isSubmittable)
    }

    /// A manifest written before the local half existed carries no source, and every
    /// such cell came from the synced catalog — so absent must read as submittable or an
    /// upgrade would silently strand pending results.
    @Test func aCellWithNoRecordedSourceIsSubmittable() {
        #expect(cell(id: "old", benchmarkId: "prefill_throughput_512", source: nil).isSubmittable)
    }

    /// The sweep drops local cells and keeps remote ones — the gate where it actually
    /// bites, rather than only on the predicate.
    @Test func theSubmitSweepSkipsLocalCells() async throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }
        let cells = [
            cell(id: "local-1", benchmarkId: "prefill_throughput_8", source: .local),
            cell(id: "remote-1", benchmarkId: "prefill_throughput_512", source: .remote),
        ]
        for c in cells {
            try writePayload(storage: storage, cellId: c.cellId, benchmarkId: c.benchmarkId)
        }
        let submitted = SubmittedIds()

        _ = await ResultSubmissionService.submit(
            manifest: manifest(cells: cells),
            registration: registrationData(),
            auth: AuthIdentity(
                clientId: ClientID("client-1"), privateKeyHex: PrivateKeyHex("k")),
            submitResultBatch: { _, _, payloadsJson in
                await submitted.record(payloadsJson)
                return try batchResponse([["index": 0, "job_id": "server-1"]])
            },
            storage: storage)

        let seen = await submitted.benchmarkIds
        #expect(seen == ["prefill_throughput_512"])
    }

    // MARK: - Fixtures

    private func cell(id: String, benchmarkId: String, source: BenchmarkSource?) -> JobCell {
        JobCell(
            cellId: CellId(id), benchmarkId: benchmarkId,
            benchmarkType: .prefillThroughput,
            runStatus: .completed, serverJobId: nil, errorMessage: nil,
            source: ggufTextFixture("test/model-GGUF", "model.gguf"),
            benchmarkSource: source)
    }

    private func manifest(cells: [JobCell]) -> JobManifest {
        JobManifest(
            jobId: "job-1",
            createdAt: "2026-06-24T00:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: cells,
            status: .completed
        )
    }
}

/// Collects the benchmark ids each submitted batch carried.
private actor SubmittedIds {
    private var ids: [String] = []

    func record(_ payloadsJson: String) {
        guard let data = payloadsJson.data(using: .utf8),
              let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return }
        ids.append(contentsOf: rows.compactMap { $0["benchmark_id"] as? String })
    }

    var benchmarkIds: [String] { ids }
}
