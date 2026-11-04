import Foundation
import Testing

@testable import Pipette

/// `benchmarkType` decodes leniently: an unknown, legacy or absent string is `nil`, never
/// a throw. A benchmark reference carries its own provenance, so `pending` is what keeps a
/// result from the generated catalog half out of the submit sweep.
@Suite
struct JobCellTests {
    // MARK: - The wire contract for `benchmarkType`

    /// `benchmarkTypeField` is already JSON-quoted, the literal `null`, or `nil` to omit
    /// the key.
    private func manifestJSON(benchmarkTypeField: String?) -> Data {
        let benchmarkId = "decode_throughput_100_100"
        let typeLine = benchmarkTypeField.map { "\"benchmarkType\": \($0)," } ?? ""
        return Data("""
            {
              "jobId": "job-x",
              "createdAt": "2026-06-10T08:30:00Z",
              "nGpuLayers": 99,
              "contextSize": 4096,
              "cells": [
                {
                  "cellId": "cell-a",
                  "benchmarkId": "\(benchmarkId)",
                  \(typeLine)
                  "modelPath": "/models/a.gguf",
                  "modelName": "A",
                  "runStatus": "completed",
                  "source": {
                    "type": "gguf_text", "source": "huggingface",
                    "org": "test",
                    "repo_name": "A-GGUF",
                    "path": "a.gguf"
                  }
                }
              ],
              "status": "completed"
            }
            """.utf8)
    }

    @Test func knownWireStringDecodesToItsCase() throws {
        let manifest = try JobManifestSchema.decode(manifestJSON(benchmarkTypeField: "\"decode_throughput\""))
        #expect(manifest.cells.first?.benchmarkType == .decodeThroughput)
    }

    @Test func unknownLegacyStringDecodesToNilNotAThrow() throws {
        let manifest = try JobManifestSchema.decode(manifestJSON(benchmarkTypeField: "\"speculative_decode\""))
        #expect(manifest.cells.first?.benchmarkType == nil)
    }

    @Test func absentBenchmarkTypeDecodesToNil() throws {
        let manifest = try JobManifestSchema.decode(manifestJSON(benchmarkTypeField: nil))
        #expect(manifest.cells.first?.benchmarkType == nil)
    }

    @Test func nullBenchmarkTypeDecodesToNil() throws {
        let manifest = try JobManifestSchema.decode(manifestJSON(benchmarkTypeField: "null"))
        #expect(manifest.cells.first?.benchmarkType == nil)
    }

    @Test func knownTypeRoundTripsToTheSameWireString() throws {
        let cell = JobCell(
            cellId: CellId("cell-rt"), benchmarkId: "prefill_throughput_512",
            benchmarkType: .prefillThroughput, runStatus: .completed,
            serverJobId: nil, errorMessage: nil,
            source: .ggufText(GgufText(source: .huggingFace(repo: try HfRepo.parseOrgRepo("test/A-GGUF"), path: try RepoSubpath("a.gguf"), sha256: nil))))

        let data = try JSONEncoder().encode(cell)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["benchmarkType"] as? String == "prefill_throughput")

        let decoded = try JSONDecoder().decode(JobCell.self, from: data)
        #expect(decoded.benchmarkType == .prefillThroughput)
    }

    @Test func nilTypeOmitsTheWireKey() throws {
        let cell = JobCell(
            cellId: CellId("cell-nil"), benchmarkId: "smoke",
            benchmarkType: nil, runStatus: .completed,
            serverJobId: nil, errorMessage: nil,
            source: .ggufText(GgufText(source: .huggingFace(repo: try HfRepo.parseOrgRepo("test/A-GGUF"), path: try RepoSubpath("a.gguf"), sha256: nil))))

        let data = try JSONEncoder().encode(cell)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["benchmarkType"] == nil)
    }

    // MARK: - The public-build memory record

    /// Local only: a manifest written before it decodes, and an unset cell writes no key.
    @Test func measuredAfterOtherWorkIsOptionalAndRoundTrips() throws {
        let manifest = try JobManifestSchema.decode(
            manifestJSON(benchmarkTypeField: "\"max_memory_usage\""))
        #expect(manifest.cells.first?.measuredAfterOtherWork == nil)

        var cell = JobCell(
            cellId: CellId("cell-mem"), benchmarkId: "max_memory_usage_512",
            benchmarkType: .maxMemoryUsage, runStatus: .completed,
            serverJobId: nil, errorMessage: nil,
            source: .ggufText(GgufText(source: .huggingFace(repo: try HfRepo.parseOrgRepo("test/A-GGUF"), path: try RepoSubpath("a.gguf"), sha256: nil))))
        let unsetData = try JSONEncoder().encode(cell)
        let unset = try #require(JSONSerialization.jsonObject(with: unsetData) as? [String: Any])
        #expect(unset["measuredAfterOtherWork"] == nil)

        cell.measuredAfterOtherWork = true
        let decoded = try JSONDecoder().decode(JobCell.self, from: JSONEncoder().encode(cell))
        #expect(decoded.measuredAfterOtherWork == true)
    }

    // MARK: - Cells built from a benchmark reference

    /// A reference carries its own source, as the CLI's `--benchmark` does, so
    /// `benchmarks run benchmark=<a local ladder id>` cannot bypass the submit gate.
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

    /// A `vl_spec_decode` id spells neither its parameters nor its corpus pin, so the store
    /// is the *only* thing that can answer for it.
    @Test func aCorpusBackedDefinitionResolvesFromTheStore() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let definition = BenchmarkDefinition.vlSpecDecode(VlSpecDecodeBenchmark(
            benchmarkId: BenchmarkId(validated: "vl_spec_decode_mmspec_test_gqa_k8"),
            parameterDraftLength: 8,
            parameterMaxTokens: 2048,
            dataset: BenchmarkDefinitionTests.sampleDataset))
        try store.cacheRemote(definition)
        #expect(BenchmarkDefinition(parsingId: definition.benchmarkId.value) == nil)

        var skipped: [String] = []
        let cells = JobCell.pending(
            benchmarkIds: ["remote/\(definition.benchmarkId)"], for: .appleFoundation,
            store: store) { skipped.append($0) }

        #expect(skipped.isEmpty)
        #expect(cells.count == 1)
        #expect(cells.first?.benchmarkType == .vlSpecDecode)
        #expect(cells.first?.isSubmittable == true)
    }

    /// A bare id means the synced catalog, the form plans and claims carry. Reading it as
    /// local would strand real results.
    @Test func aBareReferenceIsRemote() {
        let cells = JobCell.pending(
            benchmarkIds: ["prefill_throughput_512"], for: .appleFoundation)
        #expect(cells.first?.isSubmittable == true)
    }

    /// Where the crate's `benchmarks run` puts them, so the gate reads a headless cell's
    /// exactly as it reads a claim's.
    @Test func benchmarkFlagsRideOnTheSpec() throws {
        let model = DiscoveredModel(source: try ggufTextSpec("org/r-GGUF", "r-Q4_0.gguf"),
                                    path: "/tmp/r.gguf", sizeBytes: 1)
        let flags = BenchmarkFlagRef(
            benchmarkType: .prefillThroughput, runtimeType: .llamacppIosPipette,
            modelType: .ggufText, readiness: ReadinessOverrides(maxWaitSecs: 1800))

        let cells = JobCell.pending(benchmarkIds: ["prefill_throughput_512"], for: model,
                                    benchmarkFlags: flags)

        #expect(cells.first?.spec.benchmarkFlags == flags)
    }

    @Test func anUnparseableReferenceIsSkipped() {
        var skipped: [String] = []
        let cells = JobCell.pending(
            benchmarkIds: ["elsewhere/foo", "not_a_benchmark"], for: .appleFoundation
        ) { skipped.append($0) }
        #expect(cells.isEmpty)
        #expect(skipped == ["elsewhere/foo", "not_a_benchmark"])
    }

    /// The store answers first, as the CLI's `ws.benchmarks().get(ref)` does, because the
    /// `_smoke` ids and every `eval` id parse to nothing.
    @Test(arguments: [
        BenchmarkDefinition.vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "vl_throughput_smoke"),
            parameterImageWidth: 224,
            parameterImageHeight: 224,
            parameterTextTokens: 8,
            parameterDecodeTokens: 8)),
        .maxMemoryUsage(MaxMemoryUsageBenchmark(
            benchmarkId: BenchmarkId(validated: "max_memory_usage_smoke"),
            parameterPrefillTokens: 8)),
        .eval(EvalBenchmark(
            benchmarkId: BenchmarkId(validated: "eval_smoke"),
            parameterEvalId: EvalId("eval_smoke"),
            parameterDatasetName: "local",
            parameterMaxTokens: 4,
            parameterMcqChoices: nil,
            samples: nil)),
    ])
    func aStoredDefinitionResolvesWhereItsIdDoesNot(_ definition: BenchmarkDefinition) throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.putLocal(definition)
        #expect(BenchmarkDefinition(parsingId: definition.benchmarkId.value) == nil)

        var skipped: [String] = []
        let cells = JobCell.pending(
            benchmarkIds: ["local/\(definition.benchmarkId)"], for: .appleFoundation,
            store: store) { skipped.append($0) }

        #expect(skipped.isEmpty)
        #expect(cells.count == 1)
        #expect(cells.first?.benchmarkId == definition.benchmarkId.value)
        #expect(cells.first?.benchmarkType == definition.benchmarkType)
        #expect(cells.first?.isSubmittable == false)
    }

    /// The halves are different entries: a `remote/` reference must not be answered from a
    /// `local/` definition.
    @Test func theReferencedSideIsTheOneRead() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.putLocal(.vlThroughput(VlThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: "vl_throughput_smoke"),
            parameterImageWidth: 224,
            parameterImageHeight: 224,
            parameterTextTokens: 8,
            parameterDecodeTokens: 8)))

        var skipped: [String] = []
        let cells = JobCell.pending(benchmarkIds: ["remote/vl_throughput_smoke"],
                                    for: .appleFoundation, store: store) { skipped.append($0) }
        #expect(cells.isEmpty)
        #expect(skipped == ["remote/vl_throughput_smoke"])
    }

    /// So a ladder rung nothing has seeded stays runnable.
    @Test func aLadderIdStillResolvesWithNoStore() {
        let cells = JobCell.pending(benchmarkIds: ["local/decode_throughput_512_100"],
                                    for: .appleFoundation)
        #expect(cells.first?.benchmarkType == .decodeThroughput)
    }
}
