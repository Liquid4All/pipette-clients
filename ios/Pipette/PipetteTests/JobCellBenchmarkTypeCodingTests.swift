import Foundation
import Testing

@testable import Pipette

/// `JobCell.benchmarkType` is a typed `BenchmarkType?` but persists as the bare
/// type string under the unchanged `benchmarkType` key. These pin the lenient
/// wire contract: a known type decodes to its case, an unknown/legacy/absent
/// string decodes to `nil` (never a throw — the prior `?? ""` / `default`
/// tolerance), and encoding a known type round-trips the exact wire string.
///
/// Fixtures are decoded through `JobManifestSchema.decode` directly — no on-disk
/// store, no global test seam — so the suite is pure and parallel-safe.
@Suite
struct JobCellBenchmarkTypeCodingTests {
    /// A single-cell manifest fixture with `benchmarkType` set to `typeJson`
    /// (already JSON-quoted, or the literal `null`, or omitted when `nil`).
    private func manifestJSON(benchmarkTypeField: String?) -> Data {
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
                  "benchmarkId": "decode_throughput_100_100",
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
        // A type this build has never heard of must not fail the whole manifest —
        // the cell simply loses its typed kind, exactly as the pre-typed code did.
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
            source: .ggufText(GgufText(source: .huggingFace(repo: try HFRepo.parse("test/A-GGUF"), path: try RepoSubpath("a.gguf"), sha256: nil))))

        let data = try JSONEncoder().encode(cell)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        // The bare rawValue string persists under the unchanged key.
        #expect(json["benchmarkType"] as? String == "prefill_throughput")

        let decoded = try JSONDecoder().decode(JobCell.self, from: data)
        #expect(decoded.benchmarkType == .prefillThroughput)
    }

    @Test func nilTypeOmitsTheWireKey() throws {
        let cell = JobCell(
            cellId: CellId("cell-nil"), benchmarkId: "smoke",
            benchmarkType: nil, runStatus: .completed,
            serverJobId: nil, errorMessage: nil,
            source: .ggufText(GgufText(source: .huggingFace(repo: try HFRepo.parse("test/A-GGUF"), path: try RepoSubpath("a.gguf"), sha256: nil))))

        let data = try JSONEncoder().encode(cell)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["benchmarkType"] == nil)
    }
}
