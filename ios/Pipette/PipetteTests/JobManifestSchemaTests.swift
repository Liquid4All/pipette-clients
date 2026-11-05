import Foundation
import Testing
@testable import Pipette

/// Each migration step gets a fixture of the old shape and an assertion on the upgraded
/// result.
@MainActor struct JobManifestSchemaTests {
    @Test func legacyManifestWithoutSchemaVersionLoads() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        // Also carries the cell's removed `modelPath` key, which decoding ignores. It keeps
        // `source`: a manifest predating that required field is deliberately unsupported.
        try writeManifestFixture(ws: ws, jobId: "job-legacy", json: """
            {
              "jobId": "job-legacy",
              "createdAt": "2026-01-15T08:30:00Z",
              "nGpuLayers": 99,
              "contextSize": 4096,
              "cells": [
                {
                  "cellId": "cell-a",
                  "benchmarkId": "prefill",
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
            """)

        let loaded = ws.loadJobManifest(jobId: "job-legacy")
        #expect(loaded?.jobId == "job-legacy")
        #expect(loaded?.cells.map(\.cellId) == ["cell-a"])
        // Absent reads as version 1, so the ladder runs. The file keeps the old bytes until
        // something saves it: upgrade on load, never rewrite on sight.
        #expect(loaded?.schemaVersion == JobManifestSchema.currentVersion)
        let raw = try Data(contentsOf: manifestURL(ws: ws, jobId: "job-legacy"))
        let json = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        #expect(json["schemaVersion"] == nil)
    }

    /// A pre-spec cell carries its model and flags as separate keys. Nothing this build
    /// writes reaches this decode branch, so no other test exercises it.
    @Test func aPreSpecCellReassemblesIntoItsSpec() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try writeManifestFixture(ws: ws, jobId: "job-prespec", json: """
            {
              "jobId": "job-prespec",
              "createdAt": "2026-01-15T08:30:00Z",
              "nGpuLayers": 99,
              "contextSize": 4096,
              "cells": [
                {
                  "cellId": "cell-a",
                  "benchmarkId": "decode_throughput_512_100",
                  "benchmarkType": "decode_throughput",
                  "runStatus": "pending",
                  "source": {
                    "type": "gguf_text", "source": "huggingface",
                    "org": "test", "repo_name": "A-GGUF", "path": "a.gguf"
                  },
                  "runtimeFlags": {
                    "benchmark_type": "decode_throughput",
                    "runtime_type": "llamacpp_ios_pipette",
                    "model_type": "gguf_text",
                    "number_gpu_layers": 33
                  }
                }
              ],
              "status": "running"
            }
            """)

        let cell = try #require(ws.loadJobManifest(jobId: "job-prespec")?.cells.first)
        #expect(cell.spec.benchmark.value == "decode_throughput_512_100")
        #expect(cell.spec.model == cell.source)
        // The runtime is derived, not stored: a pre-spec cell never named one, and this
        // client can only be what it compiled as.
        #expect(cell.spec.runtime == Runtime.thisBuild(for: cell.source))
        #expect(cell.spec.runtimeFlags?.numberGpuLayers == 33)
    }

    // MARK: - Version 1 → 2: a cell's catalog reference splits into id + side

    /// A version-1 fixture whose single cell names `benchmarkId`, plus whatever else
    /// `extraCellFields` adds (already comma-terminated JSON).
    private func referenceFixture(_ benchmarkId: String,
                                  extraCellFields: String = "") -> String {
        """
        {
          "jobId": "job-ref",
          "createdAt": "2026-01-15T08:30:00Z",
          "nGpuLayers": 99,
          "contextSize": 4096,
          "cells": [
            {
              "cellId": "cell-a",
              "benchmarkId": "\(benchmarkId)",
              "benchmarkType": "decode_throughput",
              \(extraCellFields)
              "runStatus": "completed",
              "source": {
                "type": "gguf_text", "source": "huggingface",
                "org": "test", "repo_name": "A-GGUF", "path": "a.gguf"
              }
            }
          ],
          "status": "completed"
        }
        """
    }

    /// An older version-1 build wrote the whole reference into `benchmarkId`, and `BenchmarkId`
    /// refuses the `/` at the spec boundary, so the step splits it.
    @Test func aStoredReferenceSplitsIntoItsIdAndSide() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try writeManifestFixture(ws: ws, jobId: "job-ref",
                                 json: referenceFixture("remote/decode_throughput_512_100"))

        let cell = try #require(ws.loadJobManifest(jobId: "job-ref")?.cells.first)
        #expect(cell.benchmarkId == "decode_throughput_512_100")
        #expect(cell.spec.benchmark.value == "decode_throughput_512_100")
        #expect(BenchmarkId.validate(cell.spec.benchmark.value))
        #expect(cell.benchmarkSource == .remote)
    }

    /// The side is recovered rather than defaulted. `benchmarkSource` absent reads as
    /// `remote`, so a `local/` cell left unsplit would submit a result the server never
    /// sanctioned, the one rule that field exists to carry.
    @Test func aStoredLocalReferenceRecoversItsSide() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try writeManifestFixture(ws: ws, jobId: "job-ref",
                                 json: referenceFixture("local/decode_throughput_512_100"))

        let cell = try #require(ws.loadJobManifest(jobId: "job-ref")?.cells.first)
        #expect(cell.benchmarkId == "decode_throughput_512_100")
        #expect(cell.benchmarkSource == .local)
        #expect(cell.isSubmittable == false)
    }

    /// A side already on the cell was written deliberately and wins over the prefix.
    @Test func aRecordedSideSurvivesTheSplit() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try writeManifestFixture(
            ws: ws, jobId: "job-ref",
            json: referenceFixture("remote/decode_throughput_512_100",
                                   extraCellFields: "\"benchmarkSource\": \"local\","))

        let cell = try #require(ws.loadJobManifest(jobId: "job-ref")?.cells.first)
        #expect(cell.benchmarkId == "decode_throughput_512_100")
        #expect(cell.benchmarkSource == .local)
    }

    /// Side included: a manifest this step has nothing to do to comes out unchanged.
    @Test func aBareIdIsLeftAlone() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try writeManifestFixture(ws: ws, jobId: "job-ref",
                                 json: referenceFixture("decode_throughput_512_100"))

        let cell = try #require(ws.loadJobManifest(jobId: "job-ref")?.cells.first)
        #expect(cell.benchmarkId == "decode_throughput_512_100")
        #expect(cell.benchmarkSource == nil)
    }

    @Test func saveStampsCurrentSchemaVersion() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        ws.saveJobManifest(makeManifest(jobId: "job-stamped"))

        let loaded = ws.loadJobManifest(jobId: "job-stamped")
        #expect(loaded?.schemaVersion == JobManifestSchema.currentVersion)

        // The stamp must be in the file itself, not just the decoded struct.
        let raw = try Data(contentsOf: manifestURL(ws: ws, jobId: "job-stamped"))
        let json = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        #expect(json["schemaVersion"] as? Int == JobManifestSchema.currentVersion)
    }

    @Test func corruptManifestFailsLoudlyAndKeepsFile() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        try writeManifestFixture(ws: ws, jobId: "job-corrupt", json: "not json {{{")
        ws.saveJobManifest(makeManifest(jobId: "job-ok"))

        #expect(ws.loadJobManifest(jobId: "job-corrupt") == nil)
        // The healthy sibling still loads, and the corrupt file isn't deleted.
        #expect(ws.loadAllJobManifests().map(\.jobId) == ["job-ok"])
        #expect(FileManager.default.fileExists(atPath: manifestURL(ws: ws, jobId: "job-corrupt").path))
    }

    @Test func futureSchemaVersionDecodesBestEffort() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }

        // Simulates an app downgrade: a manifest written by a future version
        // with a higher schemaVersion and fields this build doesn't know.
        try writeManifestFixture(ws: ws, jobId: "job-future", json: """
            {
              "jobId": "job-future",
              "schemaVersion": \(JobManifestSchema.currentVersion + 1),
              "createdAt": "2026-06-10T08:30:00Z",
              "nGpuLayers": 99,
              "contextSize": 4096,
              "fieldFromTheFuture": {"nested": true},
              "cells": [],
              "status": "planned"
            }
            """)

        let loaded = ws.loadJobManifest(jobId: "job-future")
        #expect(loaded?.jobId == "job-future")
        #expect(loaded?.schemaVersion == JobManifestSchema.currentVersion + 1)
    }

    // MARK: - Helpers

    private func manifestURL(ws: PipetteWorkspace, jobId: JobId) -> URL {
        ws.jobsDir
            .appendingPathComponent(jobId.value, isDirectory: true)
            .appendingPathComponent("manifest.json")
    }

    private func writeManifestFixture(ws: PipetteWorkspace, jobId: JobId, json: String) throws {
        let dir = ws.jobsDir.appendingPathComponent(jobId.value, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: dir.appendingPathComponent("manifest.json"))
    }

    private func makeManifest(jobId: JobId) -> JobManifest {
        JobManifest(
            jobId: jobId,
            createdAt: "2026-06-10T18:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: [
                JobCell(
                    cellId: "cell-a",
                    benchmarkId: "prefill",
                    benchmarkType: .prefillThroughput,
                    runStatus: .completed,
                    serverJobId: nil,
                    errorMessage: nil,
                    source: ggufTextFixture("test/A-GGUF", "a.gguf")
                )
            ],
            status: .completed
        )
    }
}
