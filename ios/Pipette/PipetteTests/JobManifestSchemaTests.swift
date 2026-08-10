import Foundation
import Testing
@testable import Pipette

/// Versioned-manifest behavior: legacy manifests (no `schemaVersion`) keep
/// loading, every save stamps the current version, corrupt files fail loudly
/// without being deleted, and future-version manifests decode best-effort.
/// When a real migration step lands, add a fixture of the old shape here and
/// assert the upgraded result — that's the whole point of the ladder.
///
/// Each test injects its own temporary `FileStorage`, so the suite carries no
/// shared global and runs in parallel.
@MainActor struct JobManifestSchemaTests {
    @Test func legacyManifestWithoutSchemaVersionLoads() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // A manifest with no `schemaVersion` still loads — the ladder tolerates its
        // absence, as decoding tolerates the cell's removed `modelPath` key. (Cells carry
        // a `source`; a manifest predating that required field is a separate,
        // deliberately-unsupported case handled by decode failure.)
        try writeManifestFixture(storage: storage, jobId: "job-legacy", json: """
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

        let loaded = storage.loadJobManifest(jobId: "job-legacy")
        #expect(loaded?.jobId == "job-legacy")
        #expect(loaded?.cells.map(\.cellId) == ["cell-a"])
        #expect(loaded?.schemaVersion == nil)
    }

    /// A cell written before the spec existed carries its model and flags as separate
    /// keys. It reassembles into the `ClientRunSpec` the run path reads, so a job saved by
    /// an earlier build still resumes — the one decode branch nothing this build writes can
    /// reach, and therefore the one no other test exercises.
    @Test func aPreSpecCellReassemblesIntoItsSpec() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try writeManifestFixture(storage: storage, jobId: "job-prespec", json: """
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

        let cell = try #require(storage.loadJobManifest(jobId: "job-prespec")?.cells.first)
        #expect(cell.spec.benchmark == "decode_throughput_512_100")
        #expect(cell.spec.model == cell.source)
        // The runtime is derived, not stored: a pre-spec cell never named one, and this
        // client can only be what it compiled as.
        #expect(cell.spec.runtime == Runtime.thisBuild(for: cell.source))
        #expect(cell.spec.runtimeFlags?.numberGpuLayers == 33)
    }

    @Test func saveStampsCurrentSchemaVersion() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        storage.saveJobManifest(makeManifest(jobId: "job-stamped"))

        let loaded = storage.loadJobManifest(jobId: "job-stamped")
        #expect(loaded?.schemaVersion == JobManifestSchema.currentVersion)

        // The stamp must be in the file itself, not just the decoded struct.
        let raw = try Data(contentsOf: manifestURL(storage: storage, jobId: "job-stamped"))
        let json = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        #expect(json["schemaVersion"] as? Int == JobManifestSchema.currentVersion)
    }

    @Test func corruptManifestFailsLoudlyAndKeepsFile() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        try writeManifestFixture(storage: storage, jobId: "job-corrupt", json: "not json {{{")
        storage.saveJobManifest(makeManifest(jobId: "job-ok"))

        #expect(storage.loadJobManifest(jobId: "job-corrupt") == nil)
        // The healthy sibling still loads, and the corrupt file isn't deleted.
        #expect(storage.loadAllJobManifests().map(\.jobId) == ["job-ok"])
        #expect(FileManager.default.fileExists(atPath: manifestURL(storage: storage, jobId: "job-corrupt").path))
    }

    @Test func futureSchemaVersionDecodesBestEffort() throws {
        let storage = makeTemporaryStorage()
        defer { removeStorage(storage) }

        // Simulates an app downgrade: a manifest written by a future version
        // with a higher schemaVersion and fields this build doesn't know.
        try writeManifestFixture(storage: storage, jobId: "job-future", json: """
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

        let loaded = storage.loadJobManifest(jobId: "job-future")
        #expect(loaded?.jobId == "job-future")
        #expect(loaded?.schemaVersion == JobManifestSchema.currentVersion + 1)
    }

    // MARK: - Helpers

    private func manifestURL(storage: Storage, jobId: JobId) -> URL {
        storage.jobsDir
            .appendingPathComponent(jobId.value, isDirectory: true)
            .appendingPathComponent("manifest.json")
    }

    private func writeManifestFixture(storage: Storage, jobId: JobId, json: String) throws {
        let dir = storage.jobsDir.appendingPathComponent(jobId.value, isDirectory: true)
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
