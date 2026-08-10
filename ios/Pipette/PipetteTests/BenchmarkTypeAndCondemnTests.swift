import Foundation
import Testing

@testable import Pipette

/// `BenchmarkDefinition.init(from:)` now decodes the `benchmark_type` tag as the
/// `BenchmarkType` enum and switches exhaustively, so it can't drift from the
/// enum's rawValues. A known tag decodes into its typed variant; an unknown tag
/// still throws (via the synthesized rawValue decode → `DecodingError`), which is
/// the unchanged reject-unknown behavior.
struct BenchmarkDefinitionEnumTagDecodeTests {
    private func decode(_ json: String) throws -> BenchmarkDefinition {
        try JSONDecoder().decode(BenchmarkDefinition.self, from: Data(json.utf8))
    }

    @Test func knownTagDecodesIntoTypedVariant() throws {
        let def = try decode(
            #"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#)
        #expect(def == .prefillThroughput(benchmarkId: "p", prefillTokens: 512))
    }

    @Test func unknownTagStillThrows() {
        #expect(throws: (any Error).self) {
            try decode(#"{"benchmark_id":"x","benchmark_type":"mystery_metric"}"#)
        }
    }
}

/// Crash-condemnation groups siblings by the typed `source` (Model), which is the only
/// identity a cell carries. Exact for GGUF/MLX, and honest for AFM: every
/// `.appleFoundationText` cell shares one source, so a crash condemns the whole family.
struct CrashCondemnBySourceTests {
    private let ggufA: Model = ggufTextFixture("test/A-GGUF", "A-Q4_0.gguf")

    private func cell(
        id: CellId, source: Model, status: CellRunStatus, crashCount: Int? = nil
    ) -> JobCell {
        JobCell(
            cellId: id,
            benchmarkId: "decode",
            benchmarkType: .decodeThroughput,
            runStatus: status,
            serverJobId: nil,
            errorMessage: nil,
            crashCount: crashCount,
            source: source)
    }

    private func manifest(_ cells: [JobCell]) -> JobManifest {
        JobManifest(
            jobId: "job-1",
            createdAt: "2026-06-09T10:00:00Z",
            nGpuLayers: 99,
            contextSize: 4096,
            cells: cells,
            status: .running)
    }

    private func sentinel(_ cellId: CellId) -> ActiveCellSentinel {
        ActiveCellSentinel(cellId: cellId, startedAt: "2026-06-09T10:00:00Z")
    }

    /// AFM cells all share `.appleFoundationText`: a double crash condemns the sibling.
    @Test func doubleCrashCondemnsSiblingsWithSameSource() {
        var manifest = manifest([
            cell(id: "active", source: .appleFoundationText, status: .running, crashCount: 1),
            cell(id: "queued", source: .appleFoundationText, status: .pending),
        ])

        let changed = manifest.applyCrashEvidence(sentinel: sentinel("active"), payloadIsFresh: false)
        #expect(changed)

        #expect(manifest.cells[0].crashCount == 2)
        #expect(manifest.cells[0].runStatus == .failed)
        #expect(manifest.cells[1].runStatus == .failed)
    }

    /// Only the true same-source sibling is condemned; a cell of a different source
    /// running the same benchmark is left alone.
    @Test func differentSourceIsNotCrossCondemned() {
        var manifest = manifest([
            cell(id: "active", source: ggufA, status: .running, crashCount: 1),
            cell(id: "other-source", source: .appleFoundationText, status: .pending),
            cell(id: "same-source", source: ggufA, status: .pending),
        ])

        let changed = manifest.applyCrashEvidence(sentinel: sentinel("active"), payloadIsFresh: false)
        #expect(changed)

        #expect(manifest.cells[0].runStatus == .failed)
        // Different source → untouched.
        #expect(manifest.cells[1].runStatus == .pending)
        // True same-source sibling → condemned.
        #expect(manifest.cells[2].runStatus == .failed)
    }
}
