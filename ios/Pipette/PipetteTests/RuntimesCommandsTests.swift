import Foundation
import Testing

@testable import Pipette

/// `runtimes list`: the crate's `commands/runtimes.rs` rows over the engines this build
/// compiled in.
struct RuntimesCommandsTests {

    /// The value after `runtime=`, which is last on the line.
    private func runtimeColumn(_ row: String) -> String? {
        row.range(of: " runtime=").map { String(row[$0.upperBound...]) }
    }

    /// The crate's `RUNTIME` / `TYPE` / `DIGEST` columns, with the runtime as its
    /// `cli_ref`: a llama.cpp build as `version:flavor`.
    @Test func eachEngineIsListedWithItsTypeDigestAndCliRef() throws {
        let rows = try RuntimesCommands.runtimeRows(format: .name)
        // One per llama.cpp build the test app carries, then MLX, ONNX Runtime GenAI and
        // Apple Foundation.
        #expect(rows.count == LlamaPins.carried.count + 3)

        let onnx = try #require(rows.first { $0.contains("type=onnx_genai_ios_pipette ") })
        #expect(runtimeColumn(onnx)?.contains("onnxruntime-genai=") == true)

        let main = Runtime.carried(LlamaPins.main)
        let row = try #require(rows.first {
            runtimeColumn($0) == "\(LlamaPins.main.submissionVersion):ios-arm64"
        })
        #expect(row.contains("type=llamacpp_ios_pipette "))
        #expect(row.contains(" digest=\(shortDigest(try descriptorDigest(main))) "))

        let afm = try #require(rows.first { $0.contains("type=apple_foundation ") })
        #expect(runtimeColumn(afm) == "apple_foundation")
    }

    /// The column this client adds: the canonical JSON `benchmarks run runtime=` takes,
    /// since an on-device runtime has no URI for a listing to offer instead.
    @Test func theListedDescriptorIsWhatARunAccepts() throws {
        let mlx = try #require(Runtime.thisBuild(for: .mlxIosPipette))
        let row = try #require(try RuntimesCommands.runtimeRows(format: .name)
            .first { $0.contains("type=mlx_ios_pipette ") })
        let descriptor = try #require(row.range(of: " descriptor=").flatMap { start in
            row.range(of: " runtime=", range: start.upperBound..<row.endIndex)
                .map { String(row[start.upperBound..<$0.lowerBound]) }
        })
        #expect(try JSONDecoder().decode(Runtime.self, from: Data(descriptor.utf8)) == mlx)
    }

    /// The same listing twice is the same text: a capture diffed against the last one
    /// shows what changed, not a reordering.
    @Test func aListingIsStableAcrossCalls() throws {
        #expect(try RuntimesCommands.runtimeRows(format: .name) == RuntimesCommands.runtimeRows(format: .name))
    }

    /// plan-types gives the on-device runtimes no URI form, so `format=uri` falls back to
    /// the `cli_ref`, as the crate falls back for any runtime it cannot write as a URI.
    @Test func theUriFormatFallsBackToTheCliRef() throws {
        #expect(try RuntimesCommands.runtimeRows(format: .uri) == RuntimesCommands.runtimeRows(format: .name))
    }

    @Test func rowsAreSortedByTheRuntimeColumn() throws {
        let columns = try RuntimesCommands.runtimeRows(format: .name).compactMap(runtimeColumn)
        #expect(columns == columns.sorted())
    }
}
