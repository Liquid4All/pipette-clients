import XCTest
@testable import Pipette

@MainActor
final class MemoryGateTests: XCTestCase {
    private let gb: Int64 = 1024 * 1024 * 1024

    func testWarnsWhenModelCannotFitTheRemainingBudget() throws {
        let warning = try XCTUnwrap(MemoryGate.warning(
            modelName: "Qwen-Q8_0",
            modelBytes: 4 * gb,
            availableBytes: 2 * gb
        ))

        XCTAssertTrue(warning.contains("Qwen-Q8_0"))
        XCTAssertTrue(warning.contains("may kill"))
    }

    func testStaysQuietWhenModelFitsWithHeadroom() {
        XCTAssertNil(MemoryGate.warning(
            modelName: "Qwen-Q4_0",
            modelBytes: 1 * gb,
            availableBytes: 1 * gb + MemoryGate.headroomBytes
        ))
    }

    func testWarnsWhenWeightsFitButHeadroomDoesNot() {
        XCTAssertNotNil(MemoryGate.warning(
            modelName: "Qwen-Q4_0",
            modelBytes: 1 * gb,
            availableBytes: 1 * gb + MemoryGate.headroomBytes - 1
        ))
    }

    func testUnknownSizesNeverWarn() {
        XCTAssertNil(MemoryGate.warning(modelName: "m", modelBytes: 0, availableBytes: 2 * gb))
        XCTAssertNil(MemoryGate.warning(modelName: "m", modelBytes: 2 * gb, availableBytes: 0))
        XCTAssertNil(MemoryGate.warning(modelName: "m", modelBytes: -1, availableBytes: -1))
    }

    func testSnapshotIsNilForUnreadableModels() {
        // Missing path → no size → nothing to measure or record. (The other
        // nil arm — a zero jetsam budget — is the simulator's own behavior
        // and not assertable from here.)
        XCTAssertNil(MemoryGate.snapshot(modelPath: "/nonexistent/\(UUID().uuidString).gguf"))
    }

    func testSizeOfFileOrDirectoryMeasuresFilesAndSumsDirectories() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("memgate-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        let file = dir.appendingPathComponent("weights.gguf")
        try Data(count: 1024).write(to: file)
        XCTAssertEqual(MemoryGate.sizeOfFileOrDirectory(atPath: file.path), 1024)

        // MLX-style model directory: config + nested shards all count.
        let nested = dir.appendingPathComponent("shards")
        try fm.createDirectory(at: nested, withIntermediateDirectories: true)
        try Data(count: 2048).write(to: nested.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(MemoryGate.sizeOfFileOrDirectory(atPath: dir.path), 3072)

        XCTAssertEqual(MemoryGate.sizeOfFileOrDirectory(atPath: dir.appendingPathComponent("missing").path), 0)
    }
}
