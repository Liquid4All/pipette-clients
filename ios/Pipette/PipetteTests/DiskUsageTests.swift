import Foundation
import Testing

@testable import Pipette

/// The one on-disk measurement (`DiskUsage`): recursive, hidden children included,
/// symlinks counted as links rather than followed. Each test builds its own temporary
/// tree, so the suite runs in parallel.
struct DiskUsageTests {
    private func makeTree() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("disk-usage-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @discardableResult
    private func write(_ bytes: Int, to url: URL) throws -> URL {
        try Data(repeating: 0x61, count: bytes).write(to: url)
        return url
    }

    @Test func recursesIntoNestedDirectories() throws {
        let root = try makeTree()
        defer { try? FileManager.default.removeItem(at: root) }
        let nested = root.appendingPathComponent("a/b", isDirectory: true)
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try write(64_000, to: nested.appendingPathComponent("deep.bin"))

        #expect(DiskUsage.bytes(at: root) >= 64_000)
    }

    @Test func doesNotFollowASymlinkToALargeFile() throws {
        let root = try makeTree()
        defer { try? FileManager.default.removeItem(at: root) }
        let target = try write(512_000, to: root.appendingPathComponent("big.bin"))
        let linkDir = root.appendingPathComponent("links", isDirectory: true)
        try FileManager.default.createDirectory(at: linkDir, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: linkDir.appendingPathComponent("big.bin"), withDestinationURL: target)

        // The link is a few bytes of path, not another copy of the target.
        #expect(DiskUsage.bytes(at: linkDir) < 512_000)
    }

    @Test func doesNotPullInATreeLinkedFromOutside() throws {
        let outside = try makeTree()
        let root = try makeTree()
        defer {
            try? FileManager.default.removeItem(at: outside)
            try? FileManager.default.removeItem(at: root)
        }
        try write(512_000, to: outside.appendingPathComponent("big.bin"))
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("elsewhere"), withDestinationURL: outside)

        #expect(DiskUsage.bytes(at: root) < 512_000)
    }

    @Test func countsHiddenChildren() throws {
        let root = try makeTree()
        defer { try? FileManager.default.removeItem(at: root) }
        let staging = root.appendingPathComponent(".staging", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try write(64_000, to: staging.appendingPathComponent("orphan.part"))

        #expect(DiskUsage.bytes(at: root) >= 64_000)
    }

    @Test func missingPathMeasuresZero() throws {
        let root = try makeTree()
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(DiskUsage.bytes(at: root.appendingPathComponent("nope")) == 0)
    }

    /// A file occupies whole blocks, so its usage is at least its length.
    @Test func aFileMeasuresAtLeastItsLength() throws {
        let root = try makeTree()
        defer { try? FileManager.default.removeItem(at: root) }
        let file = try write(5_000, to: root.appendingPathComponent("f.bin"))

        #expect(DiskUsage.bytes(at: file) >= 5_000)
    }
}
