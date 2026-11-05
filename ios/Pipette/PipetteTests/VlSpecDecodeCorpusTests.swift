import CryptoKit
import Foundation
import Testing

@testable import Pipette

/// The Swift half of `crates/pipette-vl-spec-decode-corpus`'s suite. The archives are built
/// here rather than checked in: a fixture tar would freeze one writer's padding choices
/// into the expectations.
@Suite
struct VlSpecDecodeCorpusTests {
    // MARK: - Fixtures

    /// A `ustar` header + contents for one entry, padded to the block boundary, written the
    /// way `tar` writes it.
    static func tarEntry(name: String, contents: Data, kind: UInt8 = UInt8(ascii: "0")) -> Data {
        var header = [UInt8](repeating: 0, count: Tar.blockSize)
        func write(_ string: String, at offset: Int, width: Int) {
            for (index, byte) in Array(string.utf8).prefix(width - 1).enumerated() {
                header[offset + index] = byte
            }
        }
        // `name` and `prefix` split a long path; these fixtures stay inside 100 bytes.
        write(name, at: 0, width: 100)
        write("0000644", at: 100, width: 8)
        write(String(format: "%011o", contents.count), at: 124, width: 12)
        header[156] = kind
        write("ustar", at: 257, width: 6)
        write("00", at: 263, width: 3)
        // The checksum is computed with its own field read as spaces, then written in.
        for index in 148..<156 { header[index] = UInt8(ascii: " ") }
        let sum = header.reduce(0) { $0 + Int($1) }
        write(String(format: "%06o", sum), at: 148, width: 8)
        header[154] = 0
        header[155] = UInt8(ascii: " ")

        var out = Data(header)
        out.append(contents)
        let padding = (Tar.blockSize - contents.count % Tar.blockSize) % Tar.blockSize
        out.append(Data(repeating: 0, count: padding))
        return out
    }

    /// The two trailing zero blocks that end an archive.
    static let tarTrailer = Data(repeating: 0, count: Tar.blockSize * 2)

    static let upstreamJson = """
        {"source":"git","url":"github.com/acme/corpus",
         "revision":"0123456789abcdef0123456789abcdef01234567",
         "identity":"2923603292c7dd463ad01a119f57965817778fc3",
         "digest":"\(String(repeating: "c", count: 64))","license":"Apache-2.0"}
        """

    static func manifest(samples: Int, turns: Int) -> Data {
        Data("""
            {"format":"vl-spec-decode-corpus/1","adapter":"mmspec@1",
             "upstream":\(upstreamJson),"samples":\(samples),"turns":\(turns),
             "subsets":{"GQA":{"samples":\(samples),"turns":\(turns)}}}
            """.utf8)
    }

    /// One sample line: a system prompt and `turns` user turns, the first carrying an image.
    static func sampleLine(id: String, turns: Int) -> String {
        let bodies = (0..<turns).map { index -> String in
            index == 0
                ? """
                  [{"type":"text","text":"what is this? "},
                   {"type":"image","path":"media/aa.png","mime":"image/png","bytes":4}]
                  """
                : #"[{"type":"text","text":"and turn \#(index)?"}]"#
        }
        return """
            {"id":"\(id)","subset":"GQA","system":"You are helpful.",
             "turns":[\(bodies.joined(separator: ","))]}
            """
    }

    /// A whole corpus tar: manifest, samples and one media file.
    static func archive(samples: [(id: String, turns: Int)]) -> Data {
        let lines = samples.map { sampleLine(id: $0.id, turns: $0.turns) }
            .map { $0.replacingOccurrences(of: "\n", with: "") }
        var out = Data()
        out.append(tarEntry(name: "manifest.json",
                            contents: manifest(samples: samples.count,
                                               turns: samples.reduce(0) { $0 + $1.turns })))
        out.append(tarEntry(name: "samples.jsonl",
                            contents: Data((lines.joined(separator: "\n") + "\n").utf8)))
        out.append(tarEntry(name: "media/", contents: Data(), kind: UInt8(ascii: "5")))
        out.append(tarEntry(name: "media/aa.png", contents: Data([0x89, 0x50, 0x4E, 0x47])))
        out.append(tarTrailer)
        return out
    }

    static func scratch() throws -> URL {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "corpus-tests-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func digest(of data: Data) throws -> Sha256 {
        try Sha256(Data(SHA256.hash(data: data)).hexEncodedString())
    }

    // MARK: - The tar reader

    @Test func readsEveryEntryInTheArchivesOrder() throws {
        let entries = try Tar.entries(of: Self.archive(samples: [("s1", 2)]))
        #expect(entries.map(\.name)
            == ["manifest.json", "samples.jsonl", "media/", "media/aa.png"])
        #expect(entries[3].contents == Data([0x89, 0x50, 0x4E, 0x47]))
        #expect(entries[2].kind == .directory)
    }

    /// A truncated download that still parsed as a header would otherwise be unpacked as
    /// garbage: the digest check is *recorded*, not enforced, so this is the only thing
    /// standing in the way.
    @Test func refusesAHeaderWhoseChecksumDoesNotMatch() throws {
        var archive = Self.archive(samples: [("s1", 1)])
        archive[0] = UInt8(ascii: "X")  // corrupt the name, leaving the checksum stale
        #expect(throws: VlSpecDecodeCorpusError.corruptArchive("bad header checksum")) {
            try Tar.entries(of: archive)
        }
    }

    /// A link written as a file would be an empty media file, and followed it could point
    /// anywhere, so every type but a regular file and a directory is refused by name.
    @Test func refusesAnEntryThatIsNotAFileOrADirectory() throws {
        var archive = Self.tarEntry(name: "media/aa.png", contents: Data(),
                                    kind: UInt8(ascii: "2"))
        archive.append(Self.tarTrailer)
        #expect(throws: VlSpecDecodeCorpusError.unexpectedEntryKind(name: "media/aa.png", kind: "2")) {
            try Tar.entries(of: archive)
        }
    }

    // MARK: - Unpack

    @Test func unpackRefusesAnEntryOutsideTheFormat() throws {
        let dir = try Self.scratch()
        var archive = Self.tarEntry(name: "notes.txt", contents: Data("hi".utf8))
        archive.append(Self.tarTrailer)
        #expect(throws: VlSpecDecodeCorpusError.unexpectedEntry("notes.txt")) {
            try VlSpecDecodeCorpus.unpack(tar: archive, dest: dir)
        }
    }

    @Test func unpackRefusesAnEntryThatEscapesTheArchive() throws {
        let dir = try Self.scratch()
        var archive = Self.tarEntry(name: "media/../../escape.png", contents: Data())
        archive.append(Self.tarTrailer)
        #expect(throws: VlSpecDecodeCorpusError.escapingPath("media/../../escape.png")) {
            try VlSpecDecodeCorpus.unpack(tar: archive, dest: dir)
        }
    }

    // MARK: - Open

    @Test func opensACorpusInTheFilesOrder() throws {
        let archive = Self.archive(samples: [("s1", 2), ("s2", 1)])
        let (corpus, matched) = try VlSpecDecodeCorpus.verifyAndOpen(
            tar: archive, expected: try Self.digest(of: archive), dest: try Self.scratch())

        #expect(matched)
        #expect(corpus.samples.map(\.id) == ["s1", "s2"])
        #expect(corpus.turns == 3)
        #expect(corpus.manifest.format == .vlSpecDecodeCorpusV1)
        #expect(corpus.manifest.subsets["GQA"]?.turns == 3)
        #expect(corpus.samples[0].system == "You are helpful.")
        #expect(corpus.samples[0].turns[0].count == 2)
    }

    /// A mismatch is recorded, never rejected: the row is what tells an analyst which
    /// corpus was measured, and the *computed* digest is what gets submitted.
    @Test func aDigestMismatchIsReportedAndTheCorpusStillOpens() throws {
        let archive = Self.archive(samples: [("s1", 1)])
        let (corpus, matched) = try VlSpecDecodeCorpus.verifyAndOpen(
            tar: archive,
            expected: Sha256(validated: String(repeating: "f", count: 64)),
            dest: try Self.scratch())

        #expect(!matched)
        #expect(corpus.digest == (try Self.digest(of: archive)))
    }

    /// The manifest's counts are the server's own tally, so a disagreement means the two
    /// files describe different corpora, and the traversal would report a turn count
    /// nothing else in the row agrees with.
    @Test func refusesAManifestWhoseCountsDisagreeWithTheSamples() throws {
        var archive = Data()
        archive.append(Self.tarEntry(name: "manifest.json",
                                     contents: Self.manifest(samples: 9, turns: 9)))
        archive.append(Self.tarEntry(
            name: "samples.jsonl",
            contents: Data((Self.sampleLine(id: "s1", turns: 1)
                .replacingOccurrences(of: "\n", with: "") + "\n").utf8)))
        archive.append(Self.tarTrailer)
        let dir = try Self.scratch()
        #expect(throws: VlSpecDecodeCorpusError.countMismatch(manifestSamples: 9, manifestTurns: 9,
                                                  fileSamples: 1, fileTurns: 1)) {
            try VlSpecDecodeCorpus.verifyAndOpen(
                tar: archive, expected: try Self.digest(of: archive), dest: dir)
        }
    }

    // MARK: - Media paths

    /// Re-checked at use rather than trusted from the unpack: a corpus whose index pointed
    /// outside itself would otherwise read host files into a prompt.
    @Test func mediaPathRefusesAPathThatEscapesTheCorpus() throws {
        let archive = Self.archive(samples: [("s1", 1)])
        let (corpus, _) = try VlSpecDecodeCorpus.verifyAndOpen(
            tar: archive, expected: try Self.digest(of: archive), dest: try Self.scratch())

        #expect(throws: VlSpecDecodeCorpusError.escapingPath("../../etc/hosts")) {
            try corpus.mediaPath("../../etc/hosts")
        }
        #expect(try corpus.mediaPath("media/aa.png").lastPathComponent == "aa.png")
    }

    @Test func mediaPathRefusesAFileTheCorpusDoesNotHold() throws {
        let archive = Self.archive(samples: [("s1", 1)])
        let (corpus, _) = try VlSpecDecodeCorpus.verifyAndOpen(
            tar: archive, expected: try Self.digest(of: archive), dest: try Self.scratch())
        #expect(throws: (any Error).self) { try corpus.mediaPath("media/missing.png") }
    }

    // MARK: - Download

    /// What the client serves lands in one directory per benchmark under its datasets
    /// directory, and goes when the lease is released.
    @Test func aDownloadedCorpusIsUnpackedPerBenchmarkAndLeased() async throws {
        let archive = Self.archive(samples: [("s1", 2), ("s2", 1)])
        let client = Serving(tar: archive, datasetsDir: try Self.scratch())
        let root = client.datasetsDir.appending(path: "vl_spec_decode_test",
                                                directoryHint: .isDirectory)

        let lease = try await VlSpecDecodeCorpus.download(
            client, benchmarkId: "vl_spec_decode_test",
            dataset: BenchmarkDefinitionTests.sampleDataset)

        #expect(lease.corpus.samples.map(\.id) == ["s1", "s2"])
        #expect(lease.corpus.digest == (try Self.digest(of: archive)))
        #expect(FileManager.default.fileExists(
            atPath: root.appending(path: "samples.jsonl").path))
        try lease.release()
        #expect(!FileManager.default.fileExists(atPath: root.path))
    }

    /// A tree a killed job left behind is replaced, not unpacked over.
    @Test func aStaleCorpusIsReplaced() async throws {
        let client = Serving(tar: Self.archive(samples: [("s1", 1)]),
                             datasetsDir: try Self.scratch())
        let stale = client.datasetsDir.appending(path: "vl_spec_decode_test/media/stale.png")
        try FileManager.default.createDirectory(
            at: stale.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("old".utf8).write(to: stale)

        let lease = try await VlSpecDecodeCorpus.download(
            client, benchmarkId: "vl_spec_decode_test",
            dataset: BenchmarkDefinitionTests.sampleDataset)

        #expect(!FileManager.default.fileExists(atPath: stale.path))
        try lease.release()
    }

    // MARK: - The lease

    /// The corpus belongs to the job: the server hands it out `no-store`, and the upstream
    /// owns the durable copy.
    @Test func releasingTheLeaseDeletesTheCorpus() throws {
        let dir = try Self.scratch()
        let archive = Self.archive(samples: [("s1", 1)])
        let (corpus, _) = try VlSpecDecodeCorpus.verifyAndOpen(
            tar: archive, expected: try Self.digest(of: archive), dest: dir)
        let lease = VlSpecDecodeCorpusLease(corpus: corpus, dir: dir)

        #expect(FileManager.default.fileExists(atPath: dir.path))
        try lease.release()
        #expect(!FileManager.default.fileExists(atPath: dir.path))
        // A second release is a no-op: the `deinit` path takes it after an explicit one.
        try lease.release()
    }
}

/// A client that serves one tar, into a directory of the test's.
private nonisolated struct Serving: RunContext {
    let tar: Data
    let datasetsDir: URL

    func fetchDataset(_ dataset: Dataset) async throws -> Data { tar }

    var evalCompletions: EvalCompletionsStore {
        EvalCompletionsStore(root: datasetsDir.appendingPathComponent("evals", isDirectory: true))
    }
}
