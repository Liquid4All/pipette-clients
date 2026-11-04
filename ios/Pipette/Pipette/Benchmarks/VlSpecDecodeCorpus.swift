import CryptoKit
import Foundation

// `vl-spec-decode-corpus/1`: the corpus tar a `vl_spec_decode` job downloads, verified,
// unpacked and read back. The port of `crates/pipette-vl-spec-decode-corpus/src/lib.rs`:
// change the two together.
//
// Order is part of the measurement: the two arms are only comparable because they walk the
// same one, so `samples` keeps the file's order and nothing here sorts or shuffles it.

/// One `/v1/chat/completions` content part upstream; one mtmd marker plus its bitmap here.
nonisolated enum VlSpecDecodeCorpusPart: Equatable, Sendable, Decodable {
    case text(String)
    /// Tar-relative, `media/<sha256>.<ext>`.
    case image(path: String, mime: String, bytes: UInt64)

    private enum CodingKeys: String, CodingKey { case type, text, path, mime, bytes }
    private enum Tag: String { case text, image }

    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let raw = try c.decode(String.self, forKey: .type)
        guard let tag = Tag(rawValue: raw) else { throw VlSpecDecodeCorpusError.unknownPartType(raw) }
        switch tag {
        case .text:
            self = .text(try c.decode(String.self, forKey: .text))
        case .image:
            self = .image(path: try c.decode(String.self, forKey: .path),
                          mime: try c.decode(String.self, forKey: .mime),
                          bytes: try c.decode(UInt64.self, forKey: .bytes))
        }
    }
}

/// One `samples.jsonl` line: a whole conversation, user turns only (each arm generates the
/// replies).
nonisolated struct VlSpecDecodeCorpusSample: Equatable, Sendable, Decodable {
    let id: String
    let subset: String
    let system: String
    let turns: [[VlSpecDecodeCorpusPart]]
}

nonisolated struct VlSpecDecodeCorpusSubsetCounts: Equatable, Sendable, Decodable {
    var samples: UInt32 = 0
    var turns: UInt32 = 0
}

/// `manifest.json`.
nonisolated struct VlSpecDecodeCorpusManifest: Equatable, Sendable, Decodable {
    let format: CorpusFormat
    let adapter: DatasetAdapter
    let upstream: DatasetUpstream
    let samples: UInt32
    let turns: UInt32
    let subsets: [String: VlSpecDecodeCorpusSubsetCounts]

    private enum CodingKeys: String, CodingKey {
        case format, adapter, upstream, samples, turns, subsets
    }

    nonisolated init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        format = try c.decode(CorpusFormat.self, forKey: .format)
        adapter = try c.decode(DatasetAdapter.self, forKey: .adapter)
        upstream = try c.decode(DatasetUpstream.self, forKey: .upstream)
        samples = try c.decode(UInt32.self, forKey: .samples)
        turns = try c.decode(UInt32.self, forKey: .turns)
        // `#[serde(default)]` upstream: a corpus that keeps one subset needs no breakdown.
        subsets = try c.decodeIfPresent([String: VlSpecDecodeCorpusSubsetCounts].self, forKey: .subsets)
            ?? [:]
    }
}

nonisolated struct VlSpecDecodeCorpus: Sendable {
    let root: URL
    let manifest: VlSpecDecodeCorpusManifest
    /// In file order; see the file header.
    let samples: [VlSpecDecodeCorpusSample]
    /// The sha256 this client computed over the tar it received. Submitted as
    /// `observation_dataset_digest` whether or not it matched the catalog.
    let digest: Sha256

    var turns: Int { samples.reduce(0) { $0 + $1.turns.count } }

    /// Re-checked rather than trusted from the unpack, which checked the archive's entries,
    /// not the paths `samples.jsonl` asks for: an index pointing outside the corpus would
    /// otherwise read host files into a prompt.
    func mediaPath(_ partPath: String) throws -> URL {
        guard let relative = Self.safeRelative(partPath) else {
            throw VlSpecDecodeCorpusError.escapingPath(partPath)
        }
        let path = root.appending(path: relative, directoryHint: .notDirectory)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw VlSpecDecodeCorpusError.missingMedia(path.path)
        }
        return path
    }

    /// `nil` when `name` could land outside the corpus; upstream's `safe_relative`, which
    /// accepts only `Component::Normal` segments.
    ///
    /// One trailing `/` is dropped, because `tar` writes directory entries that way and
    /// Rust's `Path::components()` drops it too. An interior empty segment (`media//aa.png`)
    /// is still refused: nothing writes that, and it is how a crafted path hides a component.
    static func safeRelative(_ name: String) -> String? {
        var name = Substring(name)
        if name.count > 1, name.hasSuffix("/") { name = name.dropLast() }
        guard !name.isEmpty, !name.hasPrefix("/") else { return nil }
        let segments = name.split(separator: "/", omittingEmptySubsequences: false)
        guard segments.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else { return nil }
        return segments.joined(separator: "/")
    }
}

// MARK: - Verify, unpack, read back

nonisolated extension VlSpecDecodeCorpus {
    /// Download `dataset` through the client, verify it against its pin, and unpack it under
    /// `<datasetsDir>/<benchmarkId>/`.
    static func download(_ runContext: any RunContext, benchmarkId: String,
                         dataset: Dataset) async throws -> VlSpecDecodeCorpusLease {
        let tar = try await runContext.fetchDataset(dataset)
        if UInt64(tar.count) != dataset.bytes {
            // Not fatal: the digest is the real check, but this is the first clue when it fails.
            AppLog.corpus.warning(
                "corpus for \(benchmarkId) arrived as \(tar.count) bytes, the catalog pins "
                    + "\(dataset.bytes)")
        }
        let dir = runContext.datasetsDir.appending(
            path: directoryName(for: benchmarkId), directoryHint: .isDirectory)
        // A job that died between unpack and cleanup left a tree here; replace it rather
        // than unpacking over it, so a stale file cannot reach this traversal.
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
        let (corpus, matched) = try verifyAndOpen(tar: tar, expected: dataset.digest, dest: dir)
        AppLog.corpus.info(
            "corpus for \(benchmarkId): \(corpus.samples.count) samples / \(corpus.turns) "
                + "turns, digest \(matched ? "matches" : "DOES NOT MATCH") the catalog")
        return VlSpecDecodeCorpusLease(corpus: corpus, dir: dir)
    }

    /// `BenchmarkId` refuses `/`, but a catalog-side prefix is spelled with one, so it is
    /// folded rather than trusted.
    private static func directoryName(for benchmarkId: String) -> String {
        String(BenchmarkType.withoutCatalogSide(benchmarkId))
            .replacingOccurrences(of: "/", with: "_")
    }

    /// Also returns whether the digest matched. A mismatch is reported, never rejected: the
    /// row tells an analyst which corpus was measured, and refusing would turn a server-side
    /// repack into silence instead of a recorded fact.
    static func verifyAndOpen(tar: Data, expected: Sha256, dest: URL) throws -> (VlSpecDecodeCorpus, Bool) {
        let digest = try Sha256(Data(SHA256.hash(data: tar)).hexEncodedString())
        let matched = digest == expected
        if !matched {
            AppLog.corpus.warning(
                "corpus digest \(digest.value) does not match the catalog's "
                    + "\(expected.value); recording the measured digest and continuing")
        }
        try unpack(tar: tar, dest: dest)
        return (try open(root: dest, digest: digest), matched)
    }

    /// The format is closed (`manifest.json`, `samples.jsonl`, `media/*`), so an archive
    /// holding anything more is not the format the catalog pinned and is refused.
    static func unpack(tar: Data, dest: URL) throws {
        try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
        for entry in try Tar.entries(of: tar) {
            guard let relative = VlSpecDecodeCorpus.safeRelative(entry.name) else {
                throw VlSpecDecodeCorpusError.escapingPath(entry.name)
            }
            guard entry.name == "manifest.json" || entry.name == "samples.jsonl"
                    || entry.name.hasPrefix("media/") else {
                throw VlSpecDecodeCorpusError.unexpectedEntry(entry.name)
            }
            switch entry.kind {
            case .directory: continue
            case .regular: break
            }
            let out = dest.appending(path: relative, directoryHint: .notDirectory)
            try FileManager.default.createDirectory(
                at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
            try entry.contents.write(to: out, options: .atomic)
        }
    }

    static func open(root: URL, digest: Sha256) throws -> VlSpecDecodeCorpus {
        let decoder = JSONDecoder()
        let manifestPath = root.appending(path: "manifest.json", directoryHint: .notDirectory)
        let manifest: VlSpecDecodeCorpusManifest
        do {
            manifest = try decoder.decode(
                VlSpecDecodeCorpusManifest.self, from: try Data(contentsOf: manifestPath))
        } catch {
            throw VlSpecDecodeCorpusError.unreadableIndex(file: "manifest.json", reason: "\(error)")
        }
        let samplesPath = root.appending(path: "samples.jsonl", directoryHint: .notDirectory)
        guard let text = String(data: try Data(contentsOf: samplesPath), encoding: .utf8) else {
            throw VlSpecDecodeCorpusError.unreadableIndex(file: "samples.jsonl", reason: "not UTF-8")
        }
        var samples: [VlSpecDecodeCorpusSample] = []
        for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false)
            .enumerated() where !line.trimmingCharacters(in: .whitespaces).isEmpty {
            do {
                samples.append(try decoder.decode(
                    VlSpecDecodeCorpusSample.self, from: Data(line.utf8)))
            } catch {
                throw VlSpecDecodeCorpusError.unreadableIndex(
                    file: "samples.jsonl line \(index + 1)", reason: "\(error)")
            }
        }
        guard !samples.isEmpty else { throw VlSpecDecodeCorpusError.empty }

        // The manifest is the server's own tally, so a disagreement means the two files
        // describe different corpora.
        let turns = samples.reduce(0) { $0 + $1.turns.count }
        guard samples.count == Int(manifest.samples), turns == Int(manifest.turns) else {
            throw VlSpecDecodeCorpusError.countMismatch(
                manifestSamples: manifest.samples, manifestTurns: manifest.turns,
                fileSamples: samples.count, fileTurns: turns)
        }
        return VlSpecDecodeCorpus(root: root, manifest: manifest, samples: samples, digest: digest)
    }
}

/// An unpacked corpus that deletes itself: the server serves it `Cache-Control: no-store`,
/// so it belongs to the job, not the host.
///
/// A `deinit` rather than a cleanup call because a run has several ways to end (a failed
/// arm, a cancelled job, a thrown error). Upstream's lease does the same in `Drop`.
nonisolated final class VlSpecDecodeCorpusLease {
    let corpus: VlSpecDecodeCorpus
    /// `nil` once `release` has run, so a second release, or a `deinit` after one, is a
    /// no-op.
    private var dir: URL?

    init(corpus: VlSpecDecodeCorpus, dir: URL) {
        self.corpus = corpus
        self.dir = dir
    }

    /// Throws where the `deinit` path can only log.
    func release() throws {
        guard let dir else { return }
        self.dir = nil
        try FileManager.default.removeItem(at: dir)
    }

    deinit {
        do {
            try release()
        } catch {
            AppLog.corpus.warning("failed to delete the corpus: \(error)")
        }
    }
}

// MARK: - tar

/// The `ustar` reading this format needs, and nothing more.
///
/// Written here rather than taken from a package: Foundation ships no tar reader, the
/// format is an uncompressed tar of regular files and directories, and a dependency for one
/// archive shape is one more the licence-bundling step has to carry.
nonisolated enum Tar {
    /// Every other entry type is refused: a link written as a file would be an empty media
    /// file, and followed it could point anywhere.
    enum Kind { case regular, directory }

    struct Entry {
        let name: String
        let kind: Kind
        let contents: Data
    }

    static let blockSize = 512

    /// The header checksum is verified because the digest check is recorded, not enforced:
    /// a truncated download that still parsed as a header would otherwise unpack as garbage.
    static func entries(of archive: Data) throws -> [Entry] {
        var entries: [Entry] = []
        var offset = 0
        while offset + blockSize <= archive.count {
            let header = archive[archive.startIndex + offset ..< archive.startIndex + offset
                + blockSize]
            offset += blockSize
            // Two zero blocks end the archive; one is enough, since no entry can follow it.
            if header.allSatisfy({ $0 == 0 }) { break }
            guard checksumMatches(header) else { throw VlSpecDecodeCorpusError.corruptArchive("bad header checksum") }
            let name = try entryName(header)
            let size = try octal(header, at: 124, count: 12)
            guard let flag = header.dropFirst(156).first else {
                throw VlSpecDecodeCorpusError.corruptArchive("truncated header")
            }
            let kind: Kind
            switch flag {
            // `0` and NUL are both a regular file; `5` is a directory.
            case UInt8(ascii: "0"), 0: kind = .regular
            case UInt8(ascii: "5"): kind = .directory
            default:
                throw VlSpecDecodeCorpusError.unexpectedEntryKind(
                    name: name, kind: String(UnicodeScalar(flag)))
            }
            let start = archive.startIndex + offset
            guard start + Int(size) <= archive.endIndex else {
                throw VlSpecDecodeCorpusError.corruptArchive("entry \(name) runs past the end of the archive")
            }
            entries.append(Entry(name: name, kind: kind,
                                 contents: Data(archive[start ..< start + Int(size)])))
            // Contents are padded to a block boundary.
            offset += (Int(size) + blockSize - 1) / blockSize * blockSize
        }
        return entries
    }

    /// How `ustar` spells a path too long for the 100-byte name field.
    private static func entryName(_ header: Data) throws -> String {
        let name = string(header, at: 0, count: 100)
        let prefix = string(header, at: 345, count: 155)
        guard !name.isEmpty else { throw VlSpecDecodeCorpusError.corruptArchive("entry with no name") }
        return prefix.isEmpty ? name : "\(prefix)/\(name)"
    }

    /// Every byte summed with the checksum field read as spaces.
    private static func checksumMatches(_ header: Data) -> Bool {
        guard let stored = try? octal(header, at: 148, count: 8) else { return false }
        var sum = 0
        for (index, byte) in header.enumerated() {
            sum += (148..<156).contains(index) ? Int(UInt8(ascii: " ")) : Int(byte)
        }
        return sum == Int(stored)
    }

    /// A NUL/space-terminated ASCII field.
    private static func string(_ header: Data, at start: Int, count: Int) -> String {
        let bytes = header.dropFirst(start).prefix(count)
            .prefix { $0 != 0 && $0 != UInt8(ascii: " ") }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// A NUL/space-terminated octal field. Empty reads as zero, as a directory's size does.
    private static func octal(_ header: Data, at start: Int, count: Int) throws -> UInt64 {
        let text = string(header, at: start, count: count)
        if text.isEmpty { return 0 }
        guard let value = UInt64(text, radix: 8) else {
            throw VlSpecDecodeCorpusError.corruptArchive("field at \(start) is not octal: \(text)")
        }
        return value
    }
}

// MARK: - Errors

/// About the corpus bytes; `DatasetError` is about the catalog entry describing them.
nonisolated enum VlSpecDecodeCorpusError: Error, Equatable, LocalizedError {
    case unknownPartType(String)
    case escapingPath(String)
    case unexpectedEntry(String)
    case unexpectedEntryKind(name: String, kind: String)
    case corruptArchive(String)
    case unreadableIndex(file: String, reason: String)
    case missingMedia(String)
    case empty
    case countMismatch(manifestSamples: UInt32, manifestTurns: UInt32,
                       fileSamples: Int, fileTurns: Int)

    var errorDescription: String? {
        switch self {
        case let .unknownPartType(type): "unknown corpus content part type: \(type)"
        case let .escapingPath(path): "corpus path \(path) escapes the corpus"
        case let .unexpectedEntry(name): "corpus tar holds an unexpected entry: \(name)"
        case let .unexpectedEntryKind(name, kind):
            "corpus tar entry \(name) is a \(kind), not a file"
        case let .corruptArchive(reason): "the corpus is not a readable tar: \(reason)"
        case let .unreadableIndex(file, reason): "failed to parse \(file): \(reason)"
        case let .missingMedia(path): "corpus media file is missing: \(path)"
        case .empty: "the corpus holds no samples"
        case let .countMismatch(manifestSamples, manifestTurns, fileSamples, fileTurns):
            "the corpus manifest counts \(manifestSamples) samples / \(manifestTurns) "
                + "turns, but samples.jsonl holds \(fileSamples) / \(fileTurns)"
        }
    }
}
