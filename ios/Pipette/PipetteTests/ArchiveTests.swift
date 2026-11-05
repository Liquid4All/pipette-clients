import Compression
import Foundation
import Testing
import zlib

@testable import Pipette

// Mirrors `crates/pipette-artifacts/src/archive.rs`'s tests, plus the refusals that crate
// leaves to the `tar` crate.

/// Tar and gzip bytes built as mgmt's import builds them (pipette-mgmt's
/// pipette-pipeline crate, `hf_import/archive.rs`).
nonisolated enum ArchiveFixture {
    struct Entry {
        var name: String
        var contents = Data()
        var type = UInt8(ascii: "0")
        var linkname = ""
        var prefix = ""
        /// A size to write in the GNU base-256 form, in place of the size of `contents`.
        var base256Size: Int64?
    }

    /// The deterministic tar of `files`: entries in the byte order of their paths, and no
    /// directory entries.
    static func deterministicTar(_ files: [String: Data]) -> Data {
        tar(files.sorted { Array($0.key.utf8).lexicographicallyPrecedes(Array($1.key.utf8)) }
            .map { Entry(name: $0.key, contents: $0.value) })
    }

    static func tar(_ entries: [Entry]) -> Data {
        var out = Data()
        for entry in entries { out.append(self.entry(entry)) }
        out.append(Data(count: 2 * TarHeader.blockSize))
        return out
    }

    static func entry(_ entry: Entry) -> Data {
        var header = [UInt8](repeating: 0, count: TarHeader.blockSize)
        func write(_ string: String, at offset: Int) {
            for (index, byte) in string.utf8.enumerated() { header[offset + index] = byte }
        }
        write(entry.name, at: 0)
        write("0000644\0", at: 100)
        write("0000000\0", at: 108)
        write("0000000\0", at: 116)
        if let size = entry.base256Size {
            header[124] = 0x80
            for index in 0..<8 { header[128 + index] = UInt8(truncatingIfNeeded: size >> (8 * (7 - index))) }
        } else {
            write(String(format: "%011lo\0", entry.contents.count), at: 124)
        }
        write("00000000000\0", at: 136)
        header[156] = entry.type
        write(entry.linkname, at: 157)
        write("ustar\0", at: 257)
        write("00", at: 263)
        write(entry.prefix, at: 345)
        write(String(format: "%06o\0 ", checksum(header)), at: 148)
        var out = Data(header)
        out.append(entry.contents)
        out.append(Data(count: (512 - entry.contents.count % 512) % 512))
        return out
    }

    /// A PAX extended header that carries `records`, for the entry after it.
    static func pax(_ records: [(String, String)]) -> Entry {
        let body = records.map { key, value in
            let tail = " \(key)=\(value)\n"
            // The length counts its own digits.
            var length = tail.utf8.count + 1
            while "\(length)\(tail)".utf8.count != length { length += 1 }
            return "\(length)\(tail)"
        }.joined()
        return Entry(name: "PaxHeader", contents: Data(body.utf8), type: UInt8(ascii: "x"))
    }

    /// The header sum with the checksum field read as eight spaces.
    static func checksum(_ header: [UInt8]) -> Int {
        header.enumerated().reduce(0) { sum, item in
            sum + Int((148..<156).contains(item.offset) ? 0x20 : item.element)
        }
    }

    /// One gzip member of `data`. The header carries each optional field that has a value,
    /// and a header CRC when `headerCrc` is set.
    static func gzip(
        _ data: Data, name: String? = nil, extra: Data? = nil, comment: String? = nil,
        headerCrc: Bool = false
    ) throws -> Data {
        let capacity = data.count + 1024
        var deflated = Data(count: capacity)
        let size = deflated.withUnsafeMutableBytes { dst in
            data.withUnsafeBytes { src in
                guard let dst = dst.bindMemory(to: UInt8.self).baseAddress,
                      let src = src.bindMemory(to: UInt8.self).baseAddress
                else { return 0 }
                return compression_encode_buffer(
                    dst, capacity, src, data.count, nil, COMPRESSION_ZLIB)
            }
        }
        try #require(size > 0)
        let flags = (headerCrc ? 0x02 : 0) | (extra == nil ? 0 : 0x04) | (name == nil ? 0 : 0x08)
            | (comment == nil ? 0 : 0x10)
        var out = Data([0x1f, 0x8b, 8, UInt8(flags), 0, 0, 0, 0, 0, 3])
        if let extra {
            out.append(contentsOf: [UInt8(extra.count & 0xff), UInt8(extra.count >> 8)])
            out.append(extra)
        }
        if let name { out.append(Data(name.utf8) + [0]) }
        if let comment { out.append(Data(comment.utf8) + [0]) }
        if headerCrc {
            let check = crc32(out)
            out.append(contentsOf: [UInt8(check & 0xff), UInt8((check >> 8) & 0xff)])
        }
        out.append(deflated.prefix(size))
        out.append(littleEndian(crc32(data)))
        out.append(littleEndian(UInt32(truncatingIfNeeded: data.count)))
        return out
    }

    static func crc32(_ data: Data) -> UInt32 {
        data.withUnsafeBytes { raw in
            UInt32(zlib.crc32(0, raw.bindMemory(to: Bytef.self).baseAddress, uInt(raw.count)))
        }
    }

    static func littleEndian(_ value: UInt32) -> Data {
        Data((0..<4).map { UInt8(truncatingIfNeeded: value >> (8 * $0)) })
    }
}

/// A seeded generator, so a test's random bytes are the same on every run.
private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9e37_79b9_7f4a_7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z >> 31)
    }
}

/// A source over bytes in memory that records the largest read asked of it.
private final class CountingSource: ByteSource {
    private let data: Data
    private var offset = 0
    private(set) var largestRead = 0
    private(set) var reads = 0

    init(_ data: Data) { self.data = data }

    func read(upTo count: Int) throws(ArchiveError) -> Data {
        largestRead = max(largestRead, count)
        reads += 1
        let end = min(offset + count, data.count)
        defer { offset = end }
        return data.subdata(in: offset..<end)
    }
}

struct ArchiveTests {
    /// The files of a small MLX model, one of them nested.
    private static let model: [String: Data] = [
        "config.json": Data(#"{"model_type":"lfm2"}"#.utf8),
        "model.safetensors": Data(repeating: 0x5a, count: 3_000),
        "tokenizer.json": Data("{}".utf8),
        "nested/tokenizer_config.json": Data("{}".utf8),
    ]

    private func scratch() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("ArchiveTests-\(UUID().uuidString)", isDirectory: true)
    }

    private func unpack(_ bytes: Data) throws -> (Archive.Unpacked, URL) {
        let dir = scratch()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent(".vault-archive")
        try bytes.write(to: file)
        let tree = dir.appendingPathComponent("model", isDirectory: true)
        return (try Archive.unpack(file, into: tree), tree)
    }

    private func refusal(_ bytes: Data) throws -> ArchiveError? {
        do {
            _ = try unpack(bytes)
            return nil
        } catch let error as ArchiveError {
            return error
        }
    }

    private func tree(at root: URL) throws -> [String: Data] {
        let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey])
        var files: [String: Data] = [:]
        while let url = enumerator?.nextObject() as? URL {
            guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true
            else { continue }
            let relative = String(url.standardizedFileURL.path
                .dropFirst(root.standardizedFileURL.path.count + 1))
            files[relative] = try Data(contentsOf: url)
        }
        return files
    }

    @Test(arguments: [ArchiveKind.tar, .tarGz])
    func theDeterministicTarAndItsGzipUnpackToTheModel(format: ArchiveKind) throws {
        let tar = ArchiveFixture.deterministicTar(Self.model)
        let bytes = format == .tar ? tar : try ArchiveFixture.gzip(tar, name: "model.tar")

        let (unpacked, root) = try unpack(bytes)

        #expect(unpacked == Archive.Unpacked(kind: format, entries: Self.model.count))
        #expect(try tree(at: root) == Self.model)
    }

    @Test func aGzipWithEveryHeaderFieldUnpacks() throws {
        let bytes = try ArchiveFixture.gzip(
            ArchiveFixture.deterministicTar(Self.model), name: "model.tar",
            extra: Data([0x41, 0x42, 2, 0, 7, 9]), comment: "a model", headerCrc: true)

        let (unpacked, root) = try unpack(bytes)

        #expect(unpacked == Archive.Unpacked(kind: .tarGz, entries: Self.model.count))
        #expect(try tree(at: root) == Self.model)
    }

    /// A GNU long-name entry gives the next entry its path, as the `tar` crate reads it.
    @Test func aGnuLongNameNamesTheNextEntry() throws {
        let path = "nested/" + String(repeating: "d", count: 120) + "/config.json"
        let bytes = ArchiveFixture.tar([
            .init(name: "././@LongLink", contents: Data(path.utf8) + [0],
                  type: UInt8(ascii: "L")),
            .init(name: String(path.prefix(100)), contents: Data("{}".utf8)),
        ])

        let (unpacked, root) = try unpack(bytes)

        #expect(unpacked.entries == 1)
        #expect(try tree(at: root) == [path: Data("{}".utf8)])
    }

    /// The path checks read bytes, so a combining mark after a `/` keeps the separator.
    @Test func aCombiningMarkAfterASeparatorUnpacks() throws {
        let name = "\u{0301}x"
        let (unpacked, root) = try unpack(
            ArchiveFixture.tar([.init(name: name, contents: Data("x".utf8))]))

        #expect(unpacked.entries == 1)
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path))
    }

    /// A tar that ends after its last entry, with no zero end blocks, unpacks as the `tar`
    /// crate unpacks it.
    @Test func aTarWithNoEndBlocksUnpacks() throws {
        let tar = ArchiveFixture.deterministicTar(Self.model)

        let (unpacked, root) = try unpack(tar.dropLast(2 * TarHeader.blockSize))

        #expect(unpacked == Archive.Unpacked(kind: .tar, entries: Self.model.count))
        #expect(try tree(at: root) == Self.model)
    }

    /// The tar.gz that macOS `bsdtar --format pax` and `gzip -9` make: a gzip header with
    /// `FNAME`, and a PAX header with times and `LIBARCHIVE.xattr` records before each file.
    @Test func aTarGzThatMacOSMadeUnpacks() throws {
        let bytes = try #require(Data(base64Encoded: """
            H4sICOyzwWoCA2ZpeHR1cmUudGFyAO2WW0oDMRiFR/GpqwizgJk/k1tHLNKq0EKVqiD4JHGa1krnwiS1lSK4\
            MdfjJnxwqiBYraj0Qmm+lz9wOJBADudvyVFdybbK/ShNOr2ud6vTxJkvAMApRZNZMD0BCFCEWcCBECwYRYAJ\
            oeCgkbMEBtrIvLjKHB756XFrAgEUmV6sKliEGERIgXplzMoMC45LhSq/UQkIFvKJGs/2MoGajVr17KDeuDjy\
            RtKY3IvS2JNZ1ldelqd3KpFJpCrV00aV+UO/28QntcNhiYbovDA1L38ybW07zy/7j0/HO3slx/JvFpf63+cf\
            AjaVf0y4cBDY/C+csRunbdW/MveZcnfdficO3Acbi41BD6791scO8PYXPC07yqhEp7leUf8zQoXt/9X1P4fi\
            wIIZ/c+h0N63g3i21/b/uuR/Man/Q/+D+LL/C277fxkMVa97Y7QNgsVisWwYr+FCBV8AFAAA
            """))

        let (unpacked, root) = try unpack(bytes)

        #expect(unpacked == Archive.Unpacked(kind: .tarGz, entries: 2))
        #expect(try tree(at: root) == [
            "config.json": Data(#"{"model_type":"lfm2"}"#.utf8),
            "sub/model.safetensors": Data("weights".utf8),
        ])
    }

    /// A PAX `path` replaces the next entry's name, and a ustar `prefix` joins the name.
    @Test func paxPathsAndUstarPrefixesNameTheEntry() throws {
        let long = String(repeating: "d", count: 120) + "/model.safetensors"
        let bytes = ArchiveFixture.tar([
            ArchiveFixture.pax([("mtime", "1.5"), ("path", long)]),
            .init(name: "truncated-name", contents: Data("a".utf8)),
            .init(name: "config.json", contents: Data("b".utf8), prefix: "deep/er"),
            .init(name: "./", type: UInt8(ascii: "5")),
            .init(name: "empty", type: UInt8(ascii: "5")),
        ])

        let (unpacked, root) = try unpack(bytes)

        #expect(unpacked.entries == 3)
        #expect(try tree(at: root) == [long: Data("a".utf8), "deep/er/config.json": Data("b".utf8)])
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("empty").path))
    }

    /// A Mac's `tar` adds an AppleDouble `._<name>` file for each file with extended
    /// attributes. The unpack skips those, as a Mac's `tar -x` does. A `._` file of another
    /// kind is written whole.
    ///
    /// Foundation's directory listings hide `._` names on macOS, so the test checks each
    /// such path by name rather than through `tree(at:)`.
    @Test(arguments: [ArchiveKind.tar, .tarGz])
    func appleDoubleFilesAreSkipped(format: ArchiveKind) throws {
        let appleDouble = Data([0x00, 0x05, 0x16, 0x07]) + Data(repeating: 0x41, count: 159)
        let tar = ArchiveFixture.tar([
            .init(name: "._config.json", contents: appleDouble),
            .init(name: "config.json", contents: Data("{}".utf8)),
            .init(name: "sub/._model.safetensors", contents: appleDouble),
            .init(name: "sub/model.safetensors", contents: Data("weights".utf8)),
            .init(name: "._notes", contents: Data("plain text".utf8)),
            .init(name: "._x", contents: Data("ab".utf8)),
        ])
        let bytes = format == .tar ? tar : try ArchiveFixture.gzip(tar)

        let (unpacked, root) = try unpack(bytes)

        #expect(unpacked == Archive.Unpacked(kind: format, entries: 4, skipped: 2))
        #expect(try Data(contentsOf: root.appendingPathComponent("config.json")) == Data("{}".utf8))
        #expect(try Data(contentsOf: root.appendingPathComponent("sub/model.safetensors"))
            == Data("weights".utf8))
        #expect(try Data(contentsOf: root.appendingPathComponent("._notes")) == Data("plain text".utf8))
        #expect(try Data(contentsOf: root.appendingPathComponent("._x")) == Data("ab".utf8))
        for skipped in ["._config.json", "sub/._model.safetensors"] {
            #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent(skipped).path))
        }
    }

    /// mgmt writes the size of a file of 8 GiB or more in the GNU base-256 form.
    @Test func aBase256SizeReads() {
        let nineGiB: Int64 = 9 << 30
        let field = [0x80, 0, 0, 0] + (0..<8).reversed().map {
            UInt8(truncatingIfNeeded: nineGiB >> (8 * $0))
        }
        #expect(TarHeader.numeric(field[...]) == nineGiB)
        #expect(TarHeader.numeric(Array("00000000017\0".utf8)[...]) == 15)
        #expect(TarHeader.numeric(([0xff] + [UInt8](repeating: 0xff, count: 11))[...]) == nil)
    }

    struct Refusal: CustomTestStringConvertible, Sendable {
        let label: String
        let bytes: @Sendable () throws -> Data
        let expected: ArchiveError
        var testDescription: String { label }
    }

    static let refusals: [Refusal] = [
        Refusal(label: "symbolic link", bytes: {
            ArchiveFixture.tar([.init(name: "link", type: UInt8(ascii: "2"), linkname: "/etc")])
        }, expected: .unsupportedEntry(name: "link", type: "symbolic link")),
        Refusal(label: "hard link", bytes: {
            ArchiveFixture.tar([.init(name: "link", type: UInt8(ascii: "1"), linkname: "a")])
        }, expected: .unsupportedEntry(name: "link", type: "hard link")),
        Refusal(label: "parent component", bytes: {
            ArchiveFixture.tar([.init(name: "a/../../escape", contents: Data("x".utf8))])
        }, expected: .unsafePath(name: "a/../../escape")),
        Refusal(label: "absolute path", bytes: {
            ArchiveFixture.tar([.init(name: "/tmp/escape", contents: Data("x".utf8))])
        }, expected: .unsafePath(name: "/tmp/escape")),
        Refusal(label: "parent component in a PAX path", bytes: {
            ArchiveFixture.tar([
                ArchiveFixture.pax([("path", "../escape")]),
                .init(name: "safe", contents: Data("x".utf8)),
            ])
        }, expected: .unsafePath(name: "../escape")),
        Refusal(label: "PAX record of the largest length", bytes: {
            let body = "11 path=ab\n\(Int.max) x=y\n"
            return ArchiveFixture.tar([
                .init(name: "PaxHeader", contents: Data(body.utf8), type: UInt8(ascii: "x")),
            ])
        }, expected: .badHeader(offset: 0, reason: "a PAX record is malformed")),
        Refusal(label: "global PAX header of the largest size", bytes: {
            ArchiveFixture.tar([
                .init(name: "PaxGlobal", type: UInt8(ascii: "g"), base256Size: Int64.max - 100),
            ])
        }, expected: .truncated),
        Refusal(label: "zip", bytes: { Data("PK\u{03}\u{04} a zipped model".utf8) },
                expected: .zipUnsupported),
        Refusal(label: "no archive", bytes: { Data("GGUF....".utf8) }, expected: .notAnArchive),
        Refusal(label: "bad header checksum", bytes: {
            var tar = ArchiveFixture.deterministicTar(["config.json": Data("{}".utf8)])
            tar[0] = UInt8(ascii: "X")
            return tar
        }, expected: .badHeaderChecksum(offset: 0)),
        Refusal(label: "parent component before a combining mark", bytes: {
            ArchiveFixture.tar([.init(name: "../\u{0301}x", contents: Data("x".utf8))])
        }, expected: .unsafePath(name: "../\u{0301}x")),
        Refusal(label: "parent component in a GNU long name", bytes: {
            ArchiveFixture.tar([
                .init(name: "././@LongLink", contents: Data("../escape".utf8) + [0],
                      type: UInt8(ascii: "L")),
                .init(name: "safe", contents: Data("x".utf8)),
            ])
        }, expected: .unsafePath(name: "../escape")),
        Refusal(label: "bad header CRC", bytes: {
            var gzip = try ArchiveFixture.gzip(
                ArchiveFixture.deterministicTar(model), headerCrc: true)
            gzip[10] ^= 0xff
            return gzip
        }, expected: .badGzip(reason: "the header CRC does not match the header")),
        Refusal(label: "entry path that is not UTF-8", bytes: {
            var tar = ArchiveFixture.tar([.init(name: "cafe", contents: Data("x".utf8))])
            tar[3] = 0xe9
            let header = [UInt8](tar.prefix(TarHeader.blockSize))
            let sum = Data(String(format: "%06o\0 ", ArchiveFixture.checksum(header)).utf8)
            tar.replaceSubrange(148..<156, with: sum)
            return tar
        }, expected: .badHeader(offset: 0, reason: "an entry path is not UTF-8")),
        Refusal(label: "PAX path that is not UTF-8", bytes: {
            let body = Data("13 path=caf".utf8) + [0xe9] + Data("\n".utf8)
            return ArchiveFixture.tar([
                .init(name: "PaxHeader", contents: body, type: UInt8(ascii: "x")),
                .init(name: "safe", contents: Data("x".utf8)),
            ])
        }, expected: .badHeader(offset: 0, reason: "an entry path is not UTF-8")),
        Refusal(label: "PAX header with no entry after it", bytes: {
            ArchiveFixture.entry(ArchiveFixture.pax([("path", "config.json")]))
        }, expected: .truncated),
        Refusal(label: "truncated tar", bytes: {
            ArchiveFixture.deterministicTar(["model.safetensors": Data(count: 3_000)])
                .prefix(2_048)
        }, expected: .truncated),
        Refusal(label: "truncated gzip", bytes: {
            let gzip = try ArchiveFixture.gzip(ArchiveFixture.deterministicTar(model))
            return gzip.prefix(gzip.count / 2)
        }, expected: .truncated),
        Refusal(label: "bad CRC-32", bytes: {
            let tar = ArchiveFixture.deterministicTar(model)
            var gzip = try ArchiveFixture.gzip(tar)
            gzip[gzip.count - 8] ^= 0xff
            return gzip
        }, expected: .badGzip(reason: "the CRC-32 in the trailer does not match the data")),
        Refusal(label: "bad length", bytes: {
            let tar = ArchiveFixture.deterministicTar(model)
            var gzip = try ArchiveFixture.gzip(tar)
            gzip.replaceSubrange((gzip.count - 4)..., with: ArchiveFixture.littleEndian(7))
            return gzip
        }, expected: .badGzip(reason: "the length in the trailer does not match the data")),
    ]

    @Test(arguments: refusals)
    func aRefusalNamesItsCauseAndIsPermanent(_ refusal: Refusal) throws {
        let error = try #require(try self.refusal(refusal.bytes()))
        #expect(error == refusal.expected)
        #expect(error.isPermanent)
    }

    /// A refusal stops the unpack at the entry it names, so nothing lands outside the root.
    @Test func aParentComponentWritesNothingOutsideTheRoot() throws {
        let dir = scratch()
        let bytes = ArchiveFixture.tar([.init(name: "../escape", contents: Data("x".utf8))])
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent(".vault-archive")
        try bytes.write(to: file)
        #expect(throws: ArchiveError.unsafePath(name: "../escape")) {
            try Archive.unpack(file, into: dir.appendingPathComponent("model"))
        }
        #expect(!FileManager.default.fileExists(atPath: dir.appendingPathComponent("escape").path))
    }

    /// The unpack never asks its source for more than one chunk, however large the
    /// archive: each entry streams to its file, and nothing holds the whole archive.
    @Test(arguments: [ArchiveKind.tar, .tarGz])
    func theUnpackReadsInChunks(format: ArchiveKind) throws {
        let chunk = 4_096
        // Bytes that do not compress, so the tar.gz also spans many chunks.
        var generator = SplitMix64(seed: 0x5eed)
        let files = Dictionary(uniqueKeysWithValues: (0..<20).map { index in
            ("shard-\(index).safetensors",
             Data((0..<(10_000 + index)).map { _ in UInt8.random(in: .min ... .max, using: &generator) }))
        })
        let tar = ArchiveFixture.deterministicTar(files)
        let bytes = format == .tar ? tar : try ArchiveFixture.gzip(tar)
        #expect(bytes.count > 10 * chunk)
        let source = CountingSource(bytes)
        let root = scratch()

        let unpacked = try Archive.extractArchive(source, kind: format, into: root, chunkSize: chunk)

        #expect(unpacked.entries == files.count)
        #expect(source.largestRead <= chunk)
        #expect(source.reads > 1)
        #expect(try tree(at: root) == files)
    }

    /// A disk that refused a write is this device's state, and another attempt can succeed.
    /// Every other failure is the archive's.
    @Test(arguments: [
        (ENOSPC, false), (EDQUOT, false), (EACCES, false), (EPERM, false), (EROFS, false),
        (ENOMEM, false), (EBUSY, false), (EAGAIN, false), (EINTR, false), (EMFILE, false),
        (ENFILE, false), (EIO, false), (ENAMETOOLONG, true), (EISDIR, true),
    ])
    func anIoFailureIsPermanentUnlessTheHostRefused(code: Int32, permanent: Bool) {
        #expect(ArchiveError.io(path: "/m", code: code, reason: "x").isPermanent == permanent)
        let wrapped = CocoaError(.fileWriteUnknown, userInfo: [
            NSUnderlyingErrorKey: NSError(domain: NSPOSIXErrorDomain, code: Int(code)),
        ])
        #expect(ArchiveError.io("/m", wrapped).isPermanent == permanent)
    }

    /// An I/O failure that names no POSIX code is the archive's, as the crate reads one.
    /// A cancel says nothing about the bytes.
    @Test func anUncodedFailureIsPermanentAndACancelIsNot() {
        #expect(ArchiveError.io(path: "/m", code: nil, reason: "x").isPermanent)
        #expect(!ArchiveError.cancelled.isPermanent)
    }
}
