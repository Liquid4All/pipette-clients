import Foundation
import zlib

// Unpacking a vault archive (tar or tar.gz), after `crates/pipette-artifacts/src/archive.rs`.
//
// It streams in fixed chunks so memory never holds a whole model, using the system zlib
// and its own tar reader rather than a dependency (`enum Tar` in `VlSpecDecodeCorpus.swift`
// is a separate in-memory reader for a small corpus). It writes only regular files and
// directories under its root, refusing links, devices, absolute paths and `..`. It skips
// the AppleDouble `._<name>` files a Mac's `tar` adds, as `tar -x` does. mgmt's import
// writes a ustar tar with no directory entries, so parent directories are created on demand.

/// Read off the first bytes: a vault path names no file, so there is no extension.
nonisolated enum ArchiveKind: String, Sendable, Equatable {
    case tar
    case tarGz = "tar.gz"
    case zip

    /// The `ustar` magic ends at byte 262.
    static let headLength = 262

}

/// The crate's `sniff_archive_kind`.
nonisolated func sniffArchiveKind(_ head: Data) -> ArchiveKind? {
    let bytes = [UInt8](head.prefix(ArchiveKind.headLength))
    if bytes.starts(with: [0x1f, 0x8b]) { return .tarGz }
    if bytes.starts(with: [0x50, 0x4b, 0x03, 0x04]) { return .zip }
    if bytes.count >= ArchiveKind.headLength, bytes[257..<262].elementsEqual("ustar".utf8) {
        return .tar
    }
    return nil
}

nonisolated enum ArchiveError: Error, Equatable, LocalizedError {
    case notAnArchive
    case zipUnsupported
    case unsupportedEntry(name: String, type: String)
    case unsafePath(name: String)
    case badHeaderChecksum(offset: Int64)
    case badHeader(offset: Int64, reason: String)
    case truncated
    case badGzip(reason: String)
    case io(path: String, code: Int32?, reason: String)
    case cancelled

    /// The archive's bytes are pinned, so a bad format or entry fails the same way on
    /// every retry; an I/O error naming this device (see `hostCodes`) may not.
    var isPermanent: Bool {
        switch self {
        case let .io(_, code, _): !(code.map(Self.hostCodes.contains) ?? false)
        case .cancelled: false
        case .notAnArchive, .zipUnsupported, .unsupportedEntry, .unsafePath,
             .badHeaderChecksum, .badHeader, .truncated, .badGzip:
            true
        }
    }

    /// POSIX codes that name this device rather than the archive: the crate's
    /// `names_the_host`, plus transient states a phone can recover from on a later
    /// attempt (`EAGAIN`, `EINTR`, descriptor exhaustion, `EIO`).
    private static let hostCodes: Set<Int32> = [
        ENOSPC, EDQUOT, EACCES, EPERM, EROFS, ENOMEM, EBUSY, EAGAIN, EINTR, EMFILE, ENFILE, EIO,
    ]

    static func io(_ path: String, _ error: any Error) -> ArchiveError {
        .io(path: path, code: posixCode(error), reason: error.localizedDescription)
    }

    /// Foundation often wraps the POSIX code in a Cocoa error, so this also reads the
    /// underlying error.
    private static func posixCode(_ error: any Error) -> Int32? {
        let ns = error as NSError
        if ns.domain == NSPOSIXErrorDomain { return Int32(ns.code) }
        if let underlying = ns.userInfo[NSUnderlyingErrorKey] as? any Error,
           let code = posixCode(underlying) {
            return code
        }
        guard ns.domain == NSCocoaErrorDomain else { return nil }
        switch ns.code {
        case NSFileWriteOutOfSpaceError: return ENOSPC
        case NSFileWriteVolumeReadOnlyError: return EROFS
        case NSFileWriteNoPermissionError, NSFileReadNoPermissionError: return EACCES
        default: return nil
        }
    }

    var errorDescription: String? {
        switch self {
        case .notAnArchive:
            "the artifact is not a tar, tar.gz or zip archive"
        case .zipUnsupported:
            "iOS cannot unpack a zip vault archive. Upload the model as a tar or a tar.gz."
        case let .unsupportedEntry(name, type):
            "the archive entry `\(name)` is a \(type). The app unpacks only regular files "
                + "and directories."
        case let .unsafePath(name):
            "the archive entry `\(name)` names a path outside the model directory"
        case let .badHeaderChecksum(offset):
            "the tar header at byte \(offset) has a bad checksum"
        case let .badHeader(offset, reason):
            "the tar header at byte \(offset) is malformed: \(reason)"
        case .truncated:
            "the archive ends before its end marker"
        case let .badGzip(reason):
            "the gzip stream is malformed: \(reason)"
        case let .io(path, _, reason):
            "the app could not unpack into \(path): \(reason)"
        case .cancelled:
            "the unpack was cancelled"
        }
    }

}

nonisolated enum Archive {
    static let chunkSize = 1 << 20

    struct Unpacked: Equatable, Sendable {
        let kind: ArchiveKind
        /// Files and directories written.
        let entries: Int
        /// AppleDouble files skipped.
        var skipped = 0
    }

    static func unpack(_ file: URL, into dest: URL) throws(ArchiveError) -> Unpacked {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: file)
        } catch {
            throw .io(file.path, error)
        }
        defer { try? handle.close() }
        let source = FileByteSource(handle, path: file.path)
        let head = try source.read(upTo: ArchiveKind.headLength)
        do {
            try handle.seek(toOffset: 0)
        } catch {
            throw .io(file.path, error)
        }
        guard let kind = sniffArchiveKind(head) else { throw .notAnArchive }
        return try extractArchive(source, kind: kind, into: dest, chunkSize: chunkSize)
    }

    /// The crate's `extract_archive`. No read asks `source` for more than `chunkSize` bytes.
    static func extractArchive(
        _ source: any ByteSource, kind: ArchiveKind, into dest: URL, chunkSize: Int
    ) throws(ArchiveError) -> Unpacked {
        switch kind {
        case .zip:
            throw .zipUnsupported
        case .tar:
            let counts = try TarUnpacker(root: dest, chunkSize: chunkSize).unpack(source)
            return Unpacked(kind: kind, entries: counts.written, skipped: counts.skipped)
        case .tarGz:
            let gzip = GzipByteSource(source, chunkSize: chunkSize)
            let counts = try TarUnpacker(root: dest, chunkSize: chunkSize).unpack(gzip)
            // The tar stops at its end marker; read the rest so the gzip trailer's CRC-32
            // and length are checked.
            try gzip.drain()
            return Unpacked(kind: kind, entries: counts.written, skipped: counts.skipped)
        }
    }
}

// MARK: - Byte sources

nonisolated protocol ByteSource: AnyObject {
    /// At most `count` bytes; empty only at the end of the stream.
    func read(upTo count: Int) throws(ArchiveError) -> Data
}

nonisolated final class FileByteSource: ByteSource {
    private let handle: FileHandle
    private let path: String

    init(_ handle: FileHandle, path: String) {
        self.handle = handle
        self.path = path
    }

    func read(upTo count: Int) throws(ArchiveError) -> Data {
        do {
            return try handle.read(upToCount: count) ?? Data()
        } catch {
            throw .io(path, error)
        }
    }
}

/// One gzip member, inflated as it is read; bytes after the member are ignored, as with
/// the crate's `GzDecoder`. zlib checks the header and the trailer's CRC-32 and length.
nonisolated final class GzipByteSource: ByteSource {
    private let inner: any ByteSource
    private let chunkSize: Int
    private let stream = UnsafeMutablePointer<z_stream>.allocate(capacity: 1)
    private let input: UnsafeMutablePointer<UInt8>
    /// Whether `inflateInit2` succeeded, and so whether `deinit` must end the stream.
    private let ready: Bool
    private var innerEnded = false
    private var finished = false

    init(_ inner: any ByteSource, chunkSize: Int) {
        self.inner = inner
        self.chunkSize = chunkSize
        input = .allocate(capacity: chunkSize)
        stream.initialize(to: z_stream())
        // 15: a 32 KiB window. +16: expect gzip framing rather than zlib.
        ready = inflateInit2_(
            stream, 15 + 16, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK
    }

    deinit {
        if ready { inflateEnd(stream) }
        stream.deinitialize(count: 1)
        stream.deallocate()
        input.deallocate()
    }

    func read(upTo count: Int) throws(ArchiveError) -> Data {
        guard ready else { throw .badGzip(reason: "the inflater did not start") }
        guard !finished, count > 0 else { return Data() }
        var out = Data(count: count)
        var produced = 0
        while produced == 0, !finished {
            if stream.pointee.avail_in == 0 { try refill() }
            let status = out.withUnsafeMutableBytes { raw -> Int32 in
                stream.pointee.next_out = raw.baseAddress?.assumingMemoryBound(to: Bytef.self)
                stream.pointee.avail_out = uInt(count)
                return inflate(stream, Z_NO_FLUSH)
            }
            produced = count - Int(stream.pointee.avail_out)
            switch status {
            case Z_STREAM_END:
                finished = true
            case Z_OK:
                break
            case Z_BUF_ERROR:
                // No progress with the inner source spent: the member stops short.
                if innerEnded { throw .truncated }
            default:
                let message = stream.pointee.msg.map { String(cString: $0) } ?? "status \(status)"
                throw .badGzip(reason: Self.reason(message))
            }
        }
        out.count = produced
        return out
    }

    /// Reaching the end of the member is what checks the trailer.
    func drain() throws(ArchiveError) {
        while try !read(upTo: chunkSize).isEmpty {}
    }

    private func refill() throws(ArchiveError) {
        guard !innerEnded else { return }
        let data = try inner.read(upTo: chunkSize)
        guard !data.isEmpty else {
            innerEnded = true
            return
        }
        data.copyBytes(to: input, count: data.count)
        stream.pointee.next_in = input
        stream.pointee.avail_in = uInt(data.count)
    }

    private static func reason(_ message: String) -> String {
        switch message {
        case "incorrect data check": "the CRC-32 in the trailer does not match the data"
        case "incorrect length check": "the length in the trailer does not match the data"
        case "header crc mismatch": "the header CRC does not match the header"
        default: message
        }
    }
}

// MARK: - tar

/// Exact byte counts over a source that can return short reads.
nonisolated private final class ExactReader {
    private let source: any ByteSource
    private let chunkSize: Int
    private(set) var offset: Int64 = 0

    init(_ source: any ByteSource, chunkSize: Int) {
        self.source = source
        self.chunkSize = chunkSize
    }

    /// Exactly `count` bytes, or `nil` if the source ends before the first byte.
    func read(exactly count: Int) throws(ArchiveError) -> Data? {
        var out = Data()
        while out.count < count {
            let chunk = try source.read(upTo: count - out.count)
            if chunk.isEmpty {
                if out.isEmpty { return nil }
                throw .truncated
            }
            out.append(chunk)
        }
        offset += Int64(count)
        return out
    }

    func stream(_ count: Int64, _ body: (Data) throws(ArchiveError) -> Void) throws(ArchiveError) {
        var remaining = count
        while remaining > 0 {
            if Task.isCancelled { throw .cancelled }
            guard let chunk = try read(exactly: Int(min(remaining, Int64(chunkSize)))) else {
                throw .truncated
            }
            try body(chunk)
            remaining -= Int64(chunk.count)
        }
    }

    func skip(_ count: Int64) throws(ArchiveError) {
        try stream(count) { _ in }
    }
}

nonisolated struct TarHeader {
    static let blockSize = 512

    let name: String
    let size: Int64
    let type: UInt8

    init(_ block: [UInt8], offset: Int64) throws(ArchiveError) {
        guard let stored = Self.octal(block[148..<156]) else {
            throw .badHeader(offset: offset, reason: "the checksum field is not octal")
        }
        // The checksum counts its own field as eight spaces. Some writers sum signed bytes.
        let spaced = block.enumerated().map { (148..<156).contains($0) ? UInt8(0x20) : $1 }
        let unsigned = spaced.reduce(Int64(0)) { $0 + Int64($1) }
        let signed = spaced.reduce(Int64(0)) { $0 + Int64(Int8(bitPattern: $1)) }
        guard stored == unsigned || stored == signed else {
            throw .badHeaderChecksum(offset: offset)
        }
        guard let size = Self.numeric(block[124..<136]) else {
            throw .badHeader(offset: offset, reason: "the size field is not a number")
        }
        let name = try Self.path(block[0..<100], offset: offset)
        // A POSIX ustar header (`ustar\0`) splits a long path into `prefix` and `name`.
        let prefix = block[257..<263].elementsEqual("ustar\0".utf8)
            ? try Self.path(block[345..<500], offset: offset) : ""
        self.name = prefix.isEmpty ? name : "\(prefix)/\(name)"
        self.size = size
        type = block[156]
    }

    /// Octal, or GNU base-256 (which mgmt writes for files of 8 GiB or more). Base-256 reads
    /// the field's last eight bytes, as the `tar` crate does.
    static func numeric(_ field: ArraySlice<UInt8>) -> Int64? {
        guard let first = field.first else { return nil }
        guard first & 0x80 != 0 else { return octal(field) }
        guard first == 0x80, field.count >= 8 else { return nil }
        let value = field.suffix(8).reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
        return value <= UInt64(Int64.max) ? Int64(value) : nil
    }

    /// An octal field, with leading spaces and a NUL or space terminator.
    private static func octal(_ field: ArraySlice<UInt8>) -> Int64? {
        let digits = field.drop { $0 == 0x20 }.prefix { $0 != 0 && $0 != 0x20 }
        guard !digits.isEmpty else { return nil }
        var value: Int64 = 0
        for digit in digits {
            guard (0x30...0x37).contains(digit) else { return nil }
            let (shifted, overflow) = value.multipliedReportingOverflow(by: 8)
            guard !overflow else { return nil }
            value = shifted + Int64(digit - 0x30)
        }
        return value
    }

    /// Non-UTF-8 is refused: a lossy decode can give two entries the same name, and the
    /// second would overwrite the first.
    static func path<Bytes: Collection<UInt8>>(
        _ field: Bytes, offset: Int64
    ) throws(ArchiveError) -> String {
        guard let path = String(validating: field.prefix { $0 != 0 }, as: UTF8.self) else {
            throw .badHeader(offset: offset, reason: "an entry path is not UTF-8")
        }
        return path
    }
}

nonisolated struct TarUnpacker {
    let root: URL
    let chunkSize: Int

    func unpack(_ source: any ByteSource) throws(ArchiveError) -> (written: Int, skipped: Int) {
        let reader = ExactReader(source, chunkSize: chunkSize)
        try Self.createDirectory(root)
        var written = 0
        var skipped = 0
        // PAX records for the next entry.
        var pax: [String: String] = [:]
        while true {
            let offset = reader.offset
            // EOF at an entry boundary ends the archive, as in the `tar` crate, unless a
            // PAX header is still waiting for its entry.
            guard let block = try reader.read(exactly: TarHeader.blockSize) else {
                if pax.isEmpty { return (written, skipped) }
                throw .truncated
            }
            let bytes = [UInt8](block)
            // Stop at the first zero block, as the `tar` crate does.
            if bytes.allSatisfy({ $0 == 0 }) { return (written, skipped) }
            let header = try TarHeader(bytes, offset: offset)
            let pending = pax
            pax = [:]
            switch header.type {
            case UInt8(ascii: "x"):
                pax = try readPax(header, at: offset, from: reader)
                continue
            case UInt8(ascii: "L"):
                // GNU `tar` writes one for paths over 100 bytes; it names the next entry.
                pax = pending
                pax["path"] = try readLongName(header, at: offset, from: reader)
                continue
            case UInt8(ascii: "g"):
                // Global PAX defaults (times, owners) are not applied.
                try reader.skip(header.size)
                try reader.skip(Self.padding(header.size))
                continue
            default:
                break
            }
            let name = pending["path"] ?? header.name
            var size = header.size
            if let raw = pending["size"] {
                guard let parsed = Int64(raw), parsed >= 0 else {
                    throw .badHeader(offset: offset, reason: "the PAX size `\(raw)` is not a size")
                }
                size = parsed
            }
            switch header.type {
            case UInt8(ascii: "0"), 0:
                guard let dest = try destination(of: name) else { throw .unsafePath(name: name) }
                // Only a `._` file with the AppleDouble magic is skipped; any other is
                // written, with the bytes already read first.
                var head = Data()
                if dest.lastPathComponent.hasPrefix("._"), size >= Self.appleDoubleMagic.count {
                    guard let read = try reader.read(exactly: Self.appleDoubleMagic.count) else {
                        throw .truncated
                    }
                    head = read
                }
                if head == Self.appleDoubleMagic {
                    try reader.skip(size - Int64(head.count))
                    skipped += 1
                } else {
                    try write(size, to: dest, head: head, from: reader)
                    written += 1
                }
            case UInt8(ascii: "5"):
                if let dest = try destination(of: name) {
                    try Self.createDirectory(dest)
                    written += 1
                }
                try reader.skip(size)
            default:
                throw .unsupportedEntry(name: name, type: Self.typeName(header.type))
            }
            try reader.skip(Self.padding(size))
        }
    }

    /// `nil` for the root itself (a `./` entry).
    ///
    /// Compares UTF-8 bytes, as the file system does: a `Character` merges a `/` with a
    /// following combining mark and so hides the separator.
    func destination(of name: String) throws(ArchiveError) -> URL? {
        let slash = UInt8(ascii: "/")
        guard name.utf8.first != slash else { throw .unsafePath(name: name) }
        let parts = name.utf8.split(separator: slash).filter { !$0.elementsEqual(".".utf8) }
        guard !parts.contains(where: { $0.elementsEqual("..".utf8) }) else {
            throw .unsafePath(name: name)
        }
        guard !parts.isEmpty else { return nil }
        let dest = parts.reduce(root) {
            $0.appendingPathComponent(String(decoding: $1, as: UTF8.self))
        }
        let base = root.standardizedFileURL.path
        let prefix = base.hasSuffix("/") ? base : base + "/"
        guard dest.standardizedFileURL.path.utf8.starts(with: prefix.utf8) else {
            throw .unsafePath(name: name)
        }
        return dest
    }

    /// `head` is the part of the body the caller already read. `O_NOFOLLOW` refuses a
    /// pre-existing link at `dest`.
    private func write(
        _ size: Int64, to dest: URL, head: Data, from reader: ExactReader
    ) throws(ArchiveError) {
        try Self.createDirectory(dest.deletingLastPathComponent())
        let descriptor = open(dest.path, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0o644)
        guard descriptor >= 0 else {
            let code = errno
            throw .io(path: dest.path, code: code, reason: String(cString: strerror(code)))
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        let append = { (chunk: Data) throws(ArchiveError) in
            do {
                try handle.write(contentsOf: chunk)
            } catch {
                throw .io(dest.path, error)
            }
        }
        try append(head)
        try reader.stream(size - Int64(head.count), append)
        // A file system can report a failed write only at the close.
        do {
            try handle.close()
        } catch {
            throw .io(dest.path, error)
        }
    }

    private func readLongName(
        _ header: TarHeader, at offset: Int64, from reader: ExactReader
    ) throws(ArchiveError) -> String {
        guard header.size > 0, header.size <= Int64(chunkSize) else {
            throw .badHeader(offset: offset, reason: "a GNU long name of \(header.size) bytes")
        }
        guard let body = try reader.read(exactly: Int(header.size)) else { throw .truncated }
        try reader.skip(Self.padding(header.size))
        return try TarHeader.path(body, offset: offset)
    }

    /// The records of a PAX extended header (`<length> <key>=<value>\n`).
    private func readPax(
        _ header: TarHeader, at offset: Int64, from reader: ExactReader
    ) throws(ArchiveError) -> [String: String] {
        guard header.size <= Int64(chunkSize) else {
            throw .badHeader(offset: offset, reason: "a PAX header of \(header.size) bytes is too large")
        }
        let body = header.size == 0 ? Data() : try reader.read(exactly: Int(header.size))
        guard let body else { throw .truncated }
        try reader.skip(Self.padding(header.size))
        let bytes = [UInt8](body)
        var records: [String: String] = [:]
        var cursor = 0
        while cursor < bytes.count {
            guard let space = bytes[cursor...].firstIndex(of: 0x20),
                  let length = Int(String(decoding: bytes[cursor..<space], as: UTF8.self)),
                  length > space - cursor + 1, length <= bytes.count - cursor,
                  bytes[cursor + length - 1] == 0x0a
            else { throw .badHeader(offset: offset, reason: "a PAX record is malformed") }
            let record = bytes[(space + 1)..<(cursor + length - 1)]
            guard let equals = record.firstIndex(of: 0x3d) else {
                throw .badHeader(offset: offset, reason: "a PAX record has no `=`")
            }
            let key = String(decoding: record[..<equals], as: UTF8.self)
            // Only `path` must be UTF-8; other values, such as xattrs, can hold any bytes.
            records[key] = key == "path"
                ? try TarHeader.path(record[(equals + 1)...], offset: offset)
                : String(decoding: record[(equals + 1)...], as: UTF8.self)
            cursor += length
        }
        return records
    }

    private static func createDirectory(_ dir: URL) throws(ArchiveError) {
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            throw .io(dir.path, error)
        }
    }

    /// The first four bytes of an AppleDouble file (RFC 1740).
    private static let appleDoubleMagic = Data([0x00, 0x05, 0x16, 0x07])

    private static func padding(_ size: Int64) -> Int64 {
        let block = Int64(TarHeader.blockSize)
        return (block - size % block) % block
    }

    private static func typeName(_ type: UInt8) -> String {
        switch type {
        case UInt8(ascii: "1"): "hard link"
        case UInt8(ascii: "2"): "symbolic link"
        case UInt8(ascii: "3"): "character device"
        case UInt8(ascii: "4"): "block device"
        case UInt8(ascii: "6"): "FIFO"
        case UInt8(ascii: "7"): "contiguous file"
        case UInt8(ascii: "K"): "GNU long link name"
        case UInt8(ascii: "L"): "GNU long name"
        default: "entry of type `\(String(UnicodeScalar(type)))`"
        }
    }
}
