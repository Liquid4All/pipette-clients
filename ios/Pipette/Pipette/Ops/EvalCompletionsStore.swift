import CryptoKit
import Foundation

/// Hex SHA-256 of a run's portable identity; file names use the first 16 characters —
/// the crate's `EvalRunDigest` (`pipette-ops/src/eval_completions.rs:36`).
nonisolated struct EvalRunDigest: Hashable, Sendable {
    let value: String

    /// The first 16 characters, which name the file.
    var prefix: String { String(value.prefix(16)) }
}

/// Operator-facing header fields, so `head -1 <file>` says what the run was. Not hashed.
nonisolated struct EvalCompletionMeta: Codable, Equatable, Sendable {
    let benchmarkId: String
    let runtime: String
    let model: String

    enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case runtime, model
    }
}

/// Resume state for eval runs — the crate's `EvalCompletionsStore`, keyed by a run's
/// portable identity rather than by the job or cell that happened to drive it.
///
/// ```text
/// state/evals/
///   <digest16>.jsonl          # header line, then one completion per line
///   <digest16>.jsonl.stale-*  # rotated when the header disagrees or will not parse
/// ```
///
/// Why this matters more here than on a desktop: a phone running a long eval gets
/// jetsam-killed, and without a checkpoint the next attempt starts again from sample
/// zero. Appending per sample means a kill costs one sample, not the whole run.
nonisolated struct EvalCompletionsStore: Sendable {
    /// The `state/evals/` directory.
    let root: URL

    /// Canonical JSON for hashing: `.sortedKeys`, so a dictionary's iteration order
    /// cannot change the digest between two encodings of the same value.
    ///
    /// Deliberately *not* `SubmissionRef`, which encodes descriptors without sorting —
    /// correct there, because the server canonicalizes before storing and the digest is
    /// taken over that canonical form. Reusing it here made the digest unstable within a
    /// single process, so every resume rotated its own checkpoint and started over.
    private static let identityEncoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    /// SHA-256 over the run's *portable* coordinates only — the declared runtime and
    /// model and the benchmark body. Never a bound host path: a re-fetched model gets a
    /// new path routinely, and a path-sensitive digest would rotate the checkpoint on
    /// every eviction, which is the failure this exists to avoid. The crate gets the same
    /// projection free from `RunRequest`'s `Serialize`.
    static func digest(of request: RunRequest) throws -> EvalRunDigest {
        var hasher = SHA256()
        hasher.update(data: try identityEncoder.encode(request.runtime.declared))
        hasher.update(data: try identityEncoder.encode(request.model.declared))
        hasher.update(data: try identityEncoder.encode(request.benchmark))
        if let flags = try? request.runtimeFlags?.submissionValue() {
            hasher.update(data: Data(flags.utf8))
        }
        return EvalRunDigest(
            value: hasher.finalize().map { String(format: "%02x", $0) }.joined())
    }

    /// Open or resume the session for this run's identity.
    func open(request: RunRequest) throws -> EvalCompletionSession {
        let digest = try Self.digest(of: request)
        let meta = EvalCompletionMeta(
            benchmarkId: request.benchmark.benchmarkId,
            runtime: try SubmissionRef.runtime(request.runtime.declared),
            model: try SubmissionRef.model(request.model.declared))
        return try EvalCompletionSession(root: root, digest: digest, meta: meta)
    }

    /// Drop every session file — the crate's `clear`.
    func clear() {
        try? FileManager.default.removeItem(at: root)
    }
}

/// A live append session for one eval run.
///
/// Append-only on purpose: the file is the checkpoint, so a process killed mid-run keeps
/// everything written up to that sample. Nothing rewrites it, so there is no window
/// where a partial rewrite loses prior work.
nonisolated final class EvalCompletionSession {
    /// One object per line, so `Coding.encoder` cannot be used: it is `.prettyPrinted`,
    /// which would spread every record across several lines and make the file
    /// unparseable as JSONL.
    private static let lineEncoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    private let path: URL
    private let handle: FileHandle
    /// Keyed by sample id, as the crate's `load_existing` keys its map: a resume looks up
    /// per sample, and an eval cell is 2000 samples, so a linear scan would be quadratic.
    private var recorded: [String: BenchmarkEvalCompletion]
    /// Insertion order, so `completions` reports samples in the order they were run.
    private var order: [String]

    var completions: [BenchmarkEvalCompletion] { order.compactMap { recorded[$0] } }

    init(root: URL, digest: EvalRunDigest, meta: EvalCompletionMeta) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        path = root.appendingPathComponent("\(digest.prefix).jsonl")

        var resumed: [BenchmarkEvalCompletion] = []
        var needsHeader = true
        if FileManager.default.fileExists(atPath: path.path) {
            switch Self.load(path, digest: digest) {
            case .some(let prior):
                resumed = prior
                needsHeader = false
            case .none:
                // A different run's identity, or a file we cannot read: keep it for
                // inspection rather than appending this run's rows onto someone else's.
                Self.rotateStale(path)
            }
        }
        recorded = Dictionary(resumed.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
        order = resumed.map(\.id)

        if needsHeader || !FileManager.default.fileExists(atPath: path.path) {
            let header = Header(digest: digest.value, meta: meta)
            try Self.lineEncoder.encode(header).appendingNewline().write(to: path, options: .atomic)
        }
        handle = try FileHandle(forWritingTo: path)
        try handle.seekToEnd()
    }

    deinit { try? handle.close() }

    /// Has this sample already been attempted? A *failed* sample counts: the crate
    /// records a poison sample precisely so a retry skips it rather than re-hitting
    /// something that reliably kills the engine (`llamacpp/execute/eval.rs:158`).
    func contains(_ sampleId: String) -> Bool { recorded[sampleId] != nil }

    /// What an earlier attempt recorded for this sample, if anything.
    func completion(for sampleId: String) -> BenchmarkEvalCompletion? { recorded[sampleId] }

    /// Record one completion, durably, before the next sample starts.
    func append(_ completion: BenchmarkEvalCompletion) throws {
        guard recorded[completion.id] == nil else { return }
        try handle.write(contentsOf: try Self.lineEncoder.encode(completion).appendingNewline())
        try handle.synchronize()
        recorded[completion.id] = completion
        order.append(completion.id)
    }

    /// Release the file, keeping the checkpoint. What a killed run leaves behind, and
    /// what the next attempt resumes from.
    func close() {
        try? handle.close()
    }

    /// End the run — the crate's `finalize`: drop the checkpoint when every sample
    /// succeeded, otherwise rewrite it holding only the failures.
    ///
    /// The failures are kept on purpose. A sample that kills the engine would otherwise
    /// be re-hit by every fresh run of the same cell; retaining it means the next run
    /// skips it, since `contains` counts a failure as attempted.
    @discardableResult
    func finalize() -> [BenchmarkEvalCompletion] {
        let all = completions
        close()
        let failed = all.filter { if case .failed = $0 { true } else { false } }
        if failed.isEmpty {
            try? FileManager.default.removeItem(at: path)
        } else {
            try? Self.rewrite(path, keeping: failed, header: headerLine())
        }
        return all
    }

    /// The header verbatim, so a rewrite keeps the digest it was opened under.
    private func headerLine() -> Data? {
        guard let text = try? String(contentsOf: path, encoding: .utf8),
              let first = text.split(separator: "\n", omittingEmptySubsequences: true).first
        else { return nil }
        return Data(first.utf8)
    }

    private static func rewrite(
        _ path: URL, keeping failed: [BenchmarkEvalCompletion], header: Data?
    ) throws {
        guard let header else { return }
        var body = header.appendingNewline()
        for completion in failed {
            body += try lineEncoder.encode(completion).appendingNewline()
        }
        try body.write(to: path, options: .atomic)
    }

    private struct Header: Codable {
        let digest: String
        let meta: EvalCompletionMeta
    }

    /// Prior completions, or nil when the header names a different run or will not parse.
    private static func load(_ path: URL, digest: EvalRunDigest) -> [BenchmarkEvalCompletion]? {
        guard let text = try? String(contentsOf: path, encoding: .utf8) else { return nil }
        var lines = text.split(separator: "\n", omittingEmptySubsequences: true).makeIterator()
        guard let headerLine = lines.next(),
              let header = try? Coding.decoder.decode(Header.self, from: Data(headerLine.utf8)),
              header.digest == digest.value
        else { return nil }
        // Drop any line that will not parse — a kill mid-write truncates the last one —
        // rather than discarding the whole checkpoint.
        return lines.compactMap {
            try? Coding.decoder.decode(BenchmarkEvalCompletion.self, from: Data($0.utf8))
        }
    }

    private static func rotateStale(_ path: URL) {
        let stamp = Int(Date().timeIntervalSince1970)
        try? FileManager.default.moveItem(
            at: path, to: path.appendingPathExtension("stale-\(stamp)"))
    }
}

private nonisolated extension Data {
    func appendingNewline() -> Data { self + Data("\n".utf8) }
}
