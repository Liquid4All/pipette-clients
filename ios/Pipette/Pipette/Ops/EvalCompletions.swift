import CryptoKit
import Foundation

/// Hex SHA-256 of a run's portable identity; file names use the first 16 characters:
/// the crate's `EvalRunDigest` (`pipette-ops/src/eval_completions.rs`).
nonisolated struct EvalRunDigest: Hashable, Sendable {
    let value: String
}

/// Operator-facing header fields, so `head -1 <file>` says what the run was. Not hashed.
/// `runtime` and `model` are the declared coordinates' `Display`, as the crate writes them.
nonisolated struct EvalCompletionMeta: Codable, Equatable, Sendable {
    let benchmarkId: String
    let runtime: String
    let model: String

    enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case runtime, model
    }
}

/// The crate's header also records the HTTP timeout, to run failed rows again under a new one;
/// no iOS ref declares a timeout.
private nonisolated struct Header: Codable {
    let digest: String
    let meta: EvalCompletionMeta
}

/// Resume state for eval runs: the crate's `EvalCompletionsStore`, keyed by a run's
/// portable identity rather than by the job or cell that drove it.
///
/// ```text
/// state/evals/
///   <digest16>.jsonl          # header line, then one completion per line
///   <digest16>.jsonl.stale-*  # rotated when the header disagrees or will not parse
/// ```
///
/// A phone running a long eval gets jetsam-killed; appending per sample means a kill costs
/// one sample, not the run. An engine appends and never deletes: the client finalizes the
/// file once the result is saved or accepted, so a result lost on its way is rebuilt from
/// the checkpoint.
nonisolated struct EvalCompletionsStore: Sendable {
    /// The `state/evals/` directory.
    let root: URL

    func open(_ req: RunRequest) throws -> EvalCompletionSession {
        let digest = try runRequestDigest(req)
        let meta = EvalCompletionMeta(
            benchmarkId: req.benchmark.benchmarkId.value,
            runtime: req.runtime.declared.description,
            model: req.model.declared.description)
        return try EvalCompletionSession(root: root, digest: digest, meta: meta)
    }

    /// Retire the checkpoint of a run whose result is safe: delete it, or keep only its failed
    /// rows, which later runs of the cell skip.
    ///
    /// Reads the file rather than a session, since the engine that wrote it has returned: the
    /// last row per id is the one that counts, as on resume.
    func finalize(_ digest: EvalRunDigest) throws {
        let path = sessionPath(root, digest)
        guard FileManager.default.fileExists(atPath: path.path) else { return }
        switch loadExisting(path, expected: digest) {
        case let .resumed(header, completions):
            let failed = completions.filter(\.isFailed)
            if failed.isEmpty {
                try FileManager.default.removeItem(at: path)
            } else {
                try writeCheckpoint(path, header, failed)
            }
        case .otherRun:
            // Rotated away by the load.
            return
        case let .unreadable(reason):
            throw EvalCompletionsError.unreadable(path: path.path, reason: reason)
        }
    }

    /// Remove every session file and stale rotation under this store: the crate's `clear`.
    func clear() throws {
        guard FileManager.default.fileExists(atPath: root.path) else { return }
        try FileManager.default.removeItem(at: root)
    }
}

/// Canonical JSON for hashing: sorted keys and no escaped slashes, the crate's
/// `write_canonical`, so a dictionary's iteration order cannot change the digest
/// between two encodings of the same value.
private nonisolated let identityEncoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return encoder
}()

/// SHA-256 over the run's *portable* coordinates only: the crate's `run_request_digest`.
/// Never a bound host path: a re-fetched model gets a new path routinely, and a
/// path-sensitive digest would rotate the checkpoint on every eviction.
nonisolated func runRequestDigest(_ request: RunRequest) throws -> EvalRunDigest {
    let identity = ResumeIdentity(
        runtime: request.runtime.declared,
        // A vault label names nothing, so a labelled claim resumes an unlabelled run.
        model: request.model.declared.withoutCustomLabel,
        runtimeFlags: request.runtimeFlags.map(resumeRelevantFlags),
        benchmarkFlags: request.benchmarkFlags.flatMap(resumeRelevantBenchmarkFlags),
        benchmark: request.benchmark)
    let canonical = try identityEncoder.encode(identity)
    return EvalRunDigest(
        value: SHA256.hash(data: canonical).map { String(format: "%02x", $0) }.joined())
}

/// The portable axes cell identity is hashed over: the crate's `ResumeIdentity`. A struct
/// so a field renamed on `RunRequest` fails to build instead of silently leaving the
/// identity. Every key is written, `null` included, so absent runtime flags are a distinct
/// identity from empty ones.
private nonisolated struct ResumeIdentity: Encodable {
    let runtime: Runtime
    let model: Model
    let runtimeFlags: RuntimeFlagRef?
    let benchmarkFlags: BenchmarkFlagRef?
    let benchmark: BenchmarkDefinition

    enum CodingKeys: String, CodingKey {
        case runtime, model, benchmark
        case runtimeFlags = "runtime_flags"
        case modelFlags = "model_flags"
        case benchmarkFlags = "benchmark_flags"
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(runtime, forKey: .runtime)
        try c.encode(model, forKey: .model)
        try c.encode(benchmark, forKey: .benchmark)
        if let runtimeFlags { try c.encode(runtimeFlags, forKey: .runtimeFlags) } else {
            try c.encodeNil(forKey: .runtimeFlags)
        }
        // Always `null`, even when the request carries model flags: the claim refuses every
        // knob a variant has, so none can vary the identity. The key keeps the crate's shape.
        try c.encodeNil(forKey: .modelFlags)
        if let benchmarkFlags { try c.encode(benchmarkFlags, forKey: .benchmarkFlags) } else {
            try c.encodeNil(forKey: .benchmarkFlags)
        }
    }
}

/// `runtime_flags` with the throughput-only knobs cleared, so trimming `ctx_size` does not
/// re-run a dataset that may be hours in: the crate's `resume_relevant_flags`. It does not
/// change what a completion is; a truncation shows up per sample as `stop_reason`. The crate clears `parallel` too, which no iOS
/// variant carries. A typed field rather than a JSON key, so a rename breaks the build.
private nonisolated func resumeRelevantFlags(_ flags: RuntimeFlags) -> RuntimeFlagRef {
    var ref = RuntimeFlagRef(flags)
    ref.ctxSize = nil
    return ref
}

/// Benchmark flags with no knob set hash as absent, as the crate's do once it clears
/// the timeouts, which no iOS ref declares. Comparing against the bare axes keeps a knob
/// added later in the identity.
private nonisolated func resumeRelevantBenchmarkFlags(_ flags: BenchmarkFlags) -> BenchmarkFlagRef? {
    let ref = BenchmarkFlagRef(flags)
    let axes = BenchmarkFlagRef(benchmarkType: ref.benchmarkType, runtimeType: ref.runtimeType,
                                modelType: ref.modelType)
    return ref == axes ? nil : ref
}

/// A live append session for one eval run: the crate's `EvalCompletionSession`. Closing it,
/// or `intoCompletions`, keeps the file for the next open of the same digest; only the
/// store's `finalize` retires it.
///
/// Append-only on disk on purpose: the file is the checkpoint, so a process killed mid-run
/// keeps everything written up to that sample. In memory the last write for an id wins.
nonisolated final class EvalCompletionSession {
    private let path: URL
    private let handle: FileHandle
    /// Keyed by sample id, as the crate's `load_existing` keys its map: a resume looks up
    /// per sample, so a linear scan would be quadratic.
    private var recorded: [String: BenchmarkEvalCompletion]
    /// Insertion order, so `completions` reports samples in the order they were recorded.
    private var order: [String]
    /// How many recorded completions are failures, kept exact across a failure that a
    /// later success overwrites.
    private(set) var failedCount: Int

    var completions: [BenchmarkEvalCompletion] { order.compactMap { recorded[$0] } }

    /// Every id with a recorded completion, failed or not: the crate's `done_ids`.
    var doneIds: [String] { order }

    /// The ids whose recorded completion is a failure: the crate's `failed_ids`.
    var failedIds: [String] { completions.filter(\.isFailed).map(\.id) }

    init(root: URL, digest: EvalRunDigest, meta: EvalCompletionMeta) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        path = sessionPath(root, digest)

        var resumed: [BenchmarkEvalCompletion]?
        if FileManager.default.fileExists(atPath: path.path) {
            switch loadExisting(path, expected: digest) {
            case let .resumed(_, prior):
                resumed = prior
            case .otherRun:
                break
            case let .unreadable(reason):
                // Kept for inspection rather than appended onto.
                AppLog.jobRun.warning("eval completions unreadable; rotating stale "
                    + "path=\(path.path) error=\(reason)")
                rotateStale(path)
            }
        }
        if resumed == nil {
            try writeCheckpoint(path, Header(digest: digest.value, meta: meta), [])
        }
        let rows = resumed ?? []
        recorded = Dictionary(uniqueKeysWithValues: rows.map { ($0.id, $0) })
        order = rows.map(\.id)
        failedCount = rows.filter(\.isFailed).count

        handle = try FileHandle(forWritingTo: path)
        let end = try handle.seekToEnd()
        // A jetsam kill between `write` and `synchronize` can leave a last line with no
        // newline. Appending onto it would splice two records into one unparseable line, so
        // the next resume would lose the good record too. Close the torn line instead.
        if end > 0, Self.endsMidLine(path) {
            AppLog.jobRun.warning(
                "eval completions: last line of \(path.path) was torn by an interrupted "
                    + "write; closing it before appending")
            try handle.write(contentsOf: Data.lineTerminator)
            try handle.synchronize()
        }
        if !order.isEmpty {
            AppLog.jobRun.info("resuming eval completions completed=\(order.count) path=\(path.path)")
        }
    }

    deinit { try? handle.close() }

    /// Whether the file's last byte is something other than a line terminator. Its own
    /// read-only handle, since reading the session's write-only `handle` fails; seeked to
    /// the last byte, so it reads one byte rather than the whole checkpoint.
    private static func endsMidLine(_ path: URL) -> Bool {
        guard let reader = try? FileHandle(forReadingFrom: path) else { return false }
        defer { try? reader.close() }
        guard let end = try? reader.seekToEnd(), end > 0,
              (try? reader.seek(toOffset: end - 1)) != nil,
              let last = try? reader.read(upToCount: 1)
        else { return false }
        return last != Data.lineTerminator
    }

    /// Record one completion, durably, before the next sample starts: the crate's `append`.
    ///
    /// Replacing a *failed* row is a retry landing: the failure stops counting. Replacing a
    /// completed one means one dataset id ran twice, which is logged, and the later write
    /// still wins so the counts stay exact either way.
    func append(_ completion: BenchmarkEvalCompletion) throws {
        try handle.write(contentsOf: try lineEncoder.encode(completion).appendingNewline())
        try handle.synchronize()

        if let old = recorded[completion.id] {
            order.removeAll { $0 == completion.id }
            if old.isFailed {
                failedCount -= 1
            } else {
                AppLog.jobRun.warning(
                    "duplicate eval completion id (last write wins) sample_id=\(completion.id)")
            }
        }
        recorded[completion.id] = completion
        order.append(completion.id)
        if completion.isFailed { failedCount += 1 }
    }

    /// Has this sample already been attempted? A *failed* sample counts: the crate
    /// records a poison sample precisely so a retry skips it rather than re-hitting
    /// something that reliably kills the engine.
    func contains(_ sampleId: String) -> Bool { recorded[sampleId] != nil }

    /// Release the file, keeping the checkpoint. What a killed run leaves behind, and
    /// what the next attempt resumes from.
    func close() {
        try? handle.close()
    }

    /// Close the session and hand back every row it holds, completed and failed alike: the
    /// crate's `into_completions`. The file stays until the client finalizes it.
    func intoCompletions() -> [BenchmarkEvalCompletion] {
        close()
        return completions
    }
}

private nonisolated func sessionPath(_ root: URL, _ digest: EvalRunDigest) -> URL {
    root.appendingPathComponent("\(digestPrefix(digest)).jsonl")
}

nonisolated func digestPrefix(_ digest: EvalRunDigest) -> String {
    String(digest.value.prefix(16))
}

/// One object per line, so `Coding.encoder` cannot be used: it is `.prettyPrinted`, which
/// would spread every record across several lines and make the file unparseable as JSONL.
private nonisolated let lineEncoder: JSONEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return encoder
}()

/// The crate's `write_checkpoint`. `.atomic` writes through a sibling file, so a kill
/// mid-write cannot leave half a checkpoint.
private nonisolated func writeCheckpoint(_ path: URL, _ header: Header,
                                         _ rows: [BenchmarkEvalCompletion]) throws {
    var body = try lineEncoder.encode(header).appendingNewline()
    for row in rows {
        body += try lineEncoder.encode(row).appendingNewline()
    }
    try body.write(to: path, options: .atomic)
}

private nonisolated enum Loaded {
    case resumed(Header, [BenchmarkEvalCompletion])
    case otherRun
    case unreadable(String)
}

/// The header and the last row per id, or why there are none to resume: the crate's
/// `load_existing`. A header naming another digest rotates the file away.
private nonisolated func loadExisting(_ path: URL, expected: EvalRunDigest) -> Loaded {
    guard let text = try? String(contentsOf: path, encoding: .utf8) else {
        return .unreadable("cannot read the file")
    }
    var lines = text.split(separator: "\n", omittingEmptySubsequences: true).makeIterator()
    guard let headerLine = lines.next() else { return .unreadable("empty completion file") }
    guard let header = try? Coding.decoder.decode(Header.self, from: Data(headerLine.utf8))
    else { return .unreadable("failed to parse completion header") }
    guard header.digest == expected.value else {
        AppLog.jobRun.info("eval completions digest mismatch; rotating path=\(path.path) "
            + "digest=\(header.digest) expected_digest=\(expected.value)")
        rotateStale(path)
        return .otherRun
    }
    // Skip a line that will not parse (a kill mid-write truncates the last one) rather than
    // discarding the whole checkpoint.
    var completions: [BenchmarkEvalCompletion] = []
    for (index, line) in lines.enumerated() {
        if let completion = try? Coding.decoder.decode(BenchmarkEvalCompletion.self,
                                                      from: Data(line.utf8)) {
            completions.append(completion)
        } else {
            AppLog.jobRun.warning(
                "skipping corrupt completion line line=\(index + 2) path=\(path.path)")
        }
    }
    // A kill between a failure and the success that replaced it leaves both lines on disk;
    // the id is recorded once, at its first position.
    let latest = Dictionary(completions.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
    var seen = Set<String>()
    let ids = completions.map(\.id).filter { seen.insert($0).inserted }
    return .resumed(header, ids.compactMap { latest[$0] })
}

private nonisolated func rotateStale(_ path: URL) {
    let stamp = Int(Date().timeIntervalSince1970)
    let destination = path.appendingPathExtension("stale-\(stamp)")
    do {
        try FileManager.default.moveItem(at: path, to: destination)
    } catch {
        AppLog.jobRun.warning("failed to rotate stale completions from=\(path.path) "
            + "to=\(destination.path) error=\(error)")
    }
}

/// A checkpoint the client cannot finalize: the crate's `load_existing` bails, with their
/// messages.
nonisolated enum EvalCompletionsError: Error, Equatable, LocalizedError {
    case unreadable(path: String, reason: String)

    var errorDescription: String? {
        switch self {
        case let .unreadable(path, reason): "\(reason): \(path)"
        }
    }
}

private nonisolated extension BenchmarkEvalCompletion {
    var isFailed: Bool {
        if case .failed = self { true } else { false }
    }
}

private nonisolated extension Data {
    func appendingNewline() -> Data { self + Data.lineTerminator }
    /// The one byte that separates two records in the checkpoint file, so the writer and
    /// the torn-line repair on resume cannot disagree about what ends a line.
    static let lineTerminator = Data("\n".utf8)
}
