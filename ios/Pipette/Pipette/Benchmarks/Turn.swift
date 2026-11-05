import Foundation

// The `vl_spec_decode` traversal: `crates/pipette-vl-spec-decode-corpus/src/turn.rs`.
//
// Where upstream runs a `llama-server` per row, this runs in process over a `VisionChat`.
// The methodology (corpus order, warm-up, gate placement, counters) is upstream's, and the
// numbers are comparable to the CLI's because the decoding loop is llama.cpp's own
// (`ios/native/pipette_vlspec.h`).
//
// One sequence, always: this witness holds one KV cache and one conversation, so a batched
// entry is refused before anything loads (`BenchmarkMeasurement.requireOneInFlight`).
// Running it serially would file a one-sequence measurement under the batched benchmark id.

/// The seam the traversal measures against (the crate's `Send`), so it is testable with no
/// model and no drafter.
///
/// A peer of `VisionInference`, which prefills one message for `vl_throughput`. This one is
/// a conversation: later turns carry the earlier replies, and the images come off disk.
nonisolated struct VisionChat {
    /// One call per corpus sample.
    var begin: (_ system: String) throws -> Void
    /// `text` carries one media marker per image. Every turn prefills the whole conversation
    /// from an empty cache, as the CLI's `cache_prompt: false` does. Returns everything but
    /// `e2eMs`, which the traversal stamps around this call.
    var turn: (_ text: String, _ images: [String], _ maxTokens: Int) throws -> TurnRecord
    /// Whether a drafter is actually attached. Asserted on the speculative arm: a drafter
    /// that never attached submits a speedup of one, which reads as a finding rather than
    /// as a broken run.
    var hasDrafter: Bool
}

nonisolated enum VlSpecDecodeCorpusTurn {
    /// `open` loads the model and hands back its witness plus a teardown, so the traversal
    /// never holds a model handle and a test can drive the whole methodology with neither.
    static func run(
        speculation: VlSpecDecode.Speculation,
        corpus: VlSpecDecodeCorpus,
        draftLength: UInt32,
        maxTokens: UInt32,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver,
        isCancelled: () -> Bool = { false },
        progress: (BenchmarkProgress) -> Void = { _ in },
        open: () throws -> (chat: VisionChat, close: () -> Void)
    ) async throws -> (data: BenchmarkResultData, traversal: Traversal) {
        if speculation.enabled, draftLength == 0 {
            throw RuntimeError.engine("parameter_draft_length must be at least 1, got 0")
        }
        guard maxTokens > 0 else {
            throw RuntimeError.engine("parameter_max_tokens must be at least 1, got 0")
        }

        AppLog.jobRun.info(
            "\(speculation.label): \(corpus.samples.count) samples / \(corpus.turns) turns")

        let traversal = try await traverse(
            speculation: speculation, corpus: corpus, maxTokens: Int(maxTokens), readinessGate: readinessGate,
            observer: observer, isCancelled: isCancelled, progress: progress, open: open)
        return (try VlSpecDecode.counters(speculation: speculation, turns: traversal.turns),
                traversal)
    }

    struct Traversal {
        var turns: [TurnRecord]
        var observations: [TurnObservation]
        var outputDigest: String
    }

    static func traverse(
        speculation: VlSpecDecode.Speculation,
        corpus: VlSpecDecodeCorpus,
        maxTokens: Int,
        readinessGate: ReadinessGate,
        observer: MeasurementObserver,
        isCancelled: () -> Bool,
        progress: (BenchmarkProgress) -> Void,
        open: () throws -> (chat: VisionChat, close: () -> Void)
    ) async throws -> Traversal {
        guard let first = corpus.samples.first else { throw VlSpecDecodeCorpusError.empty }
        let (chat, close) = try open()
        defer { close() }

        // Each arm is refused the other's load, or its row would be filed under the wrong
        // arm. Checked after the load, because only then is it known whether a drafter
        // attached.
        if speculation.enabled, !chat.hasDrafter {
            throw RuntimeError.engine(
                "\(speculation.label) loaded without a drafter, so it would measure the "
                    + "baseline under the speculative run's name")
        }
        if !speculation.enabled, chat.hasDrafter {
            throw RuntimeError.engine(
                "\(speculation.label) loaded with a drafter attached, so it would measure "
                    + "speculation under the baseline's name")
        }

        // Untimed warm-up on the corpus's own first turn, so kernel selection and the vision
        // tower's first-shape cost are paid before counting, on the shapes a real sample uses.
        AppLog.jobRun.info("\(speculation.label): warm-up")
        try chat.begin(first.system)
        _ = try oneTurn(chat, corpus: corpus, sample: first, index: 0, maxTokens: maxTokens)

        try readinessGate()

        // The observer pair brackets the whole traversal, not each sample: the readings say
        // whether two results are comparable, and one pair per traversal answers that.
        try observer.onBeforeRepStart()
        var records: [TurnRecord] = []
        var observations: [TurnObservation] = []
        records.reserveCapacity(corpus.turns)
        observations.reserveCapacity(corpus.turns)
        for (index, sample) in corpus.samples.enumerated() {
            // Before every sample, never between the turns of one: a wait inside a
            // conversation would split what the corpus defines as a unit.
            if index > 0 {
                try checkCancelled(isCancelled)
                try readinessGate()
            }
            do {
                try chat.begin(sample.system)
                for turnIndex in sample.turns.indices {
                    let record = try oneTurn(chat, corpus: corpus, sample: sample,
                                             index: turnIndex, maxTokens: maxTokens)
                    observations.append(
                        TurnObservation(sampleId: sample.id, turn: UInt32(turnIndex),
                                        record: record))
                    records.append(record)
                }
            } catch {
                throw RuntimeError.engine(
                    "\(speculation.label): sample \(sample.id) (\(index + 1)/"
                        + "\(corpus.samples.count)) failed: \(error.localizedDescription)")
            }
            progress(.sample(completed: index + 1, total: corpus.samples.count))
            VlSpecDecode.logProgress(speculation: speculation, done: index + 1,
                                     total: corpus.samples.count, records: records)
        }
        try observer.onAfterRepEnd()
        return Traversal(turns: records, observations: observations,
                         outputDigest: VlSpecDecode.outputDigest(records))
    }

    /// The conversation's earlier turns are not passed: the witness holds them, because the
    /// engine's chat template is what renders them and a reply has to re-enter the prompt
    /// exactly as the model emitted it.
    private static func oneTurn(_ chat: VisionChat, corpus: VlSpecDecodeCorpus,
                                sample: VlSpecDecodeCorpusSample, index: Int,
                                maxTokens: Int) throws -> TurnRecord {
        let (text, images) = try prompt(for: sample.turns[index])
        let paths = try resolve(images: images, in: corpus)
        let clock = ContinuousClock()
        let started = clock.now
        var record = try chat.turn(text, paths, maxTokens)
        // Measured around the whole call, never `ttft + decode`: the difference is overhead
        // the engine never sees, and the server rejects a reconstructed value.
        record.e2eMs = durationMs(clock.now - started)
        // At the turn, so the failure names which one; otherwise the server would trust a
        // row whose counters contradict each other.
        try record.validate()
        try record.checkLength(maxTokens: maxTokens)
        return record
    }

    /// The crate's `content_parts`, flattened: each image becomes a media marker in place,
    /// which is how the server flattens upstream's `/chat/completions` content parts.
    static func prompt(for parts: [VlSpecDecodeCorpusPart]) throws -> (text: String, images: [String]) {
        var text = ""
        var images: [String] = []
        for part in parts {
            switch part {
            case let .text(value):
                text += value
            case let .image(path, _, _):
                images.append(path)
                text += LlamaMtmd.defaultMediaMarker
            }
        }
        guard !text.isEmpty else { throw RuntimeError.engine("a corpus turn carries no content") }
        return (text, images)
    }

    /// Separate from `prompt` so prompt shaping stays testable with no corpus on disk.
    static func resolve(images: [String], in corpus: VlSpecDecodeCorpus) throws -> [String] {
        try images.map { try corpus.mediaPath($0).path }
    }

    private static func checkCancelled(_ isCancelled: () -> Bool) throws {
        if isCancelled() { throw RuntimeError.cancelled }
    }
}

nonisolated extension TurnObservation {
    /// The reply's sha256 rather than the reply, so the file says which turns matched
    /// without holding text.
    init(sampleId: String, turn: UInt32, record: TurnRecord) {
        self.init(
            sampleId: sampleId, turn: turn,
            textSha256: VlSpecDecode.textSha256(record.text),
            generatedTokens: record.generatedTokens,
            draftedTokens: record.draftedTokens,
            acceptedTokens: record.acceptedTokens,
            countedForwards: record.countedForwards,
            endedOnEos: record.endedOnEos,
            promptMs: record.promptMs,
            predictedMs: record.predictedMs,
            ttftMs: record.ttftMs,
            decodeMs: record.decodeMs,
            e2eMs: record.e2eMs)
    }
}
