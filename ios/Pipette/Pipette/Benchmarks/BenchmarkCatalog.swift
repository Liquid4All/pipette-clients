import Foundation

/// One catalog entry. `rawJson` keeps the full server-shaped payload so the UI and CSV
/// readers can read parameter fields directly.
nonisolated struct BenchmarkItem: Identifiable {
    var id: String { benchmarkId }
    let benchmarkId: String
    let benchmarkType: String
    /// A `local` entry is generated on this device, so a cell built from it is not
    /// submitted (see ``JobCell/isSubmittable``).
    var source: BenchmarkSource = .remote
    let sampleCount: Int?
    let rawJson: [String: Any]

    /// Decoded once at parse so the run path does not re-decode `rawJson` per cell. `nil`
    /// if the entry isn't a known benchmark type.
    let definition: BenchmarkDefinition?

    /// `nil` for non-eval benchmarks.
    var evalId: EvalId? {
        (rawJson["parameter_eval_id"] as? String).map(EvalId.init)
    }

    /// `nil` for an unknown or legacy type.
    var type: BenchmarkType? { BenchmarkType(rawValue: benchmarkType) }
}

extension BenchmarkItem {
    /// An item from a definition alone (a local entry, a claimed definition, or the
    /// structured-id fallback on a catalog miss): only the `parameter_*` fields and no
    /// sample count.
    nonisolated init(parsed definition: BenchmarkDefinition, source: BenchmarkSource = .remote) {
        self.init(
            benchmarkId: definition.benchmarkId.value,
            benchmarkType: definition.benchmarkType.rawValue,
            source: source,
            sampleCount: nil,
            rawJson: definition.parameterFields,
            definition: definition)
    }
}

/// The benchmark catalog: the synced `remote/` half plus the generated `local/` half. There
/// is no bundled fallback, so the picker is empty until a sync or a local seed.
///
/// `all` is the id lookup table; `selectable` narrows it to what the New Job picker
/// offers, for visibility only. Rows are parsed with `JSONSerialization` so numeric fields
/// arrive as `NSNumber` and `as? Int` casts hold.
enum BenchmarkCatalog {
    /// Recomputed per access, so there is no cache to invalidate. That costs a read of the
    /// synced list, a directory listing and one read per local definition: resolve a single
    /// id with ``item(forId:store:)`` instead.
    ///
    /// Remote wins a duplicate id: preferring `local/` would make its results silently
    /// unsubmittable.
    static func all(store: BenchmarkStore) -> [BenchmarkItem] {
        let remote = parse(remoteRows(store))
        let remoteIds = Set(remote.map(\.benchmarkId))
        let local = store.list(.local)
            .filter { !remoteIds.contains($0.benchmarkId.value) }
            .map { BenchmarkItem(parsed: $0, source: .local) }
        return remote + local
    }

    /// Resolve one id without listing the catalog: the synced list's row when this build
    /// converts it, then the local file, then a definition parsed from a structured id. Some
    /// callers are SwiftUI computed properties, where `all` would list a directory per body
    /// pass.
    static func item(forId id: String, store: BenchmarkStore) -> BenchmarkItem? {
        if let row = remoteRows(store).first(where: { $0["benchmark_id"] as? String == id }),
           let item = parse([row]).first {
            return item
        }
        do {
            if let benchmarkId = try? BenchmarkId(id), let local = try store.get(.local(benchmarkId)) {
                return BenchmarkItem(parsed: local, source: .local)
            }
        } catch {
            AppLog.benchmarkSync.error("unreadable local benchmark \(id): \(error)")
        }
        return BenchmarkDefinition(parsingId: id).map { BenchmarkItem(parsed: $0) }
    }

    /// Every row the server sent; ``parse(_:)`` drops the ones this build cannot convert. A
    /// list that does not read is logged and lists nothing, so a corrupt file costs the
    /// synced half, not the screen.
    private static func remoteRows(_ store: BenchmarkStore) -> [[String: Any]] {
        do {
            return try store.listRemoteIndex().compactMap {
                try? JSONSerialization.jsonObject(with: $0.rawJson) as? [String: Any]
            }
        } catch {
            AppLog.benchmarkSync.error("unreadable remote benchmark index: \(error)")
            return []
        }
    }

    /// Pure, so catalog parsing is testable without a store.
    static func merged(syncedEntries: [[String: Any]]) -> [BenchmarkItem] {
        parse(syncedEntries)
    }

    /// `catalog` first, else a definition parsed from a structured id; `nil` when the id is
    /// neither listed nor parseable.
    static func item(forId id: String, in catalog: [String: BenchmarkItem]) -> BenchmarkItem? {
        catalog[id] ?? BenchmarkDefinition(parsingId: id).map { BenchmarkItem(parsed: $0) }
    }

    /// The definition a run reference names, as the CLI's `ws.benchmarks().get(run_ref)`,
    /// so a stored entry's own `benchmarkType` wins over a guess from the id's spelling.
    ///
    /// Only the named side is read: `local/<id>` and `remote/<id>` are different entries.
    /// A missing entry falls back to the structured id, so an unseeded ladder rung still
    /// resolves. A read that throws is logged and treated as absent, so one unreadable
    /// entry costs that reference, not the whole run.
    static func definition(for reference: SourcedBenchmarkId,
                           store: BenchmarkStore?) -> BenchmarkDefinition? {
        if let store {
            do {
                if let stored = try store.get(reference) { return stored }
            } catch {
                AppLog.benchmarkSync.error("unreadable benchmark \(reference): \(error)")
            }
        }
        return BenchmarkDefinition(parsingId: reference.id.value)
    }

    /// `eval` is excluded: its datasets don't run on this client's ladder path.
    private static let selectableTypes: Set<BenchmarkType> = [
        .prefillThroughput, .decodeThroughput, .endToEndLatency, .maxMemoryUsage,
        .vlThroughput, .vlMaxMemory,
    ]

    /// In tokens. Keeps the 8192 rung, which jetsam-OOMs on the supported devices, out of
    /// the picker.
    private static let contextLimit: UInt32 = 5000

    /// A visibility filter only: hidden benchmarks stay in `all` and remain runnable from
    /// the headless CLI or a server-assigned job. Use `all` for lookups and execution.
    static func selectable(store: BenchmarkStore) -> [BenchmarkItem] {
        selectable(from: all(store: store))
    }

    /// The cap on a vision rung's estimated context. The estimate over-counts the documented
    /// encoders, so a rung under it loads no more than this; one over it may count smaller,
    /// but the picker cannot know before a load.
    private static let visionContextLimit: UInt32 = 8192

    /// Pure, so the filter is testable without a store. Caps each rung by its own workload
    /// rather than by hardcoded token values. The text arms compute that workload here,
    /// not from the engine's sizing, because they ask how big the rung is, not what a cell
    /// loads with; the vision arms read the estimate, because the projector's count exists
    /// only once a cell loads.
    static func selectable(from items: [BenchmarkItem]) -> [BenchmarkItem] {
        items
            .filter { $0.type.map(selectableTypes.contains) ?? false }
            .filter {
                switch $0.definition {
                case let .prefillThroughput(b):
                    b.parameterPrefillTokens.value.magnitude < contextLimit
                case let .maxMemoryUsage(b):
                    b.parameterPrefillTokens.value.magnitude < contextLimit
                case let .decodeThroughput(b):
                    fits(prefill: b.parameterPrefillTokens.value.magnitude,
                         decode: b.parameterDecodeTokens.value.magnitude)
                case let .endToEndLatency(b):
                    fits(prefill: b.parameterPrefillTokens.value.magnitude,
                         decode: b.parameterDecodeTokens.value.magnitude)
                case let .vlThroughput(b):
                    LlamaRuntimeFlags.ctxSize(for: .vlThroughput(b)) <= visionContextLimit
                case let .vlMaxMemory(b):
                    LlamaRuntimeFlags.ctxSize(for: .vlMaxMemory(b)) <= visionContextLimit
                // The corpus cells run here but take a corpus the management server serves,
                // so there is no local form to offer: a device runs them by claiming the cell.
                case .eval, .vlSpecDecode, .none:
                    false
                // Plans only: the audio count comes from the projector, so no rung is sized.
                case .asrThroughput, .asrMaxMemory:
                    false
                }
            }
            .sorted { $0.benchmarkId < $1.benchmarkId }
    }

    /// A sum that overflows is out.
    private static func fits(prefill: UInt32, decode: UInt32) -> Bool {
        let (total, overflow) = prefill.addingReportingOverflow(decode)
        return !overflow && total < contextLimit
    }

    /// Drops any entry that does not decode into a `BenchmarkDefinition`, so every synced
    /// item has a non-nil `definition`. This is the filter: the synced list keeps every row
    /// the server sent, so a `304` never hands this build a list another build filtered.
    private static func parse(_ array: [[String: Any]]) -> [BenchmarkItem] {
        array
            .compactMap { obj -> BenchmarkItem? in
                guard let id = obj["benchmark_id"] as? String,
                      let type = obj["benchmark_type"] as? String,
                      let data = try? JSONSerialization.data(withJSONObject: obj),
                      let definition = try? JSONDecoder().decode(BenchmarkDefinition.self, from: data)
                else { return nil }
                return BenchmarkItem(
                    benchmarkId: id,
                    benchmarkType: type,
                    sampleCount: (obj["samples"] as? [[String: Any]])?.count,
                    rawJson: obj,
                    definition: definition
                )
            }
            .sorted { $0.benchmarkId < $1.benchmarkId }
    }
}

// MARK: - Benchmark × runtime capability

/// Which `(runtime, benchmark)` pairs this client can run, keyed on the runtime case only:
/// support is about what a runtime can measure, not how it is tuned. `AFMRuntime` defers to
/// this so the offered and accepted sets cannot drift.
///
/// - AFM can't isolate prefill, can't observe its out-of-process memory, and is text-only.
/// - ONNX Runtime GenAI runs in process, so `ProcessMemory` sees its peak. `eval` is out
///   because `OGATokenizer` applies no chat template; the catalog's ONNX models are text-only.
/// - llama.cpp vision kinds are also gated per model by an mmproj
///   (`NewJobView.isVlCompatible`), which this does not model.
nonisolated func isBenchmarkSupported(_ benchmark: BenchmarkType, on runtime: RuntimeKind) -> Bool {
    switch (runtime, benchmark) {
    case (.afm, .eval), (.afm, .decodeThroughput), (.afm, .endToEndLatency): return true
    case (.afm, _):             return false
    // No MLX vision or audio path in this app.
    case (.mlx, .vlThroughput), (.mlx, .vlMaxMemory), (.mlx, .vlSpecDecode),
         (.mlx, .asrThroughput), (.mlx, .asrMaxMemory): return false
    case (.mlx, _):             return true
    case (.onnxGenai, .prefillThroughput), (.onnxGenai, .decodeThroughput),
         (.onnxGenai, .endToEndLatency), (.onnxGenai, .maxMemoryUsage): return true
    case (.onnxGenai, _):       return false
    case (.llamaCpp, _):        return true
    }
}

// MARK: - Benchmark-type presentation

/// Labels, descriptions and ordering for benchmark types by `benchmark_type` string, with
/// a generic fallback for unknown or legacy types.
extension BenchmarkCatalog {
    static func displayName(for type: String) -> String {
        if let t = BenchmarkType(rawValue: type) { return t.displayName }
        return type
            .split(separator: "_")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    /// For dense table column headers.
    static func shortName(for type: String) -> String {
        BenchmarkType(rawValue: type)?.shortName ?? displayName(for: type)
    }

    static func description(for type: String) -> String {
        BenchmarkType(rawValue: type)?.summary ?? "Benchmark variants for this metric."
    }

    /// Lower sorts first; unknown types sort last.
    static func typeRank(for type: String) -> Int {
        BenchmarkType(rawValue: type)?.rank ?? unknownTypeRank
    }

    /// Derived rather than a literal, so adding a type cannot rank it below unknown ones.
    private static let unknownTypeRank =
        (BenchmarkType.allCases.map(\.rank).max() ?? 0) + 1
}
