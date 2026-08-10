import Foundation

/// A single benchmark definition parsed from the synced catalog.
///
/// `rawJson` preserves the full server-shaped payload so downstream consumers
/// (`NewJobView`, `JobDetailView`) can read parameter fields directly.
nonisolated struct BenchmarkItem: Identifiable {
    var id: String { benchmarkId }
    let benchmarkId: String
    let benchmarkType: String
    /// Which catalog half this came from. A `local` entry is generated on this device,
    /// so a cell built from it is not submitted (see ``JobCell/isSubmittable``).
    var source: BenchmarkSource = .remote
    let sampleCount: Int?
    let rawJson: [String: Any]

    /// The strict, typed definition — decoded once when the catalog is parsed, so
    /// the run path consumes it directly instead of re-serializing `rawJson` to a
    /// string and re-decoding per cell. `nil` if the entry isn't a known benchmark
    /// type. (`rawJson` is retained for the UI / CSV / ctx-size readers for now.)
    let definition: BenchmarkDefinition?

    /// Typed eval id, parsed from `parameter_eval_id`; `nil` for non-eval
    /// benchmarks. Mirrors the Rust `BenchmarkDefinition::Eval.parameter_eval_id`.
    var evalId: EvalId? {
        (rawJson["parameter_eval_id"] as? String).map(EvalId.init)
    }

    /// The typed benchmark kind, mapped from the stored `benchmark_type` string;
    /// `nil` for an unknown/legacy type. The typed lens the context-size and
    /// selection APIs read instead of re-parsing the string per call.
    var type: BenchmarkType? { BenchmarkType(rawValue: benchmarkType) }
}

extension BenchmarkItem {
    /// Reconstruct an item from a definition parsed off a structured id — the id,
    /// its type, and the minimal `parameter_*` shape, no sample count. The
    /// catalog-miss fallback shared by the run path and CSV export.
    nonisolated init(parsed definition: BenchmarkDefinition, source: BenchmarkSource = .remote) {
        self.init(
            benchmarkId: definition.benchmarkId,
            benchmarkType: definition.benchmarkType,
            source: source,
            sampleCount: nil,
            rawJson: definition.parameterFields,
            definition: definition)
    }
}

/// The benchmark catalog, driven entirely by the server-synced catalog
/// (`BenchmarkSync`). `all` is the synced set parsed from disk; there is no
/// bundled fallback, so the picker is empty until the first sync completes and
/// then tracks the server exactly.
///
/// `all` is the all-time id lookup table: a historical result resolves its
/// type/params here by id as long as the server still lists that benchmark.
///
/// `selectable` narrows `all` to what the New Job picker advertises — the four
/// supported ladder types, capped to rungs whose required context stays under
/// ~5k (see `selectable(from:)`). Visibility only; hidden benchmarks stay runnable.
///
/// The JSON shape mirrors the server payload exactly so downstream consumers
/// (`NewJobView`, `JobDetailView`) keep reading `rawJson` the same way. Parse via
/// `JSONSerialization` — same path the disk cache uses — so numeric fields arrive
/// as `NSNumber` and `as? Int` casts hold.
enum BenchmarkCatalog {
    /// Both halves of the catalog: the synced `remote/` index plus the generated
    /// `local/` definitions. Recomputed per access, so it always reflects the latest
    /// sync and seed without a cache to invalidate.
    ///
    /// Not free once the local half is seeded: one `index.json` read plus a directory
    /// listing *and one read per local definition* (`init-local` writes 34). Callers
    /// resolving a single id should take ``item(forId:store:)``, which reads one file.
    ///
    /// Remote wins a duplicate id: the same benchmark defined on both sides is the
    /// submittable one, and preferring `local/` would make its results silently
    /// unsubmittable.
    static func all(store: BenchmarkStore) -> [BenchmarkItem] {
        let remote = parse(store.listRemoteIndex().map(\.rawJson))
        let remoteIds = Set(remote.map(\.benchmarkId))
        let local = store.list(.local)
            .filter { !remoteIds.contains($0.benchmarkId) }
            .map { BenchmarkItem(parsed: $0, source: .local) }
        return remote + local
    }

    /// Resolve one id without listing the catalog: the synced index, then the local
    /// half's single file, then a definition reconstructed from a structured id.
    ///
    /// Exists because the whole-catalog read is per-access and some callers are SwiftUI
    /// computed properties — `all` there is a directory listing per body pass.
    static func item(forId id: String, store: BenchmarkStore) -> BenchmarkItem? {
        if let row = store.listRemoteIndex().first(where: { $0.benchmarkId == id }),
           let item = parse([row.rawJson]).first {
            return item
        }
        if let local = store.get(.local(id)) {
            return BenchmarkItem(parsed: local, source: .local)
        }
        return BenchmarkDefinition(parsingId: id).map { BenchmarkItem(parsed: $0) }
    }

    /// Parses synced list entries into `BenchmarkItem`s. Pure — exposed so catalog
    /// consumption is testable without a store. There is no bundled fallback: an
    /// empty/unparseable input yields an empty catalog.
    static func merged(syncedEntries: [[String: Any]]) -> [BenchmarkItem] {
        parse(syncedEntries)
    }

    /// Resolve a benchmark id to an item: the synced catalog first, else a
    /// definition reconstructed from the structured id (the four ladder types).
    /// `nil` when the id is neither listed nor parseable. The catalog-independent
    /// resolver shared by the run path (`JobExecutor`) and CSV export.
    static func item(forId id: String, in catalog: [String: BenchmarkItem]) -> BenchmarkItem? {
        catalog[id] ?? BenchmarkDefinition(parsingId: id).map { BenchmarkItem(parsed: $0) }
    }

    /// The benchmark types the New Job picker offers. `eval` and `vl_throughput`
    /// are excluded — they don't run on this client's ladder path.
    private static let selectableTypes: Set<BenchmarkType> = [
        .prefillThroughput, .decodeThroughput, .endToEndLatency, .maxMemoryUsage,
    ]

    /// Upper bound (tokens) on the context a selectable benchmark may require — the
    /// per-type ladder cap. Keeps the working set under ~5k so the heaviest rung
    /// (8192), which jetsam-OOMs on the supported devices, isn't offered.
    private static let contextLimit: UInt32 = 5000

    /// Benchmarks advertised in the New Job picker — a UI-visibility filter only.
    /// Hidden benchmarks (unsupported types, or rungs over the context cap) stay in
    /// `all` and remain runnable via the headless CLI or a server-assigned job;
    /// they're just not offered for manual selection. Use `all` for id lookups and
    /// execution.
    static func selectable(store: BenchmarkStore) -> [BenchmarkItem] {
        selectable(from: all(store: store))
    }

    /// `items` narrowed to the New Job picker's offering. Pure — exposed for
    /// testing the filter without the store or the `LocalStorage` global.
    ///
    /// 1. Keep only the four supported ladder types (drop `eval`, `vl_throughput`).
    /// 2. Keep each rung whose required context stays under `contextLimit`, where
    ///    "context" is the rung's own workload (prefill for prefill/max-memory,
    ///    prefill + decode for decode/e2e). This caps the synced ladder by workload
    ///    rather than hardcoding token values.
    ///
    ///    Stated here rather than read from the engine's sizing: this asks how big the
    ///    rung is, not what a cell will load with, and the two answers diverge on the
    ///    types step 1 has already dropped.
    ///
    /// Output is stably sorted by `benchmarkId`.
    static func selectable(from items: [BenchmarkItem]) -> [BenchmarkItem] {
        items
            .filter { $0.type.map(selectableTypes.contains) ?? false }
            .filter {
                switch $0.definition {
                case let .prefillThroughput(_, prefill), let .maxMemoryUsage(_, prefill):
                    prefill < contextLimit
                case let .decodeThroughput(_, prefill, decode),
                     let .endToEndLatency(_, prefill, decode):
                    prefill.addingReportingOverflow(decode).1
                        ? false : prefill + decode < contextLimit
                case .eval, .vlThroughput, .none:
                    false
                }
            }
            .sorted { $0.benchmarkId < $1.benchmarkId }
    }

    /// Parse synced entries into items, **ignoring any we can't fully decode** into
    /// a `BenchmarkDefinition`: an unknown `benchmark_type`, or a known type with a
    /// schema mismatch / missing params, is dropped rather than surfaced. The sync
    /// layer already filters these (`BenchmarkSync.keepParseable`); repeating the
    /// guard here means a stale or hand-edited on-disk cache can't smuggle an
    /// unparseable benchmark into the catalog. Every item in `all` therefore has a
    /// non-nil `definition`.
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

/// The single source of truth for which `(runtime, benchmark)` pairs this client
/// can actually run — an enum-driven capability check rather than per-site boolean
/// guesses. Support is config-independent (it's about what a runtime *can measure*,
/// not how it's tuned), so it's keyed on the runtime *case* only. Lives next to
/// `BenchmarkType` to keep `PlanTypes/Runtime.swift` free of benchmark policy.
///
/// - AFM: only `eval`, `decode_throughput`, `end_to_end_latency` — it can't isolate
///   prefill, can't observe its out-of-process memory, and is text-only (see
///   `AFMRuntime`, which defers to this so the offered and accepted sets can't drift).
/// - MLX: everything except `vl_throughput` (no MLX VL in this app).
/// - llama.cpp: everything — VL is *additionally* gated per-model by an mmproj at the
///   call site (`NewJobView.isVlCompatible`), which this function doesn't model.
nonisolated func isBenchmarkSupported(_ benchmark: BenchmarkType, on runtime: RuntimeKind) -> Bool {
    switch (runtime, benchmark) {
    case (.afm, .eval), (.afm, .decodeThroughput), (.afm, .endToEndLatency): return true
    case (.afm, _):             return false   // AFM: no prefill / max-memory / VL
    case (.mlx, .vlThroughput): return false   // no MLX VL in this app
    case (.mlx, _):             return true
    case (.llamaCpp, _):        return true     // VL further gated per-model by mmproj
    }
}

// MARK: - Benchmark-type presentation

/// Single source of truth for how benchmark *types* are labelled, described, and
/// ordered across the app. These delegate to `BenchmarkType` for known types and
/// fall back to generic formatting for unknown/legacy ones, so every screen reads
/// the same labels by `benchmark_type` string.
extension BenchmarkCatalog {
    /// Full, human-facing name — section headers and detail pages.
    static func displayName(for type: String) -> String {
        if let t = BenchmarkType(rawValue: type) { return t.displayName }
        return type
            .split(separator: "_")
            .map { $0.capitalized }
            .joined(separator: " ")
    }

    /// Compact name for dense table column headers, where the full name is too
    /// wide to fit.
    static func shortName(for type: String) -> String {
        BenchmarkType(rawValue: type)?.shortName ?? displayName(for: type)
    }

    /// One-line description of what the benchmark measures.
    static func description(for type: String) -> String {
        BenchmarkType(rawValue: type)?.summary ?? "Benchmark variants for this metric."
    }

    /// Canonical ordering of benchmark types across every screen (selection,
    /// review, and detail). Lower sorts first; unknown/legacy types sink to the
    /// bottom.
    static func typeRank(for type: String) -> Int {
        BenchmarkType(rawValue: type)?.rank ?? 6
    }
}
