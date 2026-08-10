import Foundation

/// The `benchmarks` group — `list`, `show` and `init-local`, over both halves of the
/// catalog: the synced `remote/` one and the generated `local/` one.
enum BenchmarkCommands {
    /// The kinds this client can execute, so `init-local` seeds only benchmarks a phone
    /// can actually run. `vl_throughput` is included — the llama.cpp engine runs vision
    /// cells — while the desktop clients pass their own sets.
    static let runnableKinds: [BenchmarkType] = [
        .prefillThroughput, .decodeThroughput, .endToEndLatency, .maxMemoryUsage,
        .eval, .vlThroughput,
    ]

    /// `benchmarks [type=]`: the catalog. Empty before the first sync or seed — reported
    /// as `count=0` rather than as an error, since "nothing yet" is a state, not a
    /// failure.
    static func list(type: BenchmarkType?, storage: Storage) {
        let all = BenchmarkCatalog.all(store: storage.benchmarks)
        let items = type.map { wanted in all.filter { $0.type == wanted } } ?? all
        HeadlessRunner.log("benchmarks count=\(items.count)")
        for item in items {
            HeadlessRunner.log("benchmark id=\(item.benchmarkId) type=\(item.benchmarkType) "
                + "samples=\(item.sampleCount.map(String.init) ?? "-")")
        }
    }

    /// `benchmarks show benchmark=<id>`: one benchmark's resolved definition. Returns
    /// `false` when the id is not in the catalog, so the process exits non-zero — an id
    /// that does not resolve is the mistake this verb exists to catch before a run.
    static func show(id: String, storage: Storage) -> Bool {
        let all = BenchmarkCatalog.all(store: storage.benchmarks)
        guard let item = all.first(where: { $0.benchmarkId == id }) else {
            HeadlessRunner.log("benchmarks show ERROR no benchmark id=\(id) in the catalog "
                + "(\(all.count) synced); run `headlessrun benchmarks` to list them")
            return false
        }
        HeadlessRunner.log("benchmark id=\(item.benchmarkId) type=\(item.benchmarkType) "
            + "samples=\(item.sampleCount.map(String.init) ?? "-") "
            + "eval=\(item.evalId?.rawValue ?? "-") "
            + "parsed=\(item.definition == nil ? "no" : "yes")")
        return true
    }

    /// `benchmarks init-local`: write the standard ladder + smoke set into the local
    /// half — the CLI's `benchmarks init-local`.
    ///
    /// These never reach the server, so their results stay on the device (see
    /// ``BenchmarkSource``). Idempotent: a second run updates rather than duplicating.
    @discardableResult
    static func initLocal(storage: Storage) -> Bool {
        do {
            let summary = try StandardBenchmarks.seedLocal(
                into: storage.benchmarks, kinds: runnableKinds)
            HeadlessRunner.log(
                "benchmarks init-local created=\(summary.created) updated=\(summary.updated)")
            return true
        } catch {
            HeadlessRunner.log("benchmarks init-local ERROR \(error.localizedDescription)")
            return false
        }
    }
}
