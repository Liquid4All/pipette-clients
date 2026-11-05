import Foundation

extension JobCell {
    /// Build one `.pending` cell per benchmark reference for `model`, dropping any the
    /// store does not hold and the id does not describe (reported via `onSkip`).
    ///
    /// A reference carries its provenance (`local/<id>`, `remote/<id>`, or a bare id meaning
    /// remote), which becomes `benchmarkSource` and keeps generated local results out of the
    /// submit sweep.
    ///
    /// The store is consulted before the structured id, as the CLI's `benchmarks run` does:
    /// an id is a naming convention, not a description, and the vision kinds, `eval` and
    /// every `_smoke` cell have parameters no id spells. The structured-id fallback keeps the
    /// ladder ids runnable without a seeded catalog. The server is asked earlier, by
    /// ``resolvable(benchmarkIds:store:server:http:auth:onSkip:)``, so this stays synchronous.
    ///
    /// For a claimed run that call also returns the definitions the executor runs, so a cell
    /// built here takes only its type from the store.
    ///
    /// Shared by the headless CLI and deep links. `runtimeFlags` and `benchmarkFlags` ride
    /// whole on each spec, so every knob they name (`ctx_size` and `load_mode` included)
    /// reaches the run.
    static func pending(
        benchmarkIds ids: [String], for model: DiscoveredModel,
        store: BenchmarkStore? = nil,
        runtimeFlags: RuntimeFlagRef? = nil,
        benchmarkFlags: BenchmarkFlagRef? = nil,
        onSkip: (String) -> Void = { _ in }
    ) -> [JobCell] {
        ids.compactMap { reference in
            guard let sourced = try? SourcedBenchmarkId(parsing: reference),
                  let def = BenchmarkCatalog.definition(for: sourced, store: store)
            else {
                onSkip(reference)
                return nil
            }
            return JobCell(
                cellId: CellId(UUID().uuidString), benchmarkId: sourced.id.value,
                benchmarkType: def.benchmarkType,
                runStatus: .pending, serverJobId: nil, errorMessage: nil,
                source: model.source, benchmarkSource: sourced.source,
                runtimeFlags: runtimeFlags, benchmarkFlags: benchmarkFlags)
        }
    }

    /// The references among `ids` that `pending` will build a cell for.
    ///
    /// Split from `pending` so a caller can run it before a model download: an unknown id
    /// then fails in seconds, and a benchmark published after the last `sync` can run.
    /// `local/<id>` is never fetched; the server has no copy. With no `server`, nothing is,
    /// and a reference resolves from the store and the structured id alone.
    ///
    /// With a `server` (a claimed run), each remote reference resolves as the planner worker's
    /// claim does, through ``ensureRemoteBenchmarkCached(store:server:http:auth:benchmarkId:)``,
    /// and the result carries those definitions so the executor runs them, not an older synced row.
    ///
    /// A fetch that fails in transit throws, so the caller fails the run as retriable.
    /// A definition the store would not take is skipped, since `pending` types its cell there.
    static func resolvable(
        benchmarkIds ids: [String],
        store: BenchmarkStore,
        server: ServerUrl?,
        http: HttpClient,
        auth: ClientIdentity?,
        onSkip: (_ reference: String, _ reason: String) -> Void = { _, _ in }
    ) async throws -> ResolvableBenchmarks {
        let unrecognized = "unrecognized benchmark id"
        var kept: [String] = []
        var claimed: [BenchmarkDefinition] = []
        for reference in ids {
            guard let sourced = try? SourcedBenchmarkId(parsing: reference) else {
                onSkip(reference, unrecognized)
                continue
            }
            guard sourced.source == .remote, let server else {
                if BenchmarkCatalog.definition(for: sourced, store: store) != nil {
                    kept.append(reference)
                } else {
                    onSkip(reference, unrecognized)
                }
                continue
            }
            let definition: BenchmarkDefinition
            do {
                definition = try await ensureRemoteBenchmarkCached(
                    store: store, server: server, http: http, auth: auth,
                    benchmarkId: sourced.id.value)
            } catch let unknown as UnknownRemoteBenchmarkError {
                onSkip(reference, "\(unrecognized) (\(unknown.message))")
                continue
            }
            guard BenchmarkCatalog.definition(for: sourced, store: store) != nil else {
                onSkip(reference, "fetched from \(server.value), but this device's "
                    + "benchmark catalog would not store it")
                continue
            }
            kept.append(reference)
            claimed.append(definition)
        }
        return ResolvableBenchmarks(references: kept, claimed: claimed)
    }
}

nonisolated struct ResolvableBenchmarks: Sendable {
    let references: [String]
    /// What a claimed run's cells execute (``JobExecutor``'s `claimedBenchmarks`). Empty
    /// without a server.
    let claimed: [BenchmarkDefinition]
}
