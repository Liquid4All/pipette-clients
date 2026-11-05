import Foundation
import MLXLMCommon

/// The `benchmarks` group over both halves of the catalog (synced `remote/`, generated
/// `local/`). `run` also serves the sweep forms (`bench` and the bare invocation), which
/// are `benchmarks run` over a set of benchmark ids for the fleet harness.
enum BenchmarksCommands {
    /// The kinds this client can execute, so `init-local` seeds only benchmarks a phone can
    /// run. The vision kinds are included: the llama.cpp engine runs them.
    static let runnableKinds: [BenchmarkType] = [
        .prefillThroughput, .decodeThroughput, .endToEndLatency, .maxMemoryUsage,
        .eval, .vlThroughput, .vlMaxMemory,
    ]

    /// Empty before the first sync or seed is `count=0`, not an error: "nothing yet" is a
    /// state, not a failure.
    static func list(benchmarkType: BenchmarkType?, ws: PipetteWorkspace) {
        let rows = Self.rows(benchmarkType: benchmarkType, store: ws.benchmarks)
        HeadlessRunner.log("benchmarks count=\(rows.count)")
        for row in rows { HeadlessRunner.log(row) }
    }

    /// Each row names the reference `run` and `show` take, so the same id on both halves is
    /// two rows.
    static func rows(benchmarkType: BenchmarkType?, store: BenchmarkStore) -> [String] {
        [BenchmarkSource.local, .remote].flatMap { source in
            store.list(source)
                .filter { benchmarkType == nil || $0.benchmarkType == benchmarkType }
                .map { definition in
                    "benchmark benchmark=\(SourcedBenchmarkId(source: source, id: definition.benchmarkId)) "
                        + "type=\(definition.benchmarkType.rawValue)"
                }
        }
    }

    /// `false` for a reference the catalog does not hold: catching that before a run is
    /// what this verb is for.
    static func show(_ reference: SourcedBenchmarkId, ws: PipetteWorkspace) -> Bool {
        let stored: BenchmarkDefinition?
        do {
            stored = try ws.benchmarks.get(reference)
        } catch {
            // A body this build cannot read is an error, not "no such id".
            HeadlessRunner.log("benchmarks show ERROR reading `\(reference)`: "
                + "\(error.localizedDescription)")
            return false
        }
        guard let definition = stored else {
            HeadlessRunner.log("benchmarks show ERROR unknown benchmark reference "
                + "`\(reference)`; run `headlessrun benchmarks list` to list them")
            return false
        }
        HeadlessRunner.log(writeBenchmark(definition, source: reference.source))
        return true
    }

    /// The crate's `write_benchmark`: parameters under their wire names, so each greps the
    /// same as the definition file. An eval's samples are counted, not printed.
    static func writeBenchmark(_ definition: BenchmarkDefinition, source: BenchmarkSource) -> String {
        var fields = ["id=\(definition.benchmarkId)", "type=\(definition.benchmarkType.rawValue)",
                      "source=\(source.rawValue)"]
        switch definition {
        case let .maxMemoryUsage(b):
            fields.append("parameter_prefill_tokens=\(b.parameterPrefillTokens)")
        case let .prefillThroughput(b):
            fields += ["parameter_prefill_tokens=\(b.parameterPrefillTokens)",
                       "parameter_batch_size=\(b.parameterBatchSize)"]
        case let .decodeThroughput(b):
            fields += ["parameter_prefill_tokens=\(b.parameterPrefillTokens)",
                       "parameter_decode_tokens=\(b.parameterDecodeTokens)",
                       "parameter_batch_size=\(b.parameterBatchSize)"]
        case let .endToEndLatency(b):
            fields += ["parameter_prefill_tokens=\(b.parameterPrefillTokens)",
                       "parameter_decode_tokens=\(b.parameterDecodeTokens)",
                       "parameter_batch_size=\(b.parameterBatchSize)"]
        case let .eval(b):
            fields += ["parameter_eval_id=\(b.parameterEvalId.rawValue)",
                       "parameter_dataset_name=\(b.parameterDatasetName)",
                       "parameter_max_tokens=\(b.parameterMaxTokens)"]
            if let choices = b.parameterMcqChoices {
                fields.append("parameter_mcq_choices=\(choices.joined(separator: ","))")
            }
            if let v = b.parameterTemperature { fields.append("parameter_temperature=\(v.value)") }
            if let v = b.parameterTopP { fields.append("parameter_top_p=\(v.value)") }
            if let v = b.parameterTopK { fields.append("parameter_top_k=\(v.value)") }
            if let v = b.parameterMinP { fields.append("parameter_min_p=\(v.value)") }
            if let v = b.parameterSeed { fields.append("parameter_seed=\(v.value)") }
            if let samples = b.samples { fields.append("samples=\(samples.count)") }
        case let .vlThroughput(b):
            fields += ["parameter_image_width=\(b.parameterImageWidth)",
                       "parameter_image_height=\(b.parameterImageHeight)",
                       "parameter_text_tokens=\(b.parameterTextTokens)",
                       "parameter_decode_tokens=\(b.parameterDecodeTokens)",
                       "parameter_num_images=\(b.parameterNumImages)"]
        case let .vlMaxMemory(b):
            fields += ["parameter_image_width=\(b.parameterImageWidth)",
                       "parameter_image_height=\(b.parameterImageHeight)",
                       "parameter_text_tokens=\(b.parameterTextTokens)"]
            if let v = b.parameterDecodeTokens { fields.append("parameter_decode_tokens=\(v)") }
            fields.append("parameter_num_images=\(b.parameterNumImages)")
        // The corpus is printed as its pin: the digest is what a row's
        // `observation_dataset_digest` is compared against.
        case let .vlSpecDecode(b):
            fields += ["parameter_draft_length=\(b.parameterDraftLength)",
                       "parameter_max_tokens=\(b.parameterMaxTokens)",
                       "parameter_batch_size=\(b.parameterBatchSize)",
                       "parameter_temperature=\(b.parameterTemperature.value)"]
                + writeCorpus(b.dataset)
        case let .asrThroughput(b):
            fields += asrFields(b.parameterAudioDurationMs, b.parameterAudioSampleRateHz,
                                b.parameterAudioChannels, b.parameterTextTokens,
                                b.parameterDecodeTokens)
        case let .asrMaxMemory(b):
            fields += asrFields(b.parameterAudioDurationMs, b.parameterAudioSampleRateHz,
                                b.parameterAudioChannels, b.parameterTextTokens,
                                b.parameterDecodeTokens)
        }
        return "benchmark " + fields.joined(separator: " ")
    }

    private static func asrFields(
        _ durationMs: AudioDurationMillis, _ sampleRate: AudioSampleRate, _ channels: AudioChannels,
        _ text: NonNegativeCount, _ decode: AsrDecodeTokens
    ) -> [String] {
        ["parameter_audio_duration_ms=\(durationMs.value)",
         "parameter_audio_sample_rate_hz=\(sampleRate.value)",
         "parameter_audio_channels=\(channels.value)",
         "parameter_text_tokens=\(text)",
         "parameter_decode_tokens=\(decode.value)"]
    }

    private static func writeCorpus(_ dataset: Dataset) -> [String] {
        var fields = ["dataset_format=\(dataset.format.rawValue)",
                      "dataset_digest=\(dataset.digest.value)",
                      "dataset_bytes=\(dataset.bytes)",
                      "dataset_upstream=\(dataset.upstream.reference)"]
        if let subset = dataset.subset { fields.append("dataset_subset=\(subset)") }
        // The server stamps the URL on the detail route alone, so a definition read from
        // elsewhere has a pin it cannot fetch. `-` shows that before a run, not after.
        fields.append("dataset_url=\(dataset.url?.value ?? "-")")
        return fields
    }

    /// `benchmarks init-local`: write the standard ladder and smoke set into the local half,
    /// whose results never reach the server (``BenchmarkSource``). A rerun updates in place.
    @discardableResult
    static func initLocal(ws: PipetteWorkspace) -> Bool {
        do {
            let summary = try StandardBenchmarks.seedStandardLocal(
                ws.benchmarks, kinds: runnableKinds)
            HeadlessRunner.log(
                "benchmarks init-local created=\(summary.created) updated=\(summary.updated)")
            return true
        } catch {
            HeadlessRunner.log("benchmarks init-local ERROR \(error.localizedDescription)")
            return false
        }
    }

    /// Run catalog benchmarks against the Apple foundation model through the same
    /// `JobLauncher` path a UI-started job uses, so recording and `submit=1` behave as for
    /// the file-backed runtimes. `true` only if the job completed (the exit code).
    static func runAFM(benchmarkIds: [String], submit: Bool, claim: String? = nil,
                       claimedAt: ContinuousClock.Instant? = nil,
                       ws: PipetteWorkspace, http: HttpClient) async -> Bool {
        setvbuf(stdout, nil, _IONBF, 0)   // unbuffered, so per-token progress shows live
        HeadlessRunner.log("afm availability=\(AFMRuntime.availabilityText()) benchmarks=\(benchmarkIds) submit=\(submit)")

        guard let resolved = await resolvableBenchmarkIds(benchmarkIds, claim: claim, ws: ws,
                                                          http: http, tag: "afm") else {
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }
        // The bare `.appleFoundation` source makes the executor skip path resolution and
        // the memory gate: AFM has no weights on disk.
        let cells = JobCell.pending(benchmarkIds: resolved.references, for: .appleFoundation,
                                    store: ws.benchmarks) {
            HeadlessRunner.log("skip \($0): unrecognized benchmark id")
        }
        guard !cells.isEmpty else {
            HeadlessRunner.log("afm ERROR no benchmarks resolved from \(benchmarkIds)")
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }

        // AFM has no load knobs, so the ngl / ctx / batch values are inert placeholders.
        let final: JobManifest? = await withCheckedContinuation { cont in
            Task { @MainActor in
                let launched = JobLauncher.launch(
                    cells: cells, nGpuLayers: 99, contextSize: 4096, prefillBatch: 512,
                    contributeResults: submit, claimJobId: claim, claimedAt: claimedAt,
                    claimedBenchmarks: resolved.claimed,
                    jobRunner: JobRunner(), jobStore: JobStore(ws: ws), ws: ws,
                    onFinish: { cont.resume(returning: $0) })
                if launched == nil { cont.resume(returning: nil) }
            }
        }
        guard let final else {
            HeadlessRunner.log("afm ERROR job launch blocked: another job is running")
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }
        await writeRecordDone(final, submitRequested: submit, ws: ws)
        return finishSweep(final, claimed: claim != nil)
    }

    private static func finishSweep(_ final: JobManifest, claimed: Bool) -> Bool {
        let failed = final.cells.first { $0.runStatus == .failed }
        let done = HeadlessRunner.sweepDone(
            status: final.status, cellFailure: failed.map { $0.errorMessage ?? "the cell failed" },
            claimed: claimed)
        HeadlessRunner.benchDone(done.code, done.detail)
        return done.code == HeadlessExit.ok
    }

    // MARK: - Bench (the verb: resolve → download-if-missing → run)

    /// The model a selector names: the one it carries, or the installed model its digest
    /// prefix matches.
    static func resolve(_ selector: HeadlessCommand.ModelSelector,
                                ws: PipetteWorkspace) async -> Model? {
        switch selector {
        case let .model(model): return model
        case let .digest(prefix):
            return await MainActor.run { modelMatching(digest: prefix, ws: ws) }
        }
    }

    /// Ambiguity is an error rather than a pick: two entries sharing a prefix are two
    /// different artifacts, and guessing would bench the wrong one.
    private static func modelMatching(digest prefix: String, ws: PipetteWorkspace) -> Model? {
        do {
            return try resolveModel(
                digestPrefix: prefix, installed: try ws.models.list().map(\.declared))
        } catch {
            HeadlessRunner.log("bench ERROR \(error.localizedDescription)")
            return nil
        }
    }

    /// The catalog rows a `bench match=` / `quant=` pair selects for one runtime. More than
    /// one means the caller must narrow.
    ///
    /// `match` is a substring of the name or repo id, so a name is a candidate wherever a
    /// longer name contains it. Rows named exactly `match` win when any exist, which keeps
    /// the shorter of such a pair selectable.
    ///
    /// A tiebreak, not a guarantee: `quant=` narrows before the preference, so an
    /// exactly-named row without that quant is gone and a substring sibling wins alone.
    ///
    /// `rows` is a parameter so a test can use a fixed colliding pair: the catalog's own
    /// collisions move with every intake.
    static func resolveCatalogRows(match: String, quant: String?,
                                   runtime: RuntimeType,
                                   rows: [CatalogEntry] = CatalogEntry.catalog) -> [CatalogEntry] {
        // An absent and an empty `quant=` both mean "any quant".
        let wantedQuant = (quant?.isEmpty == false) ? quant : nil
        let candidates = rows.filter { entry in
            let nameMatches = match.isEmpty
                || entry.name.localizedCaseInsensitiveContains(match)
                || entry.repoIdentifier.localizedCaseInsensitiveContains(match)
            let quantMatches = wantedQuant.map {
                entry.quant?.caseInsensitiveCompare($0) == .orderedSame
            } ?? true
            return isCompatible(entry.source, runtime) && nameMatches && quantMatches
        }
        // Localized, to fold case as the filter above does; otherwise a candidate might
        // not be recognised as the exactly named one.
        let exact = candidates.filter {
            $0.name.localizedCaseInsensitiveCompare(match) == .orderedSame
                || $0.repoIdentifier.localizedCaseInsensitiveCompare(match) == .orderedSame
        }
        return exact.isEmpty ? candidates : exact
    }

    /// `benchmarks run` and the sweep forms: resolve a model, download it if missing, then
    /// `runSweep`. `runSweep` and `runAFM` print `BENCH_DONE`, so the early failures here
    /// print it themselves.
    ///
    /// `coordinator` defaults to `nil` rather than `.shared` because `.shared` is main-actor
    /// isolated and a default argument is not.
    static func run(spec: HeadlessCommand.ModelSelector?, modelMatch: String?, quant: String?,
                    runtime: RuntimeType, batch: Int, runtimeFlags: RuntimeFlags? = nil,
                    benchmarkFlags: BenchmarkFlags? = nil,
                    benchmarks: [String], metrics: [String], offsets: [Int],
                    submit: Bool, claim: String? = nil,
                    claimedAt: ContinuousClock.Instant? = nil,
                    ws: PipetteWorkspace, http: HttpClient,
                    coordinator: DownloadCoordinator? = nil) async -> Bool {
        let requested = resolveBenchmarkIds(metrics: metrics, offsets: offsets, explicit: benchmarks)
        guard !requested.isEmpty else {
            HeadlessRunner.log("bench ERROR no benchmarks resolved from metrics=\(metrics) offsets=\(offsets) benchmarks=\(benchmarks)")
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }

        if runtime == .appleFoundation {
            return await runAFM(benchmarkIds: requested, submit: submit, claim: claim,
                                claimedAt: claimedAt, ws: ws, http: http)
        }

        // Before the download, so an unknown id fails in seconds rather than after it.
        guard let resolvable = await resolvableBenchmarkIds(requested, claim: claim, ws: ws,
                                                            http: http, tag: "bench") else {
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }
        let benchmarkIds = resolvable.references
        guard !benchmarkIds.isEmpty else {
            HeadlessRunner.log("bench ERROR no benchmarks resolved from \(requested)")
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }

        let target: Model
        if let spec {
            switch spec {
            case let .model(model):
                target = model
            case let .digest(prefix):
                guard let matched = modelMatching(digest: prefix, ws: ws) else {
                    HeadlessRunner.benchDone(HeadlessExit.failure); return false
                }
                target = matched
            }
        } else {
            let match = modelMatch ?? ""
            let resolved = resolveCatalogRows(match: match, quant: quant, runtime: runtime)
            let runtimeTag = runtime.rawValue
            guard !resolved.isEmpty else {
                let available = CatalogEntry.catalog
                    .filter { isCompatible($0.source, runtime) }
                    .map { "\($0.name)[\($0.quant ?? "-")]" }.joined(separator: ", ")
                HeadlessRunner.log("bench ERROR no catalog model matches match=\(match) quant=\(quant ?? "-"); "
                    + "available for \(runtimeTag): \(available)")
                HeadlessRunner.benchDone(HeadlessExit.failure); return false
            }
            guard resolved.count == 1 else {
                let names = resolved.map { "\($0.name)[\($0.quant ?? "-")]" }.joined(separator: ", ")
                // Advising `quant=` to a caller who already passed one names the remedy
                // they just used; what is left to them is a longer `match=`.
                let remedy = (quant?.isEmpty == false)
                    ? "name a row in full with match="
                    : "narrow with quant="
                HeadlessRunner.log("bench ERROR ambiguous (\(resolved.count)): \(names); \(remedy)")
                HeadlessRunner.benchDone(HeadlessExit.failure); return false
            }
            target = resolved[0].source
        }

        // From the resolved model's format, so a `spec=` wins over a disagreeing
        // `runtime=`. `thisBuild(for:)` is exhaustive over `Model`, so a new format cannot
        // fall through to the wrong engine.
        let targetRuntime = RuntimeType.of(Runtime.thisBuild(for: target))

        // AFM returned above, so a resolved download target always has a repo.
        let repoLabel = target.repo?.description ?? "?"

        HeadlessRunner.log("bench ensuring \(repoLabel)")
        let transfer = DownloadProgressLog(label: "bench", what: repoLabel)
        do {
            _ = try await ensureModel(target, ws: ws, coordinator: coordinator ?? .shared,
                                      progress: transfer.report)
            transfer.finish()
        } catch {
            HeadlessRunner.log("bench FAILED \(repoLabel): \(error.localizedDescription)")
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }

        // Explicit return type so `.first(where:)` isn't taken for `MainActor.run`'s
        // closure. `asRequested`: `==` ignores a vault label, and the manifest holds the
        // label of whichever fetch created the entry.
        func discover() async -> DiscoveredModel? {
            await MainActor.run { () -> DiscoveredModel? in
                ws.availableModels().first(where: { $0.source == target })?.asRequested(target)
            }
        }

        // The store resolved it, so a miss here means the two disagree: its own line.
        guard let file = await discover() else {
            HeadlessRunner.log("bench NOT-DISCOVERED \(repoLabel)")
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }
        return await runSweep(runtime: targetRuntime, batch: batch,
                         runtimeFlags: runtimeFlags, benchmarkFlags: benchmarkFlags,
                         benchmarkIds: benchmarkIds,
                         runCoherence: false, runCalibrate: false, runPromptSeed: false,
                         offsets: offsets, resolved: file, modelMatch: nil,
                         submit: submit, claim: claim, claimedAt: claimedAt,
                         claimedBenchmarks: resolvable.claimed, ws: ws)
    }

    /// The run knobs as the command line carried them, before the cell's `BenchmarkFlags`
    /// variant decides which it accepts. No timeouts: iOS's `BenchmarkFlagRef` has none.
    nonisolated struct CellKnobs: Equatable, Sendable {
        var readinessMaxWaitSecs: UInt64?
        var readinessSkipThermal = false
    }

    /// `nil` when no knob is set.
    nonisolated static func buildBenchmarkFlags(
        benchmarkType: BenchmarkType, runtime: Runtime, model: Model, knobs: CellKnobs
    ) throws(BenchmarkFlagsError) -> BenchmarkFlags? {
        guard knobs.readinessMaxWaitSecs != nil || knobs.readinessSkipThermal else { return nil }
        // `nil` rather than `false` when unset, so an untouched knob stays absent.
        let readiness = ReadinessOverrides(maxWaitSecs: knobs.readinessMaxWaitSecs,
                                           skipThermal: knobs.readinessSkipThermal ? true : nil)
        // `resolve` rejects a cell whose variant has no readiness field.
        return try BenchmarkFlagRef(benchmarkType: benchmarkType,
                                    runtimeType: RuntimeType.of(runtime),
                                    modelType: ModelType.of(model),
                                    readiness: readiness).resolve()
    }

    /// Why a `benchmark-flags=` value was refused before its cell's variant saw it.
    nonisolated enum BenchmarkFlagsJsonError: Error, Equatable {
        case notAnObject
        case axisNotAccepted(key: String)
        case zeroWait
    }

    /// Parse `benchmark-flags=` against its cell, as `RuntimeFlags.fromCellJson` parses
    /// `runtime-flags=`. `nil` for `{}`.
    nonisolated static func parseBenchmarkFlagsJson(
        _ json: Data, benchmarkType: BenchmarkType, runtime: Runtime, model: Model
    ) throws -> BenchmarkFlags? {
        guard var fields = try JSONSerialization.jsonObject(
            with: json, options: [.fragmentsAllowed]) as? [String: Any]
        else { throw BenchmarkFlagsJsonError.notAnObject }
        if fields.isEmpty { return nil }
        let derived: [(BenchmarkFlagRef.CodingKeys, String)] = [
            (.runtimeType, RuntimeType.of(runtime).rawValue),
            (.modelType, ModelType.of(model).rawValue),
            (.benchmarkType, benchmarkType.rawValue),
        ]
        for (key, value) in derived {
            guard fields[key.rawValue] == nil else {
                throw BenchmarkFlagsJsonError.axisNotAccepted(key: key.rawValue)
            }
            fields[key.rawValue] = value
        }
        let ref = try JSONDecoder().decode(
            BenchmarkFlagRef.self, from: try JSONSerialization.data(withJSONObject: fields))
        // The shorthand's `atLeast: 1`, so the two forms accept the same waits.
        guard ref.readiness?.maxWaitSecs != 0 else { throw BenchmarkFlagsJsonError.zeroWait }
        return try ref.resolve()
    }

    /// LFM2.5-8B token ids for the `coherence` probe: no tokenizer in the loop, so greedy
    /// output compares directly to the Python `mlx_lm` reference (routing-corrected model).
    private enum ParityProbe {
        /// Arbitrary in-vocab sequence for last-token top-5 parity.
        static let tokens: [Int32] = [124894, 100, 2000, 345, 6789, 42, 999, 17,
                                      2560, 88, 12345, 5, 678, 9012, 34, 567]
        /// Compared as a set.
        static let expectedTop5: [Int] = [96, 207, 81, 34, 342]
        /// A real chat-templated prompt.
        static let realPrompt: [Int32] = [124894, 124899, 5922, 207, 2992, 355, 278, 5205,
                                          302, 3980, 39, 41774, 296, 734, 1858, 22, 124900,
                                          207, 124899, 63514, 207]
        static let realGreedy: [Int] = [124901, 207, 597, 4695, 20589, 34, 496, 2992, 355,
                                        278, 5205, 302, 3980, 39, 41774, 296, 734, 1858,
                                        2426, 8, 2083, 1946, 6119, 415]
    }

    /// A `metrics=` token to the catalog benchmark id at an offset. Add an entry (and the
    /// catalog id) to expose another benchmark.
    private static let metricBuilders: [String: (Int) -> String] = [
        "prefill": { "prefill_throughput_\($0)" },
        "decode": { "decode_throughput_\($0)_100" },
        "maxmem": { "max_memory_usage_\($0)" },
        "e2e": { "end_to_end_latency_\($0)_256" },   // catalog fixes e2e decode at 256
    ]

    /// An explicit `benchmarks=` list wins; otherwise `metrics` x `offsets`, ignoring unknown
    /// metrics. Internal so `DeepLinkRouter` resolves ids the same way.
    static func resolveBenchmarkIds(metrics: [String], offsets: [Int], explicit: [String]) -> [String] {
        if !explicit.isEmpty { return explicit }
        var ids: [String] = []
        for off in offsets {
            for met in metrics {
                guard let build = metricBuilders[met] else { continue }
                ids.append(build(off))
            }
        }
        return ids
    }

    /// ``JobCell/resolvable(benchmarkIds:store:server:http:auth:onSkip:)``, logging each
    /// skip. `nil` is a failed fetch, already logged; the caller prints `BENCH_DONE 1`.
    ///
    /// Only a claimed run asks the server, as only `pipette-cli`'s worker does: it checks a
    /// cached definition and fetches one it lacks. A plain run resolves from the store and
    /// the id.
    private static func resolvableBenchmarkIds(_ ids: [String], claim: String?,
                                               ws: PipetteWorkspace, http: HttpClient,
                                               tag: String) async -> ResolvableBenchmarks? {
        let server = claim == nil
            ? nil : CollectorEndpoint.effectiveServerUrl(registered: ws.registration.get()?.serverUrl)
        // Nil without a registration, so the production fallback is never signed for.
        let auth = ws.registration.signingIdentity()
        do {
            return try await JobCell.resolvable(benchmarkIds: ids, store: ws.benchmarks,
                                                server: server, http: http,
                                                auth: auth) { reference, reason in
                HeadlessRunner.log("skip \(reference): \(reason)")
            }
        } catch {
            HeadlessRunner.log("\(tag) ERROR fetching a benchmark definition failed: "
                + "\(error.localizedDescription)")
            return nil
        }
    }

    /// Print how each completed cell was filed and, when asked to submit, whether it was.
    /// Reads the persisted manifest, so it reports what is on disk, not what was intended.
    private static func writeRecordDone(_ manifest: JobManifest, submitRequested: Bool,
                                      ws: PipetteWorkspace) async {
        @Sendable func filings(_ manifest: JobManifest) async -> [ResultReport] {
            await MainActor.run {
                manifest.cells
                    .filter { $0.runStatus == .completed }
                    .map { ResultReport(cell: $0, store: ws.results) }
            }
        }
        var reports = await filings(manifest)
        // The same terms `shouldAutoSubmit` gates on, so the line names the one that
        // actually stopped the upload.
        let blocker = await MainActor.run {
            ResultReporter.submitBlocker(
                registered: ws.registration.get() != nil,
                online: NetworkReachability.shared.isConnected)
        }
        // Submit here, not only in the executor's pass, so the outcome can be reported (as
        // the crate's `benchmarks run` does). The drain is serialized and idempotent.
        var errors: [String] = []
        if submitRequested, blocker == nil, reports.contains(where: { $0.location == .remotePending }) {
            errors = await ResultUploader.shared.drainJob(jobId: manifest.jobId).errors
            // Re-read: the acks land on disk, so a cell that just synced must not still
            // report as pending.
            if let acked = await MainActor.run(body: { ws.loadJobManifest(jobId: manifest.jobId) }) {
                reports = await filings(acked)
            }
        }
        for line in ResultReporter.lines(reports: reports, submitRequested: submitRequested,
                                         blocker: blocker, errors: errors) {
            HeadlessRunner.log(line)
        }
    }
    // MARK: - Bench (the bare form)

    static func runSweep(runtime: RuntimeType, batch: Int,
                         runtimeFlags: RuntimeFlags? = nil,
                         benchmarkFlags: BenchmarkFlags? = nil,
                            benchmarkIds: [String],
                            runCoherence: Bool, runCalibrate: Bool, runPromptSeed: Bool,
                            offsets: [Int], resolved: DiscoveredModel? = nil,
                            modelMatch: String?, submit: Bool, claim: String? = nil,
                            claimedAt: ContinuousClock.Instant? = nil,
                            claimedBenchmarks: [BenchmarkDefinition] = [],
                            ws: PipetteWorkspace) async -> Bool {
        let engineTag = runtime.engineLabel
        let models = await MainActor.run { ws.availableModels() }
        // The entry, not a name to re-match: a pinned revision and an unpinned copy of one
        // repo share a filename, so re-matching can run the wrong weights.
        let model = resolved ?? models.first { m in
            return isCompatible(m.source, runtime)
                && (modelMatch == nil || m.name.localizedCaseInsensitiveContains(modelMatch!))
        }
        guard let model else {
            HeadlessRunner.log("ERROR no \(engineTag) model found; have: "
                + models.map { "\($0.name)[\($0.engineLabel)]" }.joined(separator: ", "))
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }
        // A handed-over `resolved` entry has not been through the search's check.
        guard isCompatible(model.source, runtime) else {
            // The format, not the engine label: two model types share one engine.
            HeadlessRunner.log(
                "ERROR \(model.name) is a \(ModelType.of(model.source).rawValue) model, which "
                    + "\(engineTag) cannot load for this cell")
            HeadlessRunner.benchDone(HeadlessExit.failure); return false
        }
        let ctx = UInt32((offsets.max() ?? 4096) + 300)
        HeadlessRunner.log("start runtime=\(engineTag) model=\(model.name) batch=\(batch) "
            + "benchmarks=\(benchmarkIds) submit=\(submit)")

        // `metrics=promptseed`: the runtime builds a prompt per target under this model's
        // tokenizer and reports the token count it got.
        if runPromptSeed {
            let targets = [100, 256, 512, 1024, 2048]
            HeadlessRunner.log("promptseed runtime=\(engineTag) model=\(model.name) targets=\(targets)")
            var allOK = false
            do {
                let results = runtime == .mlxIosPipette
                    ? try await MlxRuntime.promptSeedCounts(modelPath: model.path, targets: targets)
                    : try await LlamaExecute.promptSeedCounts(
                        modelPath: model.path, nGpuLayers: 99,
                        contextSize: ctx, nUbatch: UInt32(batch), targets: targets)
                allOK = true
                for r in results {
                    let ok = r.got == r.target
                    allOK = allOK && ok
                    HeadlessRunner.log("promptseed \(engineTag) target=\(r.target) got=\(r.got) \(ok ? "OK" : "MISMATCH")")
                }
                HeadlessRunner.log("promptseed \(allOK ? "ALL OK" : "HAD MISMATCHES")")
            } catch { HeadlessRunner.log("promptseed ERROR \(error)") }
            HeadlessRunner.benchDone(ok: allOK); return allOK
        }

        // `metrics=calibrate`: fit `IMUThermometer` against soc_temp, so the readiness gate
        // can use the IMU on builds without the private sensor. Needs the
        // PIPETTE_PRIVATE_THERMAL build (ground truth) and a stationary device.
        if runCalibrate {
            guard runtime == .mlxIosPipette else { HeadlessRunner.log("calibrate requires runtime=mlx"); HeadlessRunner.benchDone(HeadlessExit.failure); return false }
            if dieTempMaxC() == nil {
                HeadlessRunner.log("calibrate ERROR: soc_temp unavailable; build with PIPETTE_PRIVATE_THERMAL=1")
                HeadlessRunner.benchDone(HeadlessExit.failure); return false
            }
            let mlxModel: any LanguageModel
            do { mlxModel = try await MlxRuntime.loadModel(path: model.path) } catch {
                HeadlessRunner.log("ERROR load \(error)"); HeadlessRunner.benchDone(HeadlessExit.failure); return false
            }
            defer { MlxRuntime.releaseModel() }
            // Two heat/cool cycles at different peaks, so temperature is decoupled from time.
            var samples = [(imu: [Double], temp: Double)]()
            for bursts in [8, 14] {
                for _ in 0 ..< bursts {
                    do { _ = try MlxRuntime.prefillBurst(mlxModel, tokens: 4096, prefillChunk: batch) } catch {
                        HeadlessRunner.log("calibrate ERROR burst \(error)"); HeadlessRunner.benchDone(HeadlessExit.failure); return false
                    }
                }
                for _ in 0 ..< 30 {
                    let imu = IMUThermometer.averagedIMU()
                    let t = dieTempMaxC()
                    if let t { samples.append((imu, t)) }
                    HeadlessRunner.log(String(format: "calibrate sample n=%d soc_temp=%.2f", samples.count, t ?? -1))
                    if let t, t <= maxImuC { break }
                    try? await Task.sleep(for: .seconds(4))
                }
            }
            guard let rmse = IMUThermometer.calibrate(samples) else {
                HeadlessRunner.log("calibrate FAILED (n=\(samples.count), need >=7)"); HeadlessRunner.benchDone(HeadlessExit.failure); return false
            }
            HeadlessRunner.log(String(format: "calibrate DONE n=%d trainRMSE=%.2fC, persisted per-device model",
                       samples.count, rmse))

            // The IMU chip lags the SoC die under load. lag>0 means the IMU reads cooler
            // than the die: the dangerous case, where the gate could proceed while it is hot.
            HeadlessRunner.log("calibrate: LAG TEST, heating ramp (soc_temp vs imuEst)")
            for step in 0 ..< 12 {
                do { _ = try MlxRuntime.prefillBurst(mlxModel, tokens: 4096, prefillChunk: batch) } catch {
                    HeadlessRunner.log("calibrate ERROR burst \(error)"); HeadlessRunner.benchDone(HeadlessExit.failure); return false
                }
                let soc = dieTempMaxC() ?? -1
                let est = IMUThermometer.estimate() ?? -1
                HeadlessRunner.log(String(format: "lag heat step=%d soc_temp=%.2f imuEst=%.2f lag=%+.2f", step, soc, est, soc - est))
            }
            HeadlessRunner.log("calibrate: LAG TEST, settle (idle, does imuEst catch up?)")
            for step in 0 ..< 16 {
                let soc = dieTempMaxC() ?? -1
                let est = IMUThermometer.estimate() ?? -1
                HeadlessRunner.log(String(format: "lag settle step=%d soc_temp=%.2f imuEst=%.2f lag=%+.2f", step, soc, est, soc - est))
                try? await Task.sleep(for: .seconds(4))
            }
            HeadlessRunner.benchDone(HeadlessExit.ok); return true
        }

        if runCoherence {
            do {
                if runtime == .mlxIosPipette {
                    let probe = try await MlxRuntime.coherence(modelPath: model.path,
                        promptIds: ParityProbe.tokens, gen: 0, prefillChunk: batch)
                    let cont = try await MlxRuntime.coherence(modelPath: model.path,
                        promptIds: ParityProbe.realPrompt, gen: 24, prefillChunk: batch)
                    var pfx = 0
                    for (a, b) in zip(cont.ids, ParityProbe.realGreedy) { if a == b { pfx += 1 } else { break } }
                    HeadlessRunner.log("coherence mlx probe_top5=\(probe.top5) ref=\(ParityProbe.expectedTop5) "
                        + "setmatch=\(Set(probe.top5) == Set(ParityProbe.expectedTop5))")
                    HeadlessRunner.log("coherence mlx greedy=\(cont.ids)")
                    HeadlessRunner.log("coherence mlx ref_greedy=\(ParityProbe.realGreedy) prefix_match=\(pfx)/\(ParityProbe.realGreedy.count)")
                } else {
                    // The same llama.cpp ops the benchmark path uses.
                    let m = try LlamaCpp.load(path: model.path, nGpuLayers: 99,
                                              contextSize: ctx, nUbatch: UInt32(batch),
                                              threads: LlamaCpp.defaultThreads,
                                              swaFull: LlamaCpp.defaultSwaFull,
                                              loadMode: LlamaCpp.defaultLoadMode)
                    defer { m.free() }
                    try LlamaCpp.prefill(m, ParityProbe.realPrompt)
                    let text = try LlamaCpp.decodeGreedyText(m, maxTokens: 24)
                    let ids = (try? LlamaCpp.tokenize(m, text, addSpecial: false)) ?? []
                    HeadlessRunner.log("coherence llama text=\(text.replacingOccurrences(of: "\n", with: " "))")
                    HeadlessRunner.log("coherence llama greedy_ids=\(ids)")
                }
            } catch { HeadlessRunner.log("ERROR coherence \(error)") }
        }

        // A claimed cell runs `claimedBenchmarks`, the definitions the claim's server check
        // returned. The `runtime-flags=` entry rides on each cell whole, so every
        // knob it names reaches the run, not only the three the job's settings can spell.
        let cells = JobCell.pending(benchmarkIds: benchmarkIds, for: model,
                                    store: ws.benchmarks,
                                    runtimeFlags: runtimeFlags.map(RuntimeFlagRef.init),
                                    benchmarkFlags: benchmarkFlags.map(BenchmarkFlagRef.init)) {
            HeadlessRunner.log("skip \($0): unrecognized benchmark id")
        }
        // No cells is success only for a coherence-only run.
        guard !cells.isEmpty else { HeadlessRunner.benchDone(ok: runCoherence); return runCoherence }

        // The New Job screen's `JobLauncher` path, so a CLI job is identical to a UI one.
        let final: JobManifest? = await withCheckedContinuation { cont in
            Task { @MainActor in
                let launched = JobLauncher.launch(
                    cells: cells,
                    nGpuLayers: Int(runtimeFlags?.numberGpuLayers ?? HeadlessCommand.defaultGpuLayers),
                    contextSize: Int(runtimeFlags?.ctxSize ?? ctx), prefillBatch: batch,
                    threads: runtimeFlags?.threads.map(Int.init),
                    contributeResults: submit, claimJobId: claim, claimedAt: claimedAt,
                    claimedBenchmarks: claimedBenchmarks,
                    jobRunner: JobRunner(), jobStore: JobStore(ws: ws), ws: ws,
                    onFinish: { cont.resume(returning: $0) })
                if launched == nil { cont.resume(returning: nil) }
            }
        }
        guard let final else {
            // Unreachable with this process's fresh JobRunner; kept rather than force-unwrapped.
            HeadlessRunner.log("ERROR job launch blocked: another job is running")
            HeadlessRunner.benchDone(HeadlessExit.failure)
            return false
        }
        await writeRecordDone(final, submitRequested: submit, ws: ws)
        return finishSweep(final, claimed: claim != nil)
    }
}
