import ArgumentParser
import Foundation

/// Why an invocation is not a command. Reported with exit code 2 (a failed run is 1), and
/// every case names the offending token.
nonisolated enum HeadlessUsageError: Error, Equatable {
    /// No `headlessrun` marker: "nothing to do" rather than an error.
    case notHeadless
    case unknownVerb(String)
    /// Sorted so the message is deterministic.
    case unknownParameters([String])
    case missingParameter(String)
    case invalidValue(key: String, value: String)
    /// A runtime that exists in plan-types but cannot run in this process.
    case hostOnlyRuntime(String)
    /// An argument that parsed but cannot be honoured. Carries the reason because the
    /// underlying decoders raise their own error types.
    case rejected(key: String, reason: String)

    var message: String {
        switch self {
        case .notHeadless:
            return "no `headlessrun` marker in the launch arguments"
        case let .unknownVerb(verb):
            return "unknown command `\(verb)`"
        case let .unknownParameters(keys):
            let names = keys.map { "`\($0)=`" }.joined(separator: ", ")
            return "unknown parameter\(keys.count == 1 ? "" : "s") \(names)"
        case let .missingParameter(key):
            // Covers present-but-empty too: "missing" alone reads wrong next to `org=`.
            return "missing or empty required parameter `\(key)=`"
        case let .invalidValue(key, value):
            return "invalid value for `\(key)=`: `\(value)`"
        case let .hostOnlyRuntime(token):
            return "runtime `\(token)` is host-only and cannot run on this device"
        case let .rejected(key, reason):
            return "`\(key)=` rejected: \(reason)"
        }
    }
}

/// One parsed headless CLI invocation. Parsing is side-effect free so the surface is
/// unit-testable off-device; validation that needs device state stays in the handlers.
///
/// After the `headlessrun` marker, a token is a bare word (a verb, or the positional on
/// the leaves that take one) or a `key=value` parameter. No verb at all selects the bench
/// form. A `key=` the selected verb does not accept is refused, never ignored.
///
/// `submit` defaults to on for catalog-resolved runs and to off for a run that names an
/// exact model, which nothing in the catalog sanctioned.
nonisolated enum HeadlessCommand: Equatable, Sendable {

    /// Every layer on the GPU. Applied where the value is consumed, never at parse time:
    /// an absent `number_gpu_layers` must stay absent, as the crate's builder emits `-ngl`
    /// only for a `Some`.
    static let defaultGpuLayers: UInt32 = 99

    enum ProbeKind: String, Equatable, Sendable, CaseIterable {
        case token, generation, cap, enforcement
    }

    /// Mirrors the job page's two actions: retry failed cells or resume cancelled
    /// (paused) ones. `job run` without `scope=` resumes.
    enum RunScope: String, Equatable, Sendable {
        case failed
        case cancelled

        var resetTarget: CellRunStatus {
            switch self {
            case .failed: return .failed
            case .cancelled: return .cancelled
            }
        }
    }

    case authRegister(AuthCommands.RegisterArgs)
    case diagMemseq(models: [String], batch: UInt32, prefill: UInt32)
    case modelsList(format: ModelsCommands.ListFormat)
    case modelsRm(name: String?, repo: String?)
    /// A phone installs no runtimes, so `list` is the only leaf.
    case runtimesList(format: RuntimesCommands.ListFormat)
    case benchmarksList(benchmarkType: BenchmarkType?)
    case benchmarksShow(SourcedBenchmarkId)
    case benchmarksInitLocal
    case version
    case storageStatus
    case storageGc(dryRun: Bool)
    /// Pull the catalog, then submit what is pending. `jobId` narrows the submission; the
    /// CLI narrows by result id, which iOS cannot address yet. The pull is a public read,
    /// so an unregistered device can still pass `serverUrl`.
    case sync(jobId: String?, serverUrl: String?)
    case resultsList(benchmark: SourcedBenchmarkId?, benchmarkType: BenchmarkType?,
                     state: BenchmarkResultState?, limit: Int?)
    case resultsShow(resultId: ResultId)
    case resultsDelete(resultId: ResultId)
    case authMe(local: Bool)
    /// The two labels are local; `clientDetails` is server-side, as the CLI's is.
    case authSetDevice(deviceName: String?, deviceFormFactor: DeviceFormFactor?,
                       clientDetails: String?)
    case authReset(force: Bool)
    /// iOS-only: the CLI's signing key is a file, so a backup there is a file copy.
    case authExport
    /// `force` overrides the refusal over an existing signing key.
    case authImport(bundle: String, force: Bool)
    case diagProbe(kind: ProbeKind)
    case listJobs
    case removeJob(id: String)
    case runJob(id: String, scope: RunScope)
    case exportJob(id: String)
    case submitJob(id: String)
    case status
    /// `worker`, with the crate's knobs, or `settings run`, with their defaults.
    case worker(WorkerArgs)

    /// The forms `--model` takes. A `model://sha256=<prefix>` addresses the local store,
    /// which the parser cannot read, so it is carried to the runner unresolved.
    enum ModelSelector: Equatable, Sendable {
        /// A JSON `Model` or a compact `<scheme>://key=value` URI.
        case model(Model)
        /// Validated for shape but not yet matched.
        case digest(String)

        var namedModel: Model? {
            guard case let .model(model) = self else { return nil }
            return model
        }
    }

    case modelsPull(ModelSelector)
    case modelsDelete(ModelSelector)

    /// One cell, every axis named, as `pipette benchmarks run` takes it. The flags are
    /// already resolved against the cell.
    ///
    /// `llamaPin`, here and on the sweeps, is the carried llama.cpp build a `runtime=`
    /// descriptor named; `nil` runs the process's build.
    case benchmarksRun(benchmarkRef: SourcedBenchmarkId, model: ModelSelector,
                       runtime: RuntimeType, batch: Int, runtimeFlags: RuntimeFlags?,
                       sync: Bool, benchmarkFlags: BenchmarkFlags?, llamaPin: LlamaPin? = nil)
    /// This client's sweep: an exact model (`model=`, or its old name `spec=`) or a catalog
    /// selection (`match=` / `quant=`), downloaded first if it is not on device.
    ///
    /// `claim` is the server job a `pipette-ios-proxy` claimed for this run; every upload
    /// echoes it as `job_id`, which closes that job.
    case bench(spec: ModelSelector?, model: String?, quant: String?, runtime: RuntimeType,
               batch: Int, runtimeFlags: RuntimeFlags?, benchmarks: [String],
               metrics: [String], offsets: [Int], submit: Bool,
               benchmarkFlags: BenchmarkFlags?, claim: String? = nil, llamaPin: LlamaPin? = nil)
    /// No verb: `headlessrun` alone runs the default MLX prefill/decode/maxmem sweep over
    /// whatever matching model is already on device, with no download.
    ///
    /// No runtime or benchmark flags: those resolve against an exact model, which this form
    /// does not name, so the parser refuses them.
    case bareBench(runtime: RuntimeType, batch: Int, metrics: [String], offsets: [Int],
                   benchmarks: [String], model: String?, submit: Bool, llamaPin: LlamaPin? = nil)
    /// `runtime=afm`. A probe token in `metrics` runs a diagnostic that neither records nor
    /// submits; otherwise the benchmarks run through the normal job path.
    case afm(metrics: [String], benchmarks: [String], offsets: [Int], submit: Bool,
             claim: String? = nil, benchmarkFlags: BenchmarkFlags? = nil)

    static let afmProbeTokens: Set<String> = [
        "enforceprobe", "capprobe", "genprobe", "tokprobe",
    ]

    /// Whether this command will try to contribute results; the runner's registration
    /// check reads it.
    var submitsResults: Bool {
        switch self {
        case .bench(_, _, _, _, _, _, _, _, _, let submit, _, _, _): return submit
        case .bareBench(_, _, _, _, _, _, let submit, _): return submit
        // Not gated though it submits: the catalog pull is a public read, and
        // `SyncCommands` reports the skipped submit on an unregistered device, as the CLI does.
        case .sync: return false
        case .resultsList, .resultsShow, .resultsDelete: return false
        case .authMe, .authSetDevice, .authReset, .diagProbe: return false
        case .afm(let metrics, _, _, let submit, _, _):
            return submit && !metrics.contains(where: Self.afmProbeTokens.contains)
        default: return false
        }
    }

    /// `pipette://run[/<verb>[/<sub>]]?k=v&k=v`, reduced to the tokens `parse(_:)` takes so
    /// links and argv share one grammar. A non-`run` host is `.notHeadless`.
    ///
    /// Callers must still gate the result on `allowedViaDeepLink`: any app can open a URL.
    static func parse(url: URL) -> Result<HeadlessCommand, HeadlessUsageError> {
        guard url.scheme == "pipette", url.host == "run",
              let comps = URLComponents(url: url, resolvingAgainstBaseURL: false)
        else { return .failure(.notHeadless) }
        let verbs = comps.path.split(separator: "/").map(String.init)
        // `queryItems` percent-decodes, and the argv parser splits on the first `=`, so a
        // JSON or URI value survives intact.
        let params = (comps.queryItems ?? []).map { "\($0.name)=\($0.value ?? "")" }
        return parse(["headlessrun"] + verbs + params)
    }

    /// Deliberately narrow: any app, web page or QR code can open a URL, so anything with
    /// a destructive or identity side effect stays CLI-only.
    var allowedViaDeepLink: Bool {
        switch self {
        case .bench, .bareBench, .afm, .runJob, .submitJob, .exportJob, .status:
            return true
        case .benchmarksRun, .modelsPull, .modelsDelete,
             .authRegister, .diagMemseq, .modelsList, .modelsRm, .listJobs, .removeJob,
             .worker, .runtimesList, .benchmarksList, .benchmarksShow,
             .benchmarksInitLocal,
             .storageStatus, .storageGc, .sync, .version,
             .authMe, .authSetDevice, .authReset, .authExport, .authImport, .diagProbe,
             .resultsList, .resultsShow, .resultsDelete:
            // Read-only verbs stay off too: nothing needs a device's inventory from a URL.
            // A URL must never print a signing key (`auth export`) or replace one.
            return false
        }
    }

    /// `args` is the full argument vector; everything up to and including the
    /// `headlessrun` marker is ignored.
    static func parse(_ args: [String]) -> Result<HeadlessCommand, HeadlessUsageError> {
        guard let marker = args.firstIndex(of: "headlessrun") else { return .failure(.notHeadless) }
        // A global switch (`HeadlessRunner.isVerbose`), not any verb's parameter.
        let raw = Array(args[(marker + 1)...]).filter { $0 != "-v" && $0 != "--verbose" }
        let (verbs, options) = HeadlessTokens.split(raw)

        let leaf: any HeadlessLeaf
        do {
            // Not the root's default subcommand, so a failure to route a verb stays a
            // refusal instead of falling through to a benchmark run.
            if verbs.isEmpty {
                leaf = try HeadlessTree.Bare.parse(options)
            } else {
                guard let routed = try HeadlessTree.Cli.parseAsRoot(verbs + options)
                    as? any HeadlessLeaf
                else {
                    // A group named without a leaf (`auth`, `job`, `storage`, `diag`).
                    return .failure(.unknownVerb(verbs.joined(separator: " ")))
                }
                leaf = routed
            }
        } catch {
            return .failure(.unknownVerb(verbs.joined(separator: " ")))
        }
        let leafType = type(of: leaf)
        var strays = leaf.unrecognized
        var params = leaf.bag
        // `split` puts every bare word before every option and `.allUnrecognized` keeps
        // input order, so the positional is the first stray when that stray is bare.
        var positional: String?
        if leafType.positionalKey != nil, let first = strays.first, !first.hasPrefix("-") {
            positional = strays.removeFirst()
        }
        // A dashed stray is an unknown parameter; a bare one is an unknown verb.
        if let stray = strays.first(where: { !$0.hasPrefix("-") }) {
            let verbPath = leafType.verbPath
            // A leaf the caller named is reported by its canonical path; one a default
            // subcommand supplied is not, so `models frobnicate` is not reported as
            // `models list frobnicate`.
            let named = verbPath.last.map(verbs.contains) ?? false
            let path = named ? verbPath + [stray]
                             : (verbs.last == stray ? verbs : verbs + [stray])
            return .failure(.unknownVerb(path.joined(separator: " ")))
        }
        if !strays.isEmpty {
            return .failure(.unknownParameters(strays.map(headlessStrayKey).sorted()))
        }
        if let key = leafType.positionalKey, let positional {
            guard params[key] == nil else {
                return .failure(.rejected(
                    key: key, reason: "also given as a positional argument; pass one"))
            }
            params[key] = positional
        }
        do {
            return .success(try build(verbs: leafType.verbPath, params: params))
        } catch {
            return .failure(error)
        }
    }

    /// The carried build a llama.cpp descriptor names. Matches the whole descriptor, as
    /// `runtimeArgument` does for the other engines.
    static func carriedLlamaPin(_ declared: Runtime, carried: [LlamaPin] = LlamaPins.carried)
        throws(HeadlessUsageError) -> LlamaPin {
        let want = (try? SubmissionRef.runtime(declared)) ?? RuntimeType.of(declared).rawValue
        let matches = carried.filter { Runtime.carried($0) == declared }
        guard let pin = matches.first else {
            let have = carried
                .map { (try? SubmissionRef.runtime(Runtime.carried($0))) ?? $0.name }
                .joined(separator: ", ")
            throw .rejected(key: "runtime", reason: "names \(want), but this build is \(have)")
        }
        // Defensive: ios/release-manifest.py fails an archive that carries two builds
        // with one identity. A bundle built outside CI can still carry two, and a
        // descriptor cannot say which of them to run.
        guard matches.count == 1 else {
            let names = matches.map(\.name).joined(separator: " and ")
            throw .rejected(key: "runtime", reason: "names \(want), which \(names) both are")
        }
        return pin
    }

    /// Typed-throwing so the refusals a caller must handle are the enum and nothing else.
    private static func build(
        verbs unnormalized: [String],
        params: [String: String]
    ) throws(HeadlessUsageError) -> HeadlessCommand {
        let verbs = unnormalized
        /// A blank value is as unusable as an absent one: `id=` would address a job named "".
        func require(_ key: String) throws(HeadlessUsageError) -> String {
            guard let value = params[key],
                  !value.trimmingCharacters(in: .whitespaces).isEmpty
            else { throw .missingParameter(key) }
            return value
        }
        func list(_ key: String) -> [String] {
            (params[key] ?? "").split(separator: ",").map(String.init).filter { !$0.isEmpty }
        }
        /// Unlike the CLI's opt-in `--sync`, submission is the default here: a phone that
        /// measures for an hour and publishes nothing is the failure worth designing
        /// against. An exact model coordinate stays opt-in, since nothing sanctioned it;
        /// orchestrated runs (`pipette-ctl`, `pipette-ios-proxy`) pass `submit=1` explicitly.
        func submits(coordinate: ModelSelector?) -> Bool {
            (params["submit"] ?? (coordinate == nil ? "1" : "0")) != "0"
        }
        /// A claim is one cell a server sent, so a run that picks its own model refuses one.
        /// Blank or padded is refused: the upload echoes the id verbatim as `job_id`.
        func claim(picksItsModel: Bool) throws(HeadlessUsageError) -> String? {
            guard let value = params["claim"] else { return nil }
            if picksItsModel {
                throw .rejected(key: "claim", reason: "a claimed run names its model with `model=`")
            }
            guard !value.isEmpty,
                  value == value.trimmingCharacters(in: .whitespacesAndNewlines) else {
                throw .rejected(key: "claim", reason: "expected a server job id")
            }
            return value
        }
        /// llama's `n_ubatch`, MLX's chunk size. Not written as `??`: that operator is
        /// `rethrows`, which widens the thrown type to `any Error`.
        func prefillChunk(_ flags: RuntimeFlags?) throws(HeadlessUsageError) -> Int {
            if let chunk = flags?.nUbatch { return Int(chunk) }
            return try int("batch", default: 512)
        }
        /// `<jobId>/<cellId>`: iOS results live under their job, where the CLI's are flat.
        func resultId() throws(HeadlessUsageError) -> ResultId {
            let raw = try require("result")
            guard let id = ResultId.parse(raw) else {
                throw .rejected(key: "result", reason: "expected `<jobId>/<cellId>`")
            }
            return id
        }
        func runtimeCoordinate() throws(HeadlessUsageError)
            -> (type: RuntimeType, llamaPin: LlamaPin?) {
            try runtimeArgument(try require("runtime"))
        }

        /// A JSON `Runtime` or this client's short token; absent means MLX. A llama.cpp
        /// descriptor also names the carried build to run, which the runner loads.
        ///
        /// A `<scheme>://` URI is refused by name: plan-types gives the on-device runtimes
        /// no URI form, and every scheme it does define names a desktop runtime.
        func runtimeArgument(_ argument: String?) throws(HeadlessUsageError)
            -> (type: RuntimeType, llamaPin: LlamaPin?) {
            // Trimmed, as the crate trims before `parse_runtime_arg`.
            let raw = argument?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let raw, !raw.isEmpty else { return (.mlxIosPipette, nil) }
            if raw.hasPrefix("{") {
                guard let declared = try? JSONDecoder().decode(Runtime.self, from: Data(raw.utf8))
                else { throw .invalidValue(key: "runtime", value: raw) }
                let type = RuntimeType.of(declared)
                if case .llamacppIosPipette = declared {
                    return (type, try Self.carriedLlamaPin(declared))
                }
                guard let builtIn = Runtime.thisBuild(for: type) else {
                    throw .rejected(key: "runtime",
                                    reason: "`\(type.rawValue)` is not runnable on iOS")
                }
                // Compared, not just typed: a runtime names a build and this binary is one,
                // so running a cell that asked for another would record a build nobody asked
                // for. `runtimes` prints the identity that passes.
                guard declared == builtIn else {
                    let want = (try? SubmissionRef.runtime(declared)) ?? type.rawValue
                    let have = (try? SubmissionRef.runtime(builtIn)) ?? type.rawValue
                    throw .rejected(key: "runtime",
                                    reason: "names \(want), but this build is \(have)")
                }
                return (type, nil)
            }
            if raw.contains("://") {
                throw .rejected(key: "runtime",
                                reason: "a runtime URI names a runtime this build cannot be: "
                                    + "the on-device runtimes have no URI form; pass a JSON "
                                    + "`Runtime` object or the short token")
            }
            return (try RuntimeType.parseHeadless(raw), nil)
        }

        /// The exact model `model=` names, or `nil` when the caller selected from the
        /// catalog. The CLI's `--model` spellings (see ModelUri.swift); substring selection
        /// is `match=`.
        func modelCoordinate() throws(HeadlessUsageError) -> ModelSelector? {
            if params["model"] != nil, params["spec"] != nil {
                throw .rejected(key: "model",
                                reason: "`spec=` is the former name for it; pass one")
            }
            guard let raw = params["model"] ?? params["spec"] else { return nil }
            if params["match"] != nil {
                throw .rejected(key: "model",
                                reason: "`match=` selects from the catalog; pass one or the other")
            }
            // The same digest the warehouse keeps as `model_descriptor_sha256`.
            let digest: String?
            do {
                digest = try digestPrefix(raw, scheme: modelScheme)
            } catch {
                throw .rejected(key: "model", reason: error.localizedDescription)
            }
            if let digest { return .digest(digest) }
            do {
                return .model(try parseModelArg(raw))
            } catch ModelUriError.missingSchemeSeparator {
                // The likeliest cause is a bare name, which is what `match=` is for.
                throw .rejected(key: "model",
                                reason: "expected a JSON object or a `<scheme>://key=value` "
                                    + "URI; use `match=` to select from the catalog by name")
            } catch let failure as ModelUriError {
                throw .rejected(key: "model", reason: failure.localizedDescription)
            } catch {
                throw .invalidValue(key: "model", value: raw)
            }
        }
        /// `benchmark=` is the CLI's spelling; `benchmarks=` is kept because the plan runner
        /// emits it.
        func benchmarkIds() throws(HeadlessUsageError) -> [String] {
            if params["benchmark"] != nil, params["benchmarks"] != nil {
                throw .rejected(key: "benchmark",
                                reason: "`benchmarks=` is the plural alias; pass one")
            }
            return params["benchmark"] != nil ? list("benchmark") : list("benchmarks")
        }
        /// `batch=` folded into the authored knobs as `n_ubatch`. Folded into the JSON
        /// because `RuntimeFlags` is a sum type with no spare field. A non-object is handed
        /// back untouched, for `fromCellJson` to name as the wrong shape.
        func foldBatchIntoKnobs(_ json: String) throws(HeadlessUsageError) -> Data {
            let authored = Data(json.utf8)
            let key = RuntimeFlagRef.CodingKeys.nUbatch.rawValue
            guard var fields = (try? JSONSerialization.jsonObject(with: authored))
                as? [String: Any]
            else { return authored }
            // Both set the prefill chunk; picking one silently measures a size not asked for.
            if fields[key] != nil {
                guard params["batch"] == nil else {
                    throw .rejected(key: "runtime-flags",
                                    reason: "`batch=` also sets the prefill chunk; "
                                        + "pass one or the other")
                }
                return authored
            }
            let batch = try int("batch", default: 512)
            guard let chunk = UInt32(exactly: batch) else {
                throw .invalidValue(key: "batch", value: String(batch))
            }
            fields[key] = chunk
            guard let folded = try? JSONSerialization.data(withJSONObject: fields)
            else { throw .invalidValue(key: "runtime-flags", value: json) }
            return folded
        }
        /// `nil` when absent or naming no knob. Resolved through `RuntimeFlags.fromCellJson`
        /// against the cell the arguments name, so a console run is validated exactly as an
        /// orchestrated cell is.
        func resolvedRuntimeFlags(
            _ runtime: RuntimeType
        ) throws(HeadlessUsageError) -> RuntimeFlags? {
            guard let json = params["runtime-flags"] else { return nil }
            func reject(_ reason: String) -> HeadlessUsageError {
                .rejected(key: "runtime-flags", reason: reason)
            }
            let cell = try exactCell(runtime, refusingUnder: "runtime-flags")

            let payload = try foldBatchIntoKnobs(json)
            let resolved: RuntimeFlags
            do {
                guard let flags = try RuntimeFlags.fromCellJson(
                    payload, runtimeType: runtime, modelType: ModelType.of(cell.model),
                    benchmarkType: cell.benchmarkType)
                else { return nil }
                resolved = flags
            } catch let error as RuntimeFlagsError {
                switch error {
                // Named, as the CLI does, so an invocation written for the old wire says
                // what to change.
                case .notAnObject:
                    throw reject("must be a JSON object of knobs, e.g. {\"threads\":4} "
                        + "(the one-element array is no longer accepted)")
                case let .axisNotAccepted(key):
                    throw reject("must not carry `\(key)`: the cell comes from `runtime=`, "
                        + "`model=` and `benchmarks=`, which are parsed first")
                case let .knobNotAllowed(knob, _, _, _):
                    throw reject("this cell does not carry `\(knob)`")
                // The error renders its own triple, as the crate error does.
                case .noSuchCombination, .speculationWithoutDrafter:
                    throw reject(error.localizedDescription)
                }
            } catch let field as UnknownFlagField {
                throw reject("no iOS variant declares `\(field.name)`")
            } catch {
                throw .invalidValue(key: "runtime-flags", value: json)
            }
            return resolved
        }
        /// The benchmark and model a per-cell knob resolves against; anything short of one
        /// each is refused under `key`. A `match=` resolves only at run time, so its type is
        /// not known here. An unparseable id is refused under the key the caller used.
        func exactCell(_ runtime: RuntimeType, refusingUnder key: String)
            throws(HeadlessUsageError) -> (benchmarkType: BenchmarkType, model: Model) {
            let model: Model
            switch try modelCoordinate() {
            case _ where runtime == .appleFoundation: model = .appleFoundationText
            case let .model(named)?: model = named
            case .digest?:
                throw .rejected(key: key, reason: "a `model://sha256=` digest resolves "
                    + "against the store at run time, so its type is not known here; name "
                    + "the model instead")
            default:
                throw .rejected(key: key, reason: "needs an exact model to resolve against: "
                    + "pass `model=` or `spec=` rather than `match=`")
            }
            let ids = try benchmarkIds()
            guard ids.count == 1 else {
                throw .rejected(key: key, reason: "needs exactly one `benchmark=` id to "
                    + "resolve against, because cells of different types carry different "
                    + "knobs")
            }
            let idKey = params["benchmark"] != nil ? "benchmark" : "benchmarks"
            guard let reference = try? SourcedBenchmarkId(parsing: ids[0]),
                  let benchmarkType = BenchmarkType.fromId(reference.id.value)
            else { throw .invalidValue(key: idKey, value: ids[0]) }
            return (benchmarkType, model)
        }
        /// `exactCell` plus this build's runtime, which the benchmark flags resolve against.
        func builtCell(_ runtime: RuntimeType, refusingUnder key: String)
            throws(HeadlessUsageError)
            -> (benchmarkType: BenchmarkType, runtime: Runtime, model: Model) {
            let cell = try exactCell(runtime, refusingUnder: key)
            guard let built = Runtime.thisBuild(for: runtime) else {
                throw .rejected(key: "runtime", reason: "`\(runtime.rawValue)` is not runnable on iOS")
            }
            return (cell.benchmarkType, built, cell.model)
        }
        /// `nil` when no readiness knob is set. The readiness keys are shorthand for
        /// `benchmark-flags=`, so the two are refused together, as the CLI refuses them.
        func resolvedBenchmarkFlags(
            _ runtime: RuntimeType
        ) throws(HeadlessUsageError) -> BenchmarkFlags? {
            if let json = params["benchmark-flags"] {
                if let shorthand = ["readiness-max-wait-secs", "readiness-skip-thermal"]
                    .first(where: { params[$0] != nil }) {
                    throw .rejected(key: "benchmark-flags",
                                    reason: "is mutually exclusive with `\(shorthand)=`")
                }
                return try parsedBenchmarkFlags(json, runtime)
            }
            let knobs = BenchmarksCommands.CellKnobs(
                readinessMaxWaitSecs: try wholeSeconds("readiness-max-wait-secs", atLeast: 1),
                readinessSkipThermal: params["readiness-skip-thermal"].map(skipThermalFromStr) ?? false)
            let key: String
            if knobs.readinessMaxWaitSecs != nil {
                key = "readiness-max-wait-secs"
            } else if knobs.readinessSkipThermal {
                key = "readiness-skip-thermal"
            } else {
                return nil
            }
            let cell = try builtCell(runtime, refusingUnder: key)
            do {
                return try BenchmarksCommands.buildBenchmarkFlags(
                    benchmarkType: cell.benchmarkType, runtime: cell.runtime, model: cell.model,
                    knobs: knobs)
            } catch {
                throw .rejected(key: key, reason: "not valid for this benchmark: "
                    + error.localizedDescription)
            }
        }
        func parsedBenchmarkFlags(
            _ json: String, _ runtime: RuntimeType
        ) throws(HeadlessUsageError) -> BenchmarkFlags? {
            func reject(_ reason: String) -> HeadlessUsageError {
                .rejected(key: "benchmark-flags", reason: reason)
            }
            let cell = try builtCell(runtime, refusingUnder: "benchmark-flags")
            do {
                return try BenchmarksCommands.parseBenchmarkFlagsJson(
                    Data(json.utf8), benchmarkType: cell.benchmarkType, runtime: cell.runtime,
                    model: cell.model)
            } catch BenchmarksCommands.BenchmarkFlagsJsonError.notAnObject {
                throw reject("must be a JSON object of knobs, e.g. "
                    + "{\"readiness\":{\"max_wait_secs\":900}}")
            } catch let BenchmarksCommands.BenchmarkFlagsJsonError.axisNotAccepted(key) {
                throw reject("must not carry `\(key)`: the cell comes from `runtime=`, "
                    + "`model=` and `benchmarks=`, which are parsed first")
            } catch BenchmarksCommands.BenchmarkFlagsJsonError.zeroWait {
                throw reject("`readiness.max_wait_secs` must be at least 1, as "
                    + "`readiness-max-wait-secs=` requires")
            } catch let field as UnknownFlagField {
                throw reject("no iOS variant declares `\(field.name)`")
            } catch let error as BenchmarkFlagsError {
                throw reject("not valid for this benchmark: " + error.localizedDescription)
            } catch {
                throw .invalidValue(key: "benchmark-flags", value: json)
            }
        }
        /// A bare id or `remote/<id>` names the synced catalog, `local/<id>` the generated half.
        func benchmarkReference(_ raw: String) throws(HeadlessUsageError) -> SourcedBenchmarkId {
            do {
                return try SourcedBenchmarkId(parsing: raw)
            } catch {
                throw .rejected(key: "benchmark", reason: error.localizedDescription)
            }
        }
        /// A non-numeric value is refused, not defaulted: `batch=abc` must not mean 512.
        func int(_ key: String, default def: Int) throws(HeadlessUsageError) -> Int {
            guard let raw = params[key] else { return def }
            guard let value = Int(raw) else { throw .invalidValue(key: key, value: raw) }
            return value
        }
        /// `atLeast` is the crate's `value_parser!(u64).range(n..)`; below it is refused, not
        /// clamped. An integer because `Double` accepts `inf` and `1e400`, which trap in
        /// `Duration.seconds` and `random(in:)`: a crash reachable from a launch argument.
        func wholeSeconds(_ key: String, atLeast: UInt64) throws(HeadlessUsageError)
            -> UInt64? {
            guard let raw = params[key] else { return nil }
            guard let value = UInt64(raw), value >= atLeast else {
                throw .invalidValue(key: key, value: raw)
            }
            return value
        }
        /// `wholeSeconds` for the knobs `WorkerArgs` holds as a `Double`. Never round-trip
        /// the result to `UInt64`: near `UInt64.max` it rounds up past it and traps.
        func seconds(_ key: String, atLeast: UInt64) throws(HeadlessUsageError) -> Double? {
            try wholeSeconds(key, atLeast: atLeast).map { Double($0) }
        }
        /// Every element must parse, so `offsets=256,abc` is refused rather than run as one.
        func ints(_ key: String, default def: [Int]) throws(HeadlessUsageError) -> [Int] {
            guard let raw = params[key] else { return def }
            var parsed: [Int] = []
            for token in raw.split(separator: ",") {
                guard let value = Int(token) else {
                    throw .invalidValue(key: key, value: String(token))
                }
                parsed.append(value)
            }
            return parsed
        }

        switch verbs.first {
        case "auth":
            switch verbs.dropFirst().first {
            case "register":
                // Parsed now, as clap parses the crate's arguments, so a value that can never
                // register is refused before a keypair is minted or a pre-auth key spent.
                let rawServer = params["server-url"] ?? CollectorEndpoint.productionURL
                let serverUrl: ServerUrl
                do {
                    serverUrl = try ServerUrl(rawServer)
                } catch {
                    throw .rejected(key: "server-url", reason: error.localizedDescription)
                }
                let organization = try require("organization")
                let rawEmail = try require("contact-email")
                guard let contactEmail = try? ContactEmail(rawEmail) else {
                    throw .invalidValue(key: "contact-email", value: rawEmail)
                }
                let clientDetails = try require("client-details")
                var preauthKey: ClientPreauthKey?
                if let raw = params["preauth-key"] {
                    guard let key = try? ClientPreauthKey(raw) else {
                        throw .invalidValue(key: "preauth-key", value: raw)
                    }
                    preauthKey = key
                }
                var formFactor: DeviceFormFactor?
                if let raw = params["device-form-factor"] {
                    guard let parsed = DeviceFormFactor(rawValue: raw) else {
                        throw .invalidValue(key: "device-form-factor", value: raw)
                    }
                    formFactor = parsed
                }
                return .authRegister(AuthCommands.RegisterArgs(
                    serverUrl: serverUrl, organization: organization,
                    contactEmail: contactEmail, clientDetails: clientDetails,
                    preauthKey: preauthKey, deviceName: params["device-name"],
                    deviceFormFactor: formFactor))
            case "me":
                return .authMe(local: (params["local"] ?? "0") != "0")
            case "set-device":
                var factor: DeviceFormFactor?
                if let raw = params["device-form-factor"] {
                    guard let parsed = DeviceFormFactor(rawValue: raw) else {
                        throw .invalidValue(key: "device-form-factor", value: raw)
                    }
                    factor = parsed
                }
                let deviceName = params["device-name"]
                let clientDetails = params["client-details"]
                // The crate's `SetDeviceEmpty`: a relabel naming no label is a typo.
                guard deviceName != nil || factor != nil || clientDetails != nil else {
                    throw .rejected(key: "device-name",
                                    reason: "set-device needs at least one of `device-name=`, "
                                        + "`device-form-factor=` or `client-details=`")
                }
                return .authSetDevice(deviceName: deviceName, deviceFormFactor: factor,
                                      clientDetails: clientDetails)
            case "reset":
                return .authReset(force: (params["force"] ?? "0") != "0")
            case "export":
                return .authExport
            case "import":
                return .authImport(bundle: try require("bundle"),
                                   force: (params["force"] ?? "0") != "0")
            case nil:
                throw .unknownVerb("auth")
            case let other?:
                throw .unknownVerb("auth \(other)")
            }

        case "diag":
            switch verbs.dropFirst().first {
            case "memseq":
                // The prefill window: `n_ubatch` for llama, the chunk for MLX.
                guard let batch = UInt32(params["batch"] ?? "512"), batch > 0 else {
                    throw .invalidValue(key: "batch", value: params["batch"] ?? "")
                }
                guard let prefill = UInt32(params["prefill"] ?? "512"), prefill > 0 else {
                    throw .invalidValue(key: "prefill", value: params["prefill"] ?? "")
                }
                return .diagMemseq(models: list("models"), batch: batch, prefill: prefill)
            case "probe":
                let raw = try require("kind")
                guard let kind = ProbeKind(rawValue: raw) else {
                    throw .invalidValue(key: "kind", value: raw)
                }
                return .diagProbe(kind: kind)
            case nil:
                throw .unknownVerb("diag")
            case let other?:
                throw .unknownVerb("diag \(other)")
            }

        case "models":
            switch verbs.dropFirst().first {
            // Bare is this client's spelling; `list` is the crate's.
            case nil, "list":
                guard let raw = params["format"] else { return .modelsList(format: .name) }
                guard let format = ModelsCommands.ListFormat(rawValue: raw) else {
                    throw .invalidValue(key: "format", value: raw)
                }
                return .modelsList(format: format)
            case "pull":
                guard let selector = try modelCoordinate() else {
                    throw .missingParameter("model")
                }
                return .modelsPull(selector)
            case "delete":
                guard let selector = try modelCoordinate() else {
                    throw .missingParameter("model")
                }
                return .modelsDelete(selector)
            case "rm":
                let name = params["name"]
                let repo = params["repo"]
                guard name != nil || repo != nil else { throw .missingParameter("name") }
                return .modelsRm(name: name, repo: repo)
            case let other?:
                throw .unknownVerb("models \(other)")
            }

        case "runtimes":
            switch verbs.dropFirst().first {
            case nil, "list":
                guard let raw = params["format"] else { return .runtimesList(format: .name) }
                guard let format = RuntimesCommands.ListFormat(rawValue: raw) else {
                    throw .invalidValue(key: "format", value: raw)
                }
                return .runtimesList(format: format)
            case let other?:
                throw .unknownVerb("runtimes \(other)")
            }

        case "benchmarks":
            switch verbs.dropFirst().first {
            case nil, "list":
                guard let raw = params["benchmark-type"] else {
                    return .benchmarksList(benchmarkType: nil)
                }
                guard let type = try? BenchmarkType(parsing: raw) else {
                    throw .invalidValue(key: "benchmark-type", value: raw)
                }
                return .benchmarksList(benchmarkType: type)
            case "show":
                return .benchmarksShow(try benchmarkReference(try require("benchmark")))
            case "init-local":
                return .benchmarksInitLocal
            case "run":
                let (runtime, llamaPin) = try runtimeCoordinate()
                guard let model = try modelCoordinate() else {
                    throw .missingParameter("model")
                }
                let flags = try resolvedRuntimeFlags(runtime)
                let benchmark = try benchmarkReference(try require("benchmark"))
                return .benchmarksRun(
                    benchmarkRef: benchmark, model: model, runtime: runtime,
                    batch: try prefillChunk(flags), runtimeFlags: flags,
                    sync: (params["sync"] ?? "0") != "0",
                    benchmarkFlags: try resolvedBenchmarkFlags(runtime), llamaPin: llamaPin)
            case let other?:
                throw .unknownVerb("benchmarks \(other)")
            }

        case "results":
            switch verbs.dropFirst().first {
            case nil, "list":
                var type: BenchmarkType?
                if let raw = params["type"] {
                    guard let parsed = try? BenchmarkType(parsing: raw) else {
                        throw .invalidValue(key: "type", value: raw)
                    }
                    type = parsed
                }
                var state: BenchmarkResultState?
                if let raw = params["state"] {
                    guard let parsed = BenchmarkResultState(rawValue: raw) else {
                        throw .invalidValue(key: "state", value: raw)
                    }
                    state = parsed
                }
                // Carried whole: `local/x` and `remote/x` select different rows.
                var benchmark: SourcedBenchmarkId?
                if let raw = params["benchmark"] { benchmark = try benchmarkReference(raw) }
                // A negative count would trap in `prefix`.
                var limit: Int?
                if let raw = params["limit"] {
                    guard let count = Int(raw), count >= 0 else {
                        throw .invalidValue(key: "limit", value: raw)
                    }
                    limit = count
                }
                return .resultsList(benchmark: benchmark, benchmarkType: type, state: state,
                                    limit: limit)
            case "show":
                return .resultsShow(resultId: try resultId())
            case "delete":
                return .resultsDelete(resultId: try resultId())
            case let other?:
                throw .unknownVerb("results \(other)")
            }

        case "sync":
            if params["result"] != nil {
                throw .rejected(key: "result",
                                reason: "a result is not addressable yet; pass `job=<jobId>`")
            }
            // Checked at parse time, as the crate parses `--server-url`. `hasPrefix("http")`
            // would admit `httpfoo://x`.
            if let raw = params["server-url"] {
                let scheme = raw.lowercased()
                guard scheme.hasPrefix("http://") || scheme.hasPrefix("https://") else {
                    throw .rejected(key: "server-url",
                                    reason: "expected an http:// or https:// URL")
                }
            }
            return .sync(jobId: params["job"], serverUrl: params["server-url"])

        case "storage":
            switch verbs.dropFirst().first {
            case "status":
                return .storageStatus
            case "gc":
                return .storageGc(dryRun: (params["dry-run"] ?? "0") != "0")
            case nil:
                throw .unknownVerb("storage")
            case let other?:
                throw .unknownVerb("storage \(other)")
            }

        case "jobs":
            return .listJobs

        case "job":
            switch verbs.dropFirst().first {
            case "rm":
                return .removeJob(id: try require("id"))
            case "run":
                let id = try require("id")
                guard let raw = params["scope"] else { return .runJob(id: id, scope: .cancelled) }
                guard let scope = RunScope(rawValue: raw) else {
                    throw .invalidValue(key: "scope", value: raw)
                }
                return .runJob(id: id, scope: scope)
            case "export":
                return .exportJob(id: try require("id"))
            case "submit":
                return .submitJob(id: try require("id"))
            case nil:
                throw .unknownVerb("job")
            case let other?:
                throw .unknownVerb("job \(other)")
            }

        case "status":
            return .status

        case "version":
            return .version

        // Knobs and defaults are `WorkerArgs`', which are the crate's.
        case "worker":
            var options = WorkerArgs()
            if let secs = try seconds("idle-secs", atLeast: 0) { options.idleSecs = secs }
            if let secs = try seconds("idle-jitter-secs", atLeast: 0) {
                options.idleJitterSecs = secs
            }
            // A zero-second heartbeat spins rather than renews.
            options.heartbeatSecs = try seconds("heartbeat-secs", atLeast: 1)
            if let raw = params["max-jobs"] {
                guard let count = UInt(raw) else {
                    throw .invalidValue(key: "max-jobs", value: raw)
                }
                options.maxJobs = count
            }
            options.skipProfileRefresh = (params["skip-profile-refresh"] ?? "0") != "0"
            return .worker(options)

        case "settings":
            switch verbs.dropFirst().first?.lowercased() {
            case nil:
                throw .unknownVerb("settings")
            case "run":
                // This client's spelling for `worker`, named for the Settings toggle. It
                // takes no knobs; `worker` does.
                return .worker(WorkerArgs())
            case let other?:
                throw .unknownVerb("settings \(other)")
            }

        case "bench":
            let spec = try modelCoordinate()
            let metrics = (params["metrics"] ?? "prefill,decode,maxmem")
                .split(separator: ",").map(String.init)
            let offsets = try ints("offsets", default: [256, 512, 1024, 2048, 4096])
            let (runtime, llamaPin) = try runtimeArgument(params["runtime"])
            let flags = try resolvedRuntimeFlags(runtime)
            return .bench(spec: spec, model: params["match"], quant: params["quant"],
                          runtime: runtime,
                          batch: try prefillChunk(flags), runtimeFlags: flags,
                          benchmarks: try benchmarkIds(), metrics: metrics, offsets: offsets,
                          submit: submits(coordinate: spec),
                          benchmarkFlags: try resolvedBenchmarkFlags(runtime),
                          claim: try claim(picksItsModel: spec == nil && runtime != .appleFoundation),
                          llamaPin: llamaPin)

        case nil:
            let metrics = (params["metrics"] ?? "prefill,decode,maxmem")
                .split(separator: ",").map(String.init)
            let offsets = try ints("offsets", default: [256, 512, 1024, 2048, 4096])
            let (runtime, llamaPin) = try runtimeArgument(params["runtime"])
            // Before the AFM branch: plan-types defines no runtime flags for
            // `apple_foundation`, so an AFM cell carrying them is refused rather than silently
            // dropped, and only its decode and end-to-end cells take benchmark flags.
            let flags = try resolvedRuntimeFlags(runtime)
            let benchmarkFlags = try resolvedBenchmarkFlags(runtime)
            if runtime == .appleFoundation {
                return .afm(metrics: metrics, benchmarks: try benchmarkIds(), offsets: offsets,
                            submit: submits(coordinate: nil), claim: try claim(picksItsModel: false),
                            benchmarkFlags: benchmarkFlags)
            }
            // A bare invocation naming an exact model is a `bench`; the plan runner emits it.
            if let coordinate = try modelCoordinate() {
                return .bench(spec: coordinate, model: nil, quant: params["quant"],
                              runtime: runtime,
                              batch: try prefillChunk(flags), runtimeFlags: flags,
                              benchmarks: try benchmarkIds(), metrics: metrics, offsets: offsets,
                              submit: submits(coordinate: coordinate),
                              benchmarkFlags: benchmarkFlags, claim: try claim(picksItsModel: false),
                              llamaPin: llamaPin)
            }
            _ = try claim(picksItsModel: true)
            return .bareBench(
                runtime: runtime,
                batch: try prefillChunk(flags),
                metrics: metrics,
                offsets: offsets,
                benchmarks: try benchmarkIds(),
                model: params["match"],
                submit: submits(coordinate: nil),
                llamaPin: llamaPin)

        case let other?:
            throw .unknownVerb(other)
        }
    }

}

private extension String {
    nonisolated func dropSchemePrefix(_ prefix: String) -> String? {
        hasPrefix(prefix) ? String(dropFirst(prefix.count)) : nil
    }

    nonisolated func dropKeyPrefix(_ key: String) -> String? {
        hasPrefix(key) ? String(dropFirst(key.count)) : nil
    }
}
