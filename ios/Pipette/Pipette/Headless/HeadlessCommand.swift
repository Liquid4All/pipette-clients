import ArgumentParser
import Foundation

/// Why an invocation is not a command.
///
/// Distinct from a run failure, and reported with a distinct exit code (**2** against a
/// failed run's 1), so a console consumer can tell "I typed it wrong" from "the benchmark
/// failed" without scraping the message. Every case names the offending token: the reason
/// to refuse an unknown `key=` instead of ignoring it is to be able to say which one.
nonisolated enum HeadlessUsageError: Error, Equatable {
    /// No `headlessrun` marker — this launch is not a headless invocation at all, which
    /// is not an error so much as "nothing to do".
    case notHeadless
    case unknownVerb(String)
    /// Keys the selected verb does not accept, sorted so the message is deterministic.
    case unknownParameters([String])
    case missingParameter(String)
    case invalidValue(key: String, value: String)
    /// A runtime that exists in plan-types but cannot run in this process.
    case hostOnlyRuntime(String)
    /// An argument that parsed but cannot be honoured — a flag group for the wrong axes, a
    /// knob the resolved cell does not carry, a model spelling this client cannot express.
    /// Carries the reason because the underlying decoders raise their own error types, and
    /// collapsing them to "invalid" would lose the only useful part.
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
            // Covers absent and present-but-empty: `org=` conveys nothing, and reporting
            // it as merely "missing" reads wrong to someone looking at the token they
            // just typed.
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

/// One parsed headless CLI invocation — the pure argv → command mapping behind
/// `HeadlessRunner.startIfRequested()`. Parsing is side-effect free so the CLI
/// surface is unit-testable off-device; validation that needs device state
/// (does the model exist, is the job id known) stays in the handlers.
///
/// Grammar: everything after the `headlessrun` marker is either a bare word
/// (a verb, e.g. `models rm`) or a `key=value` parameter. No bare words at all
/// selects the bench command — the original CLI form.
///
/// Every refusal is a typed `HeadlessUsageError` naming the offending token: an unknown
/// verb, a missing required parameter, an out-of-range value, and — the rule the rest of
/// the surface is built on — a `key=` the selected verb does not accept. Unknown keys used
/// to be ignored, so a mistyped parameter was indistinguishable from one that had no
/// effect. Nothing emits the old accepted-and-ignored `cooldown=` any more; plan-types
/// dropped `cooldown_seconds` and has a test asserting it is never emitted.
///
/// `submit` defaults to **on** for catalog-resolved runs: a benchmark whose
/// results nobody sees is nearly always a mistake, so contributing is the
/// default and `submit=0` is the opt-out. The one exception is `spec=`, which
/// names a model the catalog never validated — an ad-hoc experiment stays
/// opt-*in* so it can't quietly publish rows nothing sanctioned.
///
/// The runner refuses to start a submitting run on a device with no
/// registration rather than measuring for an hour and dropping every result.
nonisolated enum HeadlessCommand: Equatable, Sendable {

    /// 99 — every layer on the GPU, what iOS has always loaded with. Applied where the
    /// value is *consumed*, never at parse time: an absent `number_gpu_layers` must stay
    /// absent so "asked for 99" and "asked for nothing" stay distinguishable, as the
    /// crate's builder emits `-ngl` only for a `Some`.
    static let defaultGpuLayers: UInt32 = 99

    /// The AFM probes, named for what they check rather than by the `metrics=` token that
    /// used to select them.
    enum ProbeKind: String, Equatable, Sendable, CaseIterable {
        case token, generation, cap, enforcement
    }

    /// Which terminal cells `job run` flips back to `.pending` — mirrors the
    /// two job-page actions: retry failed cells or resume cancelled (paused)
    /// ones. `job run` without `scope=` resumes (the paused page's primary
    /// action).
    enum RunScope: String, Equatable, Sendable {
        case failed
        case cancelled

        /// The terminal status this scope flips back to `.pending`: `failed`
        /// retries failed cells, `cancelled` resumes paused (cancelled) ones.
        var resetTarget: CellRunStatus {
            switch self {
            case .failed: return .failed
            case .cancelled: return .cancelled
            }
        }
    }

    /// `org` and `email` are **required**, not defaulted. They used to fall back to a
    /// hardcoded organization and a personal address, so omitting them on a fleet device
    /// registered it — silently, and permanently — under the wrong contact. `server` keeps
    /// its default: there is one production collector and naming it every time is noise.
    ///
    /// `preauth` is the crate's `--preauth-key`: a `preauth_…` key that admits a client the
    /// operator already approved, so a fleet device registers without waiting on approval.
    case register(server: String?, org: String, email: String, preauth: String?,
                  clientDetails: String?, deviceName: String?)
    case memSeq(models: [String], batch: Int)
    case listModels(format: ModelCommands.ListFormat)
    case removeModel(name: String?, repo: String?)
    /// The compiled-in engines. No leaf: a phone installs no runtimes, so there is
    /// nothing for `runtimes pull/remove/catalog/flavors` to act on.
    case listRuntimes
    case listBenchmarks(type: BenchmarkType?)
    case showBenchmark(id: String)
    case initLocalBenchmarks
    /// `version` — the crate's `pipette --version`, plus the build flags that change
    /// what a measurement means.
    case version
    case storageStatus
    /// `storage gc [dry-run=1]` — the crate's `storage gc --dry-run`.
    case storageGc(dryRun: Bool)
    /// `sync`: pull the catalog, then submit what is pending. `jobId` narrows the
    /// submission to one job — the CLI narrows by result id, which iOS cannot address yet.
    case sync(jobId: String?)
    case listResults(benchmark: String?, type: BenchmarkType?,
                     state: BenchmarkResultState?, limit: Int?)
    case showResult(id: ResultId)
    case deleteResult(id: ResultId)
    case authMe
    case authReset(force: Bool)
    /// One of the four AFM diagnostic probes. They used to ride on `metrics=`, which made
    /// a probe look like a benchmark; `diag probe kind=` says what it is.
    case diagProbe(kind: ProbeKind)
    case listJobs
    case removeJob(id: String)
    case runJob(id: String, scope: RunScope)
    case exportJob(id: String)
    case submitJob(id: String)
    case status
    /// App Settings prefs (same keys as `SettingsView` / `LocalStorage`).
    /// `show` dumps them; `setWorker` writes the planner-worker toggle.
    case settings(SettingsOp)

    /// One settings subcommand — keep ops explicit (`set …`) so get vs mutate
    /// can't share a bare token.
    enum SettingsOp: Equatable, Sendable {
        /// Print every headless-managed setting.
        case show
        /// Settings → Planner worker (preference only; process still exits).
        case setWorker(Bool)
        /// Turn worker on, start the claim loop, and keep the process alive.
        case run
    }

    /// The `bench` verb: name a model + runtime + benchmark, and the runner
    /// downloads the model first if it isn't on device. `spec` (a `Model`
    /// definition decoded from `spec=<json>`) takes precedence over the friendly
    /// catalog selectors (`model`/`quant`/`runtime`); `runtime == .appleFoundation` needs no
    /// model at all.
    /// How a model is named on the command line — the forms `--model` takes.
    ///
    /// `model://sha256=<prefix>` is not resolved while parsing: it addresses the local
    /// store, which the parser cannot read, so it is carried to the runner exactly as the
    /// crate carries it to `artifact_ref`.
    enum ModelSelector: Equatable, Sendable {
        /// A JSON `Model` or a compact `<scheme>://key=value` URI.
        case model(Model)
        /// A `model://sha256=` prefix, validated for shape but not yet matched.
        case digest(String)

        /// The model this names outright, or `nil` when it names a digest the store has
        /// yet to resolve.
        var namedModel: Model? {
            guard case let .model(model) = self else { return nil }
            return model
        }
    }

    /// `models pull` / `models delete` — a model named the way the crate names one, by a
    /// self-contained reference rather than a repo plus a filename.
    case pullModel(ModelSelector)
    case deleteModel(ModelSelector)

    /// `benchmarks run` — one cell, every axis named, as `pipette benchmarks run` takes
    /// it. The sweep form is `bench`, which this client keeps for the fleet harness.
    case benchmarksRun(benchmark: String, model: ModelSelector, runtime: RuntimeType,
                       batch: Int, nGpuLayers: UInt32?, threads: UInt32?, sync: Bool,
                       readiness: ReadinessPolicy)
    case bench(spec: ModelSelector?, model: String?, quant: String?, runtime: RuntimeType,
               batch: Int, nGpuLayers: UInt32?, threads: UInt32?, benchmarks: [String],
               metrics: [String], offsets: [Int], submit: Bool)
    /// The bare metrics×offsets form (no verb, driven entirely by k=v params):
    /// `headlessrun` alone runs the default MLX prefill/decode/maxmem sweep over
    /// whatever matching model is already on device. Distinct from the `bench`
    /// verb, which resolves + downloads a named model.
    case bareBench(runtime: RuntimeType, batch: Int, nGpuLayers: UInt32?, threads: UInt32?,
                   metrics: [String], offsets: [Int], benchmarks: [String], model: String?,
                   submit: Bool)
    /// `runtime=afm` benchmarks Apple's on-device foundation model. `metrics` selects
    /// a diagnostic probe (`tokprobe`/`enforceprobe`/`capprobe`/`genprobe`) — a direct
    /// diagnostic that neither records nor submits — or, absent one, runs the resolved
    /// catalog benchmarks through the normal `JobLauncher`/`JobExecutor` path so results
    /// persist and `submit=1` auto-submits (parity with the other bench paths).
    case afm(metrics: [String], benchmarks: [String], offsets: [Int], submit: Bool)

    /// `metrics` tokens that select an AFM diagnostic probe instead of a
    /// benchmark sweep. A probe records nothing, so it contributes nothing.
    static let afmProbeTokens: Set<String> = [
        "enforceprobe", "capprobe", "genprobe", "tokprobe",
    ]

    /// Whether this command will attempt to contribute results to the server.
    ///
    /// The one place that answers it: the runner's registration check reads it,
    /// and it is what the submit-default tests pin. An AFM diagnostic probe
    /// records nothing, so it never contributes however `submit` parsed.
    var submitsResults: Bool {
        switch self {
        case .bench(_, _, _, _, _, _, _, _, _, _, let submit): return submit
        case .bareBench(_, _, _, _, _, _, _, _, let submit): return submit
        case .sync: return true
        // Reading and deleting local results publishes nothing.
        case .listResults, .showResult, .deleteResult: return false
        // Observation and identity work contribute nothing; a probe records nothing either.
        case .authMe, .authReset, .diagProbe: return false
        case .afm(let metrics, _, _, let submit):
            return submit && !metrics.contains(where: Self.afmProbeTokens.contains)
        default: return false
        }
    }

    /// Map a `pipette://` deep link to a command by reducing it to the same
    /// token vector `parse(_:)` consumes, so the URL surface and the argv CLI
    /// share one grammar. Shape:
    ///
    ///     pipette://run[/<verb>[/<sub>]]?k=v&k=v
    ///
    /// The `run` host namespaces the scheme (other hosts return `nil`, leaving
    /// room for future `pipette://` uses); path components become the bare verbs
    /// (`/job/run` → `job run`) and query items become `k=v` params. The empty
    /// path is the bare-bench form (`pipette://run?runtime=mlx&…`). Returns
    /// `nil` for a non-`run` host or tokens that don't form a valid command.
    ///
    /// Callers must still gate the result on `allowedViaDeepLink` — parsing a
    /// command is not the same as permitting it over a URL any app can open.
    static func parse(url: URL) -> Result<HeadlessCommand, HeadlessUsageError> {
        guard url.scheme == "pipette", url.host == "run",
              let comps = URLComponents(url: url, resolvingAgainstBaseURL: false)
        else { return .failure(.notHeadless) }
        let verbs = comps.path.split(separator: "/").map(String.init)
        // `queryItems` returns percent-decoded values, so a token like
        // `spec={"type":"hf_mlx",…}` is reconstructed intact; the argv parser
        // splits each token on its *first* `=`, so `=` inside a JSON value is
        // preserved as part of the value.
        let params = (comps.queryItems ?? []).map { "\($0.name)=\($0.value ?? "")" }
        return parse(["headlessrun"] + verbs + params)
    }

    /// Whether this command may be triggered by a `pipette://` deep link. The
    /// deep-link surface is intentionally narrow — run and observe work, never
    /// mutate device identity (`register`), stored models (`models rm`, the
    /// `*get` downloads), or delete jobs. A URL can be opened by any app, web
    /// page, or QR code, so anything with a destructive or identity side effect
    /// stays CLI-only (`devicectl … headlessrun`).
    var allowedViaDeepLink: Bool {
        switch self {
        case .bench, .bareBench, .afm, .runJob, .submitJob, .exportJob, .status:
            return true
        case .benchmarksRun, .pullModel, .deleteModel,
             .register, .memSeq, .listModels, .removeModel, .listJobs, .removeJob,
             .settings, .listRuntimes, .listBenchmarks, .showBenchmark, .initLocalBenchmarks,
             .storageStatus, .storageGc, .sync, .version,
             .authMe, .authReset, .diagProbe,
             .listResults, .showResult, .deleteResult:
            // The inspect verbs are side-effect free, but they stay off the link surface
            // with the other list verbs: the allow-list is deliberately narrow, and
            // nothing needs to read a device's inventory from a URL.
            return false
        }
    }

    /// Map argv to a command. `args` is the full argument vector; everything up
    /// to and including the `headlessrun` marker is ignored (launch args arrive
    /// after the binary path / bundle id). Returns `nil` when the marker is
    /// absent or the tokens after it don't form a valid command.
    static func parse(_ args: [String]) -> Result<HeadlessCommand, HeadlessUsageError> {
        guard let marker = args.firstIndex(of: "headlessrun") else { return .failure(.notHeadless) }
        // `-v`/`--verbose` is a global switch (`HeadlessRunner.isVerbose`), not any verb's
        // parameter, so it is dropped before the grammar sees it.
        let raw = Array(args[(marker + 1)...]).filter { $0 != "-v" && $0 != "--verbose" }
        let (verbs, options) = HeadlessTokens.split(raw)

        let leaf: any HeadlessLeaf
        do {
            // No verb at all is the bare form. Parsed directly rather than as the root's
            // default subcommand, so that a *failure* to route a verb stays a refusal
            // instead of silently falling through to a benchmark run.
            if verbs.isEmpty {
                leaf = try HeadlessTree.Bare.parse(options)
            } else {
                guard let routed = try HeadlessTree.Root.parseAsRoot(verbs + options)
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
        // Strays are whatever matched no declared parameter. A dashed one is a parameter
        // this verb does not take; a bare word is a verb that does not exist.
        if let stray = leaf.unrecognized.first(where: { !$0.hasPrefix("-") }) {
            let verbPath = type(of: leaf).verbPath
            // Which path names the offence depends on how the leaf was reached. A leaf the
            // caller named is reported by its canonical path, so the alias `register now`
            // is refused as `auth register now` — the grouped spelling it means. A leaf a
            // *default* supplied was never named, so `models frobnicate` is reported as
            // typed rather than as `models list frobnicate`, a verb nobody wrote.
            let named = verbPath.last.map(verbs.contains) ?? false
            let path = named ? verbPath + [stray]
                             : (verbs.last == stray ? verbs : verbs + [stray])
            return .failure(.unknownVerb(path.joined(separator: " ")))
        }
        if !leaf.unrecognized.isEmpty {
            return .failure(.unknownParameters(leaf.unrecognized.map(headlessStrayKey).sorted()))
        }
        do {
            return .success(try build(verbs: type(of: leaf).verbPath, params: leaf.bag))
        } catch {
            return .failure(error)
        }
    }

    /// The verb/parameter mapping itself. Split from `parse` so the tokenizer stays
    /// separate from the grammar, and typed-throwing so the set of refusals a caller must
    /// handle is the enum and nothing else.
    private static func build(
        verbs unnormalized: [String],
        params: [String: String]
    ) throws(HeadlessUsageError) -> HeadlessCommand {
        let verbs = unnormalized
        /// Every required parameter is an identifier, so a present-but-blank one is as
        /// unusable as an absent one — `email=` would otherwise register a device with no
        /// contact, and `id=` would address a job named "".
        func require(_ key: String) throws(HeadlessUsageError) -> String {
            guard let value = params[key],
                  !value.trimmingCharacters(in: .whitespaces).isEmpty
            else { throw .missingParameter(key) }
            return value
        }
        func list(_ key: String) -> [String] {
            (params[key] ?? "").split(separator: ",").map(String.init).filter { !$0.isEmpty }
        }
        /// Whether this run contributes its results.
        ///
        /// A divergence from the CLI, deliberately: `--sync` is opt-*in* there, while a
        /// phone that measures for an hour and publishes nothing is the failure mode worth
        /// designing against, so submission is the default here.
        ///
        /// The exception is a caller who named an exact coordinate rather than selecting
        /// from the catalog: nothing sanctioned that model, so an ad-hoc experiment stays
        /// opt-in. The CLI gates provenance on `BenchmarkSource` instead — over benchmarks
        /// rather than models — which this client now also has, for the catalog half.
        func submits(coordinate: ModelSelector?) -> Bool {
            (params["submit"] ?? (coordinate == nil ? "1" : "0")) != "0"
        }
        /// The prefill chunk: llama's `n_ubatch`, MLX's chunk size. An authored
        /// `runtime-flags` entry wins over `batch=`; the two together are already refused.
        ///
        /// Not written as `??`: that operator is `rethrows`, which widens the thrown type
        /// to `any Error` and defeats the typed throws this function promises.
        func prefillChunk(_ flags: RuntimeFlags?) throws(HeadlessUsageError) -> Int {
            if let chunk = flags?.nUbatch { return Int(chunk) }
            return try int("batch", default: 512)
        }
        /// `result=<jobId>/<cellId>` — the pair iOS addresses a result by. The CLI takes a
        /// flat id; iOS results live under their job, so the pair is the id.
        func resultId() throws(HeadlessUsageError) -> ResultId {
            let raw = try require("result")
            guard let id = ResultId.parse(raw) else {
                throw .rejected(key: "result", reason: "expected `<jobId>/<cellId>`")
            }
            return id
        }
        /// `benchmarks run`'s `runtime=`, which requires the key.
        func runtimeCoordinate() throws(HeadlessUsageError) -> RuntimeType {
            try runtimeArgument(try require("runtime"))
        }

        /// The `runtime=` forms, shared by every verb that takes one: a JSON `Runtime`, or
        /// this client's short token. Absent means MLX, which only the bench forms reach.
        ///
        /// A `<scheme>://` URI is refused by name. plan-types marks the on-device runtimes
        /// `NotRepresentable` as URIs — they are OS-bundled or reached over an app
        /// transport, never imported — and every scheme that *is* representable names a
        /// desktop runtime this build cannot be.
        ///
        /// The JSON is compared against `Runtime.thisBuild`, not merely typed: a
        /// descriptor has to record what actually ran.
        func runtimeArgument(_ raw: String?) throws(HeadlessUsageError) -> RuntimeType {
            guard let raw, !raw.isEmpty else { return .mlxIosPipette }
            if raw.hasPrefix("{") {
                guard let declared = try? JSONDecoder().decode(Runtime.self, from: Data(raw.utf8))
                else { throw .invalidValue(key: "runtime", value: raw) }
                let type = RuntimeType.of(declared)
                guard let builtIn = Runtime.thisBuild(for: type) else {
                    throw .rejected(key: "runtime",
                                    reason: "`\(type.rawValue)` is not runnable on iOS")
                }
                // Compared, not just typed. A runtime names a build, and this binary is
                // one build: running a cell that asked for another and then recording
                // what actually ran would answer a question nobody asked. `runtimes`
                // prints the identity that passes.
                guard declared == builtIn else {
                    let want = (try? SubmissionRef.runtime(declared)) ?? type.rawValue
                    let have = (try? SubmissionRef.runtime(builtIn)) ?? type.rawValue
                    throw .rejected(key: "runtime",
                                    reason: "names \(want), but this build is \(have)")
                }
                guard type.isIosRunnable else { throw .hostOnlyRuntime(type.rawValue) }
                return type
            }
            if raw.contains("://") {
                throw .rejected(key: "runtime",
                                reason: "a runtime URI names a runtime this build cannot be: "
                                    + "the on-device runtimes have no URI form; pass a JSON "
                                    + "`Runtime` object or the short token")
            }
            return try RuntimeType.parseHeadless(raw)
        }

        /// The exact model a `model=` argument names, or `nil` when the caller selected
        /// from the catalog instead.
        ///
        /// The CLI's `--model` spellings: a JSON `Model` or the compact URI — see
        /// `ModelUri` for why `Display` is not among them. Substring selection lives on
        /// `match=`; it used to live here, which made `model=` mean two unrelated things
        /// and left a plan-supplied coordinate silently matching nothing.
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
            // `model://sha256=<prefix>` addresses the local store by descriptor digest —
            // the same id the warehouse keeps as `model_descriptor_sha256`, so a prefix
            // copied out of `models` addresses the same artifact there.
            if let body = raw.trimmingCharacters(in: .whitespaces).dropSchemePrefix("model://") {
                guard let hex = body.dropKeyPrefix("sha256=") else {
                    throw .rejected(key: "model",
                                    reason: "`model://` addresses the store by digest; "
                                        + "write `model://sha256=<hex>`")
                }
                let prefix = hex.lowercased()
                guard prefix.count >= Descriptor.digestMinPrefixLength else {
                    throw .rejected(key: "model",
                                    reason: "digest `\(prefix)` is too short; give at least "
                                        + "\(Descriptor.digestMinPrefixLength) hex chars")
                }
                guard prefix.allSatisfy(\.isHexDigit) else {
                    throw .rejected(key: "model", reason: "digest `\(prefix)` is not hex")
                }
                return .digest(prefix)
            }
            if raw.hasPrefix("{") {
                guard let decoded = try? JSONDecoder().decode(Model.self, from: Data(raw.utf8))
                else { throw .invalidValue(key: "model", value: raw) }
                return .model(decoded)
            }
            do {
                return .model(try ModelUri.parse(raw))
            } catch ModelUri.Failure.notAUri {
                // The likeliest cause is a bare name, which is what `match=` is for.
                throw .rejected(key: "model",
                                reason: "expected a JSON object or a `<scheme>://key=value` "
                                    + "URI; use `match=` to select from the catalog by name")
            } catch let failure as ModelUri.Failure {
                throw .rejected(key: "model", reason: failure.reason)
            } catch {
                throw .invalidValue(key: "model", value: raw)
            }
        }
        /// The benchmark ids to run. `benchmark=` is the CLI's spelling; `benchmarks=` is
        /// kept because the plan runner emits it.
        func benchmarkIds() throws(HeadlessUsageError) -> [String] {
            if params["benchmark"] != nil, params["benchmarks"] != nil {
                throw .rejected(key: "benchmark",
                                reason: "`benchmarks=` is the plural alias; pass one")
            }
            return params["benchmark"] != nil ? list("benchmark") : list("benchmarks")
        }
        /// The resolved `runtime-flags=` entry, or `nil` when the parameter is absent.
        ///
        /// `runtime-flags=` is the plan runner's JSON *array* — the shape the CLI's
        /// `--runtime-flags` takes — collapsed to the at-most-one entry a single run
        /// resolves to. It is decoded and resolved through `RuntimeFlagRef`, the same path
        /// a claim takes, so a console experiment is validated exactly as an orchestrated
        /// cell is: a field no iOS variant declares is an unknown field, and a knob the
        /// resolved variant does not carry is refused by name.
        ///
        /// `ctx_size` is refused rather than accepted and dropped: the per-cell context
        /// window is computed from the benchmark and would overwrite an authored value.
        /// Honouring it means giving the computed value the role of a floor, which is a
        /// change to the run path rather than to this parser.
        func resolvedRuntimeFlags(
            _ runtime: RuntimeType
        ) throws(HeadlessUsageError) -> RuntimeFlags? {
            guard let json = params["runtime-flags"] else { return nil }
            func reject(_ reason: String) -> HeadlessUsageError {
                .rejected(key: "runtime-flags", reason: reason)
            }
            // The cell, derived from the arguments already parsed. AFM supplies its own
            // model; every other runtime needs one named exactly, because a `match=` is
            // resolved against the catalog at run time and its type is not known here.
            let modelType: ModelType
            switch try modelCoordinate() {
            case _ where runtime == .appleFoundation: modelType = .appleFoundationText
            case let .model(model)?: modelType = ModelType.of(model)
            case .digest?:
                // A digest names one artifact, but which *kind* it is only the store
                // knows, and it is not consulted until the run.
                throw reject("a `model://sha256=` digest resolves against the store at run "
                    + "time, so its type is not known here; name the model instead")
            default:
                throw reject("needs an exact model to resolve against: pass `model=` or "
                    + "`spec=` rather than `match=`")
            }
            let ids = try benchmarkIds()
            guard ids.count == 1 else {
                throw reject("needs exactly one `benchmarks=` id to resolve against, "
                    + "because cells of different types carry different knobs")
            }
            // Not `try?`: an unparseable id is a different mistake from the wrong number of
            // them, and the error already names the id.
            let benchmarkType: BenchmarkType
            do {
                benchmarkType = try BenchmarkType(benchmarkId: ids[0])
            } catch {
                throw .invalidValue(key: "benchmarks", value: ids[0])
            }

            let ref: RuntimeFlagRef
            do {
                ref = try RuntimeFlagRef.knobs(
                    from: Data(json.utf8),
                    axes: (benchmark: benchmarkType, runtime: runtime,
                           model: modelType))
            } catch is RuntimeFlagsNotAnObject {
                // Named rather than left to read as malformed, matching the CLI: an
                // invocation written against the old wire should say what to change.
                throw reject("must be a JSON object of knobs, e.g. {\"threads\":4} "
                    + "(the one-element array is no longer accepted)")
            } catch let field as UnknownFlagField {
                // Covers an axis key too: the cell is derived, so naming it here is as
                // unsupported as naming a knob no iOS variant declares.
                throw reject("no iOS variant declares `\(field.name)`")
            } catch {
                throw .invalidValue(key: "runtime-flags", value: json)
            }
            if ref.ctxSize != nil {
                throw reject("`ctx_size` is not applied yet: the per-cell context window is "
                    + "computed from the benchmark and would overwrite it")
            }
            let resolved: RuntimeFlags
            do {
                resolved = try ref.resolve()
            } catch let error as RuntimeFlagResolveError {
                switch error {
                // The error renders its own triple, as the crate error does, so this no
                // longer rebuilds the message from the ref it was thrown about.
                case let .knobNotAllowed(knob, _, _, _):
                    throw reject("this cell does not carry `\(knob)`")
                case .noSuchCombination:
                    throw reject(error.localizedDescription)
                }
            } catch {
                throw .invalidValue(key: "runtime-flags", value: json)
            }
            let chunk = resolved.nUbatch
            // Both would set the prefill chunk, and picking a winner silently is how a
            // caller ends up measuring a size they did not ask for.
            if chunk != nil, params["batch"] != nil {
                throw reject("`batch=` also sets the prefill chunk; pass one or the other")
            }
            return resolved
        }
        /// The readiness overrides, refused on a benchmark that does not gate.
        ///
        /// The crate rejects them on eval and max-memory for the same reason: those cells
        /// carry no readiness knob, so accepting one would take a value nothing reads.
        /// `skip_thermal` changes what "ready" means rather than how long it waits, so a
        /// cell run with it is not comparable to a gated one.
        func readinessOverrides(_ benchmarkId: String) throws(HeadlessUsageError)
            -> ReadinessPolicy {
            var overrides = ReadinessPolicy()
            let wait = params["readiness-max-wait-secs"]
            let skip = (params["readiness-skip-thermal"] ?? "0") != "0"
            guard wait != nil || skip else { return overrides }
            guard let type = try? BenchmarkType(benchmarkId: benchmarkId), type.gatesOnReadiness
            else {
                throw .rejected(key: "readiness-max-wait-secs",
                                reason: "`\(benchmarkId)` does not gate on readiness; only "
                                    + "the timing benchmarks do")
            }
            if let raw = wait {
                guard let secs = Double(raw), secs > 0 else {
                    throw .invalidValue(key: "readiness-max-wait-secs", value: raw)
                }
                overrides.maxSeconds = secs
            }
            overrides.skipThermal = skip
            return overrides
        }
        /// A non-numeric value is refused rather than defaulted. `batch=abc` silently
        /// meaning 512 is the same defect as an ignored unknown key: the run proceeds and
        /// the setting the caller asked for is nowhere.
        func int(_ key: String, default def: Int) throws(HeadlessUsageError) -> Int {
            guard let raw = params[key] else { return def }
            guard let value = Int(raw) else { throw .invalidValue(key: key, value: raw) }
            return value
        }
        /// Comma-separated integers, all of which must parse — `offsets=256,abc` used to
        /// run one offset instead of two.
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
                return .register(server: params["server-url"],
                                 org: try require("organization"),
                                 email: try require("contact-email"),
                                 preauth: params["preauth-key"],
                                 clientDetails: params["client-details"],
                                 deviceName: params["device-name"])
            case "me":
                return .authMe
            case "reset":
                return .authReset(force: (params["force"] ?? "0") != "0")
            case nil:
                throw .unknownVerb("auth")
            case let other?:
                throw .unknownVerb("auth \(other)")
            }

        case "diag":
            switch verbs.dropFirst().first {
            case "memseq":
                return .memSeq(models: list("models"), batch: try int("batch", default: 512))
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
            // Bare is this client's own spelling; `list` is the crate's, so a habit built
            // on `pipette models list` works here unchanged.
            case nil, "list":
                guard let raw = params["format"] else { return .listModels(format: .name) }
                guard let format = ModelCommands.ListFormat(rawValue: raw) else {
                    throw .invalidValue(key: "format", value: raw)
                }
                return .listModels(format: format)
            case "pull":
                guard let selector = try modelCoordinate() else {
                    throw .missingParameter("model")
                }
                return .pullModel(selector)
            case "delete":
                guard let selector = try modelCoordinate() else {
                    throw .missingParameter("model")
                }
                return .deleteModel(selector)
            case "rm":
                let name = params["name"]
                let repo = params["repo"]
                guard name != nil || repo != nil else { throw .missingParameter("name") }
                return .removeModel(name: name, repo: repo)
            case let other?:
                throw .unknownVerb("models \(other)")
            }

        case "runtimes":
            switch verbs.dropFirst().first {
            case nil, "list":
                return .listRuntimes
            case let other?:
                throw .unknownVerb("runtimes \(other)")
            }

        case "benchmarks":
            switch verbs.dropFirst().first {
            // Bare is this client's own spelling; `list` is the crate's, so a habit built
            // on `pipette benchmarks list` works here unchanged.
            case nil, "list":
                guard let raw = params["type"] else { return .listBenchmarks(type: nil) }
                guard let type = BenchmarkType(rawValue: raw) else {
                    throw .invalidValue(key: "type", value: raw)
                }
                return .listBenchmarks(type: type)
            case "show":
                return .showBenchmark(id: try require("benchmark"))
            case "init-local":
                return .initLocalBenchmarks
            case "run":
                let runtime = try runtimeCoordinate()
                guard let model = try modelCoordinate() else {
                    throw .missingParameter("model")
                }
                let flags = try resolvedRuntimeFlags(runtime)
                let benchmarkId = try require("benchmark")
                return .benchmarksRun(
                    benchmark: benchmarkId, model: model, runtime: runtime,
                    batch: try prefillChunk(flags), nGpuLayers: flags?.numberGpuLayers,
                    threads: flags?.threads,
                    sync: (params["sync"] ?? "0") != "0",
                    readiness: try readinessOverrides(benchmarkId))
            case let other?:
                throw .unknownVerb("benchmarks \(other)")
            }

        case "results":
            switch verbs.dropFirst().first {
            // Bare is this client's own spelling; `list` is the crate's, so a habit built
            // on `pipette results list` works here unchanged.
            case nil, "list":
                var type: BenchmarkType?
                if let raw = params["type"] {
                    guard let parsed = BenchmarkType(rawValue: raw) else {
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
                return .listResults(benchmark: params["benchmark"], type: type, state: state,
                                    limit: params["limit"] == nil ? nil : try int("limit", default: 0))
            case "show":
                return .showResult(id: try resultId())
            case "delete":
                return .deleteResult(id: try resultId())
            case let other?:
                throw .unknownVerb("results \(other)")
            }

        case "sync":
            if params["result"] != nil {
                throw .rejected(key: "result",
                                reason: "a result is not addressable yet; pass `job=<jobId>`")
            }
            return .sync(jobId: params["job"])

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

        // The crate's root `worker` command, reaching the same claim loop `settings run`
        // starts. Its flags have no counterpart yet — `idle_secs` and `idle_jitter_secs`
        // are the values `PlannerWorker` already hardcodes (300 + 0…60s, the same spec
        // section), `heartbeat_secs` is derived from the claim's own window, and
        // `max_jobs`/`skip_profile_refresh` are unwired — so none is accepted rather than
        // parsed and ignored.
        case "worker":
            return .settings(.run)

        case "settings":
            //   settings
            //   settings set worker=on|off
            //   settings run                 // enable + start claim loop (stays up)
            switch verbs.dropFirst().first?.lowercased() {
            case nil:
                return .settings(.show)
            case "set":
                let raw = try require("worker")
                switch raw.lowercased() {
                case "on": return .settings(.setWorker(true))
                case "off": return .settings(.setWorker(false))
                default: throw .invalidValue(key: "worker", value: raw)
                }
            case "run":
                return .settings(.run)
            case let other?:
                throw .unknownVerb("settings \(other)")
            }

        case "bench":
            // A `spec=<json>` is a full `Model` definition decoded by the type's
            // own Codable — malformed JSON (or an unknown `type`) is a usage error,
            // not a silent fallback to the catalog path.
            let spec = try modelCoordinate()
            let metrics = (params["metrics"] ?? "prefill,decode,maxmem")
                .split(separator: ",").map(String.init)
            let offsets = try ints("offsets", default: [256, 512, 1024, 2048, 4096])
            // A `spec=` names a model the catalog never validated, so it is an
            // ad-hoc experiment rather than a sanctioned contribution: it stays
            // opt-*in* while the catalog-resolved paths are opt-out. Nothing in
            // the repo emits `spec=` — the plan runner selects with `model=` and
            // passes `submit=1` explicitly — so this costs no orchestrated run.
            let runtime = try runtimeArgument(params["runtime"])
            let flags = try resolvedRuntimeFlags(runtime)
            return .bench(spec: spec, model: params["match"], quant: params["quant"],
                          runtime: runtime,
                          batch: try prefillChunk(flags),
                          nGpuLayers: flags?.numberGpuLayers,
                          threads: flags?.threads,
                          benchmarks: try benchmarkIds(), metrics: metrics, offsets: offsets,
                          submit: submits(coordinate: spec))

        case nil:
            // No bare verb: the AFM prototype (`runtime=afm`) or the original bench
            // form, both driven entirely by k=v parameters (all optional;
            // `headlessrun` alone runs the default MLX prefill/decode/maxmem sweep).
            let metrics = (params["metrics"] ?? "prefill,decode,maxmem")
                .split(separator: ",").map(String.init)
            let offsets = try ints("offsets", default: [256, 512, 1024, 2048, 4096])
            // `runtime=afm` in the bare form is the dedicated AFM prototype command,
            // not a `bareBench` variant; the other two engines share the bench sweep.
            let runtime = try runtimeArgument(params["runtime"])
            // Resolved before the AFM branch, not after: plan-types defines no
            // `RuntimeFlags` variant for `apple_foundation` on any benchmark, so an AFM
            // cell carrying flags has to be refused. Returning first would accept the
            // parameter and drop it, which is the defect this whole surface removes.
            let flags = try resolvedRuntimeFlags(runtime)
            if runtime == .appleFoundation {
                return .afm(metrics: metrics, benchmarks: try benchmarkIds(), offsets: offsets,
                            submit: submits(coordinate: nil))
            }
            // A bare invocation that names an exact model *is* a `bench` — the plan runner
            // emits this form. Routing it here rather than growing a second coordinate
            // carrier keeps one path from a named model to a run, as the CLI has one
            // `benchmarks run`.
            if let coordinate = try modelCoordinate() {
                return .bench(spec: coordinate, model: nil, quant: params["quant"],
                              runtime: runtime,
                              batch: try prefillChunk(flags),
                              nGpuLayers: flags?.numberGpuLayers,
                              threads: flags?.threads,
                              benchmarks: try benchmarkIds(), metrics: metrics, offsets: offsets,
                              submit: submits(coordinate: coordinate))
            }
            return .bareBench(
                runtime: runtime,
                batch: try prefillChunk(flags),
                nGpuLayers: flags?.numberGpuLayers,
                threads: flags?.threads,
                metrics: metrics,
                offsets: offsets,
                benchmarks: try benchmarkIds(),
                model: params["match"],
                submit: submits(coordinate: nil))

        case let other?:
            throw .unknownVerb(other)
        }
    }

}

private extension String {
    /// The body after `prefix`, or `nil` when the string does not start with it.
    nonisolated func dropSchemePrefix(_ prefix: String) -> String? {
        hasPrefix(prefix) ? String(dropFirst(prefix.count)) : nil
    }

    nonisolated func dropKeyPrefix(_ key: String) -> String? {
        hasPrefix(key) ? String(dropFirst(key.count)) : nil
    }
}
