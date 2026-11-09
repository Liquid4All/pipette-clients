import Foundation
import Testing

@testable import Pipette

/// `HeadlessCommand.parse`: the pure argv to command mapping behind the headless CLI.
struct HeadlessCommandTests {

    /// argv as `devicectl process launch` delivers it: binary path first. Discards the
    /// refusal; `refusal(_:)` keeps it.
    private func parse(_ args: String...) -> HeadlessCommand? {
        parse(args)
    }

    private func parse(_ args: [String]) -> HeadlessCommand? {
        try? HeadlessCommand.parse(["/app/Pipette"] + args).get()
    }

    private func refusal(_ args: String...) -> HeadlessUsageError? {
        refusal(args)
    }

    private func refusal(_ args: [String]) -> HeadlessUsageError? {
        switch HeadlessCommand.parse(["/app/Pipette"] + args) {
        case .success: return nil
        case let .failure(error): return error
        }
    }

    // MARK: - Back-compat: the bare bench form

    @Test func bareHeadlessrunIsTheDefaultMLXBench() {
        #expect(parse("headlessrun") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
    }

    @Test func bareBenchParsesEveryParameter() {
        #expect(parse("headlessrun", "runtime=llama", "batch=256", "metrics=prefill,decode",
                      "offsets=512,1024", "match=LFM2.5", "submit=1") == .bareBench(
            runtime: .llamacppIosPipette, batch: 256, metrics: ["prefill", "decode"], offsets: [512, 1024],
            benchmarks: [], model: "LFM2.5", submit: true))
    }

    @Test func bareRuntimeTypeMatchingIsCaseInsensitiveOnTheLlPrefix() {
        #expect(parse("headlessrun", "runtime=mlx") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
        #expect(parse("headlessrun", "runtime=LLAMA") == .bareBench(
            runtime: .llamacppIosPipette, batch: 512, metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
    }

    /// `runtime=afm` in the bare form is the dedicated AFM command, not a `bareBench`.
    @Test func bareRuntimeAfmParsesAsTheAfmCommand() {
        #expect(parse("headlessrun", "runtime=afm") == .afm(
            metrics: ["prefill", "decode", "maxmem"],
            benchmarks: [], offsets: [256, 512, 1024, 2048, 4096], submit: true))
    }

    @Test func bareBenchExplicitBenchmarkIdsOverrideNothingElse() {
        #expect(parse("headlessrun", "benchmarks=prefill_throughput_512,decode_throughput_512_100")
            == .bareBench(runtime: .mlxIosPipette, batch: 512, metrics: ["prefill", "decode", "maxmem"],
                          offsets: [256, 512, 1024, 2048, 4096],
                          benchmarks: ["prefill_throughput_512", "decode_throughput_512_100"],
                          model: nil, submit: true))
    }

    @Test(arguments: ["coherence", "calibrate", "promptseed"])
    func bareBenchDiagnosticMetricsParseAsMetrics(metric: String) {
        #expect(parse("headlessrun", "runtime=mlx", "metrics=\(metric)") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, metrics: [metric],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
    }

    /// plan-types never emits `cooldown_seconds`, and accepting a parameter that does
    /// nothing is how a mistyped knob goes unnoticed.
    @Test func theRetiredCooldownParameterIsNowRefused() {
        #expect(refusal("headlessrun", "runtime=mlx", "cooldown=0")
            == .unknownParameters(["cooldown"]))
    }

    // MARK: - The `bench` verb

    @Test func benchCatalogSelectionParsesModelQuantRuntime() {
        #expect(parse("headlessrun", "bench", "match=Qwen", "quant=Q4_0",
                      "runtime=llama", "benchmarks=x") == .bench(
            spec: nil, model: "Qwen", quant: "Q4_0", runtime: .llamacppIosPipette,
            batch: 512, runtimeFlags: nil, benchmarks: ["x"],
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096], submit: true, benchmarkFlags: nil))
    }

    @Test func benchRuntimeMlxParsesWithNilSpec() {
        #expect(parse("headlessrun", "bench", "runtime=mlx", "match=LFM") == .bench(
            spec: nil, model: "LFM", quant: nil, runtime: .mlxIosPipette,
            batch: 512, runtimeFlags: nil, benchmarks: [],
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096], submit: true, benchmarkFlags: nil))
    }

    @Test func benchSpecDecodesStraightIntoAModelVariant() throws {
        #expect(try parse("headlessrun", "bench",
                          "spec={\"type\":\"mlx\",\"source\":\"huggingface\",\"org\":\"mlx-community\",\"repo_name\":\"Qwen3.5-0.8B-4bit\"}",
                          "benchmarks=x") == .bench(
            spec: .model(.mlx(Mlx(source: .huggingFace(
                repo: HfRepo.parseOrgRepo("mlx-community/Qwen3.5-0.8B-4bit"), prefix: nil)))),
            model: nil, quant: nil, runtime: .mlxIosPipette,
            batch: 512, runtimeFlags: nil, benchmarks: ["x"],
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096], submit: false, benchmarkFlags: nil))
    }

    @Test func benchMalformedSpecFailsToParse() {
        #expect(parse("headlessrun", "bench", "spec={not valid json}", "benchmarks=x") == nil)
    }

    @Test func benchRuntimeAfmNeedsNoModel() {
        #expect(parse("headlessrun", "bench", "runtime=afm",
                      "metrics=prefill", "offsets=512") == .bench(
            spec: nil, model: nil, quant: nil, runtime: .appleFoundation,
            batch: 512, runtimeFlags: nil, benchmarks: [], metrics: ["prefill"], offsets: [512], submit: true,
            benchmarkFlags: nil))
    }

    // MARK: - auth register / diag memseq

    /// The input an `auth register` invocation parses to, with the crate's one default.
    private func registerInput(server: String = CollectorEndpoint.productionURL,
                               clientDetails: String = "ci box", preauth: String? = nil,
                               deviceName: String? = nil,
                               formFactor: DeviceFormFactor? = nil) throws
        -> HeadlessCommand {
        .authRegister(AuthCommands.RegisterArgs(
            serverUrl: try ServerUrl(server), organization: "Liquid",
            contactEmail: try ContactEmail("a@b.co"), clientDetails: clientDetails,
            preauthKey: try preauth.map { try ClientPreauthKey($0) }, deviceName: deviceName,
            deviceFormFactor: formFactor))
    }

    @Test func registerTakesTheServerOverrideOrDefaultsIt() throws {
        #expect(parse("headlessrun", "auth", "register", "organization=Liquid",
                      "contact-email=a@b.co", "client-details=ci box")
            == (try registerInput()))
        #expect(parse("headlessrun", "auth", "register", "server-url=https://collector.example",
                      "organization=Liquid", "contact-email=a@b.co", "client-details=ci box")
            == (try registerInput(server: "https://collector.example")))
    }

    /// Refused before a keypair is generated and a request attempted.
    @Test(arguments: ["mgmt.example.com", "https://mgmt.example.com?tenant=a"])
    func anUnusableServerUrlIsRefusedAtParseTime(raw: String) {
        let refused = refusal("headlessrun", "auth", "register", "server-url=\(raw)",
                              "organization=Liquid", "contact-email=a@b.co",
                              "client-details=ci box")
        guard case let .rejected(key, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))"); return
        }
        #expect(key == "server-url")
        #expect(reason.contains("mgmt.example.com"), "reason was: \(reason)")
    }

    /// `--client-details` is required upstream, and a contact address that cannot receive
    /// mail is refused before the server spends a pre-auth key on it.
    @Test func registerRequiresClientDetailsAndAUsableContact() {
        #expect(refusal("headlessrun", "auth", "register", "organization=Liquid",
                        "contact-email=a@b.co") == .missingParameter("client-details"))
        #expect(refusal("headlessrun", "auth", "register", "organization=Liquid",
                        "contact-email=nobody", "client-details=ci box")
            == .invalidValue(key: "contact-email", value: "nobody"))
    }

    @Test func registerTakesADeviceFormFactor() throws {
        #expect(parse("headlessrun", "auth", "register", "organization=Liquid",
                      "contact-email=a@b.co", "client-details=ci box",
                      "device-form-factor=tablet")
            == (try registerInput(formFactor: .tablet)))
        #expect(refusal("headlessrun", "auth", "register", "organization=Liquid",
                        "contact-email=a@b.co", "client-details=ci box",
                        "device-form-factor=fridge")
            == .invalidValue(key: "device-form-factor", value: "fridge"))
    }

    /// Registering is not reversible (correcting it rotates the signing key), so the parser
    /// refuses rather than guessing a default.
    @Test func registerRequiresOrgAndEmail() {
        #expect(refusal("headlessrun", "auth", "register") == .missingParameter("organization"))
        #expect(refusal("headlessrun", "auth", "register", "organization=Liquid") == .missingParameter("contact-email"))
        #expect(refusal("headlessrun", "auth", "register", "contact-email=a@b.c") == .missingParameter("organization"))
    }

    @Test(arguments: ["", "   "])
    func registerRejectsAnEmptyOrgOrEmail(blank: String) {
        #expect(refusal("headlessrun", "auth", "register", "organization=\(blank)", "contact-email=a@b.c")
            == .missingParameter("organization"))
        #expect(refusal("headlessrun", "auth", "register", "organization=Liquid", "contact-email=\(blank)")
            == .missingParameter("contact-email"))
    }

    @Test func memseqParsesModelListAndBatch() {
        #expect(parse("headlessrun", "diag", "memseq", "models=a,b,c", "batch=256")
            == .diagMemseq(models: ["a", "b", "c"], batch: 256, prefill: 512))
        #expect(parse("headlessrun", "diag", "memseq") == .diagMemseq(models: [], batch: 512, prefill: 512))
        #expect(refusal("headlessrun", "diag", "memseq", "batch=0")
            == .invalidValue(key: "batch", value: "0"))
        #expect(refusal("headlessrun", "diag", "memseq", "batch=-1")
            == .invalidValue(key: "batch", value: "-1"))
        #expect(parse("headlessrun", "diag", "memseq", "batch=4294967295")
            == .diagMemseq(models: [], batch: 4_294_967_295, prefill: 512))
        #expect(refusal("headlessrun", "diag", "memseq", "batch=4294967296")
            == .invalidValue(key: "batch", value: "4294967296"))
        #expect(parse("headlessrun", "diag", "memseq", "prefill=4000")
            == .diagMemseq(models: [], batch: 512, prefill: 4000))
        #expect(refusal("headlessrun", "diag", "memseq", "prefill=0")
            == .invalidValue(key: "prefill", value: "0"))
        #expect(parse("headlessrun", "diag", "memseq", "prefill=4294967295")
            == .diagMemseq(models: [], batch: 512, prefill: 4_294_967_295))
        #expect(refusal("headlessrun", "diag", "memseq", "prefill=4294967296")
            == .invalidValue(key: "prefill", value: "4294967296"))
    }

    // MARK: - Models verbs

    @Test func modelsListsAndModelsRmRequiresNameOrRepo() {
        #expect(parse("headlessrun", "models") == .modelsList(format: .name))
        #expect(parse("headlessrun", "models", "rm", "name=file.gguf")
            == .modelsRm(name: "file.gguf", repo: nil))
        #expect(parse("headlessrun", "models", "rm", "repo=org/name")
            == .modelsRm(name: nil, repo: "org/name"))
        #expect(parse("headlessrun", "models", "rm") == nil)
    }

    // MARK: - Job verbs

    @Test func jobsLists() {
        #expect(parse("headlessrun", "jobs") == .listJobs)
    }

    @Test func jobSubVerbsRequireAnId() {
        #expect(parse("headlessrun", "job", "rm", "id=abc") == .removeJob(id: "abc"))
        #expect(parse("headlessrun", "job", "export", "id=abc") == .exportJob(id: "abc"))
        #expect(parse("headlessrun", "job", "submit", "id=abc") == .submitJob(id: "abc"))
        #expect(parse("headlessrun", "job", "rm") == nil)
        #expect(parse("headlessrun", "job", "id=abc") == nil)
    }

    @Test func jobRunScopeDefaultsToCancelledAndRejectsUnknownScopes() {
        #expect(parse("headlessrun", "job", "run", "id=abc")
            == .runJob(id: "abc", scope: .cancelled))
        #expect(parse("headlessrun", "job", "run", "id=abc", "scope=failed")
            == .runJob(id: "abc", scope: .failed))
        #expect(parse("headlessrun", "job", "run", "id=abc", "scope=cancelled")
            == .runJob(id: "abc", scope: .cancelled))
        #expect(parse("headlessrun", "job", "run", "id=abc", "scope=everything") == nil)
    }

    // MARK: - Status / worker / failure modes

    @Test func statusParses() {
        #expect(parse("headlessrun", "status") == .status)
    }

    /// Only `run` is left under `settings`: the worker has no saved preference to show or set.
    @Test func settingsRunIsTheOnlySettingsVerb() {
        #expect(parse("headlessrun", "settings", "run") == .worker(WorkerArgs()))
        #expect(refusal("headlessrun", "settings") == .unknownVerb("settings"))
        #expect(refusal("headlessrun", "settings", "set", "worker=on")
            == .unknownVerb("settings set"))
    }

    @Test func unknownVerbFailsInsteadOfFallingThroughToBench() {
        #expect(parse("headlessrun", "frobnicate") == nil)
        #expect(parse("headlessrun", "models", "frobnicate") == nil)
    }

    @Test func missingHeadlessrunMarkerParsesToNothing() {
        #expect(refusal("status") == .notHeadless)
        switch HeadlessCommand.parse([]) {
        case .success: Issue.record("an empty argv is not a command")
        case let .failure(error): #expect(error == .notHeadless)
        }
    }

    // MARK: - Which runtimes a phone can be

    @Test func onlyTheInProcessRuntimesAreIosRunnable() {
        let runnable = RuntimeType.allCases.filter { Runtime.thisBuild(for: $0) != nil }

        #expect(Set(runnable) == [.llamacppIosPipette, .mlxIosPipette, .onnxGenaiIosPipette,
                                  .appleFoundation])
    }

    /// A plan `type` tag names an engine without naming which build of it (the plan sends
    /// the canonical `Runtime` for that), so only this client's short tokens parse.
    @Test func parseTakesShortTokensAndRefusesPlanTypeTags() throws {
        #expect(try RuntimeType.parseHeadless("mlx") == .mlxIosPipette)
        #expect(try RuntimeType.parseHeadless("llama") == .llamacppIosPipette)
        #expect(try RuntimeType.parseHeadless("afm") == .appleFoundation)
        #expect(try RuntimeType.parseHeadless("onnx") == .onnxGenaiIosPipette)
        // The bare `headlessrun` form depends on absent meaning MLX.
        #expect(try RuntimeType.parseHeadless(nil) == .mlxIosPipette)
        for tag in ["apple_foundation", "llamacpp_ios_pipette", "mlx_ios_pipette",
                    "onnx_genai_ios_pipette"] {
            #expect(throws: HeadlessUsageError.self) {
                try RuntimeType.parseHeadless(tag)
            }
        }
    }

    // MARK: - runtime-flags

    /// A knobs-only `runtime-flags=`; the cell's axes come from `runtime=`, `model=` and
    /// `benchmarks=`, as on the CLI.
    private func flagsRun(_ knobs: String, runtime: String = "mlx",
                          model: String? = "mlx://repo=o/r",
                          benchmark: String = "prefill_throughput_256") -> [String] {
        ["headlessrun", "bench", "runtime=\(runtime)"]
            + (model.map { ["model=\($0)"] } ?? [])
            + ["benchmarks=\(benchmark)", "runtime-flags={\(knobs)}"]
    }

    private static let ggufModel = "gguf-text://repo=o/r&path=m.gguf"

    @Test func runtimeFlagsSuppliesThePrefillChunk() {
        let parsed = parse(flagsRun(#""n_ubatch":256"#))
        guard case let .bench(_, _, _, _, batch, _, _, _, _, _, _, _, _) = parsed else {
            Issue.record("expected a bench command, got \(String(describing: parsed))")
            return
        }
        #expect(batch == 256)
    }

    /// The point of use applies the default, so "asked for 99" and "asked for nothing"
    /// stay distinguishable. Mirrors the crate, whose builder emits `-ngl` only for a `Some`.
    @Test func anAuthoredGpuLayerCountIsCarriedAndAnAbsentOneIsNotInvented() {
        let authored = parse(flagsRun(#""number_gpu_layers":10"#, runtime: "llama",
                                      model: Self.ggufModel))
        guard case let .bench(_, _, _, _, _, flags, _, _, _, _, _, _, _) = authored else {
            Issue.record("expected a bench command, got \(String(describing: authored))")
            return
        }
        #expect(flags?.numberGpuLayers == 10)
        #expect(flags?.threads == nil)

        let bare = parse("headlessrun", "bench", "runtime=llama", "model=\(Self.ggufModel)",
                         "benchmarks=prefill_throughput_256")
        guard case let .bench(_, _, _, _, _, absent, _, _, _, _, _, _, _) = bare else {
            Issue.record("expected a bench command"); return
        }
        #expect(absent == nil)
    }

    @Test func anAuthoredCtxSizeReachesTheCell() throws {
        let parsed = parse(flagsRun(#""ctx_size":4096,"threads":4"#, runtime: "llama",
                                    model: Self.ggufModel))
        guard case let .bench(_, _, _, _, batch, flags, benchmarks, _, _, _, _, _, _) = parsed else {
            Issue.record("expected a bench command, got \(String(describing: parsed))"); return
        }
        let resolved = try #require(flags)
        #expect(resolved.ctxSize == 4096)
        // `batch=`'s default folds in as the prefill chunk, as the job settings carry it.
        #expect(resolved.nUbatch == 512)
        #expect(batch == 512)

        let model = DiscoveredModel(source: try parseModelUri(Self.ggufModel),
                                    path: "/tmp/m.gguf", sizeBytes: 1)
        let cells = JobCell.pending(benchmarkIds: benchmarks, for: model,
                                    runtimeFlags: RuntimeFlagRef(resolved))
        #expect(cells.count == 1)
        #expect(cells.first?.spec.runtimeFlags?.ctxSize == 4096)
        #expect(cells.first?.spec.runtimeFlags?.threads == 4)
    }

    /// The cell is varied through the invocation, not the JSON: a flags value names knobs
    /// and nothing else.
    @Test(arguments: [
        (knobs: #""tensor_parallel_size":2"#, runtime: "mlx", model: "mlx://repo=o/r",
         reason: "no iOS variant declares `tensor_parallel_size`"),
        // The cell is derived, so an axis key is not the caller's to name.
        (knobs: #""runtime_type":"mlx_ios_pipette","n_ubatch":256"#, runtime: "mlx",
         model: "mlx://repo=o/r", reason: "must not carry `runtime_type`"),
        (knobs: #""number_gpu_layers":10"#, runtime: "mlx", model: "mlx://repo=o/r",
         reason: "this cell does not carry `number_gpu_layers`"),
        (knobs: "", runtime: "afm", model: nil, reason: "no runtime flags defined for"),
    ])
    func runtimeFlagsRefusalsReportWhy(
        knobs: String, runtime: String, model: String?, reason: String
    ) {
        let refused = refusal(flagsRun(knobs, runtime: runtime, model: model))
        guard case let .rejected(key, actual) = refused else {
            Issue.record("expected a flags refusal, got \(String(describing: refused))")
            return
        }
        #expect(key == "runtime-flags")
        #expect(actual.hasPrefix(reason), "reason was: \(actual)")
    }

    /// The legacy one-element array reads as wrong rather than malformed, so the refusal
    /// says what to change.
    @Test func theOneElementArrayIsRefusedByName() {
        let legacy = #"[{"benchmark_type":"prefill_throughput","runtime_type":"mlx_ios_pipette","model_type":"mlx","n_ubatch":256}]"#
        let refused = refusal(["headlessrun", "bench", "runtime=mlx", "model=mlx://repo=o/r",
                               "benchmarks=prefill_throughput_256",
                               "runtime-flags=\(legacy)"])
        guard case let .rejected(_, reason) = refused else {
            Issue.record("expected a flags refusal, got \(String(describing: refused))")
            return
        }
        #expect(reason.hasPrefix("must be a JSON object of knobs"), "reason was: \(reason)")
    }

    /// A `match=` is resolved against the catalog at run time, so there is no model type
    /// to resolve flags against at parse.
    @Test func flagsWithoutAnExactModelAreRefused() {
        let refused = refusal("headlessrun", "bench", "runtime=llama", "match=x",
                              "benchmarks=prefill_throughput_256",
                              #"runtime-flags={"number_gpu_layers":10}"#)
        guard case let .rejected(_, reason) = refused else {
            Issue.record("expected a flags refusal, got \(String(describing: refused))")
            return
        }
        #expect(reason.hasPrefix("needs an exact model"), "reason was: \(reason)")
    }

    /// A digest names one artifact but not its kind; only the store knows that, and it is
    /// not consulted until the run.
    @Test func flagsAgainstADigestSelectorAreRefused() {
        let refused = refusal("headlessrun", "bench", "runtime=llama",
                              "model=model://sha256=abcdef0123456789",
                              "benchmarks=prefill_throughput_256",
                              #"runtime-flags={"number_gpu_layers":10}"#)
        guard case let .rejected(_, reason) = refused else {
            Issue.record("expected a flags refusal, got \(String(describing: refused))")
            return
        }
        #expect(reason.hasPrefix("a `model://sha256=` digest"), "reason was: \(reason)")
    }

    /// The refusal names the unparseable id rather than blaming the count.
    @Test func flagsAgainstAnUnknownBenchmarkIdNameTheId() {
        #expect(refusal("headlessrun", "bench", "runtime=llama",
                        "model=\(Self.ggufModel)", "benchmarks=nonsense",
                        #"runtime-flags={"number_gpu_layers":10}"#)
            == .invalidValue(key: "benchmarks", value: "nonsense"))
    }

    /// Several benchmarks' cells carry different knobs, so which one the flags belong to
    /// would be ambiguous.
    @Test func flagsAcrossSeveralBenchmarksAreRefused() {
        let refused = refusal("headlessrun", "bench", "runtime=llama",
                              "model=\(Self.ggufModel)",
                              "benchmarks=prefill_throughput_256,decode_throughput_256_16",
                              #"runtime-flags={"number_gpu_layers":10}"#)
        guard case let .rejected(_, reason) = refused else {
            Issue.record("expected a flags refusal, got \(String(describing: refused))")
            return
        }
        #expect(reason.hasPrefix("needs exactly one"), "reason was: \(reason)")
    }

    /// Both set the prefill chunk; picking a winner silently would measure a size the caller
    /// did not ask for.
    @Test func runtimeFlagsAndBatchTogetherAreRefused() {
        let refused = refusal(flagsRun(#""n_ubatch":256"#) + ["batch=512"])
        #expect(refused == .rejected(
            key: "runtime-flags",
            reason: "`batch=` also sets the prefill chunk; pass one or the other"))
    }

    // MARK: - `benchmarks run`

    /// Resolved into the cell's benchmark flags as the crate's `build_benchmark_flags` does.
    @Test func readinessOverridesBecomeTheCellsBenchmarkFlags() {
        let parsed = parse("headlessrun", "benchmarks", "run",
                           "benchmark=prefill_throughput_256",
                           "model=\(Self.ggufModel)", "runtime=llama",
                           "readiness-max-wait-secs=30", "readiness-skip-thermal=1")
        guard case let .benchmarksRun(_, _, _, _, _, _, flags, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(flags == .prefillThroughputLlamacppIosPipetteGgufText(
            readiness: ReadinessOverrides(maxWaitSecs: 30, skipThermal: true)))
    }

    /// Read as `PIPETTE_READINESS_SKIP_THERMAL` is, so `false` names no override rather than
    /// skipping for not being `0`.
    @Test(arguments: ["0", "false", "FALSE", "off", "no"])
    func aFalseySkipThermalEnforces(raw: String) {
        let parsed = parse("headlessrun", "benchmarks", "run",
                           "benchmark=prefill_throughput_256",
                           "model=\(Self.ggufModel)", "runtime=llama",
                           "readiness-skip-thermal=\(raw)")
        guard case let .benchmarksRun(_, _, _, _, _, _, flags, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(flags == nil)
    }

    /// The dashed spelling takes the bare form the CLI does, since `--readiness-skip-thermal`
    /// carries no value there. In `key=value` a bare word is a verb, so that spelling needs
    /// `=1`; both reach the same override.
    @Test func theDashedSkipThermalNeedsNoValue() {
        let parsed = parse(["headlessrun", "benchmarks", "run",
                            "--benchmark", "prefill_throughput_256",
                            "--model", Self.ggufModel, "--runtime", "llama",
                            "--readiness-skip-thermal"])
        guard case let .benchmarksRun(_, _, _, _, _, _, flags, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(flags?.readiness?.skipThermal == true)
    }

    /// Refused where the crate refuses them: eval and max-memory cells have no variant with
    /// a readiness field, so a value there would be accepted and read by nothing.
    @Test(arguments: ["eval_smoke", "max_memory_usage_256"])
    func readinessOverridesAreRefusedOnUngatedBenchmarks(benchmark: String) {
        let refused = refusal("headlessrun", "benchmarks", "run",
                              "benchmark=\(benchmark)", "model=\(Self.ggufModel)",
                              "runtime=llama", "readiness-max-wait-secs=30")
        guard case let .rejected(_, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))")
            return
        }
        #expect(reason.hasPrefix("not valid for this benchmark"), "reason was: \(reason)")
    }

    /// The crate's `value_parser!(u64).range(1..)`.
    @Test(arguments: ["0", "1.5", "inf", "-3", "soon"])
    func readinessMaxWaitIsWholeSecondsOfAtLeastOne(raw: String) {
        #expect(refusal("headlessrun", "benchmarks", "run", "benchmark=prefill_throughput_256",
                        "model=\(Self.ggufModel)", "runtime=llama",
                        "readiness-max-wait-secs=\(raw)")
            == .invalidValue(key: "readiness-max-wait-secs", value: raw))
    }

    /// Any `u64` parses, as the crate's does; the gate's ceiling bounds it, not the parser.
    /// A `Double` round-trip would trap near `UInt64.max`.
    @Test func theLargestWaitParsesWithoutTrapping() {
        let parsed = parse("headlessrun", "benchmarks", "run",
                           "benchmark=prefill_throughput_256",
                           "model=\(Self.ggufModel)", "runtime=llama",
                           "readiness-max-wait-secs=\(UInt64.max)")
        guard case let .benchmarksRun(_, _, _, _, _, _, flags, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))"); return
        }
        #expect(flags?.readiness?.maxWaitSecs == .max)
    }

    /// The refusal names the parameter the caller passed, not always the wait.
    @Test func aSkipThermalOnAnUngatedBenchmarkIsRefusedByItsOwnName() {
        guard case let .rejected(key, _) = refusal(
            "headlessrun", "benchmarks", "run", "benchmark=eval_smoke",
            "model=\(Self.ggufModel)", "runtime=llama", "readiness-skip-thermal=1")
        else { Issue.record("expected a refusal"); return }
        #expect(key == "readiness-skip-thermal")
    }

    /// A `local/` reference reads its type from the id, not from the whole reference.
    @Test func aLocalReferenceGatesOnItsBenchmarkType() {
        let parsed = parse("headlessrun", "benchmarks", "run",
                           "benchmark=local/prefill_throughput_256",
                           "model=\(Self.ggufModel)", "runtime=llama",
                           "readiness-max-wait-secs=30")
        guard case let .benchmarksRun(benchmark, _, _, _, _, _, flags, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))"); return
        }
        #expect(benchmark == .local("prefill_throughput_256"))
        #expect(flags?.readiness?.maxWaitSecs == 30)
    }

    @Test func withoutOverridesTheCellCarriesNoBenchmarkFlags() {
        let parsed = parse("headlessrun", "benchmarks", "run",
                           "benchmark=prefill_throughput_256",
                           "model=\(Self.ggufModel)", "runtime=llama")
        guard case let .benchmarksRun(_, _, _, _, _, _, flags, _) = parsed else {
            Issue.record("expected benchmarksRun"); return
        }
        #expect(flags == nil)
    }

    // MARK: - readiness on the sweep forms

    private func sweepFlags(_ command: HeadlessCommand?) -> BenchmarkFlags? {
        guard case let .bench(_, _, _, _, _, _, _, _, _, _, flags, _, _)? = command else { return nil }
        return flags
    }

    /// The bare form with an exact model parses to `bench`, so the sweep forms carry the
    /// readiness keys too.
    @Test(arguments: [
        ["headlessrun", "runtime=mlx", "model=mlx://repo=o/r"],
        ["headlessrun", "bench", "runtime=mlx", "model=mlx://repo=o/r"],
    ])
    func sweepFormsCarryTheReadinessOverrides(_ form: [String]) {
        let parsed = parse(form + ["benchmarks=prefill_throughput_256",
                                   "readiness-max-wait-secs=1800", "readiness-skip-thermal=1"])
        #expect(sweepFlags(parsed) == .prefillThroughputMlxIosPipetteMlx(
            readiness: ReadinessOverrides(maxWaitSecs: 1800, skipThermal: true)))
    }

    /// Resolved against one cell, as `runtime-flags=` is: several benchmarks, or a
    /// `metrics=` ladder, may be cells of different types.
    @Test(arguments: [
        ["benchmarks=prefill_throughput_256,decode_throughput_256_16"],
        ["metrics=prefill,decode"],
    ])
    func sweepReadinessNeedsExactlyOneBenchmark(_ selection: [String]) {
        let refused = refusal(["headlessrun", "runtime=llama", "model=\(Self.ggufModel)"]
            + selection + ["readiness-skip-thermal=1"])
        guard case let .rejected(key, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))"); return
        }
        #expect(key == "readiness-skip-thermal")
        #expect(reason.hasPrefix("needs exactly one"), "reason was: \(reason)")
    }

    /// A `match=` resolves against the catalog at run time, so the cell's type is not known
    /// here, as for `runtime-flags=`.
    @Test(arguments: [
        ["headlessrun", "runtime=llama"],
        ["headlessrun", "runtime=llama", "match=LFM"],
        ["headlessrun", "bench", "runtime=llama", "match=LFM"],
    ])
    func sweepReadinessNeedsAnExactModel(_ form: [String]) {
        let refused = refusal(form + ["benchmarks=prefill_throughput_256",
                                      "readiness-skip-thermal=1"])
        guard case let .rejected(key, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))"); return
        }
        #expect(key == "readiness-skip-thermal")
        #expect(reason.hasPrefix("needs an exact model"), "reason was: \(reason)")
    }

    /// `readiness-skip-thermal=0` overrides nothing, so a ladder may carry it.
    @Test func anUnsetSkipThermalIsAcceptedOnALadder() {
        #expect(refusal("headlessrun", "runtime=llama", "metrics=prefill,decode",
                        "readiness-skip-thermal=0") == nil)
    }

    /// Refused on Apple Foundation, as the crate refuses it: plan-types declares no AFM
    /// prefill variant, so the result could not record a waived thermal check.
    @Test(arguments: [
        ["headlessrun", "runtime=afm", "benchmarks=prefill_throughput_256"],
        ["headlessrun", "bench", "runtime=afm", "benchmarks=prefill_throughput_256"],
        ["headlessrun", "benchmarks", "run", "runtime=afm", "benchmark=prefill_throughput_256",
         #"model={"type":"apple_foundation_text"}"#],
    ])
    func readinessOverridesAreRefusedOnAppleFoundation(_ form: [String]) {
        let refused = refusal(form + ["readiness-skip-thermal=1"])
        guard case let .rejected(key, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))"); return
        }
        #expect(key == "readiness-skip-thermal")
        #expect(reason.hasPrefix("not valid for this benchmark"), "reason was: \(reason)")
    }

    /// The one benchmark still needs a variant with a readiness field, as on `benchmarks run`.
    @Test func sweepReadinessIsRefusedOnAnUngatedBenchmark() {
        let refused = refusal("headlessrun", "runtime=llama", "model=\(Self.ggufModel)",
                              "benchmarks=max_memory_usage_256", "readiness-max-wait-secs=30")
        guard case let .rejected(key, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))"); return
        }
        #expect(key == "readiness-max-wait-secs")
        #expect(reason.hasPrefix("not valid for this benchmark"), "reason was: \(reason)")
    }

    // MARK: - `benchmark-flags=`

    /// On the one-cell form and on the bare form the proxy emits with a claim's
    /// `benchmark_flags`.
    @Test(arguments: [
        ["headlessrun", "benchmarks", "run", "benchmark=prefill_throughput_256",
         "model=gguf-text://repo=o/r&path=m.gguf", "runtime=llama"],
        ["headlessrun", "runtime=llama", "model=gguf-text://repo=o/r&path=m.gguf",
         "benchmarks=prefill_throughput_256"],
    ])
    func benchmarkFlagsBecomeTheCellsBenchmarkFlags(_ form: [String]) {
        let parsed = parse(form + [
            #"benchmark-flags={"readiness":{"max_wait_secs":1800,"skip_thermal":true}}"#,
        ])
        let flags: BenchmarkFlags?
        if case let .benchmarksRun(_, _, _, _, _, _, runFlags, _)? = parsed {
            flags = runFlags
        } else {
            flags = sweepFlags(parsed)
        }
        #expect(flags == .prefillThroughputLlamacppIosPipetteGgufText(
            readiness: ReadinessOverrides(maxWaitSecs: 1800, skipThermal: true)))
    }

    /// The attempt limit has no shorthand; `benchmark-flags=` carries it, on a memory cell too.
    @Test func benchmarkFlagsCarryTheAttemptLimit() {
        let parsed = parse("headlessrun", "benchmarks", "run", "benchmark=max_memory_usage_256",
                           "model=\(Self.ggufModel)", "runtime=llama",
                           #"benchmark-flags={"attempt_timeout_seconds":900}"#)
        guard case let .benchmarksRun(_, _, _, _, _, _, flags, _)? = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))"); return
        }
        #expect(flags == .maxMemoryUsageLlamacppIosPipetteGgufText(attemptTimeoutSeconds: 900))
    }

    /// Apple Foundation's decode and end-to-end cells take benchmark flags, and the AFM
    /// command carries them to its cells rather than dropping them.
    @Test func appleFoundationCarriesItsBenchmarkFlags() {
        let parsed = parse("headlessrun", "runtime=afm", "benchmarks=decode_throughput_256_100",
                           #"benchmark-flags={"attempt_timeout_seconds":600,"readiness":{"max_wait_secs":900}}"#)
        guard case let .afm(_, _, _, _, _, flags)? = parsed else {
            Issue.record("expected afm, got \(String(describing: parsed))"); return
        }
        #expect(flags == .decodeThroughputAppleFoundationAppleFoundationText(
            attemptTimeoutSeconds: 600, readiness: ReadinessOverrides(maxWaitSecs: 900)))
    }

    /// As the CLI's `conflicts_with_all` refuses them.
    @Test(arguments: ["readiness-max-wait-secs=30", "readiness-skip-thermal=1"])
    func benchmarkFlagsAndItsShorthandAreRefusedTogether(shorthand: String) {
        let key = String(shorthand.prefix { $0 != "=" })
        #expect(refusal("headlessrun", "benchmarks", "run", "benchmark=prefill_throughput_256",
                        "model=\(Self.ggufModel)", "runtime=llama",
                        #"benchmark-flags={"readiness":{"max_wait_secs":30}}"#, shorthand)
            == .rejected(key: "benchmark-flags", reason: "is mutually exclusive with `\(key)=`"))
    }

    @Test(arguments: [
        (#"[{"readiness":{"max_wait_secs":30}}]"#, "prefill_throughput_256",
         "must be a JSON object"),
        (#"{"benchmark_type":"prefill_throughput"}"#, "prefill_throughput_256",
         "must not carry `benchmark_type`"),
        (#"{"http_timeout_seconds":600}"#, "prefill_throughput_256",
         "no iOS variant declares `http_timeout_seconds`"),
        (#"{"readiness":{"max_wait_secs":30}}"#, "eval_smoke", "not valid for this benchmark"),
        (#"{"readiness":{"max_wait_secs":30}}"#, "max_memory_usage_256",
         "not valid for this benchmark: readiness is not a valid benchmark flag"),
        (#"{"readiness":{"max_wait_secs":0}}"#, "prefill_throughput_256",
         "`readiness.max_wait_secs` must be at least 1"),
    ])
    func benchmarkFlagsIsRefusedByName(json: String, benchmark: String, reason: String) {
        let refused = refusal("headlessrun", "benchmarks", "run", "benchmark=\(benchmark)",
                              "model=\(Self.ggufModel)", "runtime=llama",
                              "benchmark-flags=\(json)")
        guard case let .rejected(key, got) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))"); return
        }
        #expect(key == "benchmark-flags")
        #expect(got.hasPrefix(reason), "reason was: \(got)")
    }

    /// `model=` takes the two forms the crate's `--model` takes.
    @Test(arguments: [
        "gguf-text://repo=LiquidAI/LFM2-350M-GGUF&path=LFM2-350M-Q4_K_M.gguf",
        #"{"type":"gguf_text","source":"huggingface","org":"LiquidAI","#
            + #""repo_name":"LFM2-350M-GGUF","path":"LFM2-350M-Q4_K_M.gguf"}"#,
    ])
    func benchmarksRunTakesAUriOrJsonModel(_ model: String) {
        let parsed = parse("headlessrun", "benchmarks", "run",
                           "benchmark=decode_throughput_512_100",
                           "model=\(model)", "runtime=llama")

        guard case let .benchmarksRun(benchmark, resolved, runtime, _, _, sync, _, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(benchmark == .remote("decode_throughput_512_100"))
        #expect(resolved.namedModel?.repo?.description == "LiquidAI/LFM2-350M-GGUF")
        #expect(runtime == .llamacppIosPipette)
        #expect(!sync, "`sync=` is opt-in, as `--sync` is")
    }

    /// Running a cell that asked for another build and recording what actually ran would
    /// answer a question nobody asked.
    @Test func aRuntimeNamingAnotherBuildIsRefused() throws {
        let builtIn = try #require(Runtime.thisBuild(for: .llamacppIosPipette))
        let other = Runtime.llamacppIosPipette(
            source: SourceRepository(repositoryUrl: RepositoryUrl(LlamaPins.main.repositoryUrl),
                                     repositoryVersion: NonEmptyString(validated: "b0001")),
            flavor: .iosArm64, privateApis: .stock)
        #expect(other != builtIn, "the fixture has to differ for the check to mean anything")

        let refused = refusal("headlessrun", "benchmarks", "run", "--benchmark", "b",
                              "--model", "mlx://repo=o/r",
                              "--runtime", try SubmissionRef.runtime(other))

        guard case let .rejected(key, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))")
            return
        }
        #expect(key == "runtime")
        #expect(reason.contains("b0001") && reason.contains(LlamaPins.main.submissionVersion),
                "the refusal names both builds; was: \(reason)")
    }

    /// The identity `runtimes` advertises is encoded from the same value, so it passes.
    @Test func theAdvertisedRuntimeIdentityIsAccepted() throws {
        let builtIn = try #require(Runtime.thisBuild(for: .llamacppIosPipette))

        let parsed = parse("headlessrun", "benchmarks", "run", "--benchmark", "b",
                           "--model", "gguf-text://repo=o/r&path=m.gguf",
                           "--runtime", try SubmissionRef.runtime(builtIn))

        guard case let .benchmarksRun(_, _, runtime, _, _, _, _, llamaPin) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(runtime == .llamacppIosPipette)
        #expect(llamaPin == LlamaPins.main)
    }

    // A test bundle carries main alone, so these pass the carried builds in.
    private static let mainPin = LlamaPin(
        name: "llamacpp", repositoryUrl: "github.com/ggml-org/llama.cpp",
        build: .tagged(tag: "b11988", commit: "abb76ab38"))
    private static let asrPin = LlamaPin(
        name: "llamacpp-asr", repositoryUrl: "github.com/example-org/llama.cpp-asr",
        build: .untagged(commit: "2898bdc6f"))

    /// A descriptor naming a variant picks that variant, not main.
    @Test func aRuntimePicksTheCarriedVariantItNames() throws {
        let pin = try HeadlessCommand.carriedLlamaPin(Runtime.carried(Self.asrPin),
                                                      carried: [Self.mainPin, Self.asrPin])
        #expect(pin == Self.asrPin)
    }

    /// Two builds of one repository: the version in the descriptor picks between them.
    @Test func aRuntimePicksBetweenTwoBuildsOfOneRepository() throws {
        let url = "github.com/PrismML-Eng/llama.cpp"
        let older = LlamaPin(name: "llamacpp-prismml", repositoryUrl: url,
                             build: .tagged(tag: "prism-b10777-e3de715", commit: "e3de71522"))
        let newer = LlamaPin(name: "llamacpp-prismml-b10828", repositoryUrl: url,
                             build: .tagged(tag: "prism-b10828-a153697", commit: "a15369706"))
        let carried = [Self.mainPin, older, newer]
        #expect(try HeadlessCommand.carriedLlamaPin(Runtime.carried(older), carried: carried) == older)
        #expect(try HeadlessCommand.carriedLlamaPin(Runtime.carried(newer), carried: carried) == newer)
    }

    /// A descriptor no carried build matches is refused with every carried build named, and
    /// one that two carried builds match is refused with both names.
    @Test(arguments: [
        (LlamaPin(name: "llamacpp-other", repositoryUrl: "github.com/someone/llama.cpp",
                  build: .untagged(commit: "000000000")),
         [mainPin, asrPin],
         ["000000000", "b11988", "2898bdc6f"]),
        (mainPin,
         [mainPin, LlamaPin(name: "llamacpp-copy", repositoryUrl: mainPin.repositoryUrl,
                            build: mainPin.build)],
         ["llamacpp and llamacpp-copy both are"]),
    ])
    func aRuntimeNoSingleCarriedBuildMatchesIsRefused(
        declared: LlamaPin, carried: [LlamaPin], needles: [String]
    ) {
        do {
            let pin = try HeadlessCommand.carriedLlamaPin(Runtime.carried(declared),
                                                          carried: carried)
            Issue.record("expected a refusal, got \(pin.name)")
        } catch let .rejected(key, reason) {
            #expect(key == "runtime")
            #expect(needles.allSatisfy(reason.contains), "was: \(reason)")
        } catch {
            Issue.record("expected .rejected, got \(error)")
        }
    }

    /// The form `--runtime` takes for a runtime with no URI spelling. Uses the built-in MLX
    /// identity, since any other is refused.
    @Test func benchmarksRunTakesAJsonRuntime() throws {
        let builtIn = try #require(Runtime.thisBuild(for: .mlxIosPipette))

        let parsed = parse("headlessrun", "benchmarks", "run", "--benchmark", "eval_ifbench",
                           "--model", "mlx://repo=mlx-community/LFM2-350M-4bit",
                           "--runtime", try SubmissionRef.runtime(builtIn))

        guard case let .benchmarksRun(_, _, resolved, _, _, _, _, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(resolved == .mlxIosPipette)
    }

    /// plan-types marks the on-device runtimes `NotRepresentable` as URIs, and every
    /// representable scheme names a desktop runtime this build cannot be.
    @Test(arguments: ["llamacpp-ios-pipette://flavor=ios-arm64",
                      "llamacpp-cli-stock-tools://version=b5000&flavor=macos-arm64"])
    func benchmarksRunRefusesARuntimeUri(_ uri: String) {
        let refused = refusal("headlessrun", "benchmarks", "run", "benchmark=b",
                              "model=mlx://repo=o/r", "runtime=\(uri)")

        guard case let .rejected(key, reason) = refused else {
            Issue.record("expected a runtime refusal, got \(String(describing: refused))")
            return
        }
        #expect(key == "runtime")
        #expect(reason.contains("no URI form"), "reason was: \(reason)")
    }

    @Test func benchmarksRunRequiresAModel() {
        #expect(refusal("headlessrun", "benchmarks", "run", "benchmark=b", "runtime=llama")
            == .missingParameter("model"))
    }

    /// Carried, not resolved, while parsing: the store is the runner's to read.
    @Test func aDigestReferenceIsCarriedForTheRunnerToResolve() {
        let parsed = parse("headlessrun", "benchmarks", "run", "benchmark=b",
                           "model=model://sha256=A1B2C3D4E5F6", "runtime=llama")

        guard case let .benchmarksRun(_, model, _, _, _, _, _, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(model == .digest("a1b2c3d4e5f6"))
    }

    @Test(arguments: [("model://sha256=a1b2", "too short"), ("model://sha256=zzzzzzzz", "not hex"),
                      ("model://a1b2c3d4e5f6", "takes a digest")])
    func aMalformedDigestReferenceIsRefused(_ raw: String, _ reason: String) {
        let refused = refusal("headlessrun", "benchmarks", "run", "benchmark=b",
                              "model=\(raw)", "runtime=llama")

        guard case let .rejected(key, message) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))")
            return
        }
        #expect(key == "model")
        #expect(message.contains(reason), "message was: \(message)")
    }

    @Test func theCrateSpellingOfTheKeysParsesToTheSameCommand() {
        let dashed = parse("headlessrun", "benchmarks", "run",
                           "--benchmark", "decode_throughput_512_100",
                           "--model=mlx://repo=mlx-community/LFM2-350M-4bit",
                           "--runtime", "mlx", "--sync")
        let equals = parse("headlessrun", "benchmarks", "run",
                           "benchmark=decode_throughput_512_100",
                           "model=mlx://repo=mlx-community/LFM2-350M-4bit",
                           "runtime=mlx", "sync=1")

        #expect(dashed == equals)
        guard case let .benchmarksRun(_, _, runtime, _, _, sync, _, _) = dashed else {
            Issue.record("expected benchmarksRun, got \(String(describing: dashed))")
            return
        }
        #expect(runtime == .mlxIosPipette)
        #expect(sync, "a bare `--sync` is the boolean form")
    }

    @Test func verbsSurviveAlongsideDashedFlags() {
        let parsed = parse("headlessrun", "models", "rm", "--name", "a-Q4_0.gguf")

        #expect(parsed == .modelsRm(name: "a-Q4_0.gguf", repo: nil))
    }

    @Test func modelsListTakesTheFormatTheCrateTakes() {
        #expect(parse("headlessrun", "models") == .modelsList(format: .name))
        #expect(parse("headlessrun", "models", "--format", "uri") == .modelsList(format: .uri))
        #expect(refusal("headlessrun", "models", "--format", "table")
            == .invalidValue(key: "format", value: "table"))
    }

    @Test func modelsPullAndDeleteTakeAModelReference() throws {
        let uri = "mlx://repo=mlx-community/LFM2-350M-4bit"
        #expect(parse("headlessrun", "models", "pull", "--model", uri)
            == .modelsPull(.model(try parseModelUri(uri))))
        #expect(parse("headlessrun", "models", "delete", "--model=model://sha256=b121b700be77")
            == .modelsDelete(.digest("b121b700be77")))
        #expect(refusal("headlessrun", "models", "pull") == .missingParameter("model"))
    }

    /// `spec=` is accepted only on the sweep forms; these leaves take the crate's `--model`.
    @Test func modelsPullAndDeleteRefuseTheFormerSpecName() {
        let uri = "mlx://repo=mlx-community/LFM2-350M-4bit"
        #expect(refusal("headlessrun", "models", "pull", "spec=\(uri)")
            == .unknownParameters(["spec"]))
        #expect(refusal("headlessrun", "models", "delete", "spec=\(uri)")
            == .unknownParameters(["spec"]))
    }

    // MARK: - The auth and diag groups

    /// A type tag names the engine but not the build, so nothing can be checked against this
    /// binary. The plan emits the canonical `Runtime` instead.
    @Test(arguments: ["llamacpp_ios_pipette", "mlx_ios_pipette", "apple_foundation"])
    func aBareRuntimeTypeTagIsRefused(_ tag: String) {
        guard case let .rejected(key, reason) = refusal("headlessrun", "runtime=\(tag)",
                                                        "benchmarks=decode_throughput_256") else {
            Issue.record("expected a rejection for `runtime=\(tag)`")
            return
        }
        #expect(key == "runtime")
        #expect(reason.contains("not a build"))
    }

    /// The short tokens are hand-typed shorthand, never a plan spelling, so they claim no
    /// build identity.
    @Test func theShortTokensStillSelectAnEngine() {
        let parsed = parse("headlessrun", "runtime=llama", "benchmarks=decode_throughput_256")

        guard case let .bareBench(runtime, _, _, _, benchmarks, _, _, _) = parsed else {
            Issue.record("expected a bareBench, got \(String(describing: parsed))")
            return
        }
        #expect(runtime == .llamacppIosPipette)
        #expect(benchmarks == ["decode_throughput_256"])
    }

    @Test func authRegisterTakesTheCratesParameterNames() throws {
        #expect(parse(["headlessrun", "auth", "register", "--organization", "Liquid",
                       "--contact-email", "a@b.co", "--client-details", "ci box"])
            == (try registerInput()))
    }

    @Test func authRegisterTakesADeviceName() throws {
        #expect(parse("headlessrun", "auth", "register", "organization=Liquid",
                      "contact-email=a@b.co", "client-details=lab rig 3",
                      "device-name=boston-17-pro-1")
            == (try registerInput(clientDetails: "lab rig 3", deviceName: "boston-17-pro-1")))
    }

    @Test func authRegisterTakesAPreauthKey() throws {
        #expect(parse("headlessrun", "auth", "register", "organization=Liquid",
                      "contact-email=a@b.co", "client-details=ci box",
                      "preauth-key=preauth_abc.def")
            == (try registerInput(preauth: "preauth_abc.def")))
    }

    @Test func authRegisterStillRequiresOrgAndEmail() {
        #expect(refusal("headlessrun", "auth", "register") == .missingParameter("organization"))
    }

    @Test func authMeAndResetParse() {
        #expect(parse("headlessrun", "auth", "me") == .authMe(local: false))
        #expect(parse("headlessrun", "auth", "me", "local=1") == .authMe(local: true))
        #expect(parse("headlessrun", "auth", "me", "local=0") == .authMe(local: false))
        #expect(parse("headlessrun", "auth", "reset") == .authReset(force: false))
        #expect(parse("headlessrun", "auth", "reset", "force=1") == .authReset(force: true))
    }

    @Test func authSetDeviceTakesTheCratesThreeFields() {
        #expect(parse("headlessrun", "auth", "set-device", "device-name=Bench 4")
            == .authSetDevice(deviceName: "Bench 4", deviceFormFactor: nil, clientDetails: nil))
        #expect(parse("headlessrun", "auth", "set-device", "device-form-factor=tablet")
            == .authSetDevice(deviceName: nil, deviceFormFactor: .tablet, clientDetails: nil))
        #expect(parse("headlessrun", "auth", "set-device", "client-details=rack 2 slot 7")
            == .authSetDevice(deviceName: nil, deviceFormFactor: nil, clientDetails: "rack 2 slot 7"))
        #expect(parse("headlessrun", "auth", "set-device", "device-name=Bench 4",
                      "device-form-factor=phone", "client-details=fleet")
            == .authSetDevice(deviceName: "Bench 4", deviceFormFactor: .phone, clientDetails: "fleet"))
    }

    /// The crate's `SetDeviceEmpty`: a relabel that names no label reads as a typo.
    @Test func authSetDeviceRefusesTheEmptyInvocation() {
        #expect(refusal("headlessrun", "auth", "set-device")
            == .rejected(key: "device-name",
                         reason: "set-device needs at least one of `device-name=`, "
                             + "`device-form-factor=` or `client-details=`"))
    }

    /// Caught here rather than written to `device.json` and sent to a server that refuses it.
    @Test func authSetDeviceRefusesAnUnknownFormFactor() {
        #expect(refusal("headlessrun", "auth", "set-device", "device-form-factor=fridge")
            == .invalidValue(key: "device-form-factor", value: "fridge"))
    }

    /// Identity work never rides on a URL any web page can open.
    @Test func authSetDeviceIsCommandLineOnly() {
        #expect(parse("headlessrun", "auth", "set-device", "device-name=X")?
            .allowedViaDeepLink == false)
    }

    @Test func authExportAndImportParse() {
        #expect(parse("headlessrun", "auth", "export") == .authExport)
        #expect(parse("headlessrun", "auth", "import", "bundle=abc")
            == .authImport(bundle: "abc", force: false))
        #expect(parse("headlessrun", "auth", "import", "bundle=abc", "force=1")
            == .authImport(bundle: "abc", force: true))
        // `force` is a value-or-flag parameter, as `auth reset`'s is.
        #expect(parse("headlessrun", "auth", "import", "bundle=abc", "--force")
            == .authImport(bundle: "abc", force: true))
    }

    @Test func authImportRequiresABundle() {
        #expect(refusal("headlessrun", "auth", "import") == .missingParameter("bundle"))
        #expect(refusal("headlessrun", "auth", "import", "bundle=") == .missingParameter("bundle"))
        #expect(refusal("headlessrun", "auth", "import", "force=1")
            == .missingParameter("bundle"))
    }

    /// An operator who passed a parameter expects it to do something.
    @Test func authExportTakesNoParameters() {
        #expect(refusal("headlessrun", "auth", "export", "bundle=abc")
            == .unknownParameters(["bundle"]))
    }

    /// A link must not print a signing key, nor replace one.
    @Test func authExportAndImportAreNotDeepLinkable() {
        #expect(parse("headlessrun", "auth", "export")?.allowedViaDeepLink == false)
        #expect(parse("headlessrun", "auth", "import", "bundle=abc")?.allowedViaDeepLink
            == false)
    }

    @Test func authNeedsALeaf() {
        #expect(refusal("headlessrun", "auth") == .unknownVerb("auth"))
        #expect(refusal("headlessrun", "auth", "logout") == .unknownVerb("auth logout"))
    }

    /// Neither may demand a registration up front.
    @Test func authExportAndImportDoNotContribute() {
        #expect(parse("headlessrun", "auth", "export")?.submitsResults == false)
        #expect(parse("headlessrun", "auth", "import", "bundle=abc")?.submitsResults == false)
    }

    @Test(arguments: HeadlessCommand.ProbeKind.allCases)
    func diagProbeNamesWhatItChecks(kind: HeadlessCommand.ProbeKind) {
        #expect(parse("headlessrun", "diag", "probe", "kind=\(kind.rawValue)")
            == .diagProbe(kind: kind))
    }

    @Test func diagProbeRefusesAnUnknownKind() {
        #expect(refusal("headlessrun", "diag", "probe", "kind=vibes")
            == .invalidValue(key: "kind", value: "vibes"))
        #expect(refusal("headlessrun", "diag", "probe") == .missingParameter("kind"))
    }

    @Test func authAndDiagDoNotContribute() {
        #expect(parse("headlessrun", "auth", "me")?.submitsResults == false)
        #expect(parse("headlessrun", "diag", "probe", "kind=token")?.submitsResults == false)
    }

    // MARK: - results

    @Test func resultsListTakesTheCliFilters() {
        #expect(parse("headlessrun", "results") == .resultsList(
            benchmark: nil, benchmarkType: nil, state: nil, limit: nil))
        // A bare id means the synced catalog, as it does everywhere a reference is taken.
        #expect(parse("headlessrun", "results", "benchmark=prefill_throughput_512",
                      "type=prefill_throughput", "state=submitted", "limit=5")
            == .resultsList(benchmark: .remote("prefill_throughput_512"),
                            benchmarkType: .prefillThroughput, state: .submitted, limit: 5))
    }

    @Test func resultsListRefusesAnUnknownFilterValue() {
        #expect(refusal("headlessrun", "results", "state=posted")
            == .invalidValue(key: "state", value: "posted"))
        #expect(refusal("headlessrun", "results", "type=prefill")
            == .invalidValue(key: "type", value: "prefill"))
        #expect(refusal("headlessrun", "results", "limit=lots")
            == .invalidValue(key: "limit", value: "lots"))
        // `prefix(-1)` traps.
        #expect(refusal("headlessrun", "results", "limit=-1")
            == .invalidValue(key: "limit", value: "-1"))
        #expect(parse("headlessrun", "results", "limit=0")
            == .resultsList(benchmark: nil, benchmarkType: nil, state: nil, limit: 0))
    }

    /// The benchmark is carried as a whole `SourcedBenchmarkId`, so the filter compares the
    /// address the crate's does.
    @Test func resultsListTakesTheCratesSpellings() {
        #expect(parse("headlessrun", "results", "type=prefill-throughput",
                      "benchmark=remote/prefill_throughput_512")
            == .resultsList(benchmark: .remote("prefill_throughput_512"),
                            benchmarkType: .prefillThroughput, state: nil, limit: nil))
        // The named side survives the parse: the two halves are different filters.
        #expect(parse("headlessrun", "results", "benchmark=local/prefill_throughput_smoke")
            == .resultsList(benchmark: .local("prefill_throughput_smoke"),
                            benchmarkType: nil, state: nil, limit: nil))
        #expect(refusal("headlessrun", "results", "benchmark=local/a/b")
            == .rejected(key: "benchmark",
                         reason: SourcedBenchmarkIdError.shape("local/a/b").localizedDescription))
    }

    /// iOS results live under their job, so a result is addressed by the pair.
    @Test func resultsShowAndDeleteTakeTheJobCellPair() {
        let id = ResultId(jobId: JobId("job-1"), cellId: CellId("cell-2"))
        #expect(parse("headlessrun", "results", "show", "result=job-1/cell-2") == .resultsShow(resultId: id))
        #expect(parse("headlessrun", "results", "delete", "result=job-1/cell-2")
            == .resultsDelete(resultId: id))
    }

    @Test(arguments: ["job-1", "job-1/", "/cell-2", "job-1/cell-2/extra", ""])
    func resultsShowRefusesAMalformedAddress(raw: String) {
        #expect(refusal("headlessrun", "results", "show", "result=\(raw)") != nil)
    }

    @Test func resultsNeedsAKnownLeaf() {
        #expect(refusal("headlessrun", "results", "purge") == .unknownVerb("results purge"))
    }

    @Test func readingResultsDoesNotContribute() {
        #expect(parse("headlessrun", "results")?.submitsResults == false)
    }

    // MARK: - sync

    @Test func syncTakesAnOptionalJobNarrowing() {
        #expect(parse("headlessrun", "sync") == .sync(jobId: nil, serverUrl: nil))
        #expect(parse("headlessrun", "sync", "job=abc") == .sync(jobId: "abc", serverUrl: nil))
    }

    /// The catalog is a public read, so an unregistered device can name where it reads from.
    @Test func syncTakesAServerUrlForTheCatalogPull() {
        #expect(parse("headlessrun", "sync", "server-url=https://mgmt.example.com")
            == .sync(jobId: nil, serverUrl: "https://mgmt.example.com"))
    }

    /// Refused at parse time, as clap refuses a `ServerUrl`. The scheme is matched, not the
    /// prefix `http`, which admits `httpfoo://` and a bare `http`.
    @Test func syncRefusesAServerUrlThatCannotAddressAServer() {
        let reason = "expected an http:// or https:// URL"
        for raw in ["mgmt.example.com", "httpfoo://mgmt.example.com", "http", "https",
                    "ftp://mgmt.example.com"] {
            #expect(refusal("headlessrun", "sync", "server-url=\(raw)")
                == .rejected(key: "server-url", reason: reason), "\(raw)")
        }
    }

    /// A local collector is reached over plain HTTP.
    @Test func syncAcceptsEitherHttpScheme() {
        #expect(parse("headlessrun", "sync", "server-url=http://localhost:8080")
            == .sync(jobId: nil, serverUrl: "http://localhost:8080"))
        #expect(parse("headlessrun", "sync", "server-url=HTTPS://mgmt.example.com")
            == .sync(jobId: nil, serverUrl: "HTTPS://mgmt.example.com"))
    }

    /// Gating it would refuse the catalog-only form; `SyncCommands` decides whether an
    /// identity exists to submit with.
    @Test func syncIsNotGatedAsASubmittingCommand() {
        #expect(parse("headlessrun", "sync")?.submitsResults == false)
    }

    /// The CLI narrows `sync` by result id; iOS addresses whole jobs, so a CLI habit gets a
    /// real answer instead of "unknown parameter".
    @Test func syncRefusesAResultNarrowingItCannotAddress() {
        #expect(refusal("headlessrun", "sync", "result=abc/cell-1")
            == .rejected(key: "result",
                         reason: "a result is not addressable yet; pass `job=<jobId>`"))
    }

    /// The crate's `sync [RESULT_ID]`: a bare word after `sync` is refused as `result=` is.
    @Test func syncReadsTheCratesPositionalAsAResult() {
        let refused = HeadlessUsageError.rejected(
            key: "result", reason: "a result is not addressable yet; pass `job=<jobId>`")
        #expect(refusal("headlessrun", "sync", "now") == refused)
        #expect(refusal("headlessrun", "sync", "abc/cell-1") == refused)
        #expect(refusal("headlessrun", "sync", "--server-url", "https://mgmt.example.com",
                        "abc/cell-1") == refused)
        #expect(parse("headlessrun", "sync", "--server-url", "https://mgmt.example.com", "job=abc")
            == .sync(jobId: "abc", serverUrl: "https://mgmt.example.com"))
    }

    // MARK: - The inspect verbs

    @Test func inspectVerbsParseWithoutParameters() {
        #expect(parse("headlessrun", "runtimes") == .runtimesList(format: .name))
        #expect(parse("headlessrun", "benchmarks") == .benchmarksList(benchmarkType: nil))
        #expect(parse("headlessrun", "storage", "status") == .storageStatus)
    }

    @Test func theCratesListSpellingIsAccepted() {
        #expect(parse("headlessrun", "models", "list") == parse("headlessrun", "models"))
        #expect(parse("headlessrun", "benchmarks", "list") == parse("headlessrun", "benchmarks"))
        #expect(parse("headlessrun", "results", "list") == parse("headlessrun", "results"))
        #expect(parse("headlessrun", "runtimes", "list") == .runtimesList(format: .name))
        #expect(parse("headlessrun", "models", "list", "format=uri") == .modelsList(format: .uri))
        #expect(parse("headlessrun", "runtimes", "list", "format=uri")
            == .runtimesList(format: .uri))
        #expect(parse("headlessrun", "benchmarks", "list", "benchmark-type=eval")
            == .benchmarksList(benchmarkType: .eval))
        #expect(refusal("headlessrun", "runtimes", "list", "oops")
            == .unknownVerb("runtimes list oops"))
    }

    @Test func versionParsesAsABareVerb() {
        #expect(parse("headlessrun", "version") == .version)
        #expect(refusal("headlessrun", "version", "oops") == .unknownVerb("version oops"))
    }

    @Test func theCratesWorkerVerbStartsTheClaimLoop() {
        #expect(parse("headlessrun", "worker") == .worker(WorkerArgs()))
        #expect(parse("headlessrun", "worker") == parse("headlessrun", "settings", "run"))
        #expect(refusal("headlessrun", "worker", "run") == .unknownVerb("worker run"))
    }

    @Test func workerTakesTheCratesKnobs() {
        #expect(parse("headlessrun", "worker", "idle-secs=30", "idle-jitter-secs=5",
                      "heartbeat-secs=12", "max-jobs=3", "skip-profile-refresh=1")
            == .worker(WorkerArgs(idleSecs: 30, idleJitterSecs: 5,
                                     heartbeatSecs: 12, maxJobs: 3,
                                     skipProfileRefresh: true)))
        #expect(parse("headlessrun", "worker", "--skip-profile-refresh")
            == .worker(WorkerArgs(skipProfileRefresh: true)))
    }

    /// Defaulting would let the run proceed without the pacing the caller asked for.
    @Test func workerRefusesUnusableKnobValues() {
        #expect(refusal("headlessrun", "worker", "idle-secs=soon")
            == .invalidValue(key: "idle-secs", value: "soon"))
        #expect(refusal("headlessrun", "worker", "max-jobs=-1")
            == .invalidValue(key: "max-jobs", value: "-1"))
        // The crate's `range(1..)`: a zero-second heartbeat spins instead of renewing.
        #expect(refusal("headlessrun", "worker", "heartbeat-secs=0")
            == .invalidValue(key: "heartbeat-secs", value: "0"))
    }

    /// The crate's `u64`. As `Double`s, `inf` and `1e400` pass a `>= 0` guard and then trap
    /// in `Duration.seconds` or `random(in:)`, crashing the app on a launch argument.
    @Test func workerRefusesValuesThatAreNotWholeSeconds() {
        for key in ["idle-secs", "idle-jitter-secs", "heartbeat-secs"] {
            for raw in ["inf", "1e400", "nan", "3.5", "1e5", "-1"] {
                #expect(refusal("headlessrun", "worker", "\(key)=\(raw)")
                    == .invalidValue(key: key, value: raw), "\(key)=\(raw)")
            }
        }
    }

    @Test func storageGcTakesADryRunFlag() {
        #expect(parse("headlessrun", "storage", "gc") == .storageGc(dryRun: false))
        #expect(parse("headlessrun", "storage", "gc", "dry-run=1") == .storageGc(dryRun: true))
        #expect(parse("headlessrun", "storage", "gc", "dry-run=0") == .storageGc(dryRun: false))
    }

    @Test func benchmarksTakesATypeFilter() {
        #expect(parse("headlessrun", "benchmarks", "benchmark-type=eval")
            == .benchmarksList(benchmarkType: .eval))
        #expect(parse("headlessrun", "benchmarks", "benchmark-type=prefill_throughput")
            == .benchmarksList(benchmarkType: .prefillThroughput))
        #expect(parse("headlessrun", "benchmarks", "--benchmark-type", "prefill-throughput")
            == .benchmarksList(benchmarkType: .prefillThroughput))
        // `type=` is `results list`'s spelling, not this leaf's.
        #expect(refusal("headlessrun", "benchmarks", "type=eval") == .unknownParameters(["type"]))
    }

    /// A misspelled type would otherwise filter every benchmark out and report an empty
    /// catalog, which reads as "nothing synced" rather than "you typed it wrong".
    @Test func benchmarksRefusesAnUnknownType() {
        #expect(refusal("headlessrun", "benchmarks", "benchmark-type=prefill")
            == .invalidValue(key: "benchmark-type", value: "prefill"))
    }

    @Test func benchmarksInitLocalParses() {
        #expect(parse("headlessrun", "benchmarks", "init-local") == .benchmarksInitLocal)
        #expect(refusal("headlessrun", "benchmarks", "init-local", "extra")
            == .unknownVerb("benchmarks init-local extra"))
    }

    @Test func benchmarksShowTakesAReference() {
        #expect(parse("headlessrun", "benchmarks", "show", "benchmark=prefill_throughput_512")
            == .benchmarksShow(.remote("prefill_throughput_512")))
        #expect(parse("headlessrun", "benchmarks", "show", "benchmark=local/prefill_throughput_512")
            == .benchmarksShow(.local("prefill_throughput_512")))
        #expect(refusal("headlessrun", "benchmarks", "show", "benchmark=elsewhere/x")
            == .rejected(key: "benchmark",
                         reason: SourcedBenchmarkIdError.shape("elsewhere/x").localizedDescription))
        #expect(refusal("headlessrun", "benchmarks", "show") == .missingParameter("benchmark"))
    }

    /// `runtimes pull` asks to install a runtime, a different question from "what is
    /// compiled in", so it must not fall back to the bare list.
    @Test func inspectVerbsRefuseUnknownLeaves() {
        #expect(refusal("headlessrun", "storage") == .unknownVerb("storage"))
        #expect(refusal("headlessrun", "storage", "frobnicate") == .unknownVerb("storage frobnicate"))
        #expect(refusal("headlessrun", "runtimes", "pull") == .unknownVerb("runtimes pull"))
    }

    @Test(arguments: [("jobs", "frobnicate"), ("status", "now"), ("bench", "oops"),
                      ("runtimes", "pull")])
    func aStrayLeafOnAFlatVerbIsRefused(verb: String, leaf: String) {
        #expect(refusal("headlessrun", verb, leaf) == .unknownVerb("\(verb) \(leaf)"))
    }

    @Test(arguments: [("models rm", ["models", "rm", "extra"]),
                      ("job submit", ["job", "submit", "twice"]),
                      ("settings run", ["settings", "run", "hard"]),
                      ("auth register", ["auth", "register", "now"]),
                      ("diag memseq", ["diag", "memseq", "x"])])
    func aStrayWordBeyondAGroupedVerbIsRefused(path: String, tokens: [String]) {
        switch HeadlessCommand.parse(["/app/Pipette", "headlessrun"] + tokens) {
        case .success:
            Issue.record("\(tokens) should not parse")
        case let .failure(error):
            #expect(error == .unknownVerb("\(path) \(tokens.last ?? "")"))
        }
    }

    @Test func inspectVerbsDoNotContribute() {
        #expect(parse("headlessrun", "runtimes")?.submitsResults == false)
        #expect(parse("headlessrun", "benchmarks")?.submitsResults == false)
        #expect(parse("headlessrun", "storage", "status")?.submitsResults == false)
    }

    // MARK: - Refusals

    /// An ignored `key=` is indistinguishable from one that had no effect, the most
    /// expensive kind of wrong answer to chase on a device.
    @Test func anUnknownParameterIsRefusedAndNamed() {
        #expect(refusal("headlessrun", "runtime=mlx", "batsh=256")
            == .unknownParameters(["batsh"]))
        // Sorted, so the message is deterministic.
        #expect(refusal("headlessrun", "zebra=1", "alpha=2")
            == .unknownParameters(["alpha", "zebra"]))
    }

    @Test func acceptedParametersAreScopedToTheVerb() {
        #expect(refusal("headlessrun", "models", "rm", "repo=org/name") == nil)
        #expect(refusal("headlessrun", "repo=org/name") == .unknownParameters(["repo"]))
    }

    /// Falling back to MLX would run an MLX cell and submit a descriptor naming a runtime
    /// it never used.
    @Test func anUnknownRuntimeIsRefusedRatherThanDefaultingToMlx() {
        #expect(refusal("headlessrun", "runtime=mlex")
            == .invalidValue(key: "runtime", value: "mlex"))
        #expect(refusal("headlessrun", "bench", "runtime=llamaX")
            == .invalidValue(key: "runtime", value: "llamaX"))
    }

    @Test(arguments: ["llamacpp_cli_stock_tools", "mlx_macos_pipette", "docker_vllm",
                      "docker_sglang", "uv_vllm", "uv_sglang", "llamacpp_apk_pipette",
                      "uv_openvino", "uv_onnx_genai", "ovms", "trt_edgellm",
                      "core_ai_macos_pipette"])
    func aHostOnlyRuntimeIsRefusedAsHostOnly(token: String) {
        #expect(refusal("headlessrun", "runtime=\(token)") == .hostOnlyRuntime(token))
    }

    @Test func aMissingRequiredParameterNamesTheKey() {
        #expect(refusal("headlessrun", "benchmarks", "show") == .missingParameter("benchmark"))
        #expect(refusal("headlessrun", "job", "run") == .missingParameter("id"))
    }

    @Test func anUnknownVerbNamesTheVerb() {
        #expect(refusal("headlessrun", "frobnicate") == .unknownVerb("frobnicate"))
        #expect(refusal("headlessrun", "models", "frobnicate") == .unknownVerb("models frobnicate"))
        #expect(refusal("headlessrun", "job") == .unknownVerb("job"))
    }

    @Test func aNonNumericValueIsRefusedRatherThanDefaulted() {
        #expect(refusal("headlessrun", "batch=abc") == .invalidValue(key: "batch", value: "abc"))
        #expect(refusal("headlessrun", "offsets=256,abc")
            == .invalidValue(key: "offsets", value: "abc"))
        #expect(refusal("headlessrun", "bench", "batch=12.5")
            == .invalidValue(key: "batch", value: "12.5"))
        #expect(parse("headlessrun") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
    }

    /// `id=` would otherwise address a job named "".
    @Test func aBlankRequiredIdentifierIsRefused() {
        #expect(refusal("headlessrun", "auth", "register", "organization=", "contact-email=a@b.c")
            == .missingParameter("organization"))
        #expect(refusal("headlessrun", "job", "run", "id=  ") == .missingParameter("id"))
    }

    @Test func anOutOfRangeValueNamesTheKeyAndValue() {
        #expect(refusal("headlessrun", "job", "run", "id=j1", "scope=sideways")
            == .invalidValue(key: "scope", value: "sideways"))
        #expect(refusal("headlessrun", "bench", "spec={not json}")
            == .invalidValue(key: "model", value: "{not json}"))
    }

    /// The plan runner's `Cell::ios_headless_args` emits `runtime=`, `model=` as canonical
    /// JSON, `benchmarks=` and `submit=1`. An exact model makes it a `bench`, not a
    /// `bareBench`.
    @Test func thePlanRunnerInvocationParses() throws {
        let json = #"{"type":"mlx","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-MLX-4bit"}"#
        let runtimeJson = try String(
            decoding: JSONEncoder().encode(#require(Runtime.thisBuild(for: .mlxIosPipette))),
            as: UTF8.self)
        let parsed = parse("headlessrun", "runtime=\(runtimeJson)", "model=\(json)",
                           "benchmarks=decode_throughput_256", "submit=1")
        guard case let .bench(spec, match, _, runtime, _, _, benchmarks, _, _, submit, _, _, _) = parsed else {
            Issue.record("expected a bench command, got \(String(describing: parsed))")
            return
        }
        #expect(spec?.namedModel?.repo?.description == "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(match == nil)
        #expect(runtime == .mlxIosPipette)
        #expect(benchmarks == ["decode_throughput_256"])
        #expect(submit)

        #expect(refusal("headlessrun", "runtime=afm",
                        "benchmarks=decode_throughput_256", "submit=1") == nil)
    }

    /// `Display` (`{repo}[:{path}]`) is the log and warehouse identifier, not an input
    /// spelling; the crate's `--model` does not accept it either.
    @Test func theDisplayFormIsNotAModelSpelling() {
        let refused = refusal("headlessrun",
                              "model=LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q4_K_M.gguf")
        guard case let .rejected(key, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))")
            return
        }
        #expect(key == "model")
        #expect(reason.contains("use `match=`"))
    }

    /// The grammar itself is covered by `ModelUriTests`.
    @Test func theCompactUriReachesTheCommandAndItsRefusalsAreTheParsers() throws {
        let text = parse("headlessrun", "runtime=llama", "bench",
                         "model=gguf-text://repo=LiquidAI/LFM2.5-350M-GGUF&path=q4.gguf")
        guard case let .bench(spec, _, _, _, _, _, _, _, _, _, _, _, _) = text else {
            Issue.record("expected a bench command, got \(String(describing: text))")
            return
        }
        #expect(spec?.namedModel
            == (try parseModelUri("gguf-text://repo=LiquidAI/LFM2.5-350M-GGUF&path=q4.gguf")))

        let mlx = parse("headlessrun", "runtime=mlx", "bench",
                        "model=mlx://repo=mlx-community/Q-4bit&prefix=4bit&rev=v1.2.3")
        guard case let .bench(mlxSpec, _, _, _, _, _, _, _, _, _, _, _, _) = mlx,
              case let .mlx(m2) = try #require(mlxSpec?.namedModel),
              case let .huggingFace(mlxRepo, prefix) = m2.source else {
            Issue.record("expected a HuggingFace mlx model")
            return
        }
        #expect(mlxRepo.reference == "mlx-community/Q-4bit@v1.2.3")
        #expect(prefix?.value == "4bit")

        #expect(refusal("headlessrun", "runtime=mlx", "bench", "model=mlx://repo=org/r&garbage")
            == .rejected(key: "model",
                         reason: ModelUriError.malformedPair("garbage").localizedDescription))
    }

    /// `spec=` is the former name for `model=`. Passing two selectors is ambiguous rather
    /// than ranked.
    @Test func theModelSelectorsAreMutuallyExclusive() {
        let json = #"{"type":"mlx","source":"huggingface","org":"o","repo_name":"r"}"#
        #expect(refusal("headlessrun", "bench", "model=\(json)", "match=LFM")
            == .rejected(key: "model",
                         reason: "`match=` selects from the catalog; pass one or the other"))
        #expect(refusal("headlessrun", "bench", "model=\(json)", "spec=\(json)")
            == .rejected(key: "model", reason: "`spec=` is the former name for it; pass one"))
    }

    /// Nothing sanctioned an exactly named model, so an ad-hoc run's submission is opt-in.
    @Test func namingACoordinateOptsOutOfSubmissionAndSelectingDoesNot() throws {
        let json = #"{"type":"mlx","source":"huggingface","org":"o","repo_name":"r"}"#
        #expect(parse("headlessrun", "model=\(json)")?.submitsResults == false)
        #expect(parse("headlessrun", "match=LFM")?.submitsResults == true)
        #expect(parse("headlessrun", "model=\(json)", "submit=1")?.submitsResults == true)
        #expect(parse("headlessrun", "match=LFM", "submit=0")?.submitsResults == false)
    }

    /// `claim=` names the server job `pipette-ios-proxy` claimed. A run that picks its own
    /// model from the catalog refuses one rather than dropping it.
    @Test func aClaimRidesOnTheFormsThatCarryACell() {
        let json = #"{"type":"mlx","source":"huggingface","org":"o","repo_name":"r"}"#
        func claim(_ command: HeadlessCommand?) -> String? {
            switch command {
            case let .bench(_, _, _, _, _, _, _, _, _, _, _, claim, _)?: claim
            case let .afm(_, _, _, _, claim, _)?: claim
            default: nil
            }
        }
        #expect(claim(parse("headlessrun", "model=\(json)", "submit=1", "claim=job-1")) == "job-1")
        #expect(claim(parse("headlessrun", "bench", "model=\(json)", "claim=job-1")) == "job-1")
        #expect(claim(parse("headlessrun", "runtime=afm", "claim=job-1")) == "job-1")
        #expect(claim(parse("headlessrun", "model=\(json)")) == nil)
        #expect(refusal("headlessrun", "model=\(json)", "claim= ")
            == .rejected(key: "claim", reason: "expected a server job id"))
        #expect(refusal("headlessrun", "model=\(json)", "claim=job-1 ")
            == .rejected(key: "claim", reason: "expected a server job id"))
        #expect(claim(parse("headlessrun", "bench", "runtime=afm", "claim=job-1")) == "job-1")
        for picker in [["headlessrun", "match=LFM"], ["headlessrun", "bench", "match=LFM"]] {
            #expect(refusal(picker + ["claim=job-1"])
                == .rejected(key: "claim", reason: "a claimed run names its model with `model=`"))
        }
    }

    /// `benchmarks=` stays because the plan emits it.
    @Test func benchmarkSingularIsAcceptedAndConflictsAreRefused() {
        #expect(parse("headlessrun", "benchmark=prefill_throughput_512") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: ["prefill_throughput_512"], model: nil, submit: true))
        #expect(refusal("headlessrun", "benchmark=a", "benchmarks=b")
            == .rejected(key: "benchmark", reason: "`benchmarks=` is the plural alias; pass one"))
    }

    // MARK: - submit

    @Test(arguments: [
        ["headlessrun"],
        ["headlessrun", "runtime=afm"],
        ["headlessrun", "bench", "match=LFM2.5", "runtime=llama"],
    ])
    func submitDefaultsOn(_ args: [String]) {
        #expect((try? HeadlessCommand.parse(["/app/Pipette"] + args).get())?.submitsResults == true)
    }

    /// A `spec=` names a model the catalog never validated, so it must not publish rows
    /// without being asked.
    @Test func specRunsDoNotSubmitByDefault() throws {
        let spec = #"spec={"type":"mlx","source":"huggingface","org":"mlx-community","repo_name":"Qwen3.5-0.8B-4bit"}"#
        #expect(try parse("headlessrun", "bench", spec, "benchmarks=x")?
            .submitsResults == false)
        #expect(try parse("headlessrun", "bench", spec, "benchmarks=x", "submit=1")?
            .submitsResults == true)
    }

    /// The only spelling that turns it off: the parse is `!= "0"`.
    @Test(arguments: [
        ["headlessrun", "submit=0"],
        ["headlessrun", "runtime=afm", "submit=0"],
        ["headlessrun", "bench", "match=LFM2.5", "runtime=llama", "submit=0"],
    ])
    func submitZeroOptsOut(_ args: [String]) {
        #expect((try? HeadlessCommand.parse(["/app/Pipette"] + args).get())?.submitsResults == false)
    }

    /// A probe records nothing, so the runner must not demand a registration before it.
    @Test(arguments: HeadlessCommand.afmProbeTokens)
    func afmProbesNeverContribute(_ token: String) {
        #expect((try? HeadlessCommand.parse(
            ["/app/Pipette", "headlessrun", "runtime=afm", "metrics=\(token)"]
        ).get())?.submitsResults == false)
    }

    @Test func nonBenchCommandsDoNotContribute() {
        #expect(parse("headlessrun", "jobs")?.submitsResults == false)
        #expect(parse("headlessrun", "status")?.submitsResults == false)
        #expect(parse("headlessrun", "job", "submit", "id=abc")?.submitsResults == false)
    }

    // MARK: - The crate's positionals

    @Test(arguments: ["prefill_throughput_512", "remote/prefill_throughput_512",
                      "local/prefill_throughput_512"])
    func benchmarksShowTakesTheCratesPositional(reference: String) throws {
        let positional = try #require(parse("headlessrun", "benchmarks", "show", reference))
        #expect(positional == parse("headlessrun", "benchmarks", "show", "benchmark=\(reference)"))
    }

    @Test(arguments: ["elsewhere/x", "local/a/b", "remote/.hidden"])
    func aBadPositionalReferenceIsRefusedAsTheKeyIs(reference: String) throws {
        let refused = try #require(refusal("headlessrun", "benchmarks", "show", reference))
        #expect(refused == refusal("headlessrun", "benchmarks", "show", "benchmark=\(reference)"))
    }

    @Test func resultsShowAndDeleteTakeTheCratesPositional() {
        let id = ResultId(jobId: JobId("job-1"), cellId: CellId("cell-2"))
        #expect(parse("headlessrun", "results", "show", "job-1/cell-2") == .resultsShow(resultId: id))
        #expect(parse("headlessrun", "results", "delete", "job-1/cell-2")
            == .resultsDelete(resultId: id))
    }

    @Test(arguments: ["job-1", "job-1/", "/cell-2", "job-1/cell-2/extra"])
    func aMalformedPositionalPairIsRefusedAsTheKeyIs(raw: String) {
        #expect(refusal("headlessrun", "results", "show", raw)
            == .rejected(key: "result", reason: "expected `<jobId>/<cellId>`"))
        #expect(refusal("headlessrun", "results", "delete", raw)
            == refusal("headlessrun", "results", "delete", "result=\(raw)"))
    }

    @Test func anEmptyPositionalIsMissing() {
        #expect(refusal("headlessrun", "results", "show", "") == .missingParameter("result"))
        #expect(refusal("headlessrun", "benchmarks", "show", "") == .missingParameter("benchmark"))
    }

    @Test(arguments: [
        (["benchmarks", "show", "prefill_throughput_512", "benchmark=prefill_throughput_512"],
         "benchmark"),
        (["benchmarks", "show", "benchmark=a", "b"], "benchmark"),
        (["benchmarks", "show", "--benchmark", "a", "b"], "benchmark"),
        (["results", "show", "job-1/cell-2", "result=job-1/cell-2"], "result"),
        (["results", "delete", "--result=job-1/cell-2", "job-1/cell-3"], "result"),
        (["sync", "abc/cell-1", "result=abc/cell-1"], "result"),
    ])
    func aPositionalAndItsKeyTogetherAreRefused(tokens: [String], key: String) {
        #expect(refusal(["headlessrun"] + tokens)
            == .rejected(key: key, reason: "also given as a positional argument; pass one"))
    }

    @Test func aSecondPositionalIsAStray() {
        #expect(refusal("headlessrun", "results", "show", "job-1/cell-2", "job-1/cell-3")
            == .unknownVerb("results show job-1/cell-3"))
        #expect(refusal("headlessrun", "benchmarks", "show", "a", "b")
            == .unknownVerb("benchmarks show b"))
        #expect(refusal("headlessrun", "sync", "a/b", "c/d") == .unknownVerb("sync c/d"))
    }

    /// The argument parser would have dropped `--frob` on the way to a declared positional,
    /// so its value is not the positional.
    @Test func anUnknownDashedKeyDoesNotLendItsValueToThePositional() {
        #expect(refusal("headlessrun", "benchmarks", "show", "--frob", "prefill_throughput_512")
            == .unknownVerb("benchmarks show prefill_throughput_512"))
        #expect(refusal("headlessrun", "results", "show", "job-1/cell-2", "--frob=1")
            == .unknownParameters(["frob"]))
    }

    @Test(arguments: [
        (["results", "job-1/cell-2"], "results job-1/cell-2"),
        (["results", "list", "job-1/cell-2"], "results list job-1/cell-2"),
        (["benchmarks", "prefill_throughput_512"], "benchmarks prefill_throughput_512"),
        (["benchmarks", "run", "prefill_throughput_512"], "benchmarks run prefill_throughput_512"),
        (["job", "run", "abc"], "job run abc"),
    ])
    func aLeafWithoutAPositionalStillRefusesABareWord(tokens: [String], path: String) {
        #expect(refusal(["headlessrun"] + tokens) == .unknownVerb(path))
    }

    @Test func thePositionalLeavesAreTheCrates() {
        #expect(HeadlessTree.Benchmarks.Show.positionalKey == "benchmark")
        #expect(HeadlessTree.Results.Show.positionalKey == "result")
        #expect(HeadlessTree.Results.Delete.positionalKey == "result")
        #expect(HeadlessTree.Sync.positionalKey == "result")
        #expect(HeadlessTree.Results.List.positionalKey == nil)
    }
}
