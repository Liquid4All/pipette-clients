import Foundation
import Testing

@testable import Pipette

/// `HeadlessCommand.parse` — the pure argv → command mapping behind the
/// headless CLI. No device state: these tests pin the back-compat invocation
/// forms and the new verb grammar.
struct HeadlessCommandTests {

    /// argv as `devicectl process launch` delivers it: binary path first, then
    /// the launch arguments. Discards the refusal so the accept-path assertions read as
    /// `parse(…) == .command`; `refusal(_:)` is the counterpart that keeps it.
    private func parse(_ args: String...) -> HeadlessCommand? {
        parse(args)
    }

    /// The array forms, for a caller that assembles its invocation.
    private func parse(_ args: [String]) -> HeadlessCommand? {
        try? HeadlessCommand.parse(["/app/Pipette"] + args).get()
    }

    /// The error an invocation is refused with, or `nil` if it parsed.
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
            runtime: .mlxIosPipette, batch: 512, nGpuLayers: nil, threads: nil,
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
    }

    @Test func bareBenchParsesEveryParameter() {
        #expect(parse("headlessrun", "runtime=llama", "batch=256", "metrics=prefill,decode",
                      "offsets=512,1024", "match=LFM2.5", "submit=1") == .bareBench(
            runtime: .llamacppIosPipette, batch: 256, nGpuLayers: nil, threads: nil,
            metrics: ["prefill", "decode"], offsets: [512, 1024],
            benchmarks: [], model: "LFM2.5", submit: true))
    }

    @Test func bareRuntimeTypeMatchingIsCaseInsensitiveOnTheLlPrefix() {
        #expect(parse("headlessrun", "runtime=mlx") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, nGpuLayers: nil, threads: nil,
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
        #expect(parse("headlessrun", "runtime=LLAMA") == .bareBench(
            runtime: .llamacppIosPipette, batch: 512, nGpuLayers: nil, threads: nil,
            metrics: ["prefill", "decode", "maxmem"],
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
            == .bareBench(runtime: .mlxIosPipette, batch: 512, nGpuLayers: nil, threads: nil,
                          metrics: ["prefill", "decode", "maxmem"],
                          offsets: [256, 512, 1024, 2048, 4096],
                          benchmarks: ["prefill_throughput_512", "decode_throughput_512_100"],
                          model: nil, submit: true))
    }

    @Test(arguments: ["coherence", "calibrate", "promptseed"])
    func bareBenchDiagnosticMetricsParseAsMetrics(metric: String) {
        #expect(parse("headlessrun", "runtime=mlx", "metrics=\(metric)") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, nGpuLayers: nil, threads: nil, metrics: [metric],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
    }

    /// `cooldown=` was accepted-and-ignored in the original CLI form. It is now refused:
    /// plan-types dropped `cooldown_seconds` and asserts it is never emitted, so nothing
    /// sends it, and accepting a parameter that does nothing is how a mistyped knob goes
    /// unnoticed.
    @Test func theRetiredCooldownParameterIsNowRefused() {
        #expect(refusal("headlessrun", "runtime=mlx", "cooldown=0")
            == .unknownParameters(["cooldown"]))
    }

    // MARK: - The `bench` verb

    @Test func benchCatalogSelectionParsesModelQuantRuntime() {
        #expect(parse("headlessrun", "bench", "match=Qwen", "quant=Q4_0",
                      "runtime=llama", "benchmarks=x") == .bench(
            spec: nil, model: "Qwen", quant: "Q4_0", runtime: .llamacppIosPipette,
            batch: 512, nGpuLayers: nil, threads: nil, benchmarks: ["x"],
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096], submit: true))
    }

    @Test func benchRuntimeMlxParsesWithNilSpec() {
        #expect(parse("headlessrun", "bench", "runtime=mlx", "match=LFM") == .bench(
            spec: nil, model: "LFM", quant: nil, runtime: .mlxIosPipette,
            batch: 512, nGpuLayers: nil, threads: nil, benchmarks: [],
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096], submit: true))
    }

    @Test func benchSpecDecodesStraightIntoAModelVariant() throws {
        #expect(try parse("headlessrun", "bench",
                          "spec={\"type\":\"mlx\",\"source\":\"huggingface\",\"org\":\"mlx-community\",\"repo_name\":\"Qwen3.5-0.8B-4bit\"}",
                          "benchmarks=x") == .bench(
            spec: .model(.mlx(Mlx(source: .huggingFace(
                repo: HFRepo.parse("mlx-community/Qwen3.5-0.8B-4bit"), prefix: nil)))),
            model: nil, quant: nil, runtime: .mlxIosPipette,
            batch: 512, nGpuLayers: nil, threads: nil, benchmarks: ["x"],
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096], submit: false))
    }

    @Test func benchMalformedSpecFailsToParse() {
        #expect(parse("headlessrun", "bench", "spec={not valid json}", "benchmarks=x") == nil)
    }

    @Test func benchRuntimeAfmNeedsNoModel() {
        #expect(parse("headlessrun", "bench", "runtime=afm",
                      "metrics=prefill", "offsets=512") == .bench(
            spec: nil, model: nil, quant: nil, runtime: .appleFoundation,
            batch: 512, nGpuLayers: nil, threads: nil, benchmarks: [], metrics: ["prefill"], offsets: [512], submit: true))
    }

    // MARK: - auth register / diag memseq

    @Test func registerTakesTheServerOverrideOrDefaultsIt() {
        #expect(parse("headlessrun", "auth", "register", "organization=Liquid", "contact-email=a@b.c")
            == .register(server: nil, org: "Liquid", email: "a@b.c", preauth: nil,
                       clientDetails: nil, deviceName: nil))
        #expect(parse("headlessrun", "auth", "register", "server-url=https://collector.example",
                      "organization=Liquid", "contact-email=a@b.c")
            == .register(server: "https://collector.example", org: "Liquid", email: "a@b.c",
                         preauth: nil, clientDetails: nil, deviceName: nil))
    }

    /// `org`/`email` used to default to a hardcoded organization and a personal address,
    /// so `headlessrun register` on a fleet device attributed it to whoever wrote the
    /// default. Registering is not reversible — correcting it rotates the signing key —
    /// so the parser refuses rather than guessing.
    @Test func registerRequiresOrgAndEmail() {
        #expect(refusal("headlessrun", "auth", "register") == .missingParameter("organization"))
        #expect(refusal("headlessrun", "auth", "register", "organization=Liquid") == .missingParameter("contact-email"))
        #expect(refusal("headlessrun", "auth", "register", "contact-email=a@b.c") == .missingParameter("organization"))
    }

    /// A present-but-empty value is not a value. `require` alone would accept `email=`,
    /// which registers the device with a blank contact.
    @Test(arguments: ["", "   "])
    func registerRejectsAnEmptyOrgOrEmail(blank: String) {
        #expect(refusal("headlessrun", "auth", "register", "organization=\(blank)", "contact-email=a@b.c")
            == .missingParameter("organization"))
        #expect(refusal("headlessrun", "auth", "register", "organization=Liquid", "contact-email=\(blank)")
            == .missingParameter("contact-email"))
    }

    @Test func memseqParsesModelListAndBatch() {
        #expect(parse("headlessrun", "diag", "memseq", "models=a,b,c", "batch=256")
            == .memSeq(models: ["a", "b", "c"], batch: 256))
        #expect(parse("headlessrun", "diag", "memseq") == .memSeq(models: [], batch: 512))
    }

    // MARK: - Models verbs

    @Test func modelsListsAndModelsRmRequiresNameOrRepo() {
        #expect(parse("headlessrun", "models") == .listModels(format: .name))
        #expect(parse("headlessrun", "models", "rm", "name=file.gguf")
            == .removeModel(name: "file.gguf", repo: nil))
        #expect(parse("headlessrun", "models", "rm", "repo=org/name")
            == .removeModel(name: nil, repo: "org/name"))
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

    @Test func settingsShowSetAndRun() {
        #expect(parse("headlessrun", "settings") == .settings(.show))
        #expect(parse("headlessrun", "settings", "set", "worker=on")
            == .settings(.setWorker(true)))
        #expect(parse("headlessrun", "settings", "set", "worker=off")
            == .settings(.setWorker(false)))
        #expect(parse("headlessrun", "settings", "run") == .settings(.run))
        #expect(parse("headlessrun", "settings", "set") == nil)
        #expect(parse("headlessrun", "settings", "set", "worker=maybe") == nil)
        #expect(parse("headlessrun", "settings")?.allowedViaDeepLink == false)
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

    /// The admissible set, as one list rather than a second enum.
    ///
    /// This replaced a `BenchRuntime` that named the same three runtimes a second time and
    /// needed a forward map, an inverse and a tag to stay in step with `RuntimeType` — all
    /// three of which had drifted. There is nothing left to drift: the type *is* the
    /// value, and `rawValue` *is* the tag.
    @Test func onlyTheInProcessRuntimesAreIosRunnable() {
        let runnable = RuntimeType.allCases.filter(\.isIosRunnable)

        #expect(Set(runnable) == [.llamacppIosPipette, .mlxIosPipette, .appleFoundation])
    }

    /// The short tokens are this client's own shorthand and claim no build; a plan `type`
    /// tag is refused, because it names an engine without naming which build of it (the
    /// plan sends the canonical `Runtime` for that).
    @Test func parseTakesShortTokensAndRefusesPlanTypeTags() throws {
        #expect(try RuntimeType.parseHeadless("mlx") == .mlxIosPipette)
        #expect(try RuntimeType.parseHeadless("llama") == .llamacppIosPipette)
        #expect(try RuntimeType.parseHeadless("afm") == .appleFoundation)
        // Absent still means MLX — the bare `headlessrun` form depends on it.
        #expect(try RuntimeType.parseHeadless(nil) == .mlxIosPipette)
        for tag in ["apple_foundation", "llamacpp_ios_pipette", "mlx_ios_pipette"] {
            #expect(throws: HeadlessUsageError.self) {
                try RuntimeType.parseHeadless(tag)
            }
        }
    }

    // MARK: - runtime-flags

    /// A knobs-only `runtime-flags=` plus the invocation whose cell it resolves against.
    ///
    /// The axes come from `runtime=`, `model=` and `benchmarks=` — as they do on the CLI,
    /// where the cell is parsed before any flag is read. Nothing here restates them.
    private func flagsRun(_ knobs: String, runtime: String = "mlx",
                          model: String? = "mlx://repo=o/r",
                          benchmark: String = "prefill_throughput_256") -> [String] {
        ["headlessrun", "bench", "runtime=\(runtime)"]
            + (model.map { ["model=\($0)"] } ?? [])
            + ["benchmarks=\(benchmark)", "runtime-flags={\(knobs)}"]
    }

    private static let ggufModel = "gguf-text://repo=o/r&path=m.gguf"

    /// An authored `n_ubatch` becomes the prefill chunk, replacing `batch=`'s default.
    @Test func runtimeFlagsSuppliesThePrefillChunk() {
        let parsed = parse(flagsRun(#""n_ubatch":256"#))
        guard case let .bench(_, _, _, _, batch, _, _, _, _, _, _) = parsed else {
            Issue.record("expected a bench command, got \(String(describing: parsed))")
            return
        }
        #expect(batch == 256)
    }

    /// An absent knob stays absent — the point of use applies the default, so
    /// "asked for 99" and "asked for nothing" stay distinguishable. Mirrors the crate,
    /// whose builder emits `-ngl` only for a `Some`.
    @Test func anAuthoredGpuLayerCountIsCarriedAndAnAbsentOneIsNotInvented() {
        let authored = parse(flagsRun(#""number_gpu_layers":10"#, runtime: "llama",
                                      model: Self.ggufModel))
        guard case let .bench(_, _, _, _, _, nGpuLayers, _, _, _, _, _) = authored else {
            Issue.record("expected a bench command, got \(String(describing: authored))")
            return
        }
        #expect(nGpuLayers == 10)

        let bare = parse("headlessrun", "runtime=mlx")
        guard case let .bareBench(_, _, absent, _, _, _, _, _, _) = bare else {
            Issue.record("expected a bareBench command")
            return
        }
        #expect(absent == nil)
    }

    /// Every `runtime-flags=` value that cannot be honoured, by the reason reported. One
    /// table because they share a shape — the refusal *is* the payload.
    ///
    /// The cell is varied through the *invocation*, not through the JSON, which is the
    /// whole point: a flags value names knobs and nothing else.
    @Test(arguments: [
        (knobs: #""tensor_parallel_size":2"#, runtime: "mlx", model: "mlx://repo=o/r",
         reason: "no iOS variant declares `tensor_parallel_size`"),
        // An axis key is refused for the same reason a foreign knob is: the cell is
        // derived, so naming it here is not the caller's to do.
        (knobs: #""runtime_type":"mlx_ios_pipette","n_ubatch":256"#, runtime: "mlx",
         model: "mlx://repo=o/r", reason: "no iOS variant declares `runtime_type`"),
        (knobs: #""number_gpu_layers":10"#, runtime: "mlx", model: "mlx://repo=o/r",
         reason: "this cell does not carry `number_gpu_layers`"),
        (knobs: #""ctx_size":4096"#, runtime: "llama", model: ggufModel,
         reason: "`ctx_size` is not applied yet"),
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

    /// The old wire reads as wrong rather than as malformed, so an invocation copied from
    /// before the format change says what to change.
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

    /// Flags need a cell, and a `match=` is resolved against the catalog at run time —
    /// there is no model type to resolve them against at parse.
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

    /// A digest names one artifact but not its kind — only the store knows that, and it is
    /// not consulted until the run. Previously reported as "pass `model=`", which the
    /// caller had done.
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

    /// An unparseable id is a different mistake from the wrong number of them, and the
    /// refusal names the id rather than blaming the count.
    @Test func flagsAgainstAnUnknownBenchmarkIdNameTheId() {
        #expect(refusal("headlessrun", "bench", "runtime=llama",
                        "model=\(Self.ggufModel)", "benchmarks=nonsense",
                        #"runtime-flags={"number_gpu_layers":10}"#)
            == .invalidValue(key: "benchmarks", value: "nonsense"))
    }

    /// One invocation can name several benchmarks, and their cells carry different knobs —
    /// so which cell a flags value belongs to would be ambiguous.
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

    /// Both set the prefill chunk, so picking a winner silently would measure a size the
    /// caller did not ask for.
    @Test func runtimeFlagsAndBatchTogetherAreRefused() {
        let refused = refusal(flagsRun(#""n_ubatch":256"#) + ["batch=512"])
        #expect(refused == .rejected(
            key: "runtime-flags",
            reason: "`batch=` also sets the prefill chunk; pass one or the other"))
    }

    // MARK: - `benchmarks run`

    /// The crate's `--readiness-max-wait-secs` / `--readiness-skip-thermal`, reachable at
    /// last: before this the gate could only be waited out, never relaxed.
    @Test func readinessOverridesReachTheGate() {
        let parsed = parse("headlessrun", "benchmarks", "run",
                           "benchmark=prefill_throughput_256",
                           "model=\(Self.ggufModel)", "runtime=llama",
                           "readiness-max-wait-secs=30", "readiness-skip-thermal=1")
        guard case let .benchmarksRun(_, _, _, _, _, _, _, readiness) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(readiness.maxSeconds == 30)
        #expect(readiness.skipThermal)
    }

    /// The dashed spelling takes the bare form the CLI does, since `--readiness-skip-thermal`
    /// carries no value there. In `key=value` a bare word is a verb, so that spelling needs
    /// `=1`; both reach the same override.
    @Test func theDashedSkipThermalNeedsNoValue() {
        let parsed = parse(["headlessrun", "benchmarks", "run",
                            "--benchmark", "prefill_throughput_256",
                            "--model", Self.ggufModel, "--runtime", "llama",
                            "--readiness-skip-thermal"])
        guard case let .benchmarksRun(_, _, _, _, _, _, _, readiness) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(readiness.skipThermal)
    }

    /// Refused where the crate refuses them: eval and max-memory cells carry no readiness
    /// knob, so a value there would be accepted and read by nothing.
    @Test(arguments: ["eval_smoke", "max_memory_usage_256"])
    func readinessOverridesAreRefusedOnUngatedBenchmarks(benchmark: String) {
        let refused = refusal("headlessrun", "benchmarks", "run",
                              "benchmark=\(benchmark)", "model=\(Self.ggufModel)",
                              "runtime=llama", "readiness-max-wait-secs=30")
        guard case let .rejected(_, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))")
            return
        }
        #expect(reason.contains("does not gate on readiness"), "reason was: \(reason)")
    }

    /// An absent override leaves the built-in gate exactly as it was.
    @Test func withoutOverridesTheGateKeepsItsDefaults() {
        let parsed = parse("headlessrun", "benchmarks", "run",
                           "benchmark=prefill_throughput_256",
                           "model=\(Self.ggufModel)", "runtime=llama")
        guard case let .benchmarksRun(_, _, _, _, _, _, _, readiness) = parsed else {
            Issue.record("expected benchmarksRun"); return
        }
        #expect(readiness == ReadinessPolicy())
    }

    /// The crate's `benchmarks run` shape: one cell, every axis named. `model=` takes the
    /// two forms `--model` takes — a compact `<scheme>://` URI or a JSON `Model`.
    @Test(arguments: [
        "gguf-text://repo=LiquidAI/LFM2-350M-GGUF&path=LFM2-350M-Q4_K_M.gguf",
        #"{"type":"gguf_text","source":"huggingface","org":"LiquidAI","#
            + #""repo_name":"LFM2-350M-GGUF","path":"LFM2-350M-Q4_K_M.gguf"}"#,
    ])
    func benchmarksRunTakesAUriOrJsonModel(_ model: String) {
        let parsed = parse("headlessrun", "benchmarks", "run",
                           "benchmark=decode_throughput_512_100",
                           "model=\(model)", "runtime=llama")

        guard case let .benchmarksRun(benchmark, resolved, runtime, _, _, _, sync, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(benchmark == "decode_throughput_512_100")
        #expect(resolved.namedModel?.repo?.description == "LiquidAI/LFM2-350M-GGUF")
        #expect(runtime == .llamacppIosPipette)
        #expect(!sync, "`sync=` is opt-in, as `--sync` is")
    }

    /// A runtime names a build, and this binary is one build. The argument is compared
    /// against it, not merely typed: running a cell that asked for another build and then
    /// recording what actually ran would answer a question nobody asked.
    @Test func aRuntimeNamingAnotherBuildIsRefused() throws {
        let builtIn = try #require(Runtime.thisBuild(for: .llamacppIosPipette))
        let other = Runtime.llamacppIosPipette(
            source: SourceRepository(repositoryVersion: NonEmptyString(validated: "b0001")),
            flavor: .iosArm64, privateThermal: false)
        #expect(other != builtIn, "the fixture has to differ for the check to mean anything")

        let refused = refusal("headlessrun", "benchmarks", "run", "--benchmark", "b",
                              "--model", "mlx://repo=o/r",
                              "--runtime", try SubmissionRef.runtime(other))

        guard case let .rejected(key, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))")
            return
        }
        #expect(key == "runtime")
        #expect(reason.contains("b0001") && reason.contains(LlamaCppBuildInfo.submissionVersion),
                "the refusal names both builds; was: \(reason)")
    }

    /// The identity `runtimes` advertises is the one that passes — it is encoded from the
    /// same value, so the two cannot drift.
    @Test func theAdvertisedRuntimeIdentityIsAccepted() throws {
        let builtIn = try #require(Runtime.thisBuild(for: .llamacppIosPipette))

        let parsed = parse("headlessrun", "benchmarks", "run", "--benchmark", "b",
                           "--model", "gguf-text://repo=o/r&path=m.gguf",
                           "--runtime", try SubmissionRef.runtime(builtIn))

        guard case let .benchmarksRun(_, _, runtime, _, _, _, _, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        #expect(runtime == .llamacppIosPipette)
    }

    /// `runtime=` takes a JSON `Runtime` — the form `--runtime` takes for a runtime with
    /// no URI spelling. The MLX identity is the built-in one, since any other is refused.
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
    /// representable scheme names a desktop runtime this build cannot be — so a URI is
    /// refused by name rather than parsed.
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

    /// A cell needs its model named; the catalog match belongs to the sweep form.
    @Test func benchmarksRunRequiresAModel() {
        #expect(refusal("headlessrun", "benchmarks", "run", "benchmark=b", "runtime=llama")
            == .missingParameter("model"))
    }

    /// `model://sha256=<prefix>` addresses the local store by descriptor digest. It is
    /// carried, not resolved, while parsing: the store is the runner's to read.
    @Test func aDigestReferenceIsCarriedForTheRunnerToResolve() {
        let parsed = parse("headlessrun", "benchmarks", "run", "benchmark=b",
                           "model=model://sha256=A1B2C3D4E5F6", "runtime=llama")

        guard case let .benchmarksRun(_, model, _, _, _, _, _, _) = parsed else {
            Issue.record("expected benchmarksRun, got \(String(describing: parsed))")
            return
        }
        // Lower-cased, as a pasted prefix may not be.
        #expect(model == .digest("a1b2c3d4e5f6"))
    }

    /// A prefix short enough to collide, or one that is not hex, is refused rather than
    /// matched loosely.
    @Test(arguments: [("model://sha256=a1b2", "too short"), ("model://sha256=zzzzzzzz", "not hex"),
                      ("model://a1b2c3d4e5f6", "addresses the store by digest")])
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

    /// The CLI spelling of the same keys: `--key value`, `--key=value`, and a bare
    /// `--flag`. An invocation copied from `pipette` differs only in the dashes.
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
        guard case let .benchmarksRun(_, _, runtime, _, _, _, sync, _) = dashed else {
            Issue.record("expected benchmarksRun, got \(String(describing: dashed))")
            return
        }
        #expect(runtime == .mlxIosPipette)
        #expect(sync, "a bare `--sync` is the boolean form")
    }

    /// A verb is still a verb: it precedes the first flag, as it does in `pipette`.
    @Test func verbsSurviveAlongsideDashedFlags() {
        let parsed = parse("headlessrun", "models", "rm", "--name", "a-Q4_0.gguf")

        #expect(parsed == .removeModel(name: "a-Q4_0.gguf", repo: nil))
    }

    /// `models list --format` renders the model column the two ways the crate renders it,
    /// defaulting to the identity as it does.
    @Test func modelsListTakesTheFormatTheCrateTakes() {
        #expect(parse("headlessrun", "models") == .listModels(format: .name))
        #expect(parse("headlessrun", "models", "--format", "uri") == .listModels(format: .uri))
        #expect(refusal("headlessrun", "models", "--format", "table")
            == .invalidValue(key: "format", value: "table"))
    }

    /// `models pull` / `models delete` name a model the way the crate names one — a
    /// self-contained reference, not a repo and a filename.
    @Test func modelsPullAndDeleteTakeAModelReference() throws {
        let uri = "mlx://repo=mlx-community/LFM2-350M-4bit"
        #expect(parse("headlessrun", "models", "pull", "--model", uri)
            == .pullModel(.model(try ModelUri.parse(uri))))
        #expect(parse("headlessrun", "models", "delete", "--model=model://sha256=b121b700be77")
            == .deleteModel(.digest("b121b700be77")))
        #expect(refusal("headlessrun", "models", "pull") == .missingParameter("model"))
    }

    // MARK: - The auth and diag groups

    /// A runtime type is not a build: `runtime=llamacpp_ios_pipette` names which engine
    /// but not which llama.cpp, so nothing can be checked against this binary and the
    /// result would record whatever ran. The plan emits the canonical `Runtime` instead.
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

    /// The short tokens stay: they are this client's own shorthand for a hand-typed run,
    /// never a plan spelling, so they claim no build identity.
    @Test func theShortTokensStillSelectAnEngine() {
        let parsed = parse("headlessrun", "runtime=llama", "benchmarks=decode_throughput_256")

        guard case let .bareBench(runtime, _, _, _, _, _, benchmarks, _, _) = parsed else {
            Issue.record("expected a bareBench, got \(String(describing: parsed))")
            return
        }
        #expect(runtime == .llamacppIosPipette)
        #expect(benchmarks == ["decode_throughput_256"])
    }

    /// `auth register` is the CLI's spelling. The bare `register` is kept: it is in the
    /// shipped runner's documented usage and is run by hand.
    /// Parameter names are the CLI's, so an invocation copied from `pipette auth register`
    /// differs only in the dashes.
    @Test func authRegisterTakesTheCratesParameterNames() {
        #expect(parse("headlessrun", "auth", "register",
                      "organization=Liquid", "contact-email=a@b.c")
            == .register(server: nil, org: "Liquid", email: "a@b.c", preauth: nil,
                       clientDetails: nil, deviceName: nil))
    }

    /// `--client-details` and `--device-name`, the two the crate takes that this client
    /// can now honour: details go to the server, the name is stored locally.
    @Test func authRegisterTakesClientDetailsAndDeviceName() {
        #expect(parse("headlessrun", "auth", "register", "organization=Liquid",
                      "contact-email=a@b.c", "client-details=lab rig 3",
                      "device-name=boston-17-pro-1")
            == .register(server: nil, org: "Liquid", email: "a@b.c", preauth: nil,
                         clientDetails: "lab rig 3", deviceName: "boston-17-pro-1"))
    }

    /// The crate's `--preauth-key`: a key that admits an already-approved client, so a
    /// fleet device registers without waiting on manual approval.
    @Test func authRegisterTakesAPreauthKey() {
        #expect(parse("headlessrun", "auth", "register", "organization=Liquid", "contact-email=a@b.c",
                      "preauth-key=preauth_abc.def")
            == .register(server: nil, org: "Liquid", email: "a@b.c", preauth: "preauth_abc.def",
                       clientDetails: nil, deviceName: nil))
    }

    /// The group's required parameters are the leaf's, not the group's.
    @Test func authRegisterStillRequiresOrgAndEmail() {
        #expect(refusal("headlessrun", "auth", "register") == .missingParameter("organization"))
    }

    @Test func authMeAndResetParse() {
        #expect(parse("headlessrun", "auth", "me") == .authMe)
        #expect(parse("headlessrun", "auth", "reset") == .authReset(force: false))
        #expect(parse("headlessrun", "auth", "reset", "force=1") == .authReset(force: true))
    }

    @Test func authNeedsALeaf() {
        #expect(refusal("headlessrun", "auth") == .unknownVerb("auth"))
        #expect(refusal("headlessrun", "auth", "logout") == .unknownVerb("auth logout"))
    }

    /// `diag memseq` is the grouped spelling of the bare verb, which stays.
    @Test func diagMemseqMatchesTheBareSpelling() {
        #expect(parse("headlessrun", "diag", "memseq", "models=a,b", "batch=256")
            == .memSeq(models: ["a", "b"], batch: 256))
        #expect(parse("headlessrun", "diag", "memseq", "models=a,b", "batch=256")
            == .memSeq(models: ["a", "b"], batch: 256))
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

    /// A probe records nothing and identity work publishes nothing, so none of these may
    /// demand a registration up front.
    @Test func authAndDiagDoNotContribute() {
        #expect(parse("headlessrun", "auth", "me")?.submitsResults == false)
        #expect(parse("headlessrun", "diag", "probe", "kind=token")?.submitsResults == false)
    }

    // MARK: - results

    @Test func resultsListTakesTheCliFilters() {
        #expect(parse("headlessrun", "results") == .listResults(
            benchmark: nil, type: nil, state: nil, limit: nil))
        #expect(parse("headlessrun", "results", "benchmark=prefill_throughput_512",
                      "type=prefill_throughput", "state=submitted", "limit=5")
            == .listResults(benchmark: "prefill_throughput_512", type: .prefillThroughput,
                            state: .submitted, limit: 5))
    }

    @Test func resultsListRefusesAnUnknownFilterValue() {
        #expect(refusal("headlessrun", "results", "state=posted")
            == .invalidValue(key: "state", value: "posted"))
        #expect(refusal("headlessrun", "results", "type=prefill")
            == .invalidValue(key: "type", value: "prefill"))
        #expect(refusal("headlessrun", "results", "limit=lots")
            == .invalidValue(key: "limit", value: "lots"))
    }

    /// A result is addressed by the pair, because iOS results live under their job. A job
    /// id alone addresses a job, which is what `sync job=` is for.
    @Test func resultsShowAndDeleteTakeTheJobCellPair() {
        let id = ResultId(jobId: JobId("job-1"), cellId: CellId("cell-2"))
        #expect(parse("headlessrun", "results", "show", "result=job-1/cell-2") == .showResult(id: id))
        #expect(parse("headlessrun", "results", "delete", "result=job-1/cell-2")
            == .deleteResult(id: id))
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
        #expect(parse("headlessrun", "sync") == .sync(jobId: nil))
        #expect(parse("headlessrun", "sync", "job=abc") == .sync(jobId: "abc"))
    }

    /// The CLI narrows `sync` by *result* id. iOS addresses whole jobs, so a CLI habit
    /// gets a real answer instead of "unknown parameter".
    @Test func syncRefusesAResultNarrowingItCannotAddress() {
        #expect(refusal("headlessrun", "sync", "result=abc/cell-1")
            == .rejected(key: "result",
                         reason: "a result is not addressable yet; pass `job=<jobId>`"))
    }

    @Test func syncTakesNoLeaf() {
        #expect(refusal("headlessrun", "sync", "now") == .unknownVerb("sync now"))
    }

    /// Submitting is the point of `sync`, so the runner's registration gate has to cover
    /// it — otherwise an unregistered device would pull the catalog and drop every result.
    @Test func syncCountsAsContributing() {
        #expect(parse("headlessrun", "sync")?.submitsResults == true)
    }

    // MARK: - The inspect verbs

    @Test func inspectVerbsParseWithoutParameters() {
        #expect(parse("headlessrun", "runtimes") == .listRuntimes)
        #expect(parse("headlessrun", "benchmarks") == .listBenchmarks(type: nil))
        #expect(parse("headlessrun", "storage", "status") == .storageStatus)
    }

    /// The crate spells every inventory verb `<noun> list`. Both spellings resolve to the
    /// same command, so a script written against `pipette` runs unchanged on device.
    @Test func theCratesListSpellingIsAccepted() {
        #expect(parse("headlessrun", "models", "list") == parse("headlessrun", "models"))
        #expect(parse("headlessrun", "benchmarks", "list") == parse("headlessrun", "benchmarks"))
        #expect(parse("headlessrun", "results", "list") == parse("headlessrun", "results"))
        #expect(parse("headlessrun", "runtimes", "list") == .listRuntimes)
        // Parameters still land on the explicit spelling.
        #expect(parse("headlessrun", "models", "list", "format=uri") == .listModels(format: .uri))
        #expect(parse("headlessrun", "benchmarks", "list", "type=eval")
            == .listBenchmarks(type: .eval))
        // And a stray word past `list` is as refused as it is past the bare noun.
        #expect(refusal("headlessrun", "runtimes", "list", "oops")
            == .unknownVerb("runtimes list oops"))
    }

    /// The crate prints a build stamp for `pipette --version`; this client had no way to
    /// ask a device what it was running. Parsed as a bare verb, refusing a leaf.
    @Test func versionParsesAsABareVerb() {
        #expect(parse("headlessrun", "version") == .version)
        #expect(refusal("headlessrun", "version", "oops") == .unknownVerb("version oops"))
    }

    /// The crate makes the claim loop a root `worker` command; this client reached it
    /// only through `settings run`. Both spellings start the same loop.
    @Test func theCratesWorkerVerbStartsTheClaimLoop() {
        #expect(parse("headlessrun", "worker") == .settings(.run))
        #expect(parse("headlessrun", "worker") == parse("headlessrun", "settings", "run"))
        // The crate's flags are unwired here, so they are refused rather than ignored.
        #expect(refusal("headlessrun", "worker", "run") == .unknownVerb("worker run"))
    }

    /// `storage gc` mirrors the crate's `--dry-run`: reporting is opt-in, reclaiming is
    /// the default, so a caller that omits the flag gets the sweep it asked for.
    @Test func storageGcTakesADryRunFlag() {
        #expect(parse("headlessrun", "storage", "gc") == .storageGc(dryRun: false))
        #expect(parse("headlessrun", "storage", "gc", "dry-run=1") == .storageGc(dryRun: true))
        #expect(parse("headlessrun", "storage", "gc", "dry-run=0") == .storageGc(dryRun: false))
    }

    @Test func benchmarksTakesATypeFilter() {
        #expect(parse("headlessrun", "benchmarks", "type=eval")
            == .listBenchmarks(type: .eval))
        #expect(parse("headlessrun", "benchmarks", "type=prefill_throughput")
            == .listBenchmarks(type: .prefillThroughput))
    }

    /// A misspelled type would otherwise filter every benchmark out and report an empty
    /// catalog, which reads as "nothing synced" rather than "you typed it wrong".
    @Test func benchmarksRefusesAnUnknownType() {
        #expect(refusal("headlessrun", "benchmarks", "type=prefill")
            == .invalidValue(key: "type", value: "prefill"))
    }

    /// `init-local` seeds the generated half of the catalog, as the CLI's does.
    @Test func benchmarksInitLocalParses() {
        #expect(parse("headlessrun", "benchmarks", "init-local") == .initLocalBenchmarks)
        #expect(refusal("headlessrun", "benchmarks", "init-local", "extra")
            == .unknownVerb("benchmarks init-local extra"))
    }

    @Test func benchmarksShowNeedsAnId() {
        #expect(parse("headlessrun", "benchmarks", "show", "benchmark=prefill_throughput_512")
            == .showBenchmark(id: "prefill_throughput_512"))
        #expect(refusal("headlessrun", "benchmarks", "show") == .missingParameter("benchmark"))
    }

    /// `storage` alone is not a command, and a leaf a phone cannot offer must be refused
    /// rather than silently treated as the bare list — `runtimes pull` asks to install a
    /// runtime, which is a different question from "what is compiled in".
    @Test func inspectVerbsRefuseUnknownLeaves() {
        #expect(refusal("headlessrun", "storage") == .unknownVerb("storage"))
        #expect(refusal("headlessrun", "storage", "frobnicate") == .unknownVerb("storage frobnicate"))
        #expect(refusal("headlessrun", "runtimes", "pull") == .unknownVerb("runtimes pull"))
    }

    /// The same hole existed on every flat verb: a stray bare word was dropped, so
    /// `jobs frobnicate` listed jobs and `bench` typos went unnoticed.
    @Test(arguments: [("jobs", "frobnicate"), ("status", "now"), ("bench", "oops"),
                      ("runtimes", "pull"), ("sync", "now")])
    func aStrayLeafOnAFlatVerbIsRefused(verb: String, leaf: String) {
        #expect(refusal("headlessrun", verb, leaf) == .unknownVerb("\(verb) \(leaf)"))
    }

    /// A grouped verb consumes two bare words, so a third is as stray as a second is on a
    /// flat one — `models rm extra` used to drop it silently.
    @Test(arguments: [("models rm", ["models", "rm", "extra"]),
                      ("job submit", ["job", "submit", "twice"]),
                      ("settings set", ["settings", "set", "hard"]),
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

    /// Observation contributes nothing, so none of these may demand a registration.
    @Test func inspectVerbsDoNotContribute() {
        #expect(parse("headlessrun", "runtimes")?.submitsResults == false)
        #expect(parse("headlessrun", "benchmarks")?.submitsResults == false)
        #expect(parse("headlessrun", "storage", "status")?.submitsResults == false)
    }

    // MARK: - Refusals

    /// The rule the rest of the surface depends on. An ignored `key=` is
    /// indistinguishable from one that had no effect, which is the most expensive kind of
    /// wrong answer to chase on a device.
    @Test func anUnknownParameterIsRefusedAndNamed() {
        #expect(refusal("headlessrun", "runtime=mlx", "batsh=256")
            == .unknownParameters(["batsh"]))
        // Reported together, sorted, so the message is deterministic.
        #expect(refusal("headlessrun", "zebra=1", "alpha=2")
            == .unknownParameters(["alpha", "zebra"]))
    }

    /// Accepted keys are per verb, not global: `repo=` is meaningful to `models rm` and
    /// meaningless to the bench form.
    @Test func acceptedParametersAreScopedToTheVerb() {
        #expect(refusal("headlessrun", "models", "rm", "repo=org/name") == nil)
        #expect(refusal("headlessrun", "repo=org/name") == .unknownParameters(["repo"]))
    }

    /// An unrecognized runtime used to fall through to MLX, so a typo — or a host runtime
    /// a plan mispaired with this transport — ran an MLX cell and then submitted a
    /// descriptor naming a runtime it never used.
    @Test func anUnknownRuntimeIsRefusedRatherThanDefaultingToMlx() {
        #expect(refusal("headlessrun", "runtime=mlex")
            == .invalidValue(key: "runtime", value: "mlex"))
        #expect(refusal("headlessrun", "bench", "runtime=llamaX")
            == .invalidValue(key: "runtime", value: "llamaX"))
    }

    /// A host-only runtime is refused as such rather than as a typo: it is a real
    /// plan-types runtime, just not one that can run in this process.
    @Test(arguments: ["llamacpp_cli_stock_tools", "mlx_macos_pipette", "docker_vllm",
                      "uv_sglang", "llamacpp_apk_pipette", "uv_openvino"])
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

    /// A non-numeric `batch=` used to default silently to 512, and `offsets=256,abc` used
    /// to run one offset instead of two. Both are the ignored-unknown-key defect wearing a
    /// different hat: the run proceeds and the setting the caller asked for is nowhere.
    @Test func aNonNumericValueIsRefusedRatherThanDefaulted() {
        #expect(refusal("headlessrun", "batch=abc") == .invalidValue(key: "batch", value: "abc"))
        #expect(refusal("headlessrun", "offsets=256,abc")
            == .invalidValue(key: "offsets", value: "abc"))
        #expect(refusal("headlessrun", "bench", "batch=12.5")
            == .invalidValue(key: "batch", value: "12.5"))
        // The defaults still apply when the key is absent.
        #expect(parse("headlessrun") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, nGpuLayers: nil, threads: nil,
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
    }

    /// A required parameter is always an identifier, so a blank one is as unusable as an
    /// absent one — `id=` would otherwise address a job named "".
    @Test func aBlankRequiredIdentifierIsRefused() {
        #expect(refusal("headlessrun", "auth", "register", "organization=", "contact-email=a@b.c")
            == .missingParameter("organization"))
        #expect(refusal("headlessrun", "job", "run", "id=  ") == .missingParameter("id"))
    }

    @Test func anOutOfRangeValueNamesTheKeyAndValue() {
        #expect(refusal("headlessrun", "job", "run", "id=j1", "scope=sideways")
            == .invalidValue(key: "scope", value: "sideways"))
        #expect(refusal("headlessrun", "settings", "set", "worker=maybe")
            == .invalidValue(key: "worker", value: "maybe"))
        #expect(refusal("headlessrun", "bench", "spec={not json}")
            == .invalidValue(key: "model", value: "{not json}"))
    }

    /// The live plan-runner invocation: `Cell::ios_headless_args` emits `runtime=`,
    /// `model=` as canonical JSON, `benchmarks=` and `submit=1`, and nothing else.
    ///
    /// A bare invocation naming an exact model resolves to a `bench`, not a `bareBench` —
    /// the two differ only in whether the model is a coordinate or a catalog selector.
    @Test func thePlanRunnerInvocationParses() throws {
        let json = #"{"type":"mlx","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-MLX-4bit"}"#
        // `runtime=` carries the canonical `Runtime`, as `ios_headless_args` emits it.
        let runtimeJson = try String(
            decoding: JSONEncoder().encode(#require(Runtime.thisBuild(for: .mlxIosPipette))),
            as: UTF8.self)
        let parsed = parse("headlessrun", "runtime=\(runtimeJson)", "model=\(json)",
                           "benchmarks=decode_throughput_256", "submit=1")
        guard case let .bench(spec, match, _, runtime, _, _, _, benchmarks, _, _, submit) = parsed else {
            Issue.record("expected a bench command, got \(String(describing: parsed))")
            return
        }
        #expect(spec?.namedModel?.repo?.description == "LiquidAI/LFM2.5-350M-MLX-4bit")
        #expect(match == nil)
        #expect(runtime == .mlxIosPipette)
        #expect(benchmarks == ["decode_throughput_256"])
        #expect(submit)

        // AFM carries no model at all.
        #expect(refusal("headlessrun", "runtime=afm",
                        "benchmarks=decode_throughput_256", "submit=1") == nil)
    }

    /// `Display` (`{repo}[:{path}]`) is the log and warehouse identifier, not an input
    /// spelling — the crate's `--model` does not accept it either. It used to be what the
    /// plan runner emitted, which is why a dispatched GGUF cell matched nothing.
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

    /// The compact URI, the CLI's other `--model` spelling. Mirrors `model_uri.rs`'s
    /// key vocabulary per scheme.
    @Test func theCompactUriResolvesEachRepoBackedScheme() throws {
        let text = parse("headlessrun", "runtime=llama", "bench",
                         "model=gguf-text://repo=LiquidAI/LFM2.5-350M-GGUF&path=q4.gguf")
        guard case let .bench(spec, _, _, _, _, _, _, _, _, _, _) = text,
              case let .ggufText(m) = try #require(spec?.namedModel),
              case let .huggingFace(repo, path, _) = m.source else {
            Issue.record("expected a HuggingFace gguf-text model, got \(String(describing: text))")
            return
        }
        #expect(repo.description == "LiquidAI/LFM2.5-350M-GGUF")
        #expect(path.value == "q4.gguf")

        let mlx = parse("headlessrun", "runtime=mlx", "bench",
                        "model=mlx://repo=mlx-community/Q-4bit&prefix=4bit&rev=v1.2.3")
        guard case let .bench(mlxSpec, _, _, _, _, _, _, _, _, _, _) = mlx,
              case let .mlx(m2) = try #require(mlxSpec?.namedModel),
              case let .huggingFace(mlxRepo, prefix) = m2.source else {
            Issue.record("expected a HuggingFace mlx model")
            return
        }
        #expect(mlxRepo.reference == "mlx-community/Q-4bit@v1.2.3")
        #expect(prefix?.value == "4bit")
    }

    /// Spellings the `model=` selector does not accept are refused by name, not reported
    /// as a missing `repo`.
    ///
    /// The selector is deliberately narrower than `Model` itself, which decodes every
    /// source arm the crate has: a plan body has to be *readable* whatever it names, while
    /// this is a hand-typed affordance for the arms a phone can fetch.
    @Test(arguments: [
        (uri: "torch://repo=org/repo", reason: "`torch` names an engine this build does not link"),
        (uri: "openvino://repo=org/repo", reason: "`openvino` names an engine this build does not link"),
        (uri: "gguf-text://url=https://example.com/m.gguf", reason: "a `url=` source cannot be expressed"),
        (uri: "gguf-text://repo=org/repo&path=q.gguf&sha256=abc", reason: "a `sha256=` digest cannot be expressed"),
        (uri: "banana://repo=org/repo", reason: "unknown scheme `banana`"),
        (uri: "gguf-text://repo=org/repo", reason: "missing `path`"),
        (uri: "mlx://repo=org/repo&repo=other", reason: "`repo` appears more than once"),
        (uri: "mlx://repo=org/repo&nonsense=1", reason: "no such key `nonsense`"),
    ])
    func aModelUriSpellingThisClientCannotExpressIsRefused(uri: String, reason: String) {
        let refused = refusal("headlessrun", "runtime=mlx", "bench", "model=\(uri)")
        guard case let .rejected(key, actual) = refused else {
            Issue.record("expected a refusal for \(uri), got \(String(describing: refused))")
            return
        }
        #expect(key == "model")
        #expect(actual.hasPrefix(reason), "reason was: \(actual)")
    }

    /// An unknown scheme must be named as such. Key validation used to run first, so
    /// `banana://url=x` complained about the key and never mentioned the scheme.
    @Test func anUnknownSchemeIsReportedBeforeItsKeys() {
        let refused = refusal("headlessrun", "runtime=mlx", "bench", "model=banana://url=x")
        guard case let .rejected(_, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))")
            return
        }
        #expect(reason.hasPrefix("unknown scheme `banana`"))
    }

    /// A body fragment with no `=` is a malformed pair, not a mysterious key name.
    @Test func aBodyFragmentWithoutAnEqualsIsAMalformedPair() {
        let refused = refusal("headlessrun", "runtime=mlx", "bench",
                              "model=mlx://repo=org/r&garbage")
        guard case let .rejected(_, reason) = refused else {
            Issue.record("expected a refusal, got \(String(describing: refused))")
            return
        }
        #expect(reason == "`garbage` is not a `key=value` pair")
    }

    /// `model=` and `match=` answer different questions, and `spec=` is the former name
    /// for `model=`. Passing two of them is ambiguous rather than ranked.
    @Test func theModelSelectorsAreMutuallyExclusive() {
        let json = #"{"type":"mlx","source":"huggingface","org":"o","repo_name":"r"}"#
        #expect(refusal("headlessrun", "bench", "model=\(json)", "match=LFM")
            == .rejected(key: "model",
                         reason: "`match=` selects from the catalog; pass one or the other"))
        #expect(refusal("headlessrun", "bench", "model=\(json)", "spec=\(json)")
            == .rejected(key: "model", reason: "`spec=` is the former name for it; pass one"))
    }

    /// Submission is on by default, except when the caller named an exact coordinate —
    /// nothing sanctioned that model, so an ad-hoc run stays opt-in. Both branches of the
    /// rule live in one helper now; this pins them against each other.
    @Test func namingACoordinateOptsOutOfSubmissionAndSelectingDoesNot() throws {
        let json = #"{"type":"mlx","source":"huggingface","org":"o","repo_name":"r"}"#
        #expect(parse("headlessrun", "model=\(json)")?.submitsResults == false)
        #expect(parse("headlessrun", "match=LFM")?.submitsResults == true)
        // An explicit value still wins over either default.
        #expect(parse("headlessrun", "model=\(json)", "submit=1")?.submitsResults == true)
        #expect(parse("headlessrun", "match=LFM", "submit=0")?.submitsResults == false)
    }

    /// `benchmark=` is the CLI's spelling; `benchmarks=` stays because the plan emits it.
    @Test func benchmarkSingularIsAcceptedAndConflictsAreRefused() {
        #expect(parse("headlessrun", "benchmark=prefill_throughput_512") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, nGpuLayers: nil, threads: nil,
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: ["prefill_throughput_512"], model: nil, submit: true))
        #expect(refusal("headlessrun", "benchmark=a", "benchmarks=b")
            == .rejected(key: "benchmark", reason: "`benchmarks=` is the plural alias; pass one"))
    }

    // MARK: - submit

    /// Submission is the default for the catalog-resolved forms, so a scripted
    /// sweep contributes unless it opts out. The `submit: true` expectations
    /// elsewhere in this file rest on this.
    @Test(arguments: [
        ["headlessrun"],
        ["headlessrun", "runtime=afm"],
        ["headlessrun", "bench", "match=LFM2.5", "runtime=llama"],
    ])
    func submitDefaultsOn(_ args: [String]) {
        #expect((try? HeadlessCommand.parse(["/app/Pipette"] + args).get())?.submitsResults == true)
    }

    /// A `spec=` names a model the catalog never validated, so it must not
    /// publish rows without being asked: opt-*in*, unlike its catalog-resolved
    /// siblings. Nothing in the repo emits `spec=` — the plan runner selects
    /// with `model=` — so this exempts no orchestrated run.
    @Test func specRunsDoNotSubmitByDefault() throws {
        let spec = #"spec={"type":"mlx","source":"huggingface","org":"mlx-community","repo_name":"Qwen3.5-0.8B-4bit"}"#
        #expect(try parse("headlessrun", "bench", spec, "benchmarks=x")?
            .submitsResults == false)
        // …and still submits when explicitly asked.
        #expect(try parse("headlessrun", "bench", spec, "benchmarks=x", "submit=1")?
            .submitsResults == true)
    }

    /// `submit=0` is the opt-out — the only spelling that turns it off, since
    /// the parse is `!= "0"`.
    @Test(arguments: [
        ["headlessrun", "submit=0"],
        ["headlessrun", "runtime=afm", "submit=0"],
        ["headlessrun", "bench", "match=LFM2.5", "runtime=llama", "submit=0"],
    ])
    func submitZeroOptsOut(_ args: [String]) {
        #expect((try? HeadlessCommand.parse(["/app/Pipette"] + args).get())?.submitsResults == false)
    }

    /// An AFM diagnostic probe records nothing, so it contributes nothing even
    /// with submission defaulted on — the runner must not demand a registration
    /// before running one.
    @Test(arguments: HeadlessCommand.afmProbeTokens)
    func afmProbesNeverContribute(_ token: String) {
        #expect((try? HeadlessCommand.parse(
            ["/app/Pipette", "headlessrun", "runtime=afm", "metrics=\(token)"]
        ).get())?.submitsResults == false)
    }

    /// Commands that carry no `submit` never claim to contribute.
    @Test func nonBenchCommandsDoNotContribute() {
        #expect(parse("headlessrun", "jobs")?.submitsResults == false)
        #expect(parse("headlessrun", "status")?.submitsResults == false)
        #expect(parse("headlessrun", "job", "submit", "id=abc")?.submitsResults == false)
    }
}
