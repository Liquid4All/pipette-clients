import Foundation
import Testing

@testable import Pipette

/// `HeadlessCommand.parse(url:)` and `allowedViaDeepLink`: the `pipette://` deep-link
/// surface. The URL reduces to the same token vector the argv CLI parses, and the
/// allow-list keeps destructive and identity verbs off the link surface.
struct DeepLinkRouterTests {

    private func parse(_ string: String) -> HeadlessCommand? {
        guard let url = URL(string: string) else { return nil }
        return try? HeadlessCommand.parse(url: url).get()
    }

    // MARK: - URL → command grammar

    /// Empty path is the bare-bench form, mirroring `headlessrun` with no verb.
    @Test func emptyPathIsTheBareBenchDefault() {
        #expect(parse("pipette://run") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
    }

    @Test func bareBenchQueryItemsMapToParameters() {
        #expect(parse("pipette://run?runtime=llama&batch=256&metrics=prefill,decode&offsets=512,1024&match=LFM2.5&submit=1")
            == .bareBench(runtime: .llamacppIosPipette, batch: 256, metrics: ["prefill", "decode"], offsets: [512, 1024],
                          benchmarks: [], model: "LFM2.5", submit: true))
    }

    @Test func benchVerbFromFirstPathComponent() {
        #expect(parse("pipette://run/bench?match=Qwen&quant=Q4_0&runtime=llama&benchmarks=x")
            == .bench(spec: nil, model: "Qwen", quant: "Q4_0", runtime: .llamacppIosPipette,
                      batch: 512, runtimeFlags: nil, benchmarks: ["x"],
                      metrics: ["prefill", "decode", "maxmem"],
                      offsets: [256, 512, 1024, 2048, 4096], submit: true, benchmarkFlags: nil))
    }

    @Test func nestedJobVerbMapsPathComponentsToBareWords() {
        #expect(parse("pipette://run/job/run?id=abc&scope=failed")
            == .runJob(id: "abc", scope: .failed))
        #expect(parse("pipette://run/job/submit?id=abc") == .submitJob(id: "abc"))
        #expect(parse("pipette://run/job/export?id=abc") == .exportJob(id: "abc"))
    }

    @Test func statusFromPath() {
        #expect(parse("pipette://run/status") == .status)
    }

    /// A `spec=` JSON value carries `=`-bearing content through percent-encoding;
    /// the token parser splits on the first `=` only, so the value survives whole.
    @Test func specJsonValueSurvivesUrlEncoding() throws {
        var comps = URLComponents()
        comps.scheme = "pipette"
        comps.host = "run"
        comps.path = "/bench"
        comps.queryItems = [
            .init(name: "spec", value: #"{"type":"mlx","source":"huggingface","org":"mlx-community","repo_name":"Qwen3.5-0.8B-4bit"}"#),
            .init(name: "benchmarks", value: "x"),
        ]
        let url = try #require(comps.url)
        #expect(try HeadlessCommand.parse(url: url).get() == .bench(
            spec: .model(.mlx(Mlx(source: .huggingFace(
                repo: HfRepo.parseOrgRepo("mlx-community/Qwen3.5-0.8B-4bit"), prefix: nil)))),
            model: nil, quant: nil, runtime: .mlxIosPipette,
            batch: 512, runtimeFlags: nil, benchmarks: ["x"],
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096], submit: false, benchmarkFlags: nil))
    }

    // MARK: - Host / scheme gating

    @Test func nonRunHostIsRejected() {
        // The scheme is namespaced so future `pipette://` uses don't collide with the runner.
        #expect(parse("pipette://debug/bench?match=x") == nil)
        #expect(parse("pipette://status") == nil)
    }

    @Test func unknownVerbUnderRunFailsLikeTheCli() {
        #expect(parse("pipette://run/frobnicate") == nil)
    }

    // MARK: - Allow-list

    @Test(arguments: [
        "pipette://run",
        "pipette://run/bench?match=x",
        "pipette://run?runtime=afm",
        "pipette://run/job/run?id=abc",
        "pipette://run/job/submit?id=abc",
        "pipette://run/job/export?id=abc",
        "pipette://run/status",
    ])
    func inScopeCommandsAreAllowed(urlString: String) throws {
        let command = try #require(parse(urlString))
        #expect(command.allowedViaDeepLink)
    }

    /// Destructive, identity and download verbs parse but stay CLI-only.
    @Test(arguments: [
        // Fully formed on purpose: the point is that a register command the *parser*
        // accepts is still refused over a link, so this must not fail for want of a
        // required parameter.
        "pipette://run/auth/register?server-url=https://evil.example&organization=Evil&contact-email=e@evil.example&client-details=evil",
        "pipette://run/job/rm?id=abc",
        "pipette://run/models/rm?name=file.gguf",
        "pipette://run/models",
        "pipette://run/models/pull?model=mlx://repo=org/model",
        "pipette://run/diag/memseq?models=a,b",
        "pipette://run/jobs",
        "pipette://run/settings/run",
        // The crate's positional parses over a link as it does on the command line, and is
        // refused as the key form is.
        "pipette://run/benchmarks/show/prefill_throughput_512",
    ])
    func outOfScopeCommandsAreRejected(urlString: String) throws {
        let command = try #require(parse(urlString))
        #expect(!command.allowedViaDeepLink)
    }

    /// A result id's `/` splits it into two path components, so the positional form of a
    /// result does not parse over a link at all.
    @Test func aResultPositionalDoesNotParseOverALink() {
        #expect(parse("pipette://run/results/show/job-1/cell-2") == nil)
    }
}

/// `DeepLinkRouter.matchingModels`: which on-device models a bench link selects.
/// Pure, no device state: the caller turns an empty result into "Model not on device"
/// and a multi-row result into a refusal.
struct DeepLinkModelMatchTests {
    private let llama = RuntimeType.llamacppIosPipette

    private func onDisk(_ filename: String, repo: String) -> DiscoveredModel {
        DiscoveredModel(source: ggufTextFixture(repo, filename),
                        path: "/tmp/\(filename)", sizeBytes: 1_000_000)
    }

    /// Both Ministral sizes are offered by the catalog, so a device can hold both. The
    /// hint matches each, and the link must refuse rather than bench whichever the
    /// models directory yields first.
    @Test func aHintMatchingTwoInstalledModelsIsAmbiguous() {
        let models = [
            onDisk("Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
                   repo: "mistralai/Ministral-3-3B-Instruct-2512-GGUF"),
            onDisk("Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
                   repo: "mistralai/Ministral-3-8B-Instruct-2512-GGUF"),
        ]
        let matches = DeepLinkRouter.matchingModels(models, spec: nil, runtime: llama,
                                                    modelHint: "Ministral", quant: "Q4_K_M")
        #expect(matches.count == 2, "the caller refuses on more than one")
    }

    @Test func aNarrowerHintResolves() {
        let models = [
            onDisk("Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
                   repo: "mistralai/Ministral-3-3B-Instruct-2512-GGUF"),
            onDisk("Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
                   repo: "mistralai/Ministral-3-8B-Instruct-2512-GGUF"),
        ]
        let matches = DeepLinkRouter.matchingModels(models, spec: nil, runtime: llama,
                                                    modelHint: "Ministral-3-8B", quant: nil)
        #expect(matches.map(\.name) == ["Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"])
    }

    /// Nothing on device is distinct from several: the caller sends one to the Models
    /// tab and refuses the other.
    @Test func noMatchIsEmptyRatherThanAnArbitraryRow() {
        let models = [onDisk("LFM2-700M-Q4_0.gguf", repo: "LiquidAI/LFM2-700M-GGUF")]
        #expect(DeepLinkRouter.matchingModels(models, spec: nil, runtime: llama,
                                              modelHint: "Ministral", quant: nil).isEmpty)
    }

    /// An empty hint selects every row for the runtime. Whether a several-row result is
    /// then refused is the caller's decision rather than this function's: only the bare
    /// sweep form naming no model is exempt, and it runs the first match under its
    /// "whatever matching model is already on device" contract. See `refusesAsAmbiguous`.
    @Test func anEmptyHintSelectsEveryModelForTheRuntime() {
        let models = [
            onDisk("LFM2-700M-Q4_0.gguf", repo: "LiquidAI/LFM2-700M-GGUF"),
            onDisk("Qwen3.5-2B-Q4_K_M.gguf", repo: "unsloth/Qwen3.5-2B-GGUF"),
        ]
        #expect(DeepLinkRouter.matchingModels(models, spec: nil, runtime: llama,
                                              modelHint: nil, quant: nil).count == 2)
    }

    /// The tiebreak `resolveCatalogRows` already had: a filename contained by a longer one
    /// stays addressable, so the refusal's "name the model exactly" is advice the matcher
    /// honours rather than a dead end.
    @Test func exactFilenameWinsOverALongerOneContainingIt() {
        // Same quant on both, deliberately: with different quants neither filename
        // contains the other and the substring filter alone would already return one
        // row, so the test would pass with the preference removed.
        let models = [
            onDisk("Bonsai-8B-Q1_0.gguf", repo: "prism-ml/Bonsai-8B-gguf"),
            onDisk("Ternary-Bonsai-8B-Q1_0.gguf", repo: "prism-ml/Ternary-Bonsai-8B-gguf"),
        ]
        #expect("Ternary-Bonsai-8B-Q1_0.gguf".contains("Bonsai-8B-Q1_0.gguf"),
                "fixture must actually collide, or this proves nothing")
        #expect(DeepLinkRouter.matchingModels(models, spec: nil, runtime: llama,
                                              modelHint: "Bonsai-8B-Q1_0.gguf",
                                              quant: nil).map(\.name)
            == ["Bonsai-8B-Q1_0.gguf"])
    }

    /// `launchBench` is private and `@MainActor`, so this is the only place the
    /// refuse-vs-run rule can be asserted.
    @Test func onlyTheBareSweepNamingNothingEscapesTheAmbiguityRefusal() {
        // The bare form naming nothing runs the first match, per its contract.
        #expect(!DeepLinkRouter.refusesAsAmbiguous(bareSweep: true, spec: nil, modelHint: nil,
                                                   quant: nil, matchCount: 2))
        // An empty `match=` decodes to "" and means the same as absent.
        #expect(!DeepLinkRouter.refusesAsAmbiguous(bareSweep: true, spec: nil, modelHint: "",
                                                   quant: "", matchCount: 2))
        // Both halves are load-bearing. The bare form that DID name a model is refused...
        #expect(DeepLinkRouter.refusesAsAmbiguous(bareSweep: true, spec: nil,
                                                  modelHint: "Ministral", quant: nil,
                                                  matchCount: 2))
        // ...and so is a `bench` link naming nothing, which `BenchmarksCommands.run` refuses
        // too. Exempting every link that named nothing would miss this one.
        #expect(DeepLinkRouter.refusesAsAmbiguous(bareSweep: false, spec: nil, modelHint: nil,
                                                  quant: nil, matchCount: 2))
        // `quant=` alone is a selector.
        #expect(DeepLinkRouter.refusesAsAmbiguous(bareSweep: true, spec: nil, modelHint: nil,
                                                  quant: "Q4_K_M", matchCount: 2))
        // Unambiguous always runs, whichever form asked.
        #expect(!DeepLinkRouter.refusesAsAmbiguous(bareSweep: false, spec: nil,
                                                   modelHint: "Ministral", quant: nil,
                                                   matchCount: 1))
    }

    /// A hint that is nobody's full name stays ambiguous, so `launchBench` still refuses
    /// rather than the preference silently picking one.
    @Test func aPartialHintIsNotDisambiguated() {
        let models = [
            onDisk("Bonsai-8B-Q1_0.gguf", repo: "prism-ml/Bonsai-8B-gguf"),
            onDisk("Ternary-Bonsai-8B-Q2_0.gguf", repo: "prism-ml/Ternary-Bonsai-8B-gguf"),
        ]
        #expect(DeepLinkRouter.matchingModels(models, spec: nil, runtime: llama,
                                              modelHint: "Bonsai-8B", quant: nil).count == 2)
    }

    /// A link that names a model runs it as named, so a result reports the link's vault
    /// label and not the one the manifest holds. A digest names no label, so it reports
    /// none.
    @Test func aNamedModelIsCarriedAsTheLinkNamedIt() throws {
        let stored = DiscoveredModel(source: try vaultTextSpec("vault_1", label: "first fetch"),
                                     path: "/tmp/vault_1.gguf", sizeBytes: 1_000_000)
        for label in ["acme LFM3 q4", nil] {
            let named = try vaultTextSpec("vault_1", label: label)
            let matches = DeepLinkRouter.matchingModels([stored], spec: .model(named),
                                                        runtime: llama, modelHint: nil, quant: nil)
            #expect(matches.count == 1)
            let match = try #require(matches.first)
            #expect(try reportedLabel(match.source) == label)
            #expect(match.path == stored.path)
        }

        let prefix = String(try modelDescriptorDigest(stored.source).prefix(digestDisplayLen))
        let byDigest = DeepLinkRouter.matchingModels([stored], spec: .digest(prefix),
                                                     runtime: llama, modelHint: nil, quant: nil)
        let entry = try #require(byDigest.first)
        #expect(try reportedLabel(entry.source) == nil)
        #expect(entry.path == stored.path)
    }
}
