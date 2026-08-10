import Foundation
import Testing

@testable import Pipette

/// `HeadlessCommand.parse(url:)` + `allowedViaDeepLink` — the `pipette://`
/// deep-link surface. The URL reduces to the same token vector the argv CLI
/// parses, so these tests pin the URL → command mapping and the allow-list that
/// keeps destructive/identity verbs off the link surface. Pure, no device state.
struct DeepLinkCommandTests {

    private func parse(_ string: String) -> HeadlessCommand? {
        guard let url = URL(string: string) else { return nil }
        return try? HeadlessCommand.parse(url: url).get()
    }

    // MARK: - URL → command grammar

    /// Empty path is the bare-bench form, mirroring `headlessrun` with no verb.
    @Test func emptyPathIsTheBareBenchDefault() {
        #expect(parse("pipette://run") == .bareBench(
            runtime: .mlxIosPipette, batch: 512, nGpuLayers: nil, threads: nil,
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096],
            benchmarks: [], model: nil, submit: true))
    }

    @Test func bareBenchQueryItemsMapToParameters() {
        #expect(parse("pipette://run?runtime=llama&batch=256&metrics=prefill,decode&offsets=512,1024&match=LFM2.5&submit=1")
            == .bareBench(runtime: .llamacppIosPipette, batch: 256, nGpuLayers: nil, threads: nil,
                          metrics: ["prefill", "decode"], offsets: [512, 1024],
                          benchmarks: [], model: "LFM2.5", submit: true))
    }

    @Test func benchVerbFromFirstPathComponent() {
        #expect(parse("pipette://run/bench?match=Qwen&quant=Q4_0&runtime=llama&benchmarks=x")
            == .bench(spec: nil, model: "Qwen", quant: "Q4_0", runtime: .llamacppIosPipette,
                      batch: 512, nGpuLayers: nil, threads: nil, benchmarks: ["x"],
                      metrics: ["prefill", "decode", "maxmem"],
                      offsets: [256, 512, 1024, 2048, 4096], submit: true))
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
                repo: HFRepo.parse("mlx-community/Qwen3.5-0.8B-4bit"), prefix: nil)))),
            model: nil, quant: nil, runtime: .mlxIosPipette,
            batch: 512, nGpuLayers: nil, threads: nil, benchmarks: ["x"],
            metrics: ["prefill", "decode", "maxmem"],
            offsets: [256, 512, 1024, 2048, 4096], submit: false))
    }

    // MARK: - Host / scheme gating

    @Test func nonRunHostIsRejected() {
        // `debug` host (or any non-`run`) must not resolve — the scheme is
        // namespaced so future `pipette://` uses don't collide with the runner.
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

    /// Destructive / identity / download verbs parse but must be refused over a
    /// link — they stay CLI-only. (`auth register`, `models rm`, `job rm`,
    /// `diag memseq`, and the bare list verbs.)
    @Test(arguments: [
        // Fully formed on purpose: the point is that a register command the *parser*
        // accepts is still refused over a link, so this must not fail for want of a
        // required parameter.
        "pipette://run/auth/register?server-url=https://evil.example&organization=Evil&contact-email=e@evil.example",
        "pipette://run/job/rm?id=abc",
        "pipette://run/models/rm?name=file.gguf",
        "pipette://run/models",
        "pipette://run/models/pull?model=mlx://repo=org/model",
        "pipette://run/diag/memseq?models=a,b",
        "pipette://run/jobs",
        "pipette://run/settings",
        "pipette://run/settings/set?worker=on",
    ])
    func outOfScopeCommandsAreRejected(urlString: String) throws {
        let command = try #require(parse(urlString))
        #expect(!command.allowedViaDeepLink)
    }
}
