import Foundation
import Testing

@testable import Pipette

/// `bench match=` catalog resolution. `BenchmarksCommands.resolveCatalogRows` is the
/// whole rule: `BenchmarksCommands.run` guards only on its emptiness and its count, so
/// what is pinned here is what the selector does.
struct BenchmarksCommandsTests {
    private let llama = RuntimeType.llamacppIosPipette

    private func names(_ match: String, quant: String? = nil) -> [String] {
        BenchmarksCommands.resolveCatalogRows(match: match, quant: quant, runtime: llama)
            .map(\.name).sorted()
    }

    /// A fixed colliding pair: `Bonsai 8B` is a substring of `Ternary Bonsai 8B`, and both
    /// are GGUF. Authored here because the shipping catalog holds no same-runtime
    /// containment pair.
    private static let colliding: [CatalogEntry] = [
        .gguf("Bonsai 8B", "prism-ml/Bonsai-8B-gguf", "Q1_0", bytes: 1_158_654_496),
        .gguf("Ternary Bonsai 8B", "prism-ml/Ternary-Bonsai-8B-gguf", "Q2_0", bytes: 2_182_184_672),
    ]

    private func fixtureNames(_ match: String, quant: String? = nil) -> [String] {
        BenchmarksCommands.resolveCatalogRows(match: match, quant: quant, runtime: llama,
                                          rows: Self.colliding)
            .map(\.name).sorted()
    }

    @Test func exactNameWinsOverALongerRowContainingIt() {
        // The fixture must actually collide, or this passes without exercising anything.
        #expect(fixtureNames("Bonsai 8B").count == 1,
                "fixture should collide: both rows contain the shorter name")
        #expect(fixtureNames("Bonsai 8B") == ["Bonsai 8B"],
                "an exactly-named row should win over the longer rows containing it")
        // Without the preference both rows match, which is what makes the above a tiebreak
        // rather than a plain filter.
        #expect(fixtureNames("Bonsai") == ["Bonsai 8B", "Ternary Bonsai 8B"])
    }

    /// The gap `resolveCatalogRows` documents: `quant=` narrows before the preference, so an exactly
    /// named row publishing no such quant is gone and a sibling wins the selection alone.
    @Test func anExactNameLosesToASiblingWhenItPublishesNoSuchQuant() {
        #expect(fixtureNames("Bonsai 8B", quant: "Q2_0") == ["Ternary Bonsai 8B"])
    }

    /// A match that is nobody's full name is left alone, so a genuinely ambiguous
    /// selector still fails loudly instead of picking one. Asserts the rows, not a count.
    @Test func aPartialNameIsNotDisambiguated() {
        #expect(names("ministral", quant: "Q4_K_M")
            == ["Ministral 3 3B Instruct 2512", "Ministral 3 8B Instruct 2512"])
    }

    @Test func everyRowIsReachableByItsOwnCoordinate() {
        let rows = CatalogEntry.catalog.filter { isCompatible($0.source, llama) }
        #expect(!rows.isEmpty)
        for row in rows {
            let resolved = BenchmarksCommands.resolveCatalogRows(
                match: row.repoIdentifier, quant: row.quant, runtime: llama)
            #expect(resolved.map(\.id) == [row.id], "\(row.repoIdentifier):\(row.quant ?? "-")")
        }
    }

    @Test func anEmptyMatchSelectsEveryRowForTheRuntime() {
        let all = CatalogEntry.catalog.filter { isCompatible($0.source, llama) }
        #expect(BenchmarksCommands.resolveCatalogRows(match: "", quant: nil, runtime: llama).count
            == all.count)
    }
}

/// `benchmarks list` and `benchmarks show`, as pipette-cli's `commands/benchmarks.rs`
/// prints them, over both halves of the catalog.
struct BenchmarksCatalogCommandsTests {
    private let prefill = BenchmarkDefinition.prefillThroughput(PrefillThroughputBenchmark(
        benchmarkId: BenchmarkId(validated: "prefill_throughput_512"),
        parameterPrefillTokens: 512))
    private let decode = BenchmarkDefinition.decodeThroughput(DecodeThroughputBenchmark(
        benchmarkId: BenchmarkId(validated: "decode_throughput_512_100"),
        parameterPrefillTokens: 512,
        parameterDecodeTokens: 100))

    /// Each row names the reference `run` and `show` take, `local/` then `remote/`, so the
    /// same id on both halves is two rows.
    @Test func listNamesEachBenchmarkByItsReference() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.benchmarks.putLocal(prefill)
        try ws.benchmarks.cacheRemote(prefill)
        try ws.benchmarks.cacheRemote(decode)

        #expect(BenchmarksCommands.rows(benchmarkType: nil, store: ws.benchmarks) == [
            "benchmark benchmark=local/prefill_throughput_512 type=prefill_throughput",
            "benchmark benchmark=remote/decode_throughput_512_100 type=decode_throughput",
            "benchmark benchmark=remote/prefill_throughput_512 type=prefill_throughput",
        ])
        #expect(BenchmarksCommands.rows(benchmarkType: .decodeThroughput, store: ws.benchmarks)
            == ["benchmark benchmark=remote/decode_throughput_512_100 type=decode_throughput"])
    }

    /// As the crate's `write_benchmark`. The batch size is printed even when the entry
    /// omits it, so a reader need not know that absence means one in flight.
    @Test func showPrintsTheSourceAndTheParameters() {
        #expect(BenchmarksCommands.writeBenchmark(decode, source: .local)
            == "benchmark id=decode_throughput_512_100 type=decode_throughput source=local "
                + "parameter_prefill_tokens=512 parameter_decode_tokens=100 "
                + "parameter_batch_size=1")
        #expect(BenchmarksCommands.writeBenchmark(prefill, source: .remote)
            == "benchmark id=prefill_throughput_512 type=prefill_throughput source=remote "
                + "parameter_prefill_tokens=512 parameter_batch_size=1")
    }

    /// A reference is looked up on the half it names: a `local/` one does not resolve
    /// against the synced catalog, and the command fails rather than showing a neighbour.
    @Test func showResolvesOnlyTheHalfTheReferenceNames() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.benchmarks.cacheRemote(prefill)

        #expect(BenchmarksCommands.show(.remote("prefill_throughput_512"), ws: ws))
        #expect(!BenchmarksCommands.show(.local("prefill_throughput_512"), ws: ws))
    }

    /// Unknown to both `show` and `run`: the file may have lost a parameter, so it is never
    /// read, and it is left to that build.
    @Test func aReferenceOnlyAnOlderBuildCachedIsUnknown() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let legacy = try ws.benchmarks.writeLegacyImg80()
        let reference = "remote/\(img80Id)"
        var skipped: [String] = []

        #expect(!BenchmarksCommands.show(.remote(BenchmarkId(validated: img80Id)), ws: ws))
        let resolved = try await JobCell.resolvable(
            benchmarkIds: [reference], store: ws.benchmarks, server: nil, http: .shared, auth: nil
        ) { reference, _ in skipped.append(reference) }
        let cells = JobCell.pending(benchmarkIds: [reference], for: .appleFoundation,
                                    store: ws.benchmarks) { skipped.append($0) }

        #expect(resolved.references.isEmpty)
        #expect(cells.isEmpty)
        #expect(skipped == [reference, reference])
        #expect(FileManager.default.fileExists(atPath: legacy.path), "the older build's file went")
    }
}

/// `benchmarks run`'s `buildBenchmarkFlags`, as the crate's `commands/benchmarks.rs` tests
/// `build_benchmark_flags`.
struct BuildBenchmarkFlagsTests {
    /// A knob the cell's variant has no field for is an error, never a silent no-op.
    @Test(arguments: [
        (BenchmarkType.eval, readinessWait(1800)),
        (.eval, skipThermal()),
    ] as [(BenchmarkType, BenchmarksCommands.CellKnobs)])
    func aKnobTheCellDoesNotCarryIsRejected(
        benchmarkType: BenchmarkType, knobs: BenchmarksCommands.CellKnobs
    ) throws {
        let (runtime, model) = try ggufTextCell()
        #expect(throws: BenchmarkFlagsError.self) {
            try BenchmarksCommands.buildBenchmarkFlags(
                benchmarkType: benchmarkType, runtime: runtime, model: model, knobs: knobs)
        }
    }

    @Test func noKnobCarriesNoBenchmarkFlags() throws {
        let (runtime, model) = try ggufTextCell()
        #expect(try BenchmarksCommands.buildBenchmarkFlags(
            benchmarkType: .prefillThroughput, runtime: runtime, model: model,
            knobs: .init()) == nil)
    }

    @Test func readinessOnTimingCellIsAccepted() throws {
        let (runtime, model) = try ggufTextCell()
        let flags = try #require(try BenchmarksCommands.buildBenchmarkFlags(
            benchmarkType: .prefillThroughput, runtime: runtime, model: model,
            knobs: readinessWait(1800)))
        #expect(flags.readiness?.maxWaitSecs == 1800)
    }

    /// The thermal waiver alone builds a readiness override, and invents no deadline.
    @Test func readinessSkipThermalAloneIsAcceptedOnATimingCell() throws {
        let (runtime, model) = try ggufTextCell()
        let flags = try #require(try BenchmarksCommands.buildBenchmarkFlags(
            benchmarkType: .prefillThroughput, runtime: runtime, model: model,
            knobs: skipThermal()))
        let readiness = try #require(flags.readiness)
        #expect(readiness.skipThermal == true)
        #expect(readiness.maxWaitSecs == nil,
                "waiving the thermal gate must not fabricate a deadline")
    }

    /// `false` is not an override, so an unused waiver leaves no readiness knob.
    @Test func readinessSkipThermalFalseRecordsNoOverride() throws {
        let (runtime, model) = try ggufTextCell()
        let flags = try BenchmarksCommands.buildBenchmarkFlags(
            benchmarkType: .prefillThroughput, runtime: runtime, model: model, knobs: .init())
        #expect(flags?.readiness == nil,
                "an unset thermal waiver must not create a readiness override")
    }
}

/// The llama.cpp + GGUF-text cell the flag tests route through, so a case reads as its
/// benchmark type and knobs.
private nonisolated func ggufTextCell() throws -> (Runtime, Model) {
    let runtime = try #require(Runtime.thisBuild(for: .llamacppIosPipette))
    return (runtime, try parseModelUri("gguf-text://repo=unsloth/gemma-4-E2B-it-GGUF&path=g.gguf"))
}

// One knob set, everything else default. File-scope so a nonisolated `@Test(arguments:)` can
// build them.
private nonisolated func readinessWait(_ secs: UInt64) -> BenchmarksCommands.CellKnobs {
    BenchmarksCommands.CellKnobs(readinessMaxWaitSecs: secs)
}

private nonisolated func skipThermal() -> BenchmarksCommands.CellKnobs {
    BenchmarksCommands.CellKnobs(readinessSkipThermal: true)
}

/// `parseBenchmarkFlagsJson`, the CLI's `parse_benchmark_flags_json`: the JSON form of what
/// `buildBenchmarkFlags` builds from the single-setting knobs.
struct ParseBenchmarkFlagsJsonTests {
    private func parse(_ json: String, benchmarkType: BenchmarkType) throws -> BenchmarkFlags? {
        let (runtime, model) = try ggufTextCell()
        return try BenchmarksCommands.parseBenchmarkFlagsJson(
            Data(json.utf8), benchmarkType: benchmarkType, runtime: runtime, model: model)
    }

    @Test(arguments: [
        (BenchmarkType.prefillThroughput,
         #"{"readiness":{"max_wait_secs":1800,"skip_thermal":true}}"#,
         BenchmarksCommands.CellKnobs(readinessMaxWaitSecs: 1800, readinessSkipThermal: true)),
        (.decodeThroughput, #"{"readiness":{"max_wait_secs":900}}"#, readinessWait(900)),
        (.endToEndLatency, #"{"readiness":{"skip_thermal":true}}"#, skipThermal()),
        (.prefillThroughput, "{}", BenchmarksCommands.CellKnobs()),
    ] as [(BenchmarkType, String, BenchmarksCommands.CellKnobs)])
    func benchmarkFlagsJsonMatchesItsShorthand(
        benchmarkType: BenchmarkType, json: String, knobs: BenchmarksCommands.CellKnobs
    ) throws {
        let (runtime, model) = try ggufTextCell()
        #expect(try parse(json, benchmarkType: benchmarkType)
            == BenchmarksCommands.buildBenchmarkFlags(
                benchmarkType: benchmarkType, runtime: runtime, model: model, knobs: knobs))
    }

    @Test(arguments: [
        (BenchmarkType.prefillThroughput, #"[{"readiness":{"max_wait_secs":1800}}]"#,
         "notAnObject"),
        (.prefillThroughput, #""max_wait_secs=1800""#, "notAnObject"),
        (.prefillThroughput, #"{"benchmark_type":"prefill_throughput"}"#,
         #"axisNotAccepted(key: "benchmark_type")"#),
        // iOS declares neither server timeout, so the CLI's knob is an unknown field here.
        (.prefillThroughput, #"{"http_timeout_seconds":600}"#,
         #"name: "http_timeout_seconds""#),
        (.eval, #"{"readiness":{"max_wait_secs":1800}}"#, "noSuchCombination"),
        (.prefillThroughput, #"{"readiness":"#, "Code=3840"),
        (.prefillThroughput, #"{"readiness":{"max_wait_secs":0}}"#, "zeroWait"),
    ] as [(BenchmarkType, String, String)])
    func benchmarkFlagsJsonIsRefused(benchmarkType: BenchmarkType, json: String, needle: String) {
        let error = #expect(throws: (any Error).self) {
            try parse(json, benchmarkType: benchmarkType)
        }
        let got = error.map { String(describing: $0) } ?? "no error"
        #expect(got.contains(needle), "error was: \(got)")
    }
}
