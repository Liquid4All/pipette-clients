import Foundation
import Testing

@testable import Pipette

private let prefill = #"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}"#

/// The keys of ``BenchmarkStoreTests/aCachedServerObjectKeepsTheValuesALaterBuildReads()``
/// typed as a later build might type them.
private nonisolated struct LaterBuild: Decodable {
    let unitScale: Double
    let fromTheFuture: Double
    let temperature: Double
    let bigCount: Int64
    let maxCount: Int64
    let negative: Int
    let flag: Bool
    let none: Int?

    enum CodingKeys: String, CodingKey {
        case unitScale = "parameter_unit_scale"
        case fromTheFuture = "parameter_from_the_future"
        case temperature = "parameter_temperature"
        case bigCount = "parameter_big_count"
        case maxCount = "parameter_max_count"
        case negative = "parameter_negative"
        case flag = "parameter_flag"
        case none = "parameter_none"
    }
}

/// The catalog on disk, as pipette-cli's `benchmarks/store.rs` lays it out.
struct BenchmarkStoreTests {
    private func write(_ text: String, to relativePath: String, in store: BenchmarkStore) throws {
        let url = store.root.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    private func exists(_ relativePath: String, in store: BenchmarkStore) -> Bool {
        FileManager.default.fileExists(atPath: store.root.appendingPathComponent(relativePath).path)
    }

    private func localDef(_ id: String) -> BenchmarkDefinition {
        .prefillThroughput(PrefillThroughputBenchmark(
            benchmarkId: BenchmarkId(validated: id), parameterPrefillTokens: 128))
    }

    private func cachePrefill(_ id: String, in store: BenchmarkStore) throws {
        try store.cacheRemote(json: #"{"benchmark_id":"\#(id)","benchmark_type":"prefill_throughput","parameter_prefill_tokens":128}"#)
    }

    /// The bookkeeping a pull writes, so a test can see a write clobber it.
    private func writeBookkeeping(_ store: BenchmarkStore) throws {
        try store.putRemoteIndex(Coding.decoder.decode(
            [PipetteMgmtClient.BenchmarkSummary].self,
            from: Data(#"[{"benchmark_id":"index","benchmark_type":"prefill_throughput"}]"#.utf8)))
        try store.putSyncState(RemoteSyncState(
            pulledAt: "2026-01-01T00:00:00Z", benchmarkCount: 2, serverUrl: nil,
            benchmarksEtag: nil))
    }

    private func bookkeepingSurvives(_ store: BenchmarkStore) throws -> Bool {
        try store.listRemoteIndex().count == 1 && store.getSyncState() != nil
    }

    // MARK: - Layout

    /// Ids are the filename verbatim, as the crate writes them, so the same benchmark is the
    /// same file on both clients. An id that could escape the directory is refused rather
    /// than encoded into something unrecognizable.
    @Test func idsAreWrittenVerbatimAndTraversalIsRefused() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        try store.cacheRemote(json: prefill)
        #expect(exists("remote/detail/p.json", in: store))
        try store.putLocal(localDef("p"))
        #expect(exists("local/p.json", in: store))

        #expect(throws: BenchmarkStoreError.unsafeBenchmarkId("../escape")) {
            try store.putLocal(localDef("../escape"))
        }
    }

    /// As `BenchmarkId::try_new` refuses it in the crate. Otherwise a server id no plan could
    /// name back reaches disk and is listed, refused only later at the spec boundary. A server
    /// body with such an id does not convert, so there is no remote write to refuse.
    @Test func anIdThatIsNotABenchmarkIdIsRefused() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        for id in ["decode 512", "\tdecode", "decode\n", "", ".hidden", "a:b", ".", ".."] {
            #expect(throws: BenchmarkStoreError.unsafeBenchmarkId(id)) {
                try store.putLocal(localDef(id))
            }
            let sent = try Coding.decoder.decode(
                RemoteBenchmark.self, from: JSONSerialization.data(withJSONObject: [
                    "benchmark_id": id, "benchmark_type": "prefill_throughput",
                    "parameter_prefill_tokens": 8,
                ]))
            #expect(throws: (any Error).self) { try ConvertedRemote(sent) }
            #expect(try store.get(.remote(BenchmarkId(validated: id))) == nil)
            #expect(try store.get(.local(BenchmarkId(validated: id))) == nil)
        }
        #expect(store.list(.remote).isEmpty)
        #expect(store.list(.local).isEmpty)
    }

    /// The split is the whole reason the two halves are separate directories.
    @Test func clearRemoteKeepsLocal() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        try StandardBenchmarks.seedStandardLocal(store, kinds: [.prefillThroughput])
        try store.putRemoteIndex([])
        try store.cacheRemote(json: prefill)
        try store.writeLegacyImg80()
        try store.putSyncState(RemoteSyncState(
            pulledAt: "2026-01-01T00:00:00Z", benchmarkCount: 1, serverUrl: nil,
            benchmarksEtag: try EntityTag("e")))

        store.clearRemote()

        #expect(try store.listRemoteIndex().isEmpty)
        #expect(try store.get(.remote("p")) == nil)
        #expect(try store.getSyncState() == nil)
        #expect(!exists("remote", in: store))
        #expect(!store.list(.local).isEmpty)
    }

    // MARK: - get

    @Test func aMissingDefinitionIsNil() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        #expect(try store.get(.local("absent")) == nil)
    }

    /// As the crate's `get` reports it. The listing still skips it, so one bad file does not
    /// hide the rest.
    @Test func aCorruptDefinitionIsReportedNotHidden() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try write("{bad json", to: "local/broken.json", in: store)
        try store.putLocal(localDef("prefill_throughput_1"))

        #expect(throws: (any Error).self) { try store.get(.local("broken")) }
        #expect(store.list(.local).map(\.benchmarkId.value) == ["prefill_throughput_1"])
    }

    /// The retired shape; the crate's `read_def_file` refuses it by name rather than as a
    /// missing field. A remote file reads as it did on fetch, where the extra key is ignored:
    /// refusing it would make every claim refetch it.
    @Test func aNestedParametersObjectIsRefusedOnlyInALocalFile() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let body = #"{"benchmark_id":"nested","benchmark_type":"prefill_throughput","parameter_prefill_tokens":8,"parameters":{"prefill_tokens":8}}"#
        try write(body, to: "local/nested.json", in: store)
        try store.cacheRemote(json: body)

        #expect(throws: BenchmarkStoreError.nestedParameters("nested.json")) {
            try store.get(.local("nested"))
        }
        #expect(store.list(.local).isEmpty)
        #expect(try store.get(.remote("nested")) != nil)
        #expect(store.list(.remote).map(\.benchmarkId.value) == ["nested"])
    }

    // MARK: - Remote definitions

    /// A remote definition is cached as the object the server sent, every key kept (ones
    /// this build does not model, an eval's inline `samples`, a corpus benchmark's nested
    /// `dataset`), and reads back: the VL one runs its 80 images.
    @Test(arguments: [
        (#"{"benchmark_id":"vl_throughput_512x512_text1024_decode64_img80","benchmark_type":"vl_throughput","parameter_image_width":512,"parameter_image_height":512,"parameter_text_tokens":1024,"parameter_decode_tokens":64,"parameter_num_images":80,"parameter_temperature":0.7,"parameter_from_the_future":3,"org_id":"org_test"}"#, 80),
        (#"{"benchmark_id":"eval_mmlu","benchmark_type":"eval","parameter_eval_id":"mmlu_pro","parameter_dataset_name":"edge_2026.03.1","parameter_max_tokens":64,"samples":[{"id":"q1","messages":[{"role":"user","content":"2+2?"}]}]}"#, nil),
        (#"{"benchmark_id":"vl_spec_decode_mmspec_test_gqa_k8_bs8","benchmark_type":"vl_spec_decode","parameter_draft_length":8,"parameter_max_tokens":2048,"parameter_batch_size":8,"dataset":{"format":"vl-spec-decode-corpus/1","adapter":"mmspec@1","digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","bytes":15955456,"subset":"GQA","upstream":{"source":"git","url":"github.com/killthefullmoon/MMSpec","revision":"290486861eabd746075ca776adb66175636c1fc0","subpath":"dataset/MMSpec/test","identity":"2923603292c7dd463ad01a119f57965817778fc3","digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","license":"Apache-2.0"},"url":"/datasets/vl_spec_decode_mmspec_test_gqa_k8_bs8"}}"#, nil),
    ] as [(String, Int?)])
    func aRemoteDetailKeepsWhatTheServerSent(sent: String, images: Int?) throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let sentDocument = try jsonDocument(Data(sent.utf8))
        let id = try #require(sentDocument["benchmark_id"] as? String)

        try store.cacheRemote(json: sent)

        let onDisk = try jsonDocument(Data(contentsOf: store.remoteDetailPath(id)))
        #expect(onDisk == sentDocument)
        let definition = try #require(try store.get(.remote(BenchmarkId(validated: id))))
        if let images {
            guard case let .vlThroughput(vl) = definition else {
                Issue.record("expected vl_throughput, got \(definition)")
                return
            }
            #expect(Int(vl.parameterNumImages.value) == images)
        }
    }

    /// `1.0` and `0.25` read back as the same doubles, an integer past 2^53 exactly, and
    /// nested objects and arrays whole. `RawJSONValue` re-encodes the body, so the spelling
    /// may change (`1.0` is written `1`); the value may not.
    @Test func aCachedServerObjectKeepsTheValuesALaterBuildReads() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let sent = #"{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512,"parameter_unit_scale":1.0,"parameter_from_the_future":0.25,"parameter_temperature":0.7,"parameter_big_count":9007199254740993,"parameter_max_count":9223372036854775807,"parameter_negative":-5,"parameter_flag":true,"parameter_none":null,"parameter_shape":{"layers":[1,2.5,{"name":"x"}],"tag":"t"}}"#

        try store.cacheRemote(json: sent)

        let bytes = try Data(contentsOf: store.remoteDetailPath("p"))
        let read = try JSONDecoder().decode(LaterBuild.self, from: bytes)
        #expect(read.unitScale == 1.0)
        #expect(read.fromTheFuture == 0.25)
        #expect(read.temperature == 0.7)
        #expect(read.bigCount == 9_007_199_254_740_993)
        #expect(read.maxCount == Int64.max)
        #expect(read.negative == -5)
        #expect(read.flag)
        #expect(read.none == nil)
        let sentDocument = try jsonDocument(Data(sent.utf8))
        #expect(try jsonDocument(bytes) == sentDocument)
    }

    /// Neither as a definition nor as a cache hit: an older build's `remote/<id>.json` may have
    /// lost a parameter, and nothing in it says which.
    @Test func aLegacyDetailIsNeverRead() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.writeLegacyImg80()
        let id = BenchmarkId(validated: img80Id)

        #expect(try store.get(.remote(id)) == nil)
        #expect(store.list(.remote).isEmpty)
        #expect(!store.hasRemoteDetail(id: id))
    }

    /// The older build may still share the workspace.
    @Test func cachingARemoteDetailLeavesItsLegacyFile() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let legacy = try store.writeLegacyImg80()

        try store.cacheRemote(json: img80Wire)

        #expect(FileManager.default.fileExists(atPath: legacy.path), "the older build's file went")
        #expect(store.hasRemoteDetail(id: BenchmarkId(validated: img80Id)))
    }

    /// In any case, cached and pruned alike, such a benchmark caches under `detail/`, lists
    /// like any other, and leaves `state/` and an older build's files alone. `ids` is sorted,
    /// as the listing is.
    @Test(arguments: [["index", "state", "sync"], ["Index", "SYNC", "State"]])
    func aBenchmarkNamedLikeTheBookkeepingLeavesStateAlone(ids: [String]) throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try writeBookkeeping(store)
        try store.writeOlderBuildBookkeeping(
            rows: [prefill], server: "https://mgmt.test", listEtag: #""list-v1""#,
            benchmarkEtags: [:])
        let olderState = try #require(ids.first { $0.lowercased() == "state" })
        try write("{}", to: "remote/\(olderState).json", in: store)

        for id in ids { try cachePrefill(id, in: store) }

        #expect(try bookkeepingSurvives(store))
        #expect(exists("remote/index.json", in: store), "an older build's index went")
        #expect(exists("remote/sync.json", in: store), "an older build's sync state went")
        #expect(exists("remote/\(olderState).json", in: store),
                "the older build's `\(olderState)` went")
        for id in ids { #expect(exists("remote/detail/\(id).json", in: store), "\(id)") }
        #expect(store.list(.remote).map(\.benchmarkId.value) == ids)

        try store.pruneRemoteDetails(keeping: ids)
        #expect(try bookkeepingSurvives(store))
        #expect(store.list(.remote).map(\.benchmarkId.value) == ids)

        try store.pruneRemoteDetails(keeping: [])
        #expect(try bookkeepingSurvives(store))
        #expect(store.list(.remote).isEmpty)
    }

    /// No older build reads, writes or deletes `remote/state/`; `remote/index.json` and
    /// `remote/sync.json` are theirs.
    @Test func theBookkeepingLivesUnderState() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }

        try writeBookkeeping(store)

        #expect(exists("remote/state/index.json", in: store))
        #expect(exists("remote/state/sync.json", in: store))
        #expect(!exists("remote/index.json", in: store))
        #expect(!exists("remote/sync.json", in: store))
    }

    /// Not as the stored list, not as the sync state, and not, in any case, as a definition
    /// of a benchmark named `index` or `sync`.
    @Test func anOlderBuildsBookkeepingIsNeverRead() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.writeOlderBuildBookkeeping(
            rows: [prefill], server: "https://mgmt.test", listEtag: #""list-v1""#,
            benchmarkEtags: ["p": #""p-v1""#])

        #expect(try store.listRemoteIndex().isEmpty)
        #expect(try store.getSyncState() == nil)
        #expect(store.list(.remote).isEmpty)
        for id in ["index", "sync", "Index", "SYNC"] {
            #expect(try store.get(.remote(BenchmarkId(validated: id))) == nil, "\(id)")
        }
    }

    /// The remote half lists `detail/` alone, so its bookkeeping and an older build's files
    /// stay out of it.
    @Test func listSkipsWhatItCannotReadAndTheFilesBesideDetail() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        #expect(store.list(.remote).isEmpty)

        try writeBookkeeping(store)
        try cachePrefill("prefill_throughput_512", in: store)
        try write("{ not json", to: "remote/detail/broken.json", in: store)
        try write("not a catalog entry", to: "remote/detail/notes.txt", in: store)
        try write(#"{"benchmark_id":"prefill_throughput_2","benchmark_type":"prefill_throughput","parameter_prefill_tokens":1}"#,
                  to: "remote/prefill_throughput_2.json", in: store)

        #expect(store.list(.remote).map(\.benchmarkId.value) == ["prefill_throughput_512"])
        #expect(try bookkeepingSurvives(store))
    }

    // MARK: - Remote sync metadata

    /// A key this client does not model survives a write and a read, and so does a row it
    /// cannot convert.
    @Test func theIndexKeepsEveryRowAsSent() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let rows = #"[{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512,"server_only_key":"kept"},{"benchmark_id":"x","benchmark_type":"some_future_metric","parameter_future":1}]"#
        try store.putRemoteIndex(
            try Coding.decoder.decode([PipetteMgmtClient.BenchmarkSummary].self, from: Data(rows.utf8)))

        let read = try store.listRemoteIndex()
        #expect(read.map(\.benchmarkId) == ["p", "x"])
        let first = try #require(read.first)
        let object = try #require(JSONSerialization.jsonObject(with: first.rawJson) as? [String: Any])
        #expect(object["server_only_key"] as? String == "kept")
        let onDisk = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: store.statePath("index.json")))
                as? NSArray)
        let sent = try #require(JSONSerialization.jsonObject(with: Data(rows.utf8)) as? NSArray)
        #expect(onDisk == sent)
    }

    /// Not an absent one, as in the crate's `get_sync_state`.
    @Test func aCorruptSyncStateIsAnError() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try write("{", to: "remote/state/sync.json", in: store)
        #expect(throws: (any Error).self) { try store.getSyncState() }
    }

    // MARK: - Prune

    /// `state/` survives, as does every file an older build wrote directly under `remote/`:
    /// that build may still share the workspace. A benchmark named like the bookkeeping is
    /// kept while it is listed.
    @Test func pruneRemoteDetailsDropsOnlyDepartedBenchmarks() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        // The bookkeeping first, so caching the benchmarks named like it has a chance to
        // clobber it.
        try writeBookkeeping(store)
        for id in ["prefill_throughput_1", "prefill_throughput_2", "index", "sync"] {
            try cachePrefill(id, in: store)
        }
        try store.writeOlderBuildBookkeeping(
            rows: [prefill], server: "https://mgmt.test", listEtag: #""list-v1""#,
            benchmarkEtags: ["p": #""p-v1""#])
        let olderBuildFiles = ["index", "sync", "state", "prefill_throughput_1", "prefill_throughput_3"]
            .map { "remote/\($0).json" }
        for file in ["remote/state.json", "remote/prefill_throughput_1.json",
                     "remote/prefill_throughput_3.json"] {
            try write("{}", to: file, in: store)
        }

        try store.pruneRemoteDetails(keeping: ["prefill_throughput_1", "index"])

        #expect(exists("remote/detail/prefill_throughput_1.json", in: store))
        #expect(!exists("remote/detail/prefill_throughput_2.json", in: store))
        #expect(exists("remote/detail/index.json", in: store))
        #expect(!exists("remote/detail/sync.json", in: store))
        for file in olderBuildFiles { #expect(exists(file, in: store), "\(file) went") }
        #expect(try bookkeepingSurvives(store))
    }

    /// An empty catalog clears every detail but not the index.
    @Test func pruneRemoteDetailsHandlesEmptyAndMissing() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        try store.pruneRemoteDetails(keeping: ["anything"])
        try store.putRemoteIndex([])
        try store.pruneRemoteDetails(keeping: ["anything"])

        try cachePrefill("prefill_throughput_1", in: store)
        try store.pruneRemoteDetails(keeping: [])

        #expect(!store.hasRemoteDetail(id: "prefill_throughput_1"))
        #expect(exists("remote/state/index.json", in: store))
    }

    /// A body cached without an ETag drops the old one. The old one goes before the body is
    /// written, so a failed write leaves no ETag, never one for a body the file no longer
    /// holds; a directory at the file's path stands in for that failure.
    @Test func aDetailsEtagIsReplacedWithItsBody() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        let id = BenchmarkId(validated: "prefill_throughput_1")
        let body = #"{"benchmark_id":"prefill_throughput_1","benchmark_type":"prefill_throughput","parameter_prefill_tokens":128}"#
        try store.cacheRemote(json: body, etag: #""v1""#)
        #expect(store.remoteDetailEtag(id: id) == (try EntityTag(#""v1""#)))
        #expect(try String(contentsOf: store.remoteEtagPath(id.value), encoding: .utf8)
            .hasSuffix("\n" + #""v1""#))

        try store.cacheRemote(json: body, etag: #""v2""#)
        #expect(store.remoteDetailEtag(id: id) == (try EntityTag(#""v2""#)))
        try store.cacheRemote(json: body)
        #expect(store.remoteDetailEtag(id: id) == nil)
        #expect(!FileManager.default.fileExists(atPath: store.remoteEtagPath(id.value).path))

        try store.cacheRemote(json: body, etag: #""v3""#)
        #expect(try store.getRemote(id: id)?.etag == (try EntityTag(#""v3""#)))
        // Another writer's body beside this ETag, or an ETag file without a digest.
        try Data(body.replacingOccurrences(of: "128", with: "256").utf8)
            .write(to: store.remoteDetailPath(id.value))
        #expect(store.remoteDetailEtag(id: id) == nil, "an ETag recorded for another body")
        try Data(#""v3""#.utf8).write(to: store.remoteEtagPath(id.value))
        #expect(store.remoteDetailEtag(id: id) == nil, "an ETag file without a digest")

        try FileManager.default.removeItem(at: store.remoteDetailPath(id.value))
        #expect(store.remoteDetailEtag(id: id) == nil, "an ETag without its body")

        try FileManager.default.createDirectory(
            at: store.remoteDetailPath(id.value).appendingPathComponent("blocked"),
            withIntermediateDirectories: true)
        #expect(throws: (any Error).self) { try store.cacheRemote(json: body, etag: #""v4""#) }
        #expect(!FileManager.default.fileExists(atPath: store.remoteEtagPath(id.value).path))
    }

    @Test func pruningADetailDropsItsEtag() throws {
        let store = makeTemporaryBenchmarkStore()
        defer { removeStore(store) }
        for id in ["prefill_throughput_1", "prefill_throughput_3"] {
            try store.cacheRemote(
                json: #"{"benchmark_id":"\#(id)","benchmark_type":"prefill_throughput","parameter_prefill_tokens":128}"#,
                etag: #""\#(id)""#)
        }

        try store.pruneRemoteDetails(keeping: ["prefill_throughput_3"])

        #expect(!FileManager.default.fileExists(atPath: store.remoteEtagPath("prefill_throughput_1").path))
        #expect(store.remoteDetailEtag(id: "prefill_throughput_3")
                == (try EntityTag(#""prefill_throughput_3""#)))
    }
}
