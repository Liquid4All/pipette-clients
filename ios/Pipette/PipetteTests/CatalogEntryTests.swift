import Foundation
import Synchronization
import Testing

@testable import Pipette

/// The fallback builders use unchecked initializers for literals. These tests validate
/// the literals and use the format that the management server's `DescriptorCatalogShortList` writes.
@MainActor
struct CatalogEntryTests {
    // MARK: - The fallback

    @Test func fallbackIsTheLfmRows() {
        #expect(CatalogEntry.fallback.count == 22)
        #expect(CatalogEntry.fallback.allSatisfy { $0.name.hasPrefix("LFM") })
    }

    @Test func everyFallbackRowHasAValidRepoAndSize() throws {
        for entry in CatalogEntry.fallback {
            let repo = try #require(entry.source.repo, "\(entry.repoIdentifier) must have a repo")
            #expect(HfOrg.validate(repo.org.value), "invalid org in \(entry.repoIdentifier)")
            #expect(HfRepoName.validate(repo.repoName.value), "invalid repo name in \(entry.repoIdentifier)")
            #expect(entry.sizeBytes > 0)
        }
    }

    @Test func ggufFallbackRowsHaveParsableFilenameAndQuant() {
        for entry in CatalogEntry.fallback {
            guard case let .ggufText(m) = entry.source else { continue }
            guard case let .huggingFace(_, path, _) = m.source else {
                Issue.record("\(entry.repoIdentifier) should be a HuggingFace ggufText")
                continue
            }
            #expect(path.value.hasSuffix(".gguf"))
            #expect(entry.quant != nil, "\(path.value) should parse a quant")
        }
    }

    @Test func mlxFallbackRowsAreBareReposWithBitWidthQuant() {
        for entry in CatalogEntry.fallback {
            guard case .mlx = entry.source else { continue }
            // `Model.modelId` strips this suffix to rejoin an MLX row with its GGUF
            // sibling, so a mismatch silently splits the model in the UI.
            let quant = entry.quant ?? ""
            #expect(quant.hasSuffix("bit"), "\(entry.repoIdentifier) quant=\(entry.quant ?? "nil")")
            #expect(entry.repoIdentifier.lowercased().hasSuffix("-" + quant.lowercased()))
            #expect(!entry.repoIdentifier.contains(":"))
        }
    }

    /// The downloaded list and job wizard group by `modelId`, so the `-MLX-` infix
    /// (LiquidAI) must not split an MLX build from its GGUF stem.
    @Test func mlxAndGgufFallbackRowsShareModelId() {
        let ggufModelIdByName = Dictionary(
            CatalogEntry.fallback.compactMap { entry -> (String, String)? in
                guard case .ggufText = entry.source else { return nil }
                return (entry.name, entry.modelId)
            },
            uniquingKeysWith: { first, _ in first })
        for entry in CatalogEntry.fallback {
            guard case .mlx = entry.source, let gguf = ggufModelIdByName[entry.name] else { continue }
            #expect(entry.modelId == gguf, "\(entry.name): MLX \(entry.modelId) != GGUF \(gguf)")
        }
    }

    @Test func fallbackRowIdsAreUnique() {
        let ids = CatalogEntry.fallback.map(\.id)
        #expect(Set(ids).count == ids.count)
    }

    @Test func separateOffersShareOneSelectedDownload() throws {
        let source = try #require(CatalogEntry.fallback.first).source
        let first = CatalogEntry(name: "First offer", source: source, sizeBytes: 0,
                                 quant: "Q4_0", selector: "first::gguf::q4_0")
        let alias = CatalogEntry(name: "Alias", source: source, sizeBytes: 1024,
                                 quant: "Q4_0", selector: "alias::gguf::q4_0")
        let key = try #require(first.storageKey)
        #expect(first.offerID != alias.offerID)
        #expect(first.storageKey == alias.storageKey)
        let selected = CatalogEntry.selectedDownloads([first, alias], selected: [key], unavailable: [])
        #expect(selected.count == 1)
        #expect(selected.first?.sizeBytes == 1024)
        #expect(CatalogEntry.selectedDownloads([first, alias], selected: [key], unavailable: [key]).isEmpty)
    }

    // MARK: - Size filtering

    @Test func unknownDownloadSizesAlwaysRequireConfirmation() throws {
        let source = try #require(CatalogEntry.fallback.first).source
        let small = CatalogEntry(name: "Small", source: source, sizeBytes: 40, quant: nil)
        let unknown = CatalogEntry(name: "Unknown", source: source, sizeBytes: 0, quant: nil)
        let large = CatalogEntry(name: "Large", source: source, sizeBytes: 100, quant: nil)

        #expect(CatalogEntry.downloadWarning(for: [unknown], largeDownloadThreshold: 100) == .unknownSize)
        #expect(CatalogEntry.downloadWarning(for: [large, unknown], largeDownloadThreshold: 100) == .unknownSize)
        #expect(CatalogEntry.downloadWarning(for: [large, small], largeDownloadThreshold: 100) == .largeDownload)
        #expect(CatalogEntry.downloadWarning(for: [large], largeDownloadThreshold: 100) == nil)
    }

    @Test(arguments: [
        (size: Int64(0), quota: Int64(10), ram: UInt64(10), fits: true),        // unknown size
        (size: Int64(4_000), quota: Int64(10_000), ram: UInt64(10_000), fits: true),
        (size: Int64(10_000), quota: Int64(10_000), ram: UInt64(10_000), fits: true),  // at both limits
        (size: Int64(10_001), quota: Int64(10_000), ram: UInt64(20_000), fits: false), // over the quota
        (size: Int64(9_000), quota: Int64(20_000), ram: UInt64(8_000), fits: false),   // over the RAM
    ])
    func fittingKeepsOnlyRowsTheDeviceCanHold(size: Int64, quota: Int64, ram: UInt64, fits: Bool) throws {
        let row = try #require(CatalogEntry.fallback.first)
        let sized = CatalogEntry(name: row.name, source: row.source, sizeBytes: size, quant: row.quant)
        let kept = CatalogEntry.fitting([sized], quotaBytes: quota, ramBytes: ram)
        #expect(kept.count == (fits ? 1 : 0), "size \(size) quota \(quota) ram \(ram)")
    }

    // MARK: - The short list

    /// One model entry with a GGUF and an MLX member, and one runtime entry.
    private static let shortList = #"""
    {
      "items": [
        {
          "name": "LFM2.5-350M",
          "release_time": "2026-11-01T00:00:00Z",
          "descriptor_kind": "model",
          "members": [
            {
              "selector": "LFM2.5-350M::gguf::q4_0",
              "member_key": {"kind": "model", "format": "gguf", "quantization": "q4_0"},
              "digest": "d1", "added_at": "2026-11-01T00:00:00Z", "plan_toml": "",
              "descriptor": {"type": "gguf_text", "source": "huggingface", "org": "LiquidAI",
                             "repo_name": "LFM2.5-350M-GGUF", "path": "LFM2.5-350M-Q4_0.gguf"},
              "size_bytes": 219309792
            },
            {
              "selector": "LFM2.5-350M::mlx::4bit",
              "member_key": {"kind": "model", "format": "mlx", "quantization": "4bit"},
              "digest": "d2", "added_at": "2026-11-01T00:00:00Z", "plan_toml": "",
              "descriptor": {"type": "mlx", "source": "huggingface", "org": "LiquidAI",
                             "repo_name": "LFM2.5-350M-MLX-4bit"},
              "size_bytes": null
            }
          ]
        },
        {
          "name": "llamacpp-ios",
          "release_time": "2026-11-01T00:00:00Z",
          "descriptor_kind": "runtime",
          "members": [
            {
              "selector": "llamacpp-ios::ios-arm64",
              "member_key": {"kind": "runtime", "flavor": "ios-arm64"},
              "digest": "d3", "added_at": "2026-11-01T00:00:00Z", "plan_toml": "",
              "descriptor": {"type": "llamacpp_ios_pipette"}
            }
          ]
        }
      ],
      "withheld": {"member_count": 0, "entry_count": 0, "members": []}
    }
    """#

    @Test func parseMapsEachModelMemberToARow() throws {
        let entries = try ModelShortList.parse(Data(Self.shortList.utf8))
        #expect(entries.count == 2)
        #expect(entries.allSatisfy { $0.name == "LFM2.5-350M" })

        let gguf = try #require(entries.first { if case .ggufText = $0.source { true } else { false } })
        #expect(gguf.repoIdentifier == "LiquidAI/LFM2.5-350M-GGUF")
        #expect(gguf.sizeBytes == 219_309_792)
        // Derived from the filename, as for the installed copy, not the catalog's `q4_0`.
        #expect(gguf.quant == "Q4_0")

        let mlx = try #require(entries.first { if case .mlx = $0.source { true } else { false } })
        #expect(mlx.sizeBytes == 0, "a null size is unknown")
        #expect(mlx.quant == "4bit")
        #expect(mlx.modelId == gguf.modelId)
    }

    /// The catalog's family wins. Without one, or with a blank one, an entry takes the name
    /// before its first hyphen.
    @Test(arguments: [(nil, "LFM2.5"), ("LFM", "LFM"), ("", "LFM2.5"), (" ", "LFM2.5")] as [(String?, String)])
    func parseTakesTheCatalogFamily(family: String?, expected: String) throws {
        let body = family.map {
            Self.shortList.replacingOccurrences(
                of: #""descriptor_kind": "model","#,
                with: #""descriptor_kind": "model", "family": "\#($0)","#)
        } ?? Self.shortList
        let entries = try ModelShortList.parse(Data(body.utf8))
        #expect(entries.count == 2)
        #expect(entries.allSatisfy { $0.family == expected })
    }

    static let logoDigest = "c2e7d21098f9131392a4dc3789f3b8de9561b38e3ad627389872d24a5ca4fa02"

    @Test(arguments: [
        ("\"\(CatalogEntryTests.logoDigest)\"", CatalogEntryTests.logoDigest),
        ("\"\(CatalogEntryTests.logoDigest.uppercased())\"", nil),
        ("\"\(CatalogEntryTests.logoDigest.dropFirst())\"", nil),
        ("\"../\(CatalogEntryTests.logoDigest)\"", nil),
        ("42", nil),
        ("false", nil),
        ("{}", nil),
        ("null", nil),
        (nil, nil),
    ] as [(String?, String?)])
    func parseKeepsTheFamilyLogoDigest(logo: String?, expected: String?) throws {
        let body = logo.map {
            Self.shortList.replacingOccurrences(
                of: #""descriptor_kind": "model","#,
                with: #""descriptor_kind": "model", "family_logo_sha256": \#($0),"#)
        } ?? Self.shortList
        let entries = try ModelShortList.parse(Data(body.utf8))
        #expect(entries.count == 2)
        #expect(entries.allSatisfy { $0.familyLogoDigest == expected })
    }

    @Test func filteringAFamilyKeepsItsCatalogLogo() throws {
        let visible = CatalogEntry.gguf("LFM2.5-350M", "LiquidAI/LFM2.5-350M-GGUF", "Q4_0", bytes: 1)
        let hidden = CatalogEntry(name: "LFM2.5-1.2B", source: visible.source, sizeBytes: 2, quant: "Q4_0",
                                  family: "lfm2.5", familyLogoDigest: Self.logoDigest)
        let logoDigests = CatalogEntry.familyLogoDigests(in: [hidden, visible])
        let groups = PresetModelGroup.groups(from: [visible])
        let family = try #require(ModelFamily.grouping(groups, logoDigests: logoDigests).first)

        #expect(family.members.map(\.name) == [visible.name])
        #expect(family.logoDigest == Self.logoDigest)
    }

    @Test(arguments: [
        ("LFM2.5-350M", "LFM2.5"),
        ("Qwen3.5-0.8B-Instruct", "Qwen3.5"),
        ("Private MLX Model", "Private MLX Model"),
        ("-x", "-x"),
    ])
    func defaultFamilyIsTheNameBeforeTheFirstHyphen(name: String, family: String) {
        #expect(CatalogEntry.defaultFamily(name) == family)
    }

    @Test func fallbackRowsNameTheirFamilies() {
        let counts = Dictionary(grouping: CatalogEntry.fallback, by: \.family)
            .mapValues { Set($0.map(\.name)).count }
        #expect(counts == ["LFM2.5": 5, "LFM2": 3])
    }

    /// Families and their models sort by name, ignoring case. A query keeps only the
    /// matching models and drops a family with none.
    @Test(arguments: [
        ("", [["Granite": ["granite-4-h", "Granite-4-micro"]], ["LFM2.5": ["LFM2.5-1.2B", "lfm2.5-350m"]],
              ["Qwen3": ["Qwen3-0.6B"]]]),
        ("350", [["LFM2.5": ["lfm2.5-350m"]]]),
        ("granite", [["Granite": ["granite-4-h", "Granite-4-micro"]]]),
    ] as [(String, [[String: [String]]])])
    func familiesSortAndFilterTheirModels(query: String, expected: [[String: [String]]]) {
        let entries: [CatalogEntry] = [
            .gguf("Qwen3-0.6B", "Qwen/Qwen3-0.6B-GGUF", "Q4_0", bytes: 1, family: "Qwen3"),
            .gguf("lfm2.5-350m", "LiquidAI/LFM2.5-350M-GGUF", "Q4_0", bytes: 1, family: "LFM2.5"),
            .gguf("Granite-4-micro", "ibm/granite-4-micro-GGUF", "Q4_0", bytes: 1, family: "Granite"),
            .gguf("LFM2.5-1.2B", "LiquidAI/LFM2.5-1.2B-GGUF", "Q4_0", bytes: 1, family: "LFM2.5"),
            .gguf("granite-4-h", "ibm/granite-4-h-GGUF", "Q4_0", bytes: 1, family: "Granite"),
        ]
        let families = ModelFamily.grouping(PresetModelGroup.groups(from: entries).filter { $0.matches(query) })
            .map { [$0.name: $0.members.map(\.name)] }
        #expect(families == expected)
    }

    @Test func parseSkipsAMemberItCannotDecode() throws {
        let body = Self.shortList.replacingOccurrences(of: #""type": "mlx""#, with: #""type": "torch""#)
        let entries = try ModelShortList.parse(Data(body.utf8))
        #expect(entries.count == 1)
        #expect(entries.first?.repoIdentifier == "LiquidAI/LFM2.5-350M-GGUF")
    }

    @Test func aVaultOfferRetainsItsCatalogNameAndQuant() throws {
        let body = Data(#"""
        {"items":[{"name":"Private MLX Model","descriptor_kind":"model","members":[
          {"member_key":{"kind":"model","format":"mlx","quantization":"4bit"},
           "descriptor":{"type":"mlx","source":"vault_archive",
                         "url":"/vault/resources/vault_1/content","principal_id":"org_1"},
           "size_bytes":1024},
          {"member_key":{"kind":"model","format":"mlx","quantization":"8bit"},
           "descriptor":{"type":"mlx","source":"absolute_dir","dir":"/tmp/model"},
           "size_bytes":2048}
        ]}]}
        """#.utf8)
        let entries = try ModelShortList.parse(body)
        let entry = try #require(entries.first)
        #expect(entries.count == 1)
        #expect(entry.name == "Private MLX Model")
        #expect(entry.sizeBytes == 1024)
        #expect(ModelFetch.vaultFiles(entry.source)?.count == 1)
        let offered = try #require(NewJobView.offerable(downloaded: [], catalog: entries).first)
        #expect(offered.quant == "4bit")
    }

    @Test func parseRefusesABodyThatIsNotAShortList() {
        #expect(throws: (any Error).self) { try ModelShortList.parse(Data("[]".utf8)) }
    }

    @Test func parsePreservesThePinnedOnnxPackage() throws {
        let revision = String(repeating: "a", count: 40)
        let prefix = "cpu_and_mobile/cpu-int4-rtn-block-32"
        let body = Data("""
        {"items":[{"name":"Pinned ONNX Model","descriptor_kind":"model","members":[
          {"selector":"pinned-onnx::onnx_genai::int4",
           "member_key":{"kind":"model","format":"onnx_genai","quantization":"int4"},
           "descriptor":{"type":"onnx_genai","source":"huggingface","org":"example",
                         "repo_name":"example-onnx","revision":"\(revision)","prefix":"\(prefix)"},
           "size_bytes":1024}
        ]}]}
        """.utf8)
        let entries = try ModelShortList.parse(body)
        let row = try #require(entries.first)
        guard case let .onnxGenai(model) = row.source,
              case let .huggingFace(repo, package) = model.source else {
            Issue.record("Expected a Hugging Face ONNX offer")
            return
        }
        #expect(repo.revision?.value == revision)
        #expect(package?.value == prefix)
        #expect(row.quant == "cpu-int4-rtn-block-32")
        #expect(row.sizeBytes == 1024)
    }

    @Test func aSavedCatalogOnlyLoadsForItsSigningIdentity() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let route = try RoutedURLProtocol.serve { _, _ in .reply(200, Self.shortList) }
        let registration = try Self.catalogRegistration(server: route.server)
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        try ws.registration.put(registration)

        let notifications = Mutex(0)
        let observer = NotificationCenter.default.addObserver(
            forName: ModelShortList.didChange, object: ws.modelShortList, queue: nil
        ) { _ in
            notifications.withLock { $0 += 1 }
        }
        defer { NotificationCenter.default.removeObserver(observer) }

        let offered = try await ModelShortList.refresh(ws: ws, http: route.http)
        #expect(offered == 2)
        let reopened = PipetteWorkspace(root: ws.root, cacheRoot: ws.cacheRoot)
        ModelShortList.loadCache(ws: reopened)
        #expect(notifications.withLock { $0 } == 1)
        #expect(ModelShortList.current(ws: reopened).map(\.name) == ["LFM2.5-350M", "LFM2.5-350M"])

        try ws.registration.put(Self.catalogRegistration(server: ServerUrl("https://other.test")))
        #expect(ModelShortList.current(ws: ws).map(\.id) == CatalogEntry.fallback.map(\.id))
        ModelShortList.loadCache(ws: reopened)
        #expect(ModelShortList.current(ws: reopened).map(\.id) == CatalogEntry.fallback.map(\.id))

        try ws.registration.put(registration)
        try ws.registration.putPrivateKey(try PrivateKey(String(repeating: "b4", count: 32)))
        ModelShortList.loadCache(ws: reopened)
        #expect(ModelShortList.current(ws: reopened).map(\.id) == CatalogEntry.fallback.map(\.id))
    }

    @Test(arguments: [false, true])
    func catalogRefreshDiscardsAnAnswerAfterInvalidation(changeCollector: Bool) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let route = try RoutedURLProtocol.serve { _, _ in
            do {
                if changeCollector {
                    try ws.registration.put(Self.catalogRegistration(server: ServerUrl("https://other.test")))
                } else {
                    try ModelShortList.clearCache(ws: ws)
                }
                return .reply(200, Self.shortList)
            } catch {
                return .reply(500)
            }
        }
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        try ws.registration.put(Self.catalogRegistration(server: route.server))

        await #expect(throws: CancellationError.self) {
            try await ModelShortList.refresh(ws: ws, http: route.http)
        }
        #expect(!FileManager.default.fileExists(atPath: ModelShortList.cacheFile(ws).path))
        #expect(ModelShortList.current(ws: ws).map(\.id) == CatalogEntry.fallback.map(\.id))
    }

    @Test func aLogoIsCachedOnlyForItsWorkspaceAndSigningIdentity() async throws {
        let workspace = makeTemporaryWorkspace()
        let otherWorkspace = makeTemporaryWorkspace()
        defer {
            removeWorkspace(workspace)
            removeWorkspace(otherWorkspace)
        }
        let digest = Self.logoDigest
        let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="
        let route = try RoutedURLProtocol.serve { _, _ in
            .reply(200, #"{"sha256":"\#(digest)","logo_uri":"data:image/png;base64,\#(png)"}"#)
        }
        let registration = try Self.catalogRegistration(server: route.server)
        for ws in [workspace, otherWorkspace] {
            try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
            try ws.registration.put(registration)
        }

        let cache = RemoteLogos()
        let owner = try ModelShortList.Owner(workspace.registration.mgmtSession())
        let key = RemoteLogos.Key(workspace: workspace, owner: owner, digest: digest)
        let otherKey = RemoteLogos.Key(workspace: otherWorkspace, owner: owner, digest: digest)
        #expect(await cache.load(key, workspace: workspace, http: route.http) != nil)
        #expect(cache.loaded(otherKey) == nil)
        #expect(await cache.load(otherKey, workspace: otherWorkspace, http: route.http) != nil)

        try workspace.registration.putPrivateKey(try PrivateKey(String(repeating: "b4", count: 32)))
        let newOwner = try ModelShortList.Owner(workspace.registration.mgmtSession())
        let newKey = RemoteLogos.Key(workspace: workspace, owner: newOwner, digest: digest)
        let staleKey = RemoteLogos.Key(workspace: workspace, owner: owner, digest: String(repeating: "0", count: 64))
        #expect(cache.loaded(newKey) == nil)
        #expect(await cache.load(staleKey, workspace: workspace, http: route.http) == nil)
        #expect(await cache.load(newKey, workspace: workspace, http: route.http) != nil)
        #expect(route.seen.methodsAndPaths == Array(
            repeating: "GET /clients/me/descriptor-catalog/family-logos/\(digest)", count: 3))
        #expect(route.seen.requests.allSatisfy {
            $0.value(forHTTPHeaderField: "X-Client-Id") == registration.clientId.value
        })
    }

    enum ProfileRefreshCase: CaseIterable {
        case automatic, complete, legacy, legacyWithCapabilities, profileUnavailable
    }

    @Test(arguments: ProfileRefreshCase.allCases)
    func explicitCatalogRefreshRepairsOnlyMissingProfiles(scenario: ProfileRefreshCase) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let profileFields: String
        switch scenario {
        case .complete: profileFields = #","device_os_name":"ios""#
        case .legacyWithCapabilities:
            profileFields = #","capabilities":["runtime:llama_cpp"],"device_ram_bytes":4096"#
        default: profileFields = ""
        }
        let profile = """
        {"client_id":"catalog-client","organization":"example","client_details":"",
         "contact_email":"device@example.test","status":"approved"\(profileFields)}
        """
        let route = try RoutedURLProtocol.serve { request, _ in
            switch (request.url?.path, scenario) {
            case ("/clients/me/descriptor-catalog", _): .reply(200, Self.shortList)
            case ("/clients/me", .profileUnavailable): .reply(503)
            case ("/clients/me", _): .reply(200, profile)
            default: .reply(404)
            }
        }
        try ws.registration.putPrivateKey(try PrivateKey(testPrivateKeyHex))
        try ws.registration.put(Self.catalogRegistration(server: route.server))

        let offered: Int
        if scenario == .automatic {
            offered = try await ModelShortList.refresh(ws: ws, http: route.http)
        } else {
            offered = try await ModelShortList.refreshExplicitly(ws: ws, http: route.http)
        }
        #expect(offered == 2)

        var expected: [String] = []
        if scenario != .automatic { expected.append("GET /clients/me") }
        if scenario == .legacy || scenario == .legacyWithCapabilities {
            expected.append("PATCH /clients/me")
            let body = try #require(route.seen.bodies.first)
            let patch = try #require(JSONSerialization.jsonObject(with: body) as? [String: Any])
            #expect(patch["device_os_name"] != nil)
            #expect(patch["device_name"] == nil)
            if scenario == .legacy {
                let flags = try #require(patch["capabilities"] as? [String])
                #expect(flags.contains { $0.hasPrefix(runtimeBuildFlagPrefix) })
            } else {
                #expect(patch["capabilities"] == nil)
                #expect(patch["device_ram_bytes"] == nil)
            }
        }
        expected.append("GET /clients/me/descriptor-catalog")
        #expect(route.seen.methodsAndPaths == expected)
    }

    private nonisolated static func catalogRegistration(server: ServerUrl) throws -> IdentityRegistration {
        IdentityRegistration(
            clientId: try ClientId("catalog-client"), status: "approved", serverUrl: server,
            organization: "example", contactEmail: "device@example.test",
            registeredAt: "2026-11-19T00:00:00Z")
    }
}
