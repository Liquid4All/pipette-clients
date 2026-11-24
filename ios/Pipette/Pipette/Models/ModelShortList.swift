import Foundation
import Synchronization

/// The models this device offers for download. The list is the server's catalog short
/// list for this device (`GET /clients/me/descriptor-catalog`), or the copy of it that
/// the last refresh cached. When neither offers a model, the list is
/// ``CatalogEntry/fallback``.
///
/// The app loads the cache once at launch, before any screen or headless command reads
/// ``current(ws:)``. Refreshes publish only for the signing identity that requested them.
nonisolated final class ModelShortList: Sendable {
    static let didChange = Notification.Name("Pipette.ModelShortList.didChange")

    struct Owner: Codable, Hashable, Sendable {
        let server: ServerUrl
        let clientId: ClientId
        let publicKeyHex: String
        let organization: String

        init(_ session: MgmtSession) throws {
            server = session.serverUrl
            clientId = session.registration.clientId
            publicKeyHex = try PipetteMgmtClient.Auth.publicKeyHexFromPrivate(session.auth.privateKeyHex).value
            organization = session.registration.organization
        }
    }

    struct Snapshot {
        let entries: [CatalogEntry]
        let owner: Owner?
    }

    private struct Cache: Codable {
        let owner: Owner
        let body: Data
    }

    private struct State {
        var owner: Owner?
        var entries: [CatalogEntry] = []
        // A cache load or clear also advances the floor for responses allowed to publish.
        var issued = 0
        var published = 0
    }

    private let state = Mutex(State())

    static func current(ws: PipetteWorkspace) -> [CatalogEntry] {
        snapshot(ws: ws).entries
    }

    static func snapshot(ws: PipetteWorkspace) -> Snapshot {
        ws.modelShortList.state.withLock { state in
            guard !state.entries.isEmpty, let owner = state.owner else {
                return Snapshot(entries: CatalogEntry.fallback, owner: nil)
            }
            guard ws.registration.withIdentityLock({ currentOwner(ws) == owner }) else {
                state.owner = nil
                state.entries = []
                return Snapshot(entries: CatalogEntry.fallback, owner: nil)
            }
            return Snapshot(entries: state.entries, owner: owner)
        }
    }

    /// `<root>/catalog/short-list.json` holds one server answer and its owner.
    static func cacheFile(_ ws: PipetteWorkspace) -> URL {
        ws.root.appendingPathComponent("catalog", isDirectory: true)
            .appendingPathComponent("short-list.json")
    }

    /// Install the saved list only if it belongs to the current signing identity.
    static func loadCache(ws: PipetteWorkspace) {
        defer { ws.modelShortList.notifyChange() }
        ws.modelShortList.state.withLock { state in
            state.owner = nil
            state.entries = []
            state.published = state.issued
            ws.registration.withIdentityLock {
                guard let owner = currentOwner(ws),
                      let data = try? Data(contentsOf: cacheFile(ws)),
                      let cache = try? Coding.decoder.decode(Cache.self, from: data),
                      cache.owner == owner,
                      let entries = try? parse(cache.body)
                else { return }
                state.owner = owner
                state.entries = entries
                AppLog.storage.info("model catalog loaded from cache: offered=\(entries.count)")
            }
        }
    }

    /// Returns the number of models that the server offers. The app uses the parsed
    /// answer even if a cache write fails and the method throws.
    @discardableResult
    static func refresh(ws: PipetteWorkspace, http: HttpClient = .shared) async throws -> Int {
        let (session, owner, request) = try ws.modelShortList.state.withLock { state in
            try ws.registration.withIdentityLock {
                let session = try ws.registration.mgmtSession()
                let owner = try Owner(session)
                state.issued += 1
                return (session, owner, state.issued)
            }
        }
        let body = try await PipetteMgmtClient.ClientApi.descriptorCatalog(
            server: session.serverUrl, http: http, auth: session.auth)
        let entries = try parse(body)
        let data = try Coding.encoder.encode(Cache(owner: owner, body: body))
        var didPublish = false
        defer {
            if didPublish { ws.modelShortList.notifyChange() }
        }
        try ws.modelShortList.state.withLock { state in
            try ws.registration.withIdentityLock {
                try Task.checkCancellation()
                guard request > state.published, currentOwner(ws) == owner else {
                    throw CancellationError()
                }
                state.owner = owner
                state.entries = entries
                state.published = request
                didPublish = true
                let file = cacheFile(ws)
                try FileManager.default.createDirectory(
                    at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: file, options: .atomic)
            }
        }
        AppLog.storage.info("model catalog refreshed: offered=\(entries.count)")
        return entries.count
    }

    /// Backfill an older registration on a user-requested refresh. The catalog reads the
    /// saved profile directly, so it can refresh as soon as the patch succeeds.
    @discardableResult
    static func refreshExplicitly(ws: PipetteWorkspace, http: HttpClient = .shared) async throws -> Int {
        do {
            try await repairMissingProfile(ws: ws, http: http)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            AppLog.profile.warning("catalog profile repair failed: error=\(error)")
        }
        try Task.checkCancellation()
        return try await refresh(ws: ws, http: http)
    }

    private static func repairMissingProfile(ws: PipetteWorkspace, http: HttpClient) async throws {
        let registration = ws.registration
        let session = try registration.withIdentityLock { try registration.mgmtSession() }
        let owner = try Owner(session)
        let profile = try await PipetteMgmtClient.ClientApi.me(
            server: session.serverUrl, http: http, auth: session.auth)
        guard profile.deviceOsName == nil else { return }

        let detected = try registrationProfile(deviceStore: ws.device)
        guard let osName = detected.deviceOsName else { return }
        let update = UpdateClientRequest(device: DeviceProfileFields(
            deviceOsName: osName,
            deviceOsVersion: profile.deviceOsVersion == nil ? detected.deviceOsVersion : nil,
            deviceChipModel: profile.deviceChipModel == nil ? detected.deviceChipModel : nil,
            deviceRamBytes: profile.deviceRamBytes == nil ? detected.deviceRamBytes : nil,
            // Capabilities are replaced on PATCH. Preserve any profile already reported by a worker.
            capabilities: profile.capabilities.isEmpty ? detected.capabilities : nil
        ))
        try Task.checkCancellation()
        guard registration.withIdentityLock({ currentOwner(ws) == owner }) else {
            throw CancellationError()
        }
        _ = try await PipetteMgmtClient.ClientApi.updateMe(
            server: session.serverUrl, http: http, auth: session.auth, request: update)
    }

    /// The app starts a background refresh at launch or after registration.
    /// In headless mode, only `sync` fetches the catalog.
    static func refreshInBackground(ws: PipetteWorkspace, reason: String) {
        guard !HeadlessRunner.isHeadless else { return }
        Task.detached {
            do {
                try await refresh(ws: ws)
            } catch {
                AppLog.storage.warning("model catalog refresh failed: reason=\(reason) error=\(error)")
            }
        }
    }

    /// Clears the cache and installs the fallback for `auth reset` and `auth import`.
    /// The cache can contain entries private to the previous client's organization.
    static func clearCache(ws: PipetteWorkspace) throws {
        defer { ws.modelShortList.notifyChange() }
        try ws.modelShortList.state.withLock { state in
            state.owner = nil
            state.entries = []
            state.published = state.issued
            do {
                try FileManager.default.removeItem(at: cacheFile(ws))
            } catch CocoaError.fileNoSuchFile {
                // The cache is already absent.
            }
        }
    }

    static func clearCacheBestEffort(ws: PipetteWorkspace) {
        do {
            try clearCache(ws: ws)
        } catch {
            AppLog.storage.warning("model catalog cache removal failed: error=\(error)")
        }
    }

    private static func currentOwner(_ ws: PipetteWorkspace) -> Owner? {
        try? Owner(ws.registration.mgmtSession())
    }

    private func notifyChange() {
        NotificationCenter.default.post(name: Self.didChange, object: self)
    }

    /// Returns one row per model member that the app can download from Hugging Face or
    /// Vault. A body outside the short-list format throws.
    static func parse(_ body: Data) throws -> [CatalogEntry] {
        let list = try Coding.decoder.decode(ShortList.self, from: body)
        let models = list.items.filter { $0.descriptorKind == "model" }
        let entries = models.flatMap { item in
            item.members.compactMap(\.value)
                .filter { $0.descriptor.repo != nil || ModelFetch.vaultFiles($0.descriptor) != nil }
                .map { member in
                    CatalogEntry(name: item.name, source: member.descriptor,
                                 sizeBytes: member.sizeBytes ?? 0,
                                 // The derived form first, so the row's label matches
                                 // the installed copy. The catalog's label is lowercase.
                                 quant: member.descriptor.quant ?? member.memberKey.quantization,
                                 family: item.family, familyLogoDigest: item.familyLogoDigest,
                                 selector: member.selector)
                }
        }
        let dropped = models.reduce(0) { $0 + $1.members.count } - entries.count
        if dropped > 0 {
            AppLog.storage.warning("model catalog members skipped: count=\(dropped)")
        }
        AppLog.storage.info(
            "model catalog parsed: entries=\(entries.count) families=\(Set(entries.map(\.family)).count)")
        return entries
    }

    // MARK: - Wire form
    //
    // The parts of the mgmt repo's `DescriptorCatalogShortList` that the app reads.

    private struct ShortList: Decodable {
        let items: [Item]
    }

    private struct Item: Decodable {
        let name: String
        let descriptorKind: String
        /// Nil when the server sends no family. The entry then takes
        /// ``CatalogEntry/defaultFamily(_:)``.
        let family: String?
        let familyLogoSha256: Lenient<String>?
        let members: [Lenient<Member>]

        /// Restrict the digest format because the request uses it as a path segment.
        var familyLogoDigest: String? {
            familyLogoSha256?.value.flatMap { Sha256.validate($0) ? $0 : nil }
        }

        enum CodingKeys: String, CodingKey {
            case name, family, members
            case descriptorKind = "descriptor_kind"
            case familyLogoSha256 = "family_logo_sha256"
        }
    }

    private struct Member: Decodable {
        let selector: String?
        let memberKey: MemberKey
        let descriptor: Model
        let sizeBytes: Int64?

        enum CodingKeys: String, CodingKey {
            case descriptor, selector
            case memberKey = "member_key"
            case sizeBytes = "size_bytes"
        }
    }

    private struct MemberKey: Decodable {
        let quantization: String?
    }

    /// A value that decodes to `nil` when its element does not decode. The decode still
    /// consumes the element, so the next element of an array decodes as normal.
    private struct Lenient<T: Decodable>: Decodable {
        let value: T?

        init(from decoder: any Decoder) throws {
            value = try? T(from: decoder)
        }
    }
}
