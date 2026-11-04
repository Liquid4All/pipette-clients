import Foundation

/// `identity/registration.json`: the crate's `IdentityRegistration`
/// (`crates/pipette-cli/src/identity/registration.rs`).
///
/// The CLI's record holds only `client_id` and `server_url`. This one also keeps the
/// registration inputs, which the app renders without a round trip and re-registration
/// prefills from, and the Clerk link, which has no CLI counterpart.
///
/// `init(from:)` also reads the older camelCase keys, so an existing install keeps its
/// registration. Remove once no device runs a build that predates `snake_case`.
nonisolated struct IdentityRegistration: Codable, Sendable, Equatable {
    let clientId: ClientId
    let status: String
    let serverUrl: ServerUrl
    let organization: String
    let contactEmail: String
    let registeredAt: String
    let clerkUserId: String?
    let clerkSessionId: String?
    let clerkPrimaryEmail: String?
    let clerkLinkedAt: String?

    init(
        clientId: ClientId,
        status: String,
        serverUrl: ServerUrl,
        organization: String,
        contactEmail: String,
        registeredAt: String,
        clerkUserId: String? = nil,
        clerkSessionId: String? = nil,
        clerkPrimaryEmail: String? = nil,
        clerkLinkedAt: String? = nil
    ) {
        self.clientId = clientId
        self.status = status
        self.serverUrl = serverUrl
        self.organization = organization
        self.contactEmail = contactEmail
        self.registeredAt = registeredAt
        self.clerkUserId = clerkUserId
        self.clerkSessionId = clerkSessionId
        self.clerkPrimaryEmail = clerkPrimaryEmail
        self.clerkLinkedAt = clerkLinkedAt
    }

    enum CodingKeys: String, CodingKey {
        case clientId = "client_id"
        case status
        case serverUrl = "server_url"
        case organization
        case contactEmail = "contact_email"
        case registeredAt = "registered_at"
        case clerkUserId = "clerk_user_id"
        case clerkSessionId = "clerk_session_id"
        case clerkPrimaryEmail = "clerk_primary_email"
        case clerkLinkedAt = "clerk_linked_at"
    }

    /// Read-only. `status` and `organization` are spelled the same either way.
    private enum LegacyCodingKeys: String, CodingKey {
        case clientId, serverUrl, contactEmail, registeredAt
        case clerkUserId, clerkSessionId, clerkPrimaryEmail, clerkLinkedAt
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let legacy = try decoder.container(keyedBy: LegacyCodingKeys.self)
        // Per key rather than per file: a record linked to Clerk after the rename
        // carries both spellings, and an all-or-nothing branch would drop half of it.
        func value<T: Decodable>(_ key: CodingKeys, _ legacyKey: LegacyCodingKeys) throws -> T? {
            try c.decodeIfPresent(T.self, forKey: key)
                ?? legacy.decodeIfPresent(T.self, forKey: legacyKey)
        }

        guard let clientId: ClientId = try value(.clientId, .clientId),
              let serverUrl: ServerUrl = try value(.serverUrl, .serverUrl)
        else {
            throw DecodingError.keyNotFound(CodingKeys.clientId, DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "registration.json has neither client_id/server_url nor "
                    + "their legacy camelCase spellings"))
        }
        self.clientId = clientId
        self.serverUrl = serverUrl
        // Only `client_id` and `server_url` are required, as in the crate. A missing
        // convenience field must not read as "unregistered": setup would mint a new
        // keypair and orphan the device's submitted results.
        status = try c.decodeIfPresent(String.self, forKey: .status) ?? ""
        organization = try c.decodeIfPresent(String.self, forKey: .organization) ?? ""
        contactEmail = try value(.contactEmail, .contactEmail) ?? ""
        registeredAt = try value(.registeredAt, .registeredAt) ?? ""
        clerkUserId = try value(.clerkUserId, .clerkUserId)
        clerkSessionId = try value(.clerkSessionId, .clerkSessionId)
        clerkPrimaryEmail = try value(.clerkPrimaryEmail, .clerkPrimaryEmail)
        clerkLinkedAt = try value(.clerkLinkedAt, .clerkLinkedAt)
    }

    func withClerkLink(
        userId: String,
        sessionId: String?,
        primaryEmail: String?
    ) -> IdentityRegistration {
        IdentityRegistration(
            clientId: clientId,
            status: status,
            serverUrl: serverUrl,
            organization: organization,
            contactEmail: contactEmail,
            registeredAt: registeredAt,
            clerkUserId: userId,
            clerkSessionId: sessionId,
            clerkPrimaryEmail: primaryEmail,
            clerkLinkedAt: clerkLinkedAt ?? JobDateFormat.iso8601.string(from: Date())
        )
    }

    /// Drop the Clerk link on sign-out so the next account can link. The link is a local
    /// guard only (the management server never sees the Clerk identity).
    ///
    /// `clerkLinkedAt` goes too: `withClerkLink` keeps an existing value, so leaving it would
    /// date the next account's link from this one's.
    func withoutClerkLink() -> IdentityRegistration {
        IdentityRegistration(
            clientId: clientId,
            status: status,
            serverUrl: serverUrl,
            organization: organization,
            contactEmail: contactEmail,
            registeredAt: registeredAt,
            clerkUserId: nil,
            clerkSessionId: nil,
            clerkPrimaryEmail: nil,
            clerkLinkedAt: nil
        )
    }
}

/// `identity/registration.json` and the signing key beside it, created and cleared as a
/// unit. The crate's layout, minus `id_ed25519.pub`: iOS derives the public key when needed.
///
/// ```text
/// identity/
///   id_ed25519            # the Curve25519 signing key, hex, 0600
///   registration.json     # client_id + server_url + the registration inputs
/// Keychain (ai.liquid.liquid-pipette)
///   private_key_hex       # a copy of id_ed25519, see PrivateKeyMirror
///   registration_json     # a copy of registration.json, see RegistrationMirror
/// ```
///
/// Both halves are kept twice and the file answers first. The Keychain copies outlive an
/// uninstall; the files outlive a build re-signed under another team, which loses the
/// Keychain. Neither crosses a backup (`identity/` is excluded, see `IdentityFiles`). A
/// read with no file restores it from the mirror, and the launch backfills
/// restore the mirror from the file. Both halves must have both homes: the gate needs the
/// pair, and `ensureEmpty()` refuses setup over either half.
nonisolated struct ClientRegistrationStore: Sendable {
    private static let privateKeyName = "id_ed25519"
    private static let registrationName = "registration.json"

    /// The workspace `identity/` directory.
    let root: URL

    /// Defaults to `.disabled` so only `PipetteWorkspace.production` reaches (and deletes
    /// from) the device-global Keychain; a scratch store stays self-contained.
    let privateKeyMirror: PrivateKeyMirror

    /// Same terms as ``privateKeyMirror``.
    let registrationMirror: RegistrationMirror

    init(root: URL,
         privateKeyMirror: PrivateKeyMirror = .disabled,
         registrationMirror: RegistrationMirror = .disabled) {
        self.root = root
        self.privateKeyMirror = privateKeyMirror
        self.registrationMirror = registrationMirror
    }

    /// The mirror is best-effort: failing over the copy would trade a working identity for a
    /// durable one.
    func putPrivateKey(_ key: PrivateKey) throws {
        try IdentityFiles.writeSecret(root, Self.privateKeyName, Data(key.value.utf8))
        if !privateKeyMirror.save(key) {
            AppLog.storage.error(
                "identity: signing key not mirrored — it will not survive a reinstall")
        }
    }

    /// An absent or unusable file falls through to the Keychain mirror, which is how a
    /// reinstalled app finds its key. Absent and unusable are one state: neither can sign.
    func getPrivateKey() -> PrivateKey? {
        if let stored = try? IdentityFiles.readText(root, Self.privateKeyName),
           let key = try? PrivateKey(stored) {
            return key
        }
        return restorePrivateKeyFromMirror()
    }

    /// Delete both copies. Returns whether both went, so a caller cannot report "cleared"
    /// over a key that still signs.
    @discardableResult
    func deletePrivateKey() -> Bool {
        try? IdentityFiles.remove(root, Self.privateKeyName)
        let fileGone = !FileManager.default.fileExists(
            atPath: IdentityFiles.path(root, Self.privateKeyName).path)
        return privateKeyMirror.clear() && fileGone
    }

    /// An absent or unreadable file falls through to the Keychain mirror, which is how a
    /// reinstalled app finds its registration. Nil from both means unregistered.
    func get() -> IdentityRegistration? {
        if let registration: IdentityRegistration = try? IdentityFiles.readJson(
            root, Self.registrationName) {
            return registration
        }
        return restoreRegistrationFromMirror()
    }

    func put(_ registration: IdentityRegistration) throws {
        try IdentityFiles.writeJson(root, Self.registrationName, registration)
        mirror(registration)
    }

    /// Nil when either half is absent. Callers that also need the server URL use
    /// ``mgmtSession()``, which reads the record once.
    func signingIdentity() -> ClientIdentity? {
        guard let registration = get(), let privateKey = getPrivateKey() else {
            return nil
        }
        return ClientIdentity(clientId: registration.clientId, privateKeyHex: privateKey)
    }

    /// Refuse when this device holds a registration or a signing key, so `register` cannot
    /// silently rotate the key and orphan every result submitted under the first.
    ///
    /// A lone key counts, as upstream: `auth import` of its bundle would finish restoring it.
    /// Labels and settings do not, so `set-device` before `register` still works.
    func ensureEmpty() throws(IdentityError) {
        if let registration = get() {
            throw .identityExists("the registration for \(registration.clientId.value)")
        }
        if getPrivateKey() != nil {
            throw .identityExists("a signing key with no registration")
        }
    }

    /// Delete the record and the key, both copies of each. Labels and settings stay, as
    /// upstream. Returns ``deletePrivateKey()``'s answer.
    @discardableResult
    func clear() -> Bool {
        deleteRegistration()
        return deletePrivateKey()
    }

    // MARK: - iOS: the Keychain mirrors

    var isRegistered: Bool {
        self.get() != nil
    }

    /// Whether the record would survive an uninstall, reported by `headlessrun status`.
    var isRegistrationMirrored: Bool {
        registrationMirror.load() != nil
    }

    /// Copy the registration file into an empty mirror at launch, alongside
    /// ``backfillPrivateKeyMirror()``, so both halves gain the protection together.
    /// Idempotent.
    func backfillRegistrationMirror() {
        guard registrationMirror.load() == nil,
              let registration: IdentityRegistration = try? IdentityFiles.readJson(
                root, Self.registrationName)
        else { return }
        mirror(registration)
        AppLog.storage.info("identity: mirrored the existing registration to the Keychain")
    }

    /// Copy the key file into an empty mirror. Idempotent.
    ///
    /// The other healing direction from ``getPrivateKey()``: a build re-signed under a
    /// different team loses both Keychain copies but keeps the files, and without this only
    /// the record would be re-mirrored, leaving the next uninstall to take the key.
    func backfillPrivateKeyMirror() {
        guard privateKeyMirror.load() == nil,
              let stored = try? IdentityFiles.readText(root, Self.privateKeyName),
              let key = try? PrivateKey(stored)
        else { return }
        guard privateKeyMirror.save(key) else {
            AppLog.storage.error(
                "identity: existing signing key not backfilled into the Keychain")
            return
        }
        AppLog.storage.info("identity: mirrored the existing signing key to the Keychain")
    }

    private func deleteRegistration() {
        try? IdentityFiles.remove(root, Self.registrationName)
        // The mirror too, or the next `get()` restores the record this call exists to forget.
        if !registrationMirror.clear() {
            AppLog.storage.error("identity: registration mirror not cleared — it may be restored")
        }
    }

    /// Best effort, for the same reason as ``putPrivateKey(_:)``.
    private func mirror(_ registration: IdentityRegistration) {
        guard let json = try? Coding.encoder.encode(registration),
              registrationMirror.save(json)
        else {
            AppLog.storage.error(
                "identity: registration not mirrored — it will not survive a reinstall")
            return
        }
    }

    /// Rewrites the file so later reads take the ordinary path. A failed write is not fatal:
    /// the mirror keeps answering.
    private func restoreRegistrationFromMirror() -> IdentityRegistration? {
        guard let data = registrationMirror.load(),
              let registration = try? Coding.decoder.decode(IdentityRegistration.self, from: data)
        else { return nil }
        do {
            try IdentityFiles.writeJson(root, Self.registrationName, registration)
            AppLog.storage.info("identity: restored the registration from the Keychain")
        } catch {
            AppLog.storage.error("identity: registration restored but not rewritten: \(error)")
        }
        return registration
    }

    /// Rewrites the file so later reads take the ordinary path. A failed write is not fatal:
    /// the mirror keeps answering.
    private func restorePrivateKeyFromMirror() -> PrivateKey? {
        guard let key = privateKeyMirror.load() else { return nil }
        do {
            try IdentityFiles.writeSecret(root, Self.privateKeyName, Data(key.value.utf8))
            // A concurrent delete can land between the load and this write, leaving an orphan
            // file that `ensureEmpty()` refuses forever. An empty mirror means the delete won.
            guard privateKeyMirror.load() != nil else {
                try? IdentityFiles.remove(root, Self.privateKeyName)
                AppLog.storage.info("identity: signing key restore undone by a concurrent delete")
                return nil
            }
            AppLog.storage.info("identity: restored the signing key from the Keychain")
        } catch {
            AppLog.storage.error("identity: signing key restored but not rewritten: \(error)")
        }
        return key
    }
}

/// The signing key's second home, outside the app container.
///
/// An uninstall deletes the container, and the `client_id` is unrecoverable without its
/// key; a Keychain item under the same access group (team prefix + bundle id) survives.
/// It is a copy, not the only home, because the Keychain fails the other way: its access
/// group carries the team prefix, so a build re-signed under another team cannot read it,
/// while the container survives that change. ``RegistrationMirror`` carries the record on
/// the same terms.
nonisolated struct PrivateKeyMirror: Sendable {
    var load: @Sendable () -> PrivateKey?
    var save: @Sendable (PrivateKey) -> Bool
    var clear: @Sendable () -> Bool

    static let keychain = PrivateKeyMirror(
        load: { KeychainHelper.loadPrivateKey() },
        save: { KeychainHelper.savePrivateKey($0.value) },
        clear: { KeychainHelper.deletePrivateKey() }
    )

    /// No mirror: reads find nothing and writes succeed by doing nothing. The right choice
    /// for a store rooted in scratch space, which should not reach device-global state.
    static let disabled = PrivateKeyMirror(
        load: { nil },
        save: { _ in true },
        clear: { true }
    )
}

/// The registration record's second home, outside the app container, so it survives an
/// uninstall alongside ``PrivateKeyMirror``. A key that survived alone would read as
/// "unregistered" (``ClientRegistrationStore/signingIdentity()`` needs both).
///
/// A wiped device, or a reinstall signed by a different team or bundle id, loses the
/// Keychain too; recovering from that needs `auth import` or the server.
nonisolated struct RegistrationMirror: Sendable {
    var load: @Sendable () -> Data?
    var save: @Sendable (Data) -> Bool
    var clear: @Sendable () -> Bool

    static let keychain = RegistrationMirror(
        load: { KeychainHelper.loadRegistration() },
        save: { KeychainHelper.saveRegistration($0) },
        clear: { KeychainHelper.deleteRegistration() }
    )

    /// No mirror: reads find nothing and writes succeed by doing nothing. The right choice
    /// for a store rooted in scratch space, which should not reach device-global state.
    static let disabled = RegistrationMirror(
        load: { nil },
        save: { _ in true },
        clear: { true }
    )
}

/// The identity cases of the crate's `Error`, plus ``keyNotPersisted``.
nonisolated enum IdentityError: LocalizedError, Equatable {
    case registrationMissing
    case privateKeyMissing
    /// Carries what this device already holds.
    case identityExists(String)
    case setDeviceEmpty
    /// The server assigned a `client_id` that cannot sign.
    case registeredId(String)
    /// The server registered this client, but the local identity was not saved.
    case registrationPersisted(clientId: String, reason: String)
    /// No crate counterpart (it holds the keypair in memory): the Keychain did not keep the
    /// staged signing key.
    case keyNotPersisted

    var errorDescription: String? {
        switch self {
        case .registrationMissing:
            return "this device has no registration"
        case .privateKeyMissing:
            return "this device holds no signing key for its registration: "
                + "import the identity bundle again, or clear the identity and re-register"
        case let .identityExists(held):
            return "identity already exists (\(held)); run `auth reset` before re-registering"
        case .setDeviceEmpty:
            return "set-device requires at least one of device-name, device-form-factor, "
                + "or client-details"
        case let .registeredId(reason):
            return "the server registered this client under an unusable id, which cannot "
                + "sign: \(reason)"
        case let .registrationPersisted(clientId, reason):
            return "registered with the server as \(clientId), but saving the local identity "
                + "failed (\(reason)); any pre-auth key used is now spent. Ask an admin to "
                + "re-issue one before retrying"
        case .keyNotPersisted:
            return "failed to save this device's signing key"
        }
    }
}
