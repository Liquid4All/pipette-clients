import Foundation

/// Client identity store — the signing key and the registration record.
///
/// The crate's `IdentityStore` (`pipette-cli/src/identity/store.rs:52`): a concrete
/// handle over one directory, minted from the storage seam (`storage.identity`) the way
/// the CLI mints it from the workspace (`ws.identity()`). Call sites take the store, not
/// a path.
///
/// Layout differs from the CLI's in one way that matters. The desktop client keeps the
/// private key as a `0600` file beside the registration; here it is a Keychain item, so
/// this store composes two backings where the crate's composes one. It is also why
/// `clearRegistrationMaterial()` has no public-key half: nothing on iOS persists the
/// public key, which is sent once at registration and never read back.
///
/// ```text
/// metadata/
///   registration.json     # client_id + server_url + the registration inputs
///   device.json           # DeviceLabels — operator-supplied, no registration needed
///   settings.json         # ClientSettings
/// Keychain (ai.liquid.liquid-pipette)
///   private_key_hex       # the Curve25519 signing key
///   registration_json     # a copy of registration.json — see RegistrationMirror
/// ```
///
/// The registration is therefore stored twice, and the file is the one that answers: the
/// Keychain copy exists only to outlive an uninstall, which deletes the container the file
/// lives in. `getRegistration()` restores from it, so the two converge on the next read.
///
/// **`root` is not exclusive to identity.** The crate's `identity/` is a dedicated
/// `0700` directory; this one is the app's `metadata/`, shared with the synced
/// benchmark catalog under `remote_benchmarks/`. Every accessor here therefore names
/// its file — do not port the crate's directory-level `clear()`, which would take the
/// catalog with it. Splitting the directory needs an on-disk migration.
nonisolated struct IdentityStore: Sendable {
    /// The directory holding the identity files (the app's `metadata/`).
    let root: URL

    /// Where the signing key comes from. The crate keeps its key *inside* the workspace
    /// (`identity/store.rs` writes it as a file), so pointing a workspace at scratch space
    /// isolates both halves of an identity. iOS keeps it in the Keychain, which is
    /// device-global and survives any `root` — so a store carries its own source instead,
    /// and a temporary one can be as self-contained as the crate's temporary workspace.
    let privateKeySource: @Sendable () -> PrivateKeyHex?

    /// Where the registration record survives an uninstall. Defaults to `.disabled` so the
    /// Keychain — device-global, and written *and deleted* through this seam — is reached
    /// only by the store that names it, which is `FileStorage.production` and nothing else.
    let registrationMirror: RegistrationMirror

    init(root: URL,
         privateKeySource: @escaping @Sendable () -> PrivateKeyHex? = {
             KeychainHelper.loadPrivateKey()
         },
         registrationMirror: RegistrationMirror = .disabled) {
        self.root = root
        self.privateKeySource = privateKeySource
        self.registrationMirror = registrationMirror
    }

    private var registrationPath: URL {
        root.appendingPathComponent("registration.json")
    }

    private var settingsPath: URL {
        root.appendingPathComponent("settings.json")
    }

    private var deviceLabelsPath: URL {
        root.appendingPathComponent("device.json")
    }

    // MARK: - Device labels

    /// The operator-supplied labels, or empty when the file is absent or unreadable.
    /// Absent is the normal state — a device reports its probed values until named.
    func getDeviceLabels() -> DeviceLabels {
        guard let data = try? Data(contentsOf: deviceLabelsPath),
              let labels = try? Coding.decoder.decode(DeviceLabels.self, from: data)
        else { return .empty }
        return labels
    }

    func putDeviceLabels(_ labels: DeviceLabels) throws {
        try Coding.encoder.encode(labels).write(to: deviceLabelsPath, options: .atomic)
    }

    // MARK: - Registration

    var isRegistered: Bool {
        getRegistration() != nil
    }

    /// The registration record, or nil when this device has none. An absent or unreadable
    /// file falls through to the Keychain mirror, which is how a reinstalled app finds the
    /// identity its container no longer holds; nil from both means unregistered, and the
    /// app offers setup.
    func getRegistration() -> IdentityRegistration? {
        if let data = try? Data(contentsOf: registrationPath),
           let registration = try? Coding.decoder.decode(IdentityRegistration.self, from: data) {
            return registration
        }
        return restoreRegistrationFromMirror()
    }

    func putRegistration(_ registration: IdentityRegistration) throws {
        try writeRegistrationFile(registration)
        mirror(registration)
    }

    func deleteRegistration() {
        try? FileManager.default.removeItem(at: registrationPath)
        // The mirror too, or the next `getRegistration()` restores the record this call
        // exists to forget.
        if !registrationMirror.clear() {
            AppLog.storage.error("identity: registration mirror not cleared — it may be restored")
        }
    }

    // MARK: - Registration mirror

    /// Whether the record would survive an uninstall. Reported by `headlessrun status` so
    /// the answer is checkable on a device instead of inferred from the app version.
    var isRegistrationMirrored: Bool {
        registrationMirror.load() != nil
    }

    /// Copy an existing registration into the mirror when it isn't there yet, so a device
    /// that registered before the mirror existed gains the protection at launch rather than
    /// at its next re-registration — which for an already-registered device never comes.
    /// Idempotent, and a no-op on an unregistered device.
    func backfillRegistrationMirror() {
        guard registrationMirror.load() == nil,
              let data = try? Data(contentsOf: registrationPath),
              let registration = try? Coding.decoder.decode(IdentityRegistration.self, from: data)
        else { return }
        mirror(registration)
        AppLog.storage.info("identity: mirrored the existing registration to the Keychain")
    }

    /// Best effort: a device whose mirror write failed is still registered, and failing the
    /// registration over the copy would trade a working identity for a durable one.
    private func mirror(_ registration: IdentityRegistration) {
        guard let json = try? Coding.encoder.encode(registration),
              registrationMirror.save(json)
        else {
            AppLog.storage.error(
                "identity: registration not mirrored — it will not survive a reinstall")
            return
        }
    }

    /// Recover the record an uninstall deleted. The file is rewritten so every other reader
    /// — and the next launch — takes the ordinary path; a failed write is not fatal, since
    /// the mirror keeps answering until one succeeds.
    private func restoreRegistrationFromMirror() -> IdentityRegistration? {
        guard let data = registrationMirror.load(),
              let registration = try? Coding.decoder.decode(IdentityRegistration.self, from: data)
        else { return nil }
        do {
            try writeRegistrationFile(registration)
            AppLog.storage.info("identity: restored the registration from the Keychain")
        } catch {
            AppLog.storage.error("identity: registration restored but not rewritten: \(error)")
        }
        return registration
    }

    private func writeRegistrationFile(_ registration: IdentityRegistration) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Coding.encoder.encode(registration).write(to: registrationPath, options: .atomic)
    }

    // MARK: - Signing key

    /// The private signing key, minted at registration and never re-read from the
    /// server. Nil when the source has none — see `privateKeySource`.
    func getPrivateKey() -> PrivateKeyHex? {
        privateKeySource()
    }

    // MARK: - Client settings

    /// Read `settings.json`, or nil when it is absent or unreadable — the caller falls
    /// back to the built-in defaults, so a corrupt file degrades to defaults instead of
    /// trapping. The crate errors on a half-written file instead; here the only setting
    /// is a quota with a well-defined default, and refusing to launch over it would be
    /// the worse trade.
    func getSettings() -> ClientSettings? {
        guard let data = try? Data(contentsOf: settingsPath) else { return nil }
        return try? Coding.decoder.decode(ClientSettings.self, from: data)
    }

    func putSettings(_ settings: ClientSettings) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Coding.encoder.encode(settings).write(to: settingsPath, options: .atomic)
    }

    // MARK: - Composition

    /// Compose the identity that signs server requests — the crate's
    /// `signing_identity`. Nil when either half is absent.
    ///
    /// The halves are checked together on purpose: a registration whose key was cleared
    /// out from under it (or vice versa) cannot sign. Callers that also need the server
    /// URL take ``Storage/mgmtSession()`` instead, which composes both from one read.
    func signingIdentity() -> AuthIdentity? {
        guard let registration = getRegistration(), let privateKey = getPrivateKey() else {
            return nil
        }
        return AuthIdentity(clientId: registration.clientId, privateKeyHex: privateKey)
    }

    /// Forget the registration record — both copies of it, via `deleteRegistration()` —
    /// *and* the key it is useless without. The crate's `clear_registration_material`, and
    /// the one teardown path, so "forget this device" cannot come to mean two different
    /// things.
    ///
    /// Returns whether the key actually went: a Keychain delete can fail on its own, and
    /// a caller reporting "cleared" when the credential is still there is the failure
    /// mode this return value exists to prevent.
    ///
    /// `deleteKey` is injected because a Simulator test host cannot perform Keychain
    /// operations, so the outcome is otherwise unassertable off-device. The crate needs
    /// no such seam — its key is a file.
    @discardableResult
    func clearRegistrationMaterial(
        deleteKey: () -> Bool = KeychainHelper.deletePrivateKey
    ) -> Bool {
        deleteRegistration()
        return deleteKey()
    }
}

/// The registration record's second home, outside the app container.
///
/// Uninstalling an iOS app deletes its container — Application Support included — with no
/// opt-out, so `registration.json` cannot survive one. Keychain items under the same access
/// group (team prefix + bundle id) do, which is why the signing key already lived there. But
/// the two halves are checked together (``IdentityStore/signingIdentity()``), so a key that
/// survives alone reads as "unregistered" and setup mints a fresh keypair *over* it — the
/// device comes back with a new identity and its submitted results stop being attributable.
/// Mirroring the record keeps both halves on the same side of a reinstall.
///
/// This does not survive everything: a wiped device, or a reinstall signed by a different
/// team or bundle id, takes the Keychain with it. Recovering from those needs the server to
/// re-associate the device, not a local store.
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

/// Why an identity could not be composed — the crate's `Error::RegistrationMissing` /
/// `Error::PrivateKeyMissing`, which `mgmt_session()` surfaces the same way.
nonisolated enum IdentityError: LocalizedError, Equatable {
    case registrationMissing
    case privateKeyMissing

    var errorDescription: String? {
        switch self {
        case .registrationMissing:
            return "this device has no registration"
        case .privateKeyMissing:
            return "the signing key is missing from the Keychain: re-register to mint a new one"
        }
    }
}
