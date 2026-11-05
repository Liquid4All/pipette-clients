import Foundation

/// The `auth` group, as in `pipette-cli/src/commands/auth.rs`. Log keys are the wire
/// spellings (`reindex_pending`) so a console consumer greps the server's token.
enum AuthCommands {

    // MARK: - register

    /// `auth register`, parsed. Every value is already the type the request carries: the
    /// parser refuses a bad one (exit 2) before a keypair is minted or a pre-auth key spent.
    nonisolated struct RegisterArgs: Equatable, Sendable {
        var serverUrl: ServerUrl
        var organization: String
        var contactEmail: ContactEmail
        var clientDetails: String
        var preauthKey: ClientPreauthKey?
        var deviceName: String?
        var deviceFormFactor: DeviceFormFactor?
    }

    /// Refuses over an identity this device already holds (`identityExists`, exit 1): a
    /// second registration would mint a new keypair and orphan every result of the first.
    static func register(_ args: RegisterArgs, ws: PipetteWorkspace) async -> Bool {
        // Never log the key: it is a credential the server spends on this call.
        HeadlessRunner.log("auth register start server_url=\(args.serverUrl) "
            + "organization=\(args.organization) contact_email=\(args.contactEmail) "
            + "preauth_key=\(args.preauthKey == nil ? "no" : "yes")")
        discardOrphanedSigningKey(ws: ws)
        do {
            let registration = try await Pipette.register(
                registrationStore: ws.registration,
                device: ws.device,
                server: args.serverUrl,
                input: RegisterInput(
                    organization: args.organization,
                    contactEmail: args.contactEmail,
                    clientDetails: args.clientDetails,
                    deviceName: args.deviceName,
                    deviceFormFactor: args.deviceFormFactor,
                    preauthKey: args.preauthKey))
            // The catalog is public, so pulling it must not hold up the registration.
            BenchmarkSyncCoordinator.syncBestEffortInBackground(
                serverUrl: registration.serverUrl, ws: ws, reason: "registration",
                clearCacheFirst: false)
            // `status` (not in the crate's output) is the server's approval word, which a
            // pre-auth key changes.
            HeadlessRunner.log("auth register client_id=\(registration.clientId.value) "
                + "server_url=\(registration.serverUrl) status=\(registration.status)")
            return true
        } catch {
            HeadlessRunner.log("auth register ERROR \(error.localizedDescription)")
            return false
        }
    }

    // MARK: - me

    /// `auth me [local=1]`: the profile from `GET /clients/me`, on one line. `local=1` (the
    /// CLI's `--local`) answers from storage with no request, so a provisioning step can
    /// assert on it before the server approves the client.
    static func me(local: Bool = false, ws: PipetteWorkspace) async -> Bool {
        if local {
            do {
                HeadlessRunner.log(try localIdentity(ws: ws))
                return true
            } catch {
                HeadlessRunner.log("auth me ERROR \(error.localizedDescription)")
                return false
            }
        }
        do {
            let profile = try await fetchProfile(registration: ws.registration)
            HeadlessRunner.log("auth me client_id=\(profile.clientId) "
                + "organization=\(profile.organization) "
                + "client_details=\(profile.clientDetails) "
                + "contact_email=\(profile.contactEmail) "
                + "status=\(profile.status) "
                + "tags=\(formatTags(profile.tags)) "
                + "reindex_pending=\(profile.reindexPending) "
                + "capabilities=\(formatList(profile.capabilities))")
            return true
        } catch let error as IdentityError {
            // Names the missing half of the identity, not a flat "not registered".
            HeadlessRunner.log("auth me ERROR \(error.localizedDescription)")
            return false
        } catch {
            HeadlessRunner.log("auth me ERROR \(error)")
            return false
        }
    }

    /// `auth me local=1`: who this device says it is, from storage. Both halves are
    /// required: a registration with no key names a client this device cannot sign for.
    /// The signing key never prints, only the public key derived from it.
    static func localIdentity(ws: PipetteWorkspace) throws -> String {
        let session = try ws.registration.mgmtSession()
        let publicKey = try PipetteMgmtClient.Auth.publicKeyHexFromPrivate(
            session.auth.privateKeyHex)
        return "auth me client_id=\(session.registration.clientId.value) "
            + "server_url=\(session.serverUrl.value) public_key=\(publicKey)"
    }

    /// Joined without the crate's space, which would split the value across two
    /// `key=value` tokens.
    static func formatTags(_ tags: [String]) -> String {
        formatList(tags.sorted())
    }

    /// A list as one `key=value` value, `(none)` when empty so a parser sees the key and a
    /// reader does not see a blank.
    private static func formatList(_ items: [String]) -> String {
        items.isEmpty ? "(none)" : items.joined(separator: ",")
    }

    // MARK: - set-device

    /// `auth set-device`: relabel this device without re-registering. The two device labels
    /// are local (`DeviceLabels`), so an unregistered device can be named with no network;
    /// `client_details` is server-side and needs a registration and `PATCH /clients/me`.
    ///
    /// The parser has refused the empty invocation. A blank `device-name` unsets the label,
    /// handing it back to the probe.
    static func setDevice(deviceName: String?, deviceFormFactor: DeviceFormFactor?,
                          clientDetails: String?, ws: PipetteWorkspace) async -> Bool {
        if deviceName != nil || deviceFormFactor != nil {
            do {
                let labels = try ws.device.update(deviceName: deviceName,
                                                       deviceFormFactor: deviceFormFactor)
                HeadlessRunner.log("auth set-device "
                    + "device_name=\(labels.deviceName ?? "(unset)") "
                    + "device_form_factor=\(labels.deviceFormFactor?.rawValue ?? "(unset)")")
            } catch {
                HeadlessRunner.log("auth set-device ERROR \(error.localizedDescription)")
                return false
            }
        }

        guard let clientDetails else { return true }
        // After the labels, which need no server, so an unregistered device is still named
        // when this leg refuses.
        do {
            let profile = try await updateClientDetails(
                registration: ws.registration, clientDetails: clientDetails)
            HeadlessRunner.log("auth set-device client_details=\(profile.clientDetails)")
            return true
        } catch let error as IdentityError {
            HeadlessRunner.log("auth set-device ERROR \(error.localizedDescription)")
            return false
        } catch {
            HeadlessRunner.log("auth set-device ERROR \(error)")
            return false
        }
    }

    // MARK: - reset

    /// `auth reset [force=1]`: delete the identity and every cache derived from the server
    /// relationship. Local benchmarks and results are kept. Unlike the crate, it also clears
    /// the per-model Hugging Face tokens: they are this identity's download credentials.
    ///
    /// Refuses without `force=1` only when results await submission, so unsubmitted runs
    /// are not silently discarded.
    ///
    /// `clearModelTokens` is injected because a Simulator test host cannot use the Keychain.
    static func reset(
        force: Bool,
        ws: PipetteWorkspace,
        clearModelTokens: () -> Int = KeychainHelper.deleteAllModelHfTokens
    ) -> Bool {
        let results = ws.results
        let pending = results.listIds(.remotePending).count
        let targets = writeResetTargets(
            held: ws.registration.get(),
            benchmarks: ws.benchmarks.list(.remote).count,
            pending: pending, synced: results.listIds(.remoteSynced).count)

        if pending > 0, !force {
            HeadlessRunner.log("auth reset would_remove \(targets)")
            HeadlessRunner.log("auth reset ERROR \(pending) unsubmitted result(s) would be lost; "
                + "run `auth reset force=1` to delete them along with the identity, or `sync` "
                + "them first")
            return false
        }

        // A Keychain delete can fail on its own, and the line reports that rather than
        // implying the credential is gone when it is still there.
        let keyCleared = ws.clearIdentity()
        do {
            try clearRemoteState(benchmarks: ws.benchmarks, results: results,
                                 evals: ws.evalCompletions)
        } catch {
            HeadlessRunner.log("auth reset ERROR the identity is deleted but its caches are "
                + "not: \(error.localizedDescription)")
            return false
        }
        let modelTokens = clearModelTokens()
        HeadlessRunner.log("auth reset identity=deleted \(targets) "
            + "signing_key_cleared=\(keyCleared ? "yes" : "no") "
            + "model_tokens_cleared=\(modelTokens)")
        return keyCleared
    }

    /// What a reset (or a forced import) removes, as the crate's `write_reset_targets`
    /// lists it.
    static func writeResetTargets(held: IdentityRegistration?, benchmarks: Int,
                             pending: Int, synced: Int) -> String {
        "client_id=\(held?.clientId.value ?? "-") server_url=\(held?.serverUrl.value ?? "-") "
            + "remote_benchmarks=\(benchmarks) remote_results=\(pending + synced) "
            + "unsubmitted=\(pending)"
    }

    // MARK: - export

    /// `auth export`: print this client's identity as one restorable token. The summary
    /// goes to stderr and the token alone to stdout, so `auth export > bundle.txt` captures
    /// only the token.
    ///
    /// The token carries the signing key in plaintext into the terminal scrollback: the
    /// only way to copy a key the Keychain will not export, behind a USB cable and an
    /// unlocked device.
    ///
    /// Refuses when either half of the identity is missing: neither half alone restores
    /// anything. The Keychain reads are injected for the reason `reset`'s delete is.
    static func export(
        ws: PipetteWorkspace,
        loadUserToken: () -> AuthToken? = KeychainHelper.loadHfToken,
        loadModelTokens: () -> [String: String] = KeychainHelper.loadAllModelHfTokens
    ) -> Bool {
        let session: MgmtSession
        do {
            session = try ws.registration.mgmtSession()
        } catch {
            HeadlessRunner.logDiagnostic("auth export ERROR \(error.localizedDescription)")
            return false
        }
        let bundle = exportBundle(session: session, loadUserToken: loadUserToken,
                                  loadModelTokens: loadModelTokens)
        let publicKey: PublicKey
        let token: String
        do {
            // Derived before anything is printed: a key that cannot produce a public key
            // cannot sign either, and an export nothing can restore is worth refusing.
            publicKey = try bundle.publicKeyHex()
            token = try bundle.encode()
        } catch {
            HeadlessRunner.logDiagnostic("auth export ERROR \(error.localizedDescription)")
            return false
        }
        HeadlessRunner.logDiagnostic("auth export client_id=\(bundle.clientId.value) "
            + "server_url=\(bundle.serverUrl) public_key=\(publicKey) "
            + "bytes=\(token.utf8.count) hf_tokens=\(bundle.tokenCount)")
        HeadlessRunner.emitPayload(token)
        // The counts and the client id, never the key or a token: `AppLog` is `os.Logger`,
        // which persists to the device log store.
        AppLog.auth.info("auth export client_id=\(bundle.clientId.value) "
            + "hf_tokens=\(bundle.tokenCount) bytes=\(token.utf8.count)")
        return true
    }

    /// Separate from `export` so a test can assert on the bundle, not only the exit code.
    static func exportBundle(
        session: MgmtSession,
        loadUserToken: () -> AuthToken?,
        loadModelTokens: () -> [String: String]
    ) -> IdentityBundle {
        IdentityBundle(clientId: session.registration.clientId,
                       serverUrl: session.serverUrl,
                       privateKeyHex: session.auth.privateKeyHex,
                       userToken: loadUserToken()?.value,
                       modelTokens: loadModelTokens())
    }

    // MARK: - import

    /// What an import would do to this device, decided before it writes anything.
    enum ImportDisposition: Equatable {
        /// The key and the registration already hold what the bundle carries.
        case unchanged
        /// Nothing conflicting is held: a fresh device, one holding part of this same
        /// bundle, or one whose registration moved servers.
        case restores
        /// A different client is held. Its caches go with it, so this needs `force=1`.
        case replaces

        /// Either half can be the differing one: the caches are keyed to the `client_id`,
        /// and the key signs for it. A bundle matching what is held is not a replacement,
        /// so a rerun finishes a partial import without `force=1`.
        ///
        /// A key that cannot sign reads as absent (``ClientRegistrationStore/getPrivateKey()``).
        /// Unlike the crate there is no public-key file to count: iOS persists none.
        static func of(heldKey: PrivateKey?, held: IdentityRegistration?,
                       bundle: IdentityBundle) -> ImportDisposition {
            // `PrivateKey` is already trimmed and lowercased on both sides.
            let keyDiffers = heldKey.map { $0 != bundle.privateKeyHex } ?? false
            // The `client_id` alone. The same client against a different server is this
            // client moving servers: a write to make, not an identity to replace.
            let clientDiffers = held.map { $0.clientId != bundle.clientId } ?? false
            if keyDiffers || clientDiffers { return .replaces }
            let complete = heldKey != nil && held?.serverUrl == bundle.serverUrl
            return complete ? .unchanged : .restores
        }
    }

    /// `auth import bundle=<token> [force=1]`: restore an exported identity. The token comes
    /// as `bundle=` because a `devicectl` launch has no stdin. The whole bundle is validated
    /// before anything is written.
    ///
    /// Re-importing the held identity finishes a partial import, or writes nothing and
    /// reports `changed=no`. A different identity needs `force=1` and drops the caches and
    /// per-model tokens of the outgoing one, which would otherwise submit one client's runs
    /// under another's id.
    ///
    /// Device labels and settings are not carried: they describe the machine. Hugging Face
    /// tokens are restored whatever the disposition, since saving a held value is a no-op
    /// and a token that failed last time gets its retry. The Keychain writes are injected,
    /// as `export`'s reads are.
    static func `import`(
        bundle token: String,
        force: Bool,
        ws: PipetteWorkspace,
        saveUserToken: (AuthToken) -> Bool = KeychainHelper.saveHfToken,
        saveModelToken: (AuthToken, String) -> Bool = {
            KeychainHelper.saveModelHfToken($0, reference: $1)
        },
        clearModelTokens: () -> Int = KeychainHelper.deleteAllModelHfTokens
    ) -> Bool {
        let bundle: IdentityBundle
        let publicKey: PublicKey
        do {
            bundle = try IdentityBundle.decode(token)
            publicKey = try bundle.publicKeyHex()
        } catch {
            HeadlessRunner.log("auth import ERROR \(error.localizedDescription)")
            return false
        }

        let store = ws.registration
        let results = ws.results
        let held = store.get()
        let disposition = ImportDisposition.of(heldKey: store.getPrivateKey(), held: held,
                                               bundle: bundle)
        if disposition == .replaces, !force {
            // Priced, not just refused: `force=1` discards the outgoing client's
            // unsubmitted results, and the refusal says how many before it is taken.
            let heldName = held.map { "the identity \($0.clientId.value)" }
                ?? "a signing key it holds no registration for"
            HeadlessRunner.log("auth import ERROR this device already holds \(heldName), which "
                + "the bundle would replace; run `auth reset` first, or pass force=1 to "
                + "replace it and drop the caches derived from it, "
                + "\(results.listIds(.remotePending).count) unsubmitted result(s) among them")
            return false
        }

        var replaced: String?
        if disposition == .replaces {
            // Counted before the clear, and cleared before the identity goes, because this
            // is the last point that knows the caches belong to the outgoing client. Then
            // the outgoing identity, so a failure below leaves a device that reads as
            // half-restored rather than as a third identity, which a rerun of the same
            // bundle finishes without `force=1`.
            let targets = writeResetTargets(
                held: held, benchmarks: ws.benchmarks.list(.remote).count,
                pending: results.listIds(.remotePending).count,
                synced: results.listIds(.remoteSynced).count)
            do {
                try clearRemoteState(benchmarks: ws.benchmarks, results: results,
                                     evals: ws.evalCompletions)
            } catch {
                HeadlessRunner.log("auth import ERROR could not drop the outgoing identity's "
                    + "caches, so nothing was replaced: \(error.localizedDescription)")
                return false
            }
            replaced = "\(targets) model_tokens_cleared=\(clearModelTokens())"
            store.clear()
        }

        if disposition != .unchanged {
            do {
                // Each `put` writes the file and mirrors it to the Keychain, so a restored
                // identity survives a reinstall exactly as a freshly registered one does.
                try store.putPrivateKey(bundle.privateKeyHex)
            } catch {
                HeadlessRunner.log("auth import ERROR could not write the signing key: "
                    + "\(error.localizedDescription); the import wrote nothing else")
                return false
            }
            do {
                try store.put(importedRegistration(bundle, held: held))
            } catch {
                HeadlessRunner.log("auth import ERROR the signing key is written but the "
                    + "registration is not: \(error.localizedDescription). Rerun the same "
                    + "bundle to finish it")
                return false
            }
        }

        // An empty token authenticates nothing, so `AuthToken` rejects it and it is not
        // counted as restored.
        let userRestored = bundle.userToken
            .flatMap { try? AuthToken($0) }
            .map { saveUserToken($0) } ?? false
        let modelsRestored = bundle.modelTokens
            .compactMap { reference, value in (try? AuthToken(value)).map { (reference, $0) } }
            .filter { saveModelToken($0.1, $0.0) }
            .count
        let restored = (userRestored ? 1 : 0) + modelsRestored
        let missing = bundle.tokenCount - restored

        HeadlessRunner.log(importSummary(bundle: bundle, publicKey: publicKey,
                                         changed: disposition != .unchanged)
            + " hf_tokens=\(restored)" + (missing == 0 ? "" : " hf_tokens_failed=\(missing)"))
        if let replaced {
            HeadlessRunner.log("auth import replaced \(replaced)")
        }
        AppLog.auth.info("auth import client_id=\(bundle.clientId.value) "
            + "changed=\(disposition != .unchanged) hf_tokens=\(restored) "
            + "hf_tokens_failed=\(missing)")
        // A token that did not store means a gated download will 401 later, which is worth
        // an exit code rather than a line the operator has to compare against the export's.
        return missing == 0
    }

    /// `changed`, not `status`: `auth me` already prints a `status`, meaning the server's
    /// approval state.
    static func importSummary(bundle: IdentityBundle, publicKey: PublicKey,
                              changed: Bool) -> String {
        "auth import client_id=\(bundle.clientId.value) server_url=\(bundle.serverUrl) "
            + "public_key=\(publicKey) changed=\(changed ? "yes" : "no")"
    }

    /// The record an import writes. A bundle does not carry the registration inputs the app
    /// renders, so the same client keeps what the device knew and a different one starts
    /// with them blank.
    private static func importedRegistration(_ bundle: IdentityBundle,
                                             held: IdentityRegistration?) -> IdentityRegistration {
        guard let held, held.clientId == bundle.clientId else {
            return IdentityRegistration(clientId: bundle.clientId, status: "",
                                        serverUrl: bundle.serverUrl, organization: "",
                                        contactEmail: "", registeredAt: "")
        }
        return IdentityRegistration(
            clientId: held.clientId, status: held.status, serverUrl: bundle.serverUrl,
            organization: held.organization, contactEmail: held.contactEmail,
            registeredAt: held.registeredAt, clerkUserId: held.clerkUserId,
            clerkSessionId: held.clerkSessionId, clerkPrimaryEmail: held.clerkPrimaryEmail,
            clerkLinkedAt: held.clerkLinkedAt)
    }
}
