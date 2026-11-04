import Foundation

// Client registration and profile ops against the management server: `pipette-cli`'s
// `client/auth.rs`. Not `Auth.swift`: `PipetteMgmtClient/Auth.swift` has that name.

/// Typed as the request carries it, so a mistyped address is refused before a keypair is
/// minted or a pre-auth key spent.
nonisolated struct RegisterInput: Sendable {
    var organization: String
    var contactEmail: ContactEmail
    /// Server-side only: `auth me` reads it back.
    var clientDetails: String
    var deviceName: String?
    var deviceFormFactor: DeviceFormFactor?
    /// Sent with the request, never persisted.
    var preauthKey: ClientPreauthKey?
    // iOS only: the Clerk sign-in this registration is linked to.
    var clerkUserId: String?
    var clerkSessionId: String?
    var clerkPrimaryEmail: String?
}

/// The keypair is staged in the Keychain, where the crate holds it in memory, and promoted
/// into the store only after the server accepts.
nonisolated func register(
    registrationStore: ClientRegistrationStore,
    device: ClientDeviceStore,
    server: ServerUrl,
    http: HttpClient = .shared,
    input: RegisterInput
) async throws -> IdentityRegistration {
    // A second registration would orphan every result submitted under the first.
    try registrationStore.ensureEmpty()
    let profile = registrationProfile()
    AppLog.auth.info(
        "registering client against \(server) (cpu_arch=\(profile.deviceCpuArch ?? "unset"), "
            + "os_family=\(profile.deviceOsFamily ?? "unset"))")
    let prior = try device.get()
    // Before the server call: an I/O failure aborts before a client id is spent, and a
    // refused registration still leaves the device named.
    try device.put(mergeLabelsForRegister(
        deviceName: input.deviceName, deviceFormFactor: input.deviceFormFactor, prior: prior))
    // Counted, or a Keychain fault reads in the funnel as a user who wandered off.
    guard let publicKeyHex = KeychainHelper.generatePendingSigningKeyPair() else {
        throw registrationFailed(IdentityError.keyNotPersisted)
    }
    // Before anything persists: a rejected pre-auth key leaves no identity to reset.
    let response: RegisterResponse
    do {
        response = try await PipetteMgmtClient.ClientApi.register(
            server: server,
            http: http,
            request: RegisterRequest(
                publicKey: try PublicKey(publicKeyHex),
                organization: input.organization,
                contactEmail: input.contactEmail,
                clientDetails: input.clientDetails,
                clientPreauthKey: input.preauthKey,
                device: profile
            ))
    } catch {
        _ = KeychainHelper.deletePendingPrivateKey()
        throw registrationFailed(error)
    }
    // Checked rather than trusted: an id outside the server's charset would sign as nobody.
    let clientId: ClientId
    do {
        clientId = try ClientId(response.clientId)
    } catch {
        _ = KeychainHelper.deletePendingPrivateKey()
        throw registrationFailed(IdentityError.registeredId(error.localizedDescription))
    }
    let registeredAt = JobDateFormat.iso8601.string(from: Date())
    let registration = IdentityRegistration(
        clientId: clientId,
        // Stored as the wire word: the record is read back by builds that may not know a
        // status this one does, and the app's own gate compares the string.
        status: response.status.wireValue,
        serverUrl: server,
        organization: input.organization,
        contactEmail: input.contactEmail.value,
        registeredAt: registeredAt,
        clerkUserId: input.clerkUserId,
        clerkSessionId: input.clerkSessionId,
        clerkPrimaryEmail: input.clerkPrimaryEmail,
        clerkLinkedAt: input.clerkUserId == nil ? nil : registeredAt)
    // The server has committed and any pre-auth key is spent. Half an identity would make
    // `ensureEmpty` refuse every retry, and `auth reset` has no UI, so roll back whole.
    do {
        guard let staged = KeychainHelper.loadPendingPrivateKey() else {
            throw IdentityError.keyNotPersisted
        }
        try registrationStore.putPrivateKey(staged)
        try registrationStore.put(registration)
    } catch {
        registrationStore.clear()
        _ = KeychainHelper.deletePendingPrivateKey()
        // Logged because the thrown failure carries only the sentence the user reads.
        AppLog.auth.error(
            "registration: the local identity was not saved: \(error.localizedDescription)")
        throw registrationFailed(IdentityError.registrationPersisted(
            clientId: clientId.value, reason: error.localizedDescription))
    }
    _ = KeychainHelper.deletePendingPrivateKey()

    // Deliberately no email, organization or Clerk identity: the management server holds
    // those, and PostHog has no need for them.
    Analytics.identify(clientId.value)
    Analytics.capture(
        AnalyticsEvents.deviceRegistered, [AnalyticsEvents.status: response.status.wireValue])
    return registration
}

/// The device fields sent at registration: the CPU architecture and the OS family, so the
/// server derives the `arch:` and `os_family:` flags before the planner worker first
/// reports the full profile.
nonisolated func registrationProfile() -> DeviceProfileFields {
    DeviceProfileFields(deviceCpuArch: detectCpuArch(), deviceOsFamily: detectOsFamily())
}

/// An explicit label beats the one `device.json` holds, and an absent or blank one keeps
/// it: `register` may narrow a label but never erase one.
nonisolated func mergeLabelsForRegister(
    deviceName: String?,
    deviceFormFactor: DeviceFormFactor?,
    prior: DeviceLabels
) -> DeviceLabels {
    DeviceLabels(
        deviceName: deviceName.flatMap(DeviceLabels.suppliedName) ?? prior.deviceName,
        deviceFormFactor: deviceFormFactor ?? prior.deviceFormFactor)
}

/// `GET /clients/me`, against the server the signing key is registered with: any other
/// would describe a client this key does not sign for.
nonisolated func fetchProfile(
    registration: ClientRegistrationStore,
    http: HttpClient = .shared
) async throws -> ClientProfile {
    let session = try registration.mgmtSession()
    return try await PipetteMgmtClient.ClientApi.me(
        server: session.serverUrl, http: http, auth: session.auth)
}

/// `PATCH /clients/me`: the one `auth set-device` field that lives on the server, and so
/// the one that needs a registration.
nonisolated func updateClientDetails(
    registration: ClientRegistrationStore,
    http: HttpClient = .shared,
    clientDetails: String
) async throws -> ClientProfile {
    let session = try registration.mgmtSession()
    return try await PipetteMgmtClient.ClientApi.updateMe(
        server: session.serverUrl, http: http, auth: session.auth,
        request: UpdateClientRequest(clientDetails: clientDetails))
}

// MARK: - iOS: analytics, an orphaned key, and the setup screen's errors

/// Counts a failed registration by its kind only: the message can carry the server URL and
/// the contact email.
private nonisolated func registrationFailed(_ error: any Error) -> any Error {
    Analytics.capture(AnalyticsEvents.deviceRegistrationFailed, [
        AnalyticsEvents.errorKind: AnalyticsEvents.errorKind(error),
    ])
    return error
}

/// Drop a signing key no registration claims, so `register` can mint a fresh identity
/// rather than refuse forever. A state the crate cannot reach: the Keychain copy of the key
/// outlives the app container, and refusing over it would strand the device on setup with
/// `auth reset`, which has no UI, as the only way out.
///
/// Not while results await submission: the new client id would claim them, the
/// re-attribution `ensureEmpty` exists to prevent. `auth reset force=1` is the way through.
@discardableResult
nonisolated func discardOrphanedSigningKey(ws: PipetteWorkspace) -> Bool {
    guard ws.registration.get() == nil,
          ws.registration.getPrivateKey() != nil
    else { return false }
    let pending = ws.results.listIds(.remotePending).count
    guard pending == 0 else {
        AppLog.auth.error(
            "identity: a signing key with no registration is held, and \(pending) "
                + "unsubmitted result(s) are on disk; keeping both rather than "
                + "re-registering under a new id")
        return false
    }
    guard ws.registration.deletePrivateKey() else {
        AppLog.auth.error("identity: could not discard the orphaned signing key")
        return false
    }
    AppLog.auth.info(
        "identity: discarded a signing key no registration claims; registering fresh")
    return true
}

/// A registration failure as one plain sentence for the setup screen, never raw error
/// reflection.
/// `preauthContext` is set only on the registration path, where a `401`/`403` is the
/// server's pre-auth-key verdict (the mgmt repo's `docs/httpapi.md` §2.2.3). Other callers
/// (the benchmark-catalog sync) leave it off so an unrelated `401`/`403` is not mislabeled.
func humanizedRegistrationError(_ error: Error, preauthContext: Bool = false) -> String {
    // A transport error carries the message, not the `URLError` code, so there is no
    // per-code copy to pick.
    if case .transport = error as? PipetteMgmtClient.Error {
        return "Couldn't reach the registration server. Check your connection and try again."
    }
    if preauthContext, let statusCode = (error as? PipetteMgmtClient.Error)?.httpStatusCode {
        switch statusCode {
        case 401:
            return "Registration was rejected: the pre-auth key is invalid, "
                + "expired, or already used. Check the key and try again."
        case 403:
            return "This collector requires a pre-auth key to register. "
                + "Enter a valid key and try again."
        default:
            break
        }
    }
    return formatJobError(error, contextSize: 0)
}
