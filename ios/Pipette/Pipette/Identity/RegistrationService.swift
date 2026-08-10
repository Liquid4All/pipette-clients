import Foundation

/// Device-registration flow, lifted out of `SetupView` so the view only owns
/// UI state. Stages a CryptoKit signing keypair in the Keychain, registers
/// the public key with the management server, promotes the private key after
/// success, and persists the registration locally.
enum RegistrationService {
    struct Result {
        let clientId: ClientID
        let status: String
    }

    enum Failure: LocalizedError {
        case keychainSaveFailed

        var errorDescription: String? {
            switch self {
            case .keychainSaveFailed: return "Failed to save private key to Keychain"
            }
        }
    }

    static func register(
        serverUrl: ServerURL,
        organization: String,
        contactEmail: String,
        preauthKey: String? = nil,
        /// Free-text description of this client, as the crate's `--client-details`. Lives
        /// server-side only — `auth me` reads it back. Absent, the probed device name
        /// stands, which is what an unattended registration gets.
        clientDetails: String? = nil,
        clerkUserId: String? = nil,
        clerkSessionId: String? = nil,
        clerkPrimaryEmail: String? = nil,
        storage: Storage
    ) async throws -> Result {
        let priorServerUrl = storage.identity.getRegistration()?.serverUrl
        guard let publicKeyHex = KeychainHelper.generatePendingSigningKeyPair() else {
            // Instrumented like the network failure below: a device that can't mint a key never
            // registers, and leaving it uncaptured drops it from the funnel entirely: no
            // `device_registered`, no failure, which reads as a user wandering off rather than a
            // Keychain fault. Android captures at the matching Keystore guards.
            Analytics.capture(AnalyticsEvents.deviceRegistrationFailed, [
                AnalyticsEvents.errorKind: AnalyticsEvents.errorKind(Failure.keychainSaveFailed),
            ])
            throw Failure.keychainSaveFailed
        }

        // Register before persisting any identity: the pending private key is
        // only promoted and the registration written after the server accepts.
        // A rejected pre-auth key (401/403) tears down the staged key below and
        // leaves no partial identity, so re-entering a valid key just works.
        let result: ManagementClient.RegistrationResult
        do {
            result = try await ManagementClient.register(
                serverUrl: serverUrl,
                organization: organization,
                contactEmail: contactEmail,
                clientDetails: clientDetails ?? DeviceProbe.detectDeviceName(),
                publicKeyHex: publicKeyHex,
                preauthKey: preauthKey,
                // Register with a complete profile + capability set so the
                // planner can match this device immediately. Without it the
                // client would stay unmatchable until the next launch's patch —
                // the launch refresh runs before registration exists, so on a
                // first install that is a whole session away.
                device: ProfileReporter.profile()
            )
        } catch {
            _ = KeychainHelper.deletePendingPrivateKey()
            // Coarse error kind only: a registration failure message can embed the server URL
            // and the contact email, neither of which belongs in analytics.
            Analytics.capture(AnalyticsEvents.deviceRegistrationFailed, [
                AnalyticsEvents.errorKind: AnalyticsEvents.errorKind(error),
            ])
            throw error
        }

        guard KeychainHelper.promotePendingPrivateKey() else {
            _ = KeychainHelper.deletePendingPrivateKey()
            // The server accepted this device, but without a usable key it can never sign a request.
            // Uncaptured, that drop-off is invisible on the client *and* looks like a healthy
            // registration on the server.
            Analytics.capture(AnalyticsEvents.deviceRegistrationFailed, [
                AnalyticsEvents.errorKind: AnalyticsEvents.errorKind(Failure.keychainSaveFailed),
            ])
            throw Failure.keychainSaveFailed
        }

        let registration = IdentityRegistration(
            clientId: result.clientId,
            status: result.status,
            serverUrl: serverUrl,
            organization: organization,
            contactEmail: contactEmail,
            registeredAt: JobDateFormat.iso8601.string(from: Date()),
            clerkUserId: clerkUserId,
            clerkSessionId: clerkSessionId,
            clerkPrimaryEmail: clerkPrimaryEmail,
            clerkLinkedAt: clerkUserId == nil ? nil : JobDateFormat.iso8601.string(from: Date())
        )
        try storage.identity.putRegistration(registration)
        let serverChanged = priorServerUrl?.value != serverUrl.value

        // Fire-and-forget: catalog sync is public and must not block registration.
        BenchmarkSyncCoordinator.syncBestEffortInBackground(
            serverUrl: serverUrl,
            storage: storage,
            reason: "registration",
            clearCacheFirst: serverChanged
        )

        // From here on every event attributes to the server-assigned device id. Deliberately no
        // email / organization / Clerk identity: the management server already holds those, and
        // PostHog has no need for them.
        Analytics.identify(result.clientId.value)
        Analytics.capture(AnalyticsEvents.deviceRegistered, [AnalyticsEvents.status: result.status])

        return Result(clientId: result.clientId, status: result.status)
    }
}

/// Turn a registration failure into a short, human-readable sentence for the
/// setup screen. Maps common connectivity cases to plain language and unwraps
/// the underlying message for everything else (avoids raw error reflection
/// text leaking into the UI).
/// `preauthContext` is set only on the registration path, where a `401`/`403`
/// is the server's pre-auth-key verdict (see mgmt httpapi.md §2.2). Other
/// callers (e.g. the public benchmark-catalog sync) leave it off so an
/// unrelated `401`/`403` isn't mislabeled as a key problem.
func humanizedRegistrationError(_ error: Error, preauthContext: Bool = false) -> String {
    if let urlError = error as? URLError {
        switch urlError.code {
        case .notConnectedToInternet, .networkConnectionLost:
            return "No internet connection. Check your network and try again."
        case .timedOut:
            return "The request timed out. Check your connection and try again."
        case .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed:
            return "Couldn't reach the registration server. Please try again in a moment."
        default:
            return urlError.localizedDescription
        }
    }
    // Replace the raw "HTTP 401: {body}" with a clear cause so the user fixes
    // the key rather than blaming the network.
    if preauthContext, case let ManagementClientError.httpStatus(statusCode, _) = error {
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
