import Foundation

/// Local registration metadata stored in `metadata/registration.json` — the crate's
/// `IdentityRegistration` (`pipette-cli/src/identity/types.rs:16`).
///
/// The CLI's record holds only the upload identity (`client_id`, `server_url`) and
/// fetches everything else from `GET /clients/me` on demand. This one keeps the
/// registration inputs too, because the app renders them without a network round trip
/// (Settings shows the organization and contact email, the submission gate reads
/// `status`) and re-registration prefills from them. The Clerk fields have no CLI
/// counterpart at all: the desktop client has no sign-in gate.
///
/// Wire keys are `snake_case`, as every identity file the CLI writes is — see
/// ``ClientSettings``. Records written before that are still read: `init(from:)` falls
/// back to the camelCase spelling key by key, so an existing install keeps its
/// registration instead of being silently pushed back to the setup screen. The
/// fallback can go once no device is running a build that predates it.
nonisolated struct IdentityRegistration: Codable, Sendable, Equatable {
    let clientId: ClientID
    let status: String
    let serverUrl: ServerURL
    let organization: String
    let contactEmail: String
    let registeredAt: String
    let clerkUserId: String?
    let clerkSessionId: String?
    let clerkPrimaryEmail: String?
    let clerkLinkedAt: String?

    init(
        clientId: ClientID,
        status: String,
        serverUrl: ServerURL,
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

    /// The pre-`snake_case` spelling. Read-only: nothing writes these keys any more.
    /// `status` and `organization` are absent because they were already spelled the
    /// same either way.
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

        guard let clientId: ClientID = try value(.clientId, .clientId),
              let serverUrl: ServerURL = try value(.serverUrl, .serverUrl)
        else {
            throw DecodingError.keyNotFound(CodingKeys.clientId, DecodingError.Context(
                codingPath: decoder.codingPath,
                debugDescription: "registration.json has neither client_id/server_url nor "
                    + "their legacy camelCase spellings"))
        }
        self.clientId = clientId
        self.serverUrl = serverUrl
        // Only the upload identity above is required, matching the crate's record,
        // which holds `client_id` and `server_url` and nothing else. The rest are
        // convenience copies of state the server owns — they prefill the setup form
        // and fill Settings rows — so a record missing one degrades that row rather
        // than reading as "unregistered" and pushing a registered device back through
        // setup, which would mint a new keypair and orphan its submitted results.
        // No legacy lookup for these two — they were never renamed.
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

    /// Drop the Clerk link, keeping `clientId`, the registration inputs, and (elsewhere) the
    /// signing key. Applied on an explicit sign-out: the link is a purely local guard, since
    /// the Clerk identity is never sent to the management server, so un-pinning the device
    /// costs nothing server-side and lets the next account link instead of colliding with
    /// this one.
    ///
    /// `clerkLinkedAt` goes too. It dates the *current* link, and `withClerkLink` preserves
    /// an existing value, so leaving it would date the next account's link from this one's.
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
