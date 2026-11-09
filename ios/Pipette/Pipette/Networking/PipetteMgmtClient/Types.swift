import Foundation

// The values the management wire carries: the crate's `types.rs`.
//
// The validated newtypes refuse a wrong shape at the field an operator typed,
// rather than after a registration has spent a pre-auth key. `Decodable` runs the
// same initializer, so a record off the wire is checked by the same rule.
//
// The crate's user-account types (user keys, `UserStatus`, the `/users/*` bodies)
// are not mirrored: this app only authenticates as a device.

// MARK: - Conditional requests

nonisolated struct InvalidHeaderValue: LocalizedError {
    var errorDescription: String? { "ETag is not a valid HTTP header value" }
}

nonisolated struct EntityTag: Codable, Hashable, Sendable {
    let value: String

    init(_ value: String) throws(InvalidHeaderValue) {
        // A header value is printable ASCII plus tab.
        guard value.utf8.allSatisfy({ $0 == 0x09 || (0x20...0x7e).contains($0) }) else {
            throw InvalidHeaderValue()
        }
        self.value = value
    }

    static func fromHeaderValue(_ value: String) -> EntityTag? {
        try? EntityTag(value)
    }

    init(from decoder: any Decoder) throws {
        try self.init(decoder.singleValueContainer().decode(String.self))
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }
}

/// Its own type so the tag being sent can't be crossed with the one that came back.
nonisolated struct IfNoneMatch: Hashable, Sendable {
    let headerValue: String

    init(_ etag: EntityTag) {
        headerValue = etag.value
    }
}

// MARK: - Credentials

/// A client pre-auth key, which admits a registering client already `approved`.
/// Trimmed (a pasted key often carries a trailing newline); the format is the
/// server's to validate. Printed redacted so no request dump or `\(key)` leaks it.
nonisolated struct ClientPreauthKey: Encodable, Hashable, Sendable,
    CustomStringConvertible, CustomDebugStringConvertible {
    let value: String

    init(_ raw: String) throws(EmptyClientPreauthKeyError) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw EmptyClientPreauthKeyError() }
        value = trimmed
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }

    var description: String { "<redacted>" }
    var debugDescription: String { "ClientPreauthKey(<redacted>)" }
}

nonisolated struct EmptyClientPreauthKeyError: LocalizedError {
    var errorDescription: String? { "pre-auth key must not be empty" }
}

/// An Ed25519 private key is its 32-byte seed. Not `private`:
/// ``PipetteMgmtClient/Auth`` re-checks the length where it turns a key into a seed.
nonisolated let privateKeyLengthBytes = 32

/// The Ed25519 private key a client signs as, in hex. Checked at the store it was
/// read from rather than at the first signed request.
///
/// Lowercased so equality means "the same key": otherwise an import could refuse
/// to replace a key with itself over casing.
///
/// Unlike the crate's, not `Codable` and rendered redacted: this app exports
/// through ``IdentityBundle``, which reads the hex explicitly.
nonisolated struct PrivateKey: Hashable, Sendable, CustomStringConvertible,
    CustomDebugStringConvertible {
    let value: String

    init(_ raw: String) throws(PrivateKeyError) {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let bytes = Data(hexEncoded: normalized) else { throw .notHex }
        guard bytes.count == privateKeyLengthBytes else { throw .length(found: bytes.count) }
        value = normalized
    }

    var description: String { "<redacted>" }
    var debugDescription: String { "PrivateKey(<redacted>)" }
}

/// Each case names the value's shape, never the value.
nonisolated enum PrivateKeyError: Error, Sendable, Equatable {
    case notHex
    case length(found: Int)
    /// The right length, still refused by CryptoKit. The crate has no such case:
    /// a signing key from 32 bytes cannot fail there.
    case notAKey
}

nonisolated extension PrivateKeyError {
    var message: String {
        switch self {
        case .notHex: return "private key is not hex"
        case .length(let found):
            return "private key must be \(privateKeyLengthBytes) bytes of hex, not \(found)"
        case .notAKey: return "private key is not a Curve25519 signing key"
        }
    }
}

/// Its own constant: equal to the private key's by the curve, not by definition.
nonisolated let publicKeyLengthBytes = 32

/// The Ed25519 public key a client registers under, in hex. Validated here because
/// the server would refuse an unusable key only after the keypair is in the Keychain.
/// Not redacted: this is the half you may publish.
nonisolated struct PublicKey: Encodable, Hashable, Sendable, CustomStringConvertible {
    let value: String

    init(_ raw: String) throws(PublicKeyError) {
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let bytes = Data(hexEncoded: normalized) else { throw .notHex }
        guard bytes.count == publicKeyLengthBytes else { throw .length(found: bytes.count) }
        value = normalized
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }

    var description: String { value }
}

nonisolated enum PublicKeyError: LocalizedError {
    case notHex
    case length(found: Int)

    var errorDescription: String? {
        switch self {
        case .notHex: return "public key is not hex"
        case .length(let found):
            return "public key must be \(publicKeyLengthBytes) bytes of hex, not \(found)"
        }
    }
}

/// A shape check, not RFC 5322: it catches what an operator mistypes (a missing `@`,
/// a bare username, a trailing comma), and anything narrower would reject addresses
/// that deliver.
nonisolated struct ContactEmail: Codable, Hashable, Sendable, CustomStringConvertible {
    let value: String

    init(_ raw: String) throws(ContactEmailError) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw .empty }
        guard !trimmed.contains(where: \.isWhitespace) else { throw .whitespace }
        let parts = trimmed.split(separator: "@", omittingEmptySubsequences: false)
        guard parts.count == 2 else { throw .noAt }
        guard !parts[0].isEmpty, !parts[1].isEmpty else { throw .emptyPart }
        // A dot separates a real domain from a local hostname.
        let domain = parts[1]
        guard domain.contains("."), !domain.hasPrefix("."), !domain.hasSuffix(".") else {
            throw .domain
        }
        // Hostname characters only, which catches the punctuation a copied
        // address drags along. The local part stays permissive.
        guard domain.allSatisfy({
            $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-" || $0 == ".")
        }) else { throw .domain }
        value = trimmed
    }

    init(from decoder: any Decoder) throws {
        try self.init(decoder.singleValueContainer().decode(String.self))
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }

    var description: String { value }
}

nonisolated enum ContactEmailError: LocalizedError {
    case empty
    case whitespace
    case noAt
    case emptyPart
    case domain

    var errorDescription: String? {
        switch self {
        case .empty: return "contact email is empty"
        case .whitespace: return "contact email contains whitespace"
        case .noAt: return "contact email needs exactly one `@`"
        case .emptyPart:
            return "contact email needs a local part and a domain either side of the `@`"
        case .domain:
            return "contact email needs a dotted hostname for a domain, e.g. `example.com`"
        }
    }
}

// MARK: - Instants

/// An instant the server stamped, carried as the RFC 3339 string it arrived as:
/// nothing does arithmetic on these, and a `Date` would re-render it differently.
nonisolated struct Timestamp: Codable, Hashable, Sendable, CustomStringConvertible {
    let value: String

    init(_ raw: String) throws(TimestampError) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard Self.parses(trimmed) else { throw TimestampError(value: trimmed) }
        value = trimmed
    }

    /// Two formatters: the server sends either form, and one accepts only one.
    /// `nonisolated(unsafe)` because `ISO8601DateFormatter` is documented thread-safe.
    private nonisolated(unsafe) static let plain: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private nonisolated(unsafe) static let fractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static func parses(_ raw: String) -> Bool {
        plain.date(from: raw) != nil || fractional.date(from: raw) != nil
    }

    init(from decoder: any Decoder) throws {
        try self.init(decoder.singleValueContainer().decode(String.self))
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }

    var description: String { value }
}

nonisolated struct TimestampError: LocalizedError {
    let value: String
    var errorDescription: String? { "timestamp is not RFC 3339: \(value)" }
}

nonisolated extension KeyedDecodingContainer {
    /// Absent, `null` and unparseable all read as `nil`. Only for a field the
    /// client does not act on; anything a caller branches on should be strict.
    func decodeLenientTimestamp(forKey key: Key) -> Timestamp? {
        guard let raw = try? decodeIfPresent(String.self, forKey: key) else { return nil }
        return try? Timestamp(raw)
    }
}

// MARK: - The server

/// The management server's base URL. Normalized (lowercase scheme and host, no
/// trailing `/`) so joining an endpoint path is a concatenation and the signed
/// target matches what the server receives. A path prefix (`https://host/api`) is
/// kept, since deployments behind one sign over it.
///
/// The lowercasing matches the crate's `url::Url::parse`: without it, a stored
/// record spelled `HTTPS://Host` would stop reading as a registration.
nonisolated struct ServerUrl: Codable, Hashable, Sendable, CustomStringConvertible {
    let value: String

    init(_ raw: String) throws(ServerUrlError) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // The `source` strings are what `url::ParseError` prints, so an operator
        // reads one message on either client.
        guard var components = URLComponents(string: trimmed) else {
            throw .notAUrl(url: trimmed, source: "not a URL")
        }
        guard let scheme = components.scheme else {
            throw .notAUrl(url: trimmed, source: "relative URL without a base")
        }
        // The scheme before the host: what is wrong with `mailto:` is the scheme,
        // not a host it was never going to have.
        components.scheme = scheme.lowercased()
        guard components.scheme == "http" || components.scheme == "https" else {
            throw .scheme(url: trimmed)
        }
        guard let host = components.host, !host.isEmpty else {
            throw .notAUrl(url: trimmed, source: "empty host")
        }
        // DNS is case-insensitive: two spellings of one host are one server.
        components.host = host.lowercased()
        // An appended endpoint path would land inside a query or after a
        // fragment, sending every request to `/`.
        guard components.query == nil, components.fragment == nil else {
            throw .notABase(url: trimmed)
        }
        guard var value = components.string else {
            throw .notAUrl(url: trimmed, source: "not a URL")
        }
        while value.hasSuffix("/") { value.removeLast() }
        self.value = value
    }

    init(from decoder: any Decoder) throws {
        try self.init(decoder.singleValueContainer().decode(String.self))
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(value)
    }

    var description: String { value }
}

/// Every case carries the trimmed input, so an operator with several deployments
/// sees which URL it is about.
nonisolated enum ServerUrlError: LocalizedError, Equatable {
    /// A bare host (`mgmt.example.com`), or a typo'd scheme.
    case notAUrl(url: String, source: String)
    case scheme(url: String)
    case notABase(url: String)

    var errorDescription: String? {
        switch self {
        case .notAUrl(let url, let source):
            return "invalid management server URL `\(url)`: \(source)"
        case .scheme(let url):
            return "management server URL `\(url)` must use http or https"
        case .notABase(let url):
            return "management server URL `\(url)` must not carry a query or fragment"
        }
    }
}

// MARK: - Statuses

/// Whether an operator has vetted this client (the mgmt repo's
/// `docs/authentication.md` §4). An unknown status keeps the server's word, for
/// `auth me` to show, and is never ``approved``.
nonisolated enum ClientStatus: Codable, Hashable, Sendable, CustomStringConvertible {
    /// A submission is held, never scored, when the server allows unverified
    /// submissions, and refused otherwise.
    case pending
    /// The only status that may claim work.
    case approved
    case unknown(String)

    init(wireValue: String) {
        switch wireValue {
        case "pending": self = .pending
        case "approved": self = .approved
        default: self = .unknown(wireValue)
        }
    }

    var wireValue: String {
        switch self {
        case .pending: return "pending"
        case .approved: return "approved"
        case .unknown(let raw): return raw
        }
    }

    init(from decoder: any Decoder) throws {
        self.init(wireValue: try decoder.singleValueContainer().decode(String.self))
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(wireValue)
    }

    var description: String { wireValue }
}

nonisolated extension PipetteMgmtClient {
    /// How far a submitted job has got. Nested because ``JobManifest`` has a
    /// `JobStatus` of its own, a different vocabulary.
    enum JobStatus: Codable, Hashable, Sendable, CustomStringConvertible {
        case incoming
        /// An eval in the score queue, awaiting or finishing the scoring call.
        case scoring
        /// The only status that carries metrics.
        case processed
        /// A failure record; its metrics are not surfaced.
        case failed
        case unknown(String)

        init(wireValue: String) {
            switch wireValue {
            case "incoming": self = .incoming
            case "scoring": self = .scoring
            case "processed": self = .processed
            case "failed": self = .failed
            default: self = .unknown(wireValue)
            }
        }

        var wireValue: String {
            switch self {
            case .incoming: return "incoming"
            case .scoring: return "scoring"
            case .processed: return "processed"
            case .failed: return "failed"
            case .unknown(let raw): return raw
            }
        }

        init(from decoder: any Decoder) throws {
            self.init(wireValue: try decoder.singleValueContainer().decode(String.self))
        }

        func encode(to encoder: any Encoder) throws {
            var container = encoder.singleValueContainer()
            try container.encode(wireValue)
        }

        var description: String { wireValue }
    }
}

// MARK: - Clients

/// `POST /clients/register` body, with the device fields flat beside the
/// credentials so a first-time client is matchable without a follow-up patch
/// (the mgmt repo's `docs/client-integration.md` §2).
nonisolated struct RegisterRequest: Encodable, Sendable {
    let publicKey: PublicKey
    let organization: String
    let contactEmail: ContactEmail
    let clientDetails: String
    /// Skipped on the wire when absent. Never persisted or logged.
    let clientPreauthKey: ClientPreauthKey?
    var device = DeviceProfileFields()

    enum CodingKeys: String, CodingKey {
        case publicKey = "public_key"
        case organization
        case contactEmail = "contact_email"
        case clientDetails = "client_details"
        case clientPreauthKey = "client_preauth_key"
    }

    /// Swift has no `#[serde(flatten)]`: handing `device` the same encoder writes
    /// its keys into this container, the flat object the server expects.
    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(publicKey, forKey: .publicKey)
        try container.encode(organization, forKey: .organization)
        try container.encode(contactEmail, forKey: .contactEmail)
        try container.encode(clientDetails, forKey: .clientDetails)
        try container.encodeIfPresent(clientPreauthKey, forKey: .clientPreauthKey)
        try device.encode(to: encoder)
    }
}

nonisolated struct RegisterResponse: Decodable, Sendable, Equatable {
    /// Raw: the caller that will sign with it makes it a ``ClientId``.
    let clientId: String
    let status: ClientStatus

    enum CodingKeys: String, CodingKey {
        case clientId = "client_id"
        case status
    }
}

/// Fields shared by `POST /clients/register` and `PATCH /clients/me`, flattened
/// into either (see ``RegisterRequest/encode(to:)``), so none may share a key with
/// the request around them (`client_details` is the request's). An absent field
/// leaves the stored value unchanged on PATCH.
nonisolated struct DeviceProfileFields: Encodable, Sendable, Equatable {
    var deviceName: String?
    /// Typed narrower than the crate's `Option<String>`: the server refuses a
    /// word outside its enum, so a misspelling is caught at compile time.
    var deviceFormFactor: DeviceFormFactor?
    var deviceOsName: String?
    var deviceOsVersion: String?
    /// The CPU architecture as the platform spells it (`arm64`). The server derives
    /// the `arch:` flag from it.
    var deviceCpuArch: String?
    /// The OS family as the platform spells it (`iOS`). The server derives the
    /// `os_family:` flag from it.
    var deviceOsFamily: String?
    var deviceChipModel: String?
    var deviceRamBytes: UInt64?
    var deviceGpuModel: String?
    var deviceGpuVramBytes: UInt64?
    var deviceNpuModel: String?
    var deviceNpuVramBytes: UInt64?
    /// Replaces the stored set wholesale when non-nil. Lowercase flags
    /// (`runtime:llama_cpp`), never in a server-owned namespace (`os:`, `device:`).
    var capabilities: [String]?

    enum CodingKeys: String, CodingKey {
        case deviceName = "device_name"
        case deviceFormFactor = "device_form_factor"
        case deviceOsName = "device_os_name"
        case deviceOsVersion = "device_os_version"
        case deviceCpuArch = "device_cpu_arch"
        case deviceOsFamily = "device_os_family"
        case deviceChipModel = "device_chip_model"
        case deviceRamBytes = "device_ram_bytes"
        case deviceGpuModel = "device_gpu_model"
        case deviceGpuVramBytes = "device_gpu_vram_bytes"
        case deviceNpuModel = "device_npu_model"
        case deviceNpuVramBytes = "device_npu_vram_bytes"
        case capabilities
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(deviceName, forKey: .deviceName)
        try container.encodeIfPresent(deviceFormFactor, forKey: .deviceFormFactor)
        try container.encodeIfPresent(deviceOsName, forKey: .deviceOsName)
        try container.encodeIfPresent(deviceOsVersion, forKey: .deviceOsVersion)
        try container.encodeIfPresent(deviceCpuArch, forKey: .deviceCpuArch)
        try container.encodeIfPresent(deviceOsFamily, forKey: .deviceOsFamily)
        try container.encodeIfPresent(deviceChipModel, forKey: .deviceChipModel)
        try container.encodeIfPresent(deviceRamBytes, forKey: .deviceRamBytes)
        try container.encodeIfPresent(deviceGpuModel, forKey: .deviceGpuModel)
        try container.encodeIfPresent(deviceGpuVramBytes, forKey: .deviceGpuVramBytes)
        try container.encodeIfPresent(deviceNpuModel, forKey: .deviceNpuModel)
        try container.encodeIfPresent(deviceNpuVramBytes, forKey: .deviceNpuVramBytes)
        try container.encodeIfPresent(capabilities, forKey: .capabilities)
    }
}

/// `PATCH /clients/me` body. Only present values update, so a device field
/// cannot be cleared once set.
nonisolated struct UpdateClientRequest: Encodable, Sendable, Equatable {
    var clientDetails: String?
    var device = DeviceProfileFields()

    enum CodingKeys: String, CodingKey {
        case clientDetails = "client_details"
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(clientDetails, forKey: .clientDetails)
        try device.encode(to: encoder)
    }
}

/// `GET`/`PATCH /clients/me`
nonisolated struct ClientProfile: Decodable, Sendable, Equatable {
    let clientId: String
    let organization: String
    let clientDetails: String
    let contactEmail: ContactEmail
    let status: ClientStatus
    /// Mgmt-assigned tags (`team/mobile/ios`), display-only. Defaulted, so this
    /// still reads against a server that predates the field.
    let tags: [String]
    /// `true` while re-evaluation after a profile or capability change is
    /// pending. While set, plan operations are refused.
    let reindexPending: Bool
    /// The flags the client reported; server-derived `device_*` flags are not echoed.
    let capabilities: [String]
    let deviceName: String?
    let deviceFormFactor: String?
    let deviceOsName: String?
    let deviceOsVersion: String?
    let deviceChipModel: String?
    let deviceRamBytes: UInt64?
    let deviceGpuModel: String?
    let deviceGpuVramBytes: UInt64?
    let deviceNpuModel: String?
    let deviceNpuVramBytes: UInt64?

    enum CodingKeys: String, CodingKey {
        case clientId = "client_id"
        case organization
        case clientDetails = "client_details"
        case contactEmail = "contact_email"
        case status
        case tags
        case reindexPending = "reindex_pending"
        case capabilities
        case deviceName = "device_name"
        case deviceFormFactor = "device_form_factor"
        case deviceOsName = "device_os_name"
        case deviceOsVersion = "device_os_version"
        case deviceChipModel = "device_chip_model"
        case deviceRamBytes = "device_ram_bytes"
        case deviceGpuModel = "device_gpu_model"
        case deviceGpuVramBytes = "device_gpu_vram_bytes"
        case deviceNpuModel = "device_npu_model"
        case deviceNpuVramBytes = "device_npu_vram_bytes"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        clientId = try c.decode(String.self, forKey: .clientId)
        organization = try c.decode(String.self, forKey: .organization)
        clientDetails = try c.decode(String.self, forKey: .clientDetails)
        contactEmail = try c.decode(ContactEmail.self, forKey: .contactEmail)
        status = try c.decode(ClientStatus.self, forKey: .status)
        tags = try c.decodeIfPresent([String].self, forKey: .tags) ?? []
        reindexPending = try c.decodeIfPresent(Bool.self, forKey: .reindexPending) ?? false
        capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities) ?? []
        deviceName = try c.decodeIfPresent(String.self, forKey: .deviceName)
        deviceFormFactor = try c.decodeIfPresent(String.self, forKey: .deviceFormFactor)
        deviceOsName = try c.decodeIfPresent(String.self, forKey: .deviceOsName)
        deviceOsVersion = try c.decodeIfPresent(String.self, forKey: .deviceOsVersion)
        deviceChipModel = try c.decodeIfPresent(String.self, forKey: .deviceChipModel)
        deviceRamBytes = try c.decodeIfPresent(UInt64.self, forKey: .deviceRamBytes)
        deviceGpuModel = try c.decodeIfPresent(String.self, forKey: .deviceGpuModel)
        deviceGpuVramBytes = try c.decodeIfPresent(UInt64.self, forKey: .deviceGpuVramBytes)
        deviceNpuModel = try c.decodeIfPresent(String.self, forKey: .deviceNpuModel)
        deviceNpuVramBytes = try c.decodeIfPresent(UInt64.self, forKey: .deviceNpuVramBytes)
    }
}

// MARK: - Plans

/// A job leased via `POST /plans/claim`. `spec` stays raw (typed later by
/// `runSpecFromClaim(_:)`): an unreadable payload must not fail the envelope, which
/// identifies the job, and reporting the job is the only way it stops being re-served.
nonisolated struct ClaimedJob: Decodable, Sendable, Equatable {
    let jobId: String
    let benchmarkId: String
    /// Lease increment as an ISO 8601 duration (`"PT10M"`). Heartbeat at half this;
    /// each success extends the lease by this much.
    let timeWindow: String
    /// Lenient: nothing acts on it beyond a log line, so a bad value must not
    /// fail the claim.
    let expiresAt: Timestamp?
    /// `nil` only when absent; an unreadable `spec` is kept so the refusal
    /// (`UnrunnableClaimError`) can say which it was.
    let spec: RawJSONValue?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case benchmarkId = "benchmark_id"
        case timeWindow = "time_window"
        case expiresAt = "expires_at"
        case spec
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        jobId = try c.decode(String.self, forKey: .jobId)
        benchmarkId = try c.decode(String.self, forKey: .benchmarkId)
        timeWindow = try c.decode(String.self, forKey: .timeWindow)
        expiresAt = c.decodeLenientTimestamp(forKey: .expiresAt)
        spec = try c.decodeIfPresent(RawJSONValue.self, forKey: .spec)
    }
}

/// `POST /benchmarks` failure body for a plan-attached job (the mgmt repo's
/// `docs/httpapi.md` §2.7.2). Identity and reason only: the server holds the job
/// body, and echoing the cell back could carry a plan-supplied token.
/// `clientVersion` is sent because the job body cannot tell the server which build
/// failed.
nonisolated struct FailureSubmission: Encodable, Sendable, Equatable {
    let messageType = "failure"
    let jobId: String
    let benchmarkId: String
    let failureReason: String
    let retriable: Bool
    /// Left out when `nil`: mgmt refuses a `null`.
    var failureCode: FailureCode?
    let clientVersion: String
    /// Milliseconds from the claim to the failure being ready to submit, on a monotonic
    /// clock, excluding the submit and its retries. `nil` unless a caller timed the job.
    var deviceTimeMs: UInt64?

    enum CodingKeys: String, CodingKey {
        case messageType = "message_type"
        case jobId = "job_id"
        case benchmarkId = "benchmark_id"
        case failureReason = "failure_reason"
        case retriable
        case failureCode = "failure_code"
        case clientVersion = "client_version"
        case deviceTimeMs = "device_time_ms"
    }

    /// Reads only the envelope, so a job whose spec never parsed still reports.
    @MainActor static func fromClaim(
        _ job: ClaimedJob,
        failureReason: String,
        retriable: Bool
    ) -> FailureSubmission {
        fromClaim(
            job, failureReason: failureReason, retriable: retriable,
            clientVersion: Bundle.main.appVersionDisplayString)
    }

    static func fromClaim(
        _ job: ClaimedJob,
        failureReason: String,
        retriable: Bool,
        clientVersion: String
    ) -> FailureSubmission {
        FailureSubmission(
            jobId: job.jobId,
            benchmarkId: job.benchmarkId,
            failureReason: failureReason,
            retriable: retriable,
            clientVersion: clientVersion
        )
    }
}

/// mgmt's grouping of a failure (the mgmt repo's `docs/httpapi.md`, `failure_code`): the
/// crate's `pipette_plan_types::FailureCode` less `unspecified`, which a body says by
/// leaving the key out.
nonisolated enum FailureCode: String, Codable, Sendable, Equatable {
    case outOfMemory = "out_of_memory"
    case attemptTimeout = "attempt_timeout"
    case runtimeError = "runtime_error"
    case deviceError = "device_error"
    case harnessError = "harness_error"
}

// MARK: - Benchmarks

nonisolated extension PipetteMgmtClient {
    /// One entry of `GET /benchmarks`, no eval samples. The crate keeps unknown
    /// keys in a `#[serde(flatten)]` bag; Swift has no flatten, so ``rawJson``
    /// carries the object's own bytes and an unknown parameter or type survives.
    struct BenchmarkSummary: Decodable, Sendable, Equatable {
        let benchmarkId: String
        let benchmarkType: String
        /// Keys sorted.
        let rawJson: Data

        enum CodingKeys: String, CodingKey {
            case benchmarkId = "benchmark_id"
            case benchmarkType = "benchmark_type"
        }

        init(from decoder: any Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            benchmarkId = try c.decode(String.self, forKey: .benchmarkId)
            benchmarkType = try c.decode(String.self, forKey: .benchmarkType)
            rawJson = try RawJSONValue(from: decoder).data
        }
    }
}

/// `GET /benchmarks/{benchmark_id}`, loose so an unknown type or parameter never
/// breaks sync. The strict view is `BenchmarkDefinition`, decoded from ``rawJson``.
nonisolated struct RemoteBenchmark: Decodable, Sendable, Equatable {
    let benchmarkId: String
    let benchmarkType: String
    /// Keys sorted, samples and unknown parameters included.
    let rawJson: Data

    enum CodingKeys: String, CodingKey {
        case benchmarkId = "benchmark_id"
        case benchmarkType = "benchmark_type"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        benchmarkId = try c.decode(String.self, forKey: .benchmarkId)
        benchmarkType = try c.decode(String.self, forKey: .benchmarkType)
        rawJson = try RawJSONValue(from: decoder).data
    }
}

// MARK: - Submissions

nonisolated struct SubmitResponse: Decodable, Sendable, Equatable {
    let jobId: String

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
    }

    init(jobId: String) {
        self.jobId = jobId
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        jobId = try c.decode(String.self, forKey: .jobId)
    }
}

nonisolated struct BatchSubmitItemResponse: Decodable, Sendable, Equatable {
    let index: Int
    let jobId: String?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case index
        case jobId = "job_id"
        case error
    }

    init(index: Int, jobId: String? = nil, error: String? = nil) {
        self.index = index
        self.jobId = jobId
        self.error = error
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        index = try c.decode(Int.self, forKey: .index)
        jobId = try c.decodeIfPresent(String.self, forKey: .jobId)
        error = try c.decodeIfPresent(String.self, forKey: .error)
    }
}

nonisolated struct BatchSubmitResponse: Decodable, Sendable, Equatable {
    let results: [BatchSubmitItemResponse]

    init(results: [BatchSubmitItemResponse]) {
        self.results = results
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        results = try c.decodeIfPresent([BatchSubmitItemResponse].self, forKey: .results) ?? []
    }

    enum CodingKeys: String, CodingKey {
        case results
    }
}

// MARK: - Vault

/// The digest is a pin, not an observation: mgmt never reads the object, so a download
/// that hashes to something else means the object was replaced after the record was made.
nonisolated struct VaultResource: Decodable, Sendable, Equatable {
    let resourceId: VaultResourceId
    let name: String
    let sha256: Sha256
    let bytes: UInt64

    enum CodingKeys: String, CodingKey {
        case resourceId = "resource_id"
        case name, sha256, bytes
    }
}

// MARK: - Jobs

nonisolated struct BenchmarkJobMetric: Decodable, Sendable, Equatable {
    let metric: String
    let value: Double
    let unit: String
}

/// `GET /jobs/{job_id}`
nonisolated struct JobResponse: Decodable, Sendable, Equatable {
    let status: PipetteMgmtClient.JobStatus
    /// Present once `status` is `processed`. Lenient: polled on every sync, and
    /// nothing branches on it.
    let scoredAt: Timestamp?
    let metrics: [BenchmarkJobMetric]?

    enum CodingKeys: String, CodingKey {
        case status
        case scoredAt = "scored_at"
        case metrics
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        status = try c.decode(PipetteMgmtClient.JobStatus.self, forKey: .status)
        scoredAt = c.decodeLenientTimestamp(forKey: .scoredAt)
        metrics = try c.decodeIfPresent([BenchmarkJobMetric].self, forKey: .metrics)
    }
}

// MARK: - Untyped JSON

/// Any JSON value held as its own bytes: `Sendable` and `Equatable`, which
/// `[String: Any]` is not. Accepts non-objects, so an unreadable `spec` cannot lose
/// the whole claim (see ``ClaimedJob``).
nonisolated struct RawJSONValue: Decodable, Sendable, Equatable {
    /// Sorted keys, so equality and logging do not depend on the server's order.
    let data: Data

    init(from decoder: Decoder) throws {
        data = try JSONSerialization.data(
            withJSONObject: JSONFragment(from: decoder).value,
            options: [.sortedKeys, .fragmentsAllowed]
        )
    }

    var object: [String: Any]? {
        let value = try? JSONSerialization.jsonObject(
            with: data,
            options: [.fragmentsAllowed]
        )
        return value as? [String: Any]
    }

    /// Every `auth_token` replaced: a plan carries a gated repo's token inside the
    /// model spec, and it must not reach a log or a submission.
    var redactedDescription: String {
        guard let value = try? JSONSerialization.jsonObject(
            with: data,
            options: [.fragmentsAllowed]
        ), let bytes = try? JSONSerialization.data(
            withJSONObject: Self.redact(value),
            options: [.sortedKeys, .fragmentsAllowed]
        ) else { return "<unprintable>" }
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func redact(_ value: Any) -> Any {
        if let object = value as? [String: Any] {
            return object.reduce(into: [String: Any]()) { acc, entry in
                acc[entry.key] = entry.key == "auth_token"
                    ? "<redacted>"
                    : redact(entry.value)
            }
        }
        if let array = value as? [Any] {
            return array.map(redact)
        }
        return value
    }
}

/// The bridge from `Decodable` to `JSONSerialization`, which the claim consumers speak.
private nonisolated struct JSONFragment: Decodable {
    let value: Any

    init(from decoder: Decoder) throws {
        if let container = try? decoder.container(keyedBy: AnyKey.self) {
            value = try Self.decodeObject(from: container)
        } else if var container = try? decoder.unkeyedContainer() {
            value = try Self.decodeArray(from: &container)
        } else {
            // Kept rather than thrown: the code that knows the schema decides
            // what an unreadable payload means.
            value = Self.decodeScalar(from: try decoder.singleValueContainer())
        }
    }

    private static func decodeScalar(from container: SingleValueDecodingContainer) -> Any {
        if container.decodeNil() { return NSNull() }
        if let value = try? container.decode(Bool.self) { return value }
        if let value = try? container.decode(Int.self) { return value }
        if let value = try? container.decode(Double.self) { return value }
        if let value = try? container.decode(String.self) { return value }
        return NSNull()
    }

    private struct AnyKey: CodingKey {
        let stringValue: String
        let intValue: Int? = nil
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue _: Int) { nil }
    }

    private static func decodeObject(
        from container: KeyedDecodingContainer<AnyKey>
    ) throws -> [String: Any] {
        try container.allKeys.reduce(into: [String: Any]()) { acc, key in
            acc[key.stringValue] = try decodeValue(from: container, key: key)
        }
    }

    private static func decodeValue(
        from container: KeyedDecodingContainer<AnyKey>,
        key: AnyKey
    ) throws -> Any {
        if let value = try? container.decode(Bool.self, forKey: key) { return value }
        if let value = try? container.decode(Int.self, forKey: key) { return value }
        if let value = try? container.decode(Double.self, forKey: key) { return value }
        if let value = try? container.decode(String.self, forKey: key) { return value }
        if let nested = try? container.nestedContainer(keyedBy: AnyKey.self, forKey: key) {
            return try decodeObject(from: nested)
        }
        if var nested = try? container.nestedUnkeyedContainer(forKey: key) {
            return try decodeArray(from: &nested)
        }
        return NSNull()
    }

    private static func decodeArray(
        from container: inout UnkeyedDecodingContainer
    ) throws -> [Any] {
        var items: [Any] = []
        while !container.isAtEnd {
            if let value = try? container.decode(Bool.self) {
                items.append(value)
            } else if let value = try? container.decode(Int.self) {
                items.append(value)
            } else if let value = try? container.decode(Double.self) {
                items.append(value)
            } else if let value = try? container.decode(String.self) {
                items.append(value)
            } else if let nested = try? container.nestedContainer(keyedBy: AnyKey.self) {
                items.append(try decodeObject(from: nested))
            } else if var nested = try? container.nestedUnkeyedContainer() {
                items.append(try decodeArray(from: &nested))
            } else {
                _ = try container.decode(AnyNull.self)
                items.append(NSNull())
            }
        }
        return items
    }

    private struct AnyNull: Decodable {}
}
