import Foundation

// Wire DTOs for the planner / clients/me surface on ManagementClient.
// CodingKeys are explicit snake_case (same convention as RegisterRequest).

/// A job leased via `POST /plans/claim`: the server-owned envelope wrapped
/// around the work payload the server never interprets. Mirrors pipette-mgmt
/// httpapi §2.9 and the Rust `ClaimedJob` in `pipette-mgmt-client`.
///
/// The envelope is the subset the server acts on: identity, lease, expiry. The
/// cell lives in `spec`, carried as raw bytes and typed in one place
/// (`ClientRunSpec.runSpec(from:)`).
///
/// `spec` is kept opaque for the same reason the Rust client does: a payload
/// this build cannot understand must not fail the decode of the envelope around
/// it, because the envelope is what identifies the leased job, and reporting
/// that job is the only way it stops being re-served.
///
/// The `model_*` / `runtime_*` grouping labels a body may still carry are the
/// server's own bookkeeping; they are ignored here, as is any future addition.
struct ClaimedJob: Decodable, Sendable, Equatable {
    let jobId: String
    let benchmarkId: String
    /// ISO 8601 duration (e.g. `"PT10M"`). Heartbeat at half this by default.
    let timeWindow: String
    let expiresAt: String?
    /// The cell to run, as it arrived. `nil` only when the body carried no
    /// `spec` at all; a `spec` that is present but unreadable is kept, so the
    /// refusal names which of the two it was (`UnrunnableClaim`).
    let spec: RawJSONValue?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case benchmarkId = "benchmark_id"
        case timeWindow = "time_window"
        case expiresAt = "expires_at"
        case spec
    }
}

/// Any JSON value held as its own bytes, decoded on demand — the counterpart of
/// the Rust claim's `serde_json::Value` payload.
///
/// `Sendable` and `Equatable` (which `[String: Any]` is not) so it can sit on a
/// `ClaimedJob` crossing actor boundaries, while still deferring interpretation
/// to the code that knows the schema.
///
/// Deliberately accepts values that are *not* objects. A `spec` this build
/// cannot make sense of has to survive decoding: the envelope around it names
/// the leased job, and reporting that job is the only way it stops being
/// re-served. Throwing here would lose the whole claim, not just the payload.
///
/// `nonisolated` so the claim loop can read and log a payload off the main
/// actor, as `DeviceProfileUpdate` below is for the same reason.
nonisolated struct RawJSONValue: Decodable, Sendable, Equatable {
    /// Canonical bytes, re-encoded with sorted keys so equality and logging are
    /// stable regardless of the order the server sent.
    let data: Data

    init(from decoder: Decoder) throws {
        data = try JSONSerialization.data(
            withJSONObject: JSONFragment(from: decoder).value,
            options: [.sortedKeys, .fragmentsAllowed]
        )
    }

    /// The object, or `nil` when the payload was not a JSON object.
    var object: [String: Any]? {
        let value = try? JSONSerialization.jsonObject(
            with: data,
            options: [.fragmentsAllowed]
        )
        return value as? [String: Any]
    }

    /// The payload as text, with every `auth_token` replaced — a plan carries
    /// the token for a gated repo inside the model spec, and it must not reach
    /// a log or a submission.
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

/// Decodes any JSON value without interpreting it — the bridge from `Decodable`
/// to `JSONSerialization`, which is what the claim consumers already speak.
private nonisolated struct JSONFragment: Decodable {
    let value: Any

    init(from decoder: Decoder) throws {
        if let container = try? decoder.container(keyedBy: AnyKey.self) {
            value = try Self.decodeObject(from: container)
        } else if var container = try? decoder.unkeyedContainer() {
            value = try Self.decodeArray(from: &container)
        } else {
            // A scalar, or a value this decoder cannot name. Kept rather than
            // thrown: the disposition of an unreadable payload is decided by the
            // code that knows the schema, not by losing the claim here.
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

/// `GET/PATCH /clients/me` profile — forward-compatible with older servers.
struct ClientProfile: Decodable, Sendable, Equatable {
    let clientId: String
    let organization: String
    let clientDetails: String
    let contactEmail: String
    let status: String
    let tags: [String]
    let reindexPending: Bool
    let capabilities: [String]

    enum CodingKeys: String, CodingKey {
        case clientId = "client_id"
        case organization
        case clientDetails = "client_details"
        case contactEmail = "contact_email"
        case status
        case tags
        case reindexPending = "reindex_pending"
        case capabilities
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        clientId = try c.decode(String.self, forKey: .clientId)
        organization = try c.decode(String.self, forKey: .organization)
        clientDetails = try c.decode(String.self, forKey: .clientDetails)
        contactEmail = try c.decode(String.self, forKey: .contactEmail)
        status = try c.decode(String.self, forKey: .status)
        tags = try c.decodeIfPresent([String].self, forKey: .tags) ?? []
        reindexPending = try c.decodeIfPresent(Bool.self, forKey: .reindexPending) ?? false
        capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities) ?? []
    }
}

/// Device profile + capabilities for `PATCH /clients/me` / register.
///
/// `nonisolated` because the project defaults to MainActor isolation, which is
/// meaningless for an immutable-by-convention `Sendable` payload: it exists to be
/// built, encoded, and handed to a request. Without this, constructing one is
/// pinned to the MainActor for no reason (and default-argument positions, which
/// are nonisolated, can't build one at all).
nonisolated struct DeviceProfileUpdate: Encodable, Sendable, Equatable {
    var clientDetails: String?
    var deviceName: String?
    /// The enum, not a token string — a misspelling here is rejected by the server's
    /// own enum, so it is worth catching at compile time.
    var deviceFormFactor: DeviceFormFactor?
    var deviceOsName: String?
    var deviceOsVersion: String?
    var deviceChipModel: String?
    var deviceRamBytes: UInt64?
    var capabilities: [String]?

    enum CodingKeys: String, CodingKey {
        case clientDetails = "client_details"
        case deviceName = "device_name"
        case deviceFormFactor = "device_form_factor"
        case deviceOsName = "device_os_name"
        case deviceOsVersion = "device_os_version"
        case deviceChipModel = "device_chip_model"
        case deviceRamBytes = "device_ram_bytes"
        case capabilities
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(clientDetails, forKey: .clientDetails)
        try c.encodeIfPresent(deviceName, forKey: .deviceName)
        try c.encodeIfPresent(deviceFormFactor, forKey: .deviceFormFactor)
        try c.encodeIfPresent(deviceOsName, forKey: .deviceOsName)
        try c.encodeIfPresent(deviceOsVersion, forKey: .deviceOsVersion)
        try c.encodeIfPresent(deviceChipModel, forKey: .deviceChipModel)
        try c.encodeIfPresent(deviceRamBytes, forKey: .deviceRamBytes)
        try c.encodeIfPresent(capabilities, forKey: .capabilities)
    }
}

/// Plan-attached failure body (httpapi §2.7.2).
///
/// Identity and reason only. Every `model_*` / `runtime_*` field the endpoint
/// accepts is optional, and the server already holds the job body this `job_id`
/// names — echoing the cell back would restate what it can read, in a spelling
/// this client had to invent, and would risk carrying a plan-supplied token.
///
/// `clientVersion` is the exception that proves the rule: it is not in the job
/// body, so the server cannot recover it, and a failure is precisely when
/// "which build reported this" is the question. It defaults from the bundle so
/// no call site can omit it, and stays injectable for tests — the Rust client
/// passes its version in instead, because that wire crate is a library with no
/// bundle to read.
struct FailureSubmission: Encodable, Sendable, Equatable {
    let messageType = "failure"
    let jobId: String
    let benchmarkId: String
    let failureReason: String
    let retriable: Bool
    let clientVersion: String

    enum CodingKeys: String, CodingKey {
        case messageType = "message_type"
        case jobId = "job_id"
        case benchmarkId = "benchmark_id"
        case failureReason = "failure_reason"
        case retriable
        case clientVersion = "client_version"
    }

    static func fromClaim(
        _ job: ClaimedJob,
        reason: String,
        retriable: Bool,
        clientVersion: String = Bundle.main.appVersionDisplayString
    ) -> FailureSubmission {
        FailureSubmission(
            jobId: job.jobId,
            benchmarkId: job.benchmarkId,
            failureReason: reason,
            retriable: retriable,
            clientVersion: clientVersion
        )
    }
}
