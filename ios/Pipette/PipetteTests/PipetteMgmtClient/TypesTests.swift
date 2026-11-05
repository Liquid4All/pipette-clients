import Foundation
import Testing

@testable import Pipette

/// Mirrors the crate's `types.rs` tests.
@Suite
struct TypesTests {
    // MARK: - Statuses

    /// A newer server reaching an older client is version skew: the response
    /// must still decode, and `auth me` should print the word it was sent.
    @Test(arguments: [
        ("\"pending\"", ClientStatus.pending),
        ("\"approved\"", ClientStatus.approved),
        ("\"quarantined\"", ClientStatus.unknown("quarantined")),
    ])
    func aClientStatusDecodesOrCarriesTheUnknown(body: String, expected: ClientStatus) throws {
        let status = try JSONDecoder().decode(ClientStatus.self, from: Data(body.utf8))
        #expect(status == expected)
        #expect(String(decoding: try JSONEncoder().encode(status), as: UTF8.self) == body)
    }

    @Test(arguments: [
        ("\"incoming\"", PipetteMgmtClient.JobStatus.incoming),
        ("\"scoring\"", PipetteMgmtClient.JobStatus.scoring),
        ("\"processed\"", PipetteMgmtClient.JobStatus.processed),
        ("\"failed\"", PipetteMgmtClient.JobStatus.failed),
        ("\"requeued\"", PipetteMgmtClient.JobStatus.unknown("requeued")),
    ])
    func aJobStatusDecodesOrCarriesTheUnknown(
        body: String, expected: PipetteMgmtClient.JobStatus
    ) throws {
        let status = try JSONDecoder().decode(
            PipetteMgmtClient.JobStatus.self, from: Data(body.utf8))
        #expect(status == expected)
        #expect(String(decoding: try JSONEncoder().encode(status), as: UTF8.self) == body)
    }

    /// The planner worker gates claiming on `.approved`, so it stays closed on a
    /// word this build has never seen.
    @Test func anUnknownStatusNeverEqualsAKnownOne() {
        #expect(ClientStatus.unknown("approved-ish") != .approved)
        #expect(PipetteMgmtClient.JobStatus.unknown("processing") != .processed)
    }

    @Test func aStatusDisplaysAsTheWireWord() {
        #expect("\(ClientStatus.approved)" == "approved")
        #expect("\(PipetteMgmtClient.JobStatus.processed)" == "processed")
        #expect("\(PipetteMgmtClient.JobStatus.unknown("requeued"))" == "requeued")
    }

    // MARK: - Keys and contact email

    /// Casing and whitespace are normalized, so a key copied out of a terminal
    /// still equals the one on disk.
    @Test(arguments: [
        (String(repeating: "a3f8", count: 16), true),
        (String(repeating: "A3F8", count: 16), true),
        ("  \(String(repeating: "a3f8", count: 16))\n", true),
        (String(repeating: "a3f8", count: 8), false),
        (String(repeating: "a3f8", count: 32), false),
        (String(repeating: "zz", count: 32), false),
        ("", false),
    ])
    func publicKeyAcceptsOnly32BytesOfHex(raw: String, accepted: Bool) {
        #expect(((try? PublicKey(raw)) != nil) == accepted, "\(raw)")
    }

    @Test func publicKeyNormalizesCasingAndPadding() throws {
        let plain = try PublicKey(String(repeating: "a3f8", count: 16))
        let noisy = try PublicKey("  \(String(repeating: "A3F8", count: 16))\n")
        #expect(plain == noisy, "the same key however it was pasted")
    }

    /// It must refuse setup-form typos and accept everything that delivers; the
    /// plus and subdomain cases are what a stricter rule would wrongly reject.
    @Test(arguments: [
        ("ops@example.com", true),
        ("ops+fleet@example.com", true),
        ("ops@mail.corp.example.com", true),
        ("  ops@example.com  ", true),
        ("ops.example.com", false),
        ("ops@example@com", false),
        ("@example.com", false),
        ("ops@", false),
        ("ops@localhost", false),
        ("ops@example.com.", false),
        ("ops @example.com", false),
        ("ops@example.com,", false),
        ("", false),
    ])
    func contactEmailChecksShapeNotTheRfc(raw: String, accepted: Bool) {
        #expect(((try? ContactEmail(raw)) != nil) == accepted, "\(raw)")
    }

    // MARK: - Instants

    @Test(arguments: [
        ("2026-01-02T03:04:05Z", true),
        ("2026-01-02T03:04:05+02:00", true),
        ("2026-01-02T03:04:05.123Z", true),
        ("2026-01-02", false),
        ("2026-01-02T03:04:05", false),
        ("last tuesday", false),
        ("", false),
    ])
    func timestampAcceptsRfc3339(raw: String, accepted: Bool) {
        #expect(((try? Timestamp(raw)) != nil) == accepted, "\(raw)")
    }

    /// Nothing consults the expiry, and the envelope names the leased job, whose
    /// report is the only way it stops being re-served.
    @Test(arguments: [
        (#""expires_at":"2026-01-02T03:04:05Z","#, true),
        (#""expires_at":"tomorrow-ish","#, false),
        (#""expires_at":null,"#, false),
        ("", false),
    ])
    func aClaimSurvivesAnUnreadableExpiry(field: String, some: Bool) throws {
        let body = #"{\#(field)"job_id":"job-1","benchmark_id":"b1","time_window":"PT5M"}"#
        let job = try Coding.decoder.decode(ClaimedJob.self, from: Data(body.utf8))
        #expect(job.jobId == "job-1")
        #expect((job.expiresAt != nil) == some, "\(body)")
    }

    /// Lenient for the same reason: polled on every sync, and nothing branches
    /// on the value.
    @Test(arguments: [
        (#""2026-01-02T03:04:05Z""#, true),
        (#""whenever""#, false),
        ("null", false),
    ])
    func aJobResponseSurvivesAnUnreadableScoredAt(value: String, some: Bool) throws {
        let body = #"{"status":"processed","scored_at":\#(value)}"#
        let response = try Coding.decoder.decode(JobResponse.self, from: Data(body.utf8))
        #expect(response.status == .processed)
        #expect((response.scoredAt != nil) == some, "\(body)")
    }

    // MARK: - Conditional requests

    @Test func entityTagSerializesAsString() throws {
        let etag = try EntityTag("\"abc123\"")
        let json = String(decoding: try JSONEncoder().encode(etag), as: UTF8.self)
        #expect(json == #""\"abc123\"""#)
        #expect(try JSONDecoder().decode(EntityTag.self, from: Data(json.utf8)) == etag)
    }

    @Test func ifNoneMatchWrapsEntityTagAsHeaderValue() throws {
        #expect(IfNoneMatch(try EntityTag("\"abc123\"")).headerValue == "\"abc123\"")
    }

    // MARK: - Conditional requests

    /// What a sync state persists is what goes back out as `If-None-Match`.
    @Test func entityTagSerializesAsAString() throws {
        let etag = try EntityTag("\"abc123\"")
        let encoded = try JSONEncoder().encode(etag)
        #expect(String(decoding: encoded, as: UTF8.self) == #""\"abc123\"""#)
        #expect(try JSONDecoder().decode(EntityTag.self, from: encoded) == etag)
    }

    /// No server could have sent such a tag, so it is refused rather than carried.
    @Test func entityTagRefusesWhatIsNotAHeaderValue() {
        #expect((try? EntityTag("\"v1\"")) != nil)
        #expect((try? EntityTag("\"v1\u{0}\"")) == nil)
        #expect(EntityTag.fromHeaderValue("\"v1\n\"") == nil)
    }

    @Test func ifNoneMatchWrapsTheEntityTagAsAHeaderValue() throws {
        #expect(IfNoneMatch(try EntityTag("\"abc123\"")).headerValue == "\"abc123\"")
    }

    // MARK: - Profiles

    /// The worker waits on `reindexPending` before claiming, so a missing field
    /// must be `false`, not a decode failure.
    @Test func clientProfileDefaultsWhatAnOlderServerOmits() throws {
        let bare = """
            {"client_id":"ev1_abc","organization":"o","client_details":"d",
             "contact_email":"a@b.com","status":"approved"}
            """
        let profile = try Coding.decoder.decode(ClientProfile.self, from: Data(bare.utf8))
        #expect(profile.tags.isEmpty)
        #expect(!profile.reindexPending)
        #expect(profile.capabilities.isEmpty)
    }

    @Test func clientProfileReadsTheFieldsAServerDoesSend() throws {
        let full = """
            {"client_id":"ev1_abc","organization":"o","client_details":"d",
             "contact_email":"a@b.com","status":"approved",
             "tags":["team/mobile/ios","batch/2026-q3"],"reindex_pending":true,
             "capabilities":["runtime:llama_cpp"],"future_field":1}
            """
        let profile = try Coding.decoder.decode(ClientProfile.self, from: Data(full.utf8))
        #expect(profile.tags == ["team/mobile/ios", "batch/2026-q3"])
        #expect(profile.reindexPending)
        #expect(profile.capabilities == ["runtime:llama_cpp"])
    }

    // MARK: - Claims

    /// The envelope identifies the leased job, and reporting it is the only way it
    /// stops being re-served.
    @Test func claimedJobToleratesServerLabelsAndUnknownFields() throws {
        let body = """
            {"job_id":"job-1","benchmark_id":"prefill_throughput_256",
             "time_window":"PT10M","model_name":"m","future_field":42,
             "spec":{"benchmark":"prefill_throughput_256","model":"nope"}}
            """
        let job = try Coding.decoder.decode(ClaimedJob.self, from: Data(body.utf8))

        #expect(job.jobId == "job-1")
        #expect(job.timeWindow == "PT10M")
        // Kept, not dropped, so the refusal can tell unreadable from absent.
        #expect(job.spec != nil)

        let failure = FailureSubmission.fromClaim(
            job, failureReason: "[ts] bad spec", retriable: false,
            clientVersion: "9.9.9 (build t)")
        #expect(failure.jobId == "job-1")
        #expect(failure.benchmarkId == "prefill_throughput_256")
        #expect(!failure.retriable)
        #expect(failure.clientVersion == "9.9.9 (build t)")
    }

    // MARK: - Registration

    private func encodedFields(
        preauthKey: String? = nil,
        device: DeviceProfileFields = DeviceProfileFields()
    ) throws -> [String: Any] {
        let request = RegisterRequest(
            publicKey: try PublicKey(String(repeating: "a3f8", count: 16)),
            organization: "LiquidAI",
            contactEmail: try ContactEmail("user@example.com"),
            clientDetails: "iPhone",
            clientPreauthKey: try preauthKey.map { try ClientPreauthKey($0) },
            device: device
        )
        return try #require(
            JSONSerialization.jsonObject(with: try JSONEncoder().encode(request)) as? [String: Any]
        )
    }

    /// Dropped entirely, not sent as `null`.
    @Test func registerRequestOmitsPreauthKeyWhenNil() throws {
        let json = try encodedFields()
        #expect(json["client_preauth_key"] == nil)
        #expect(json["public_key"] as? String == String(repeating: "a3f8", count: 16))
        #expect(json["client_details"] as? String == "iPhone")
    }

    /// The server reads `preauth_key` as an alias, but both names in one body
    /// are a `400`.
    @Test func registerRequestSendsThePreauthKeyUnderTheNameTheServerAccepts() throws {
        let json = try encodedFields(preauthKey: "client_preauth_abc.secret")
        #expect(json["client_preauth_key"] as? String == "client_preauth_abc.secret")
        #expect(json["preauth_key"] == nil)
    }

    @Test func registerRequestOmitsDeviceProfileWhenEmpty() throws {
        let json = try encodedFields()
        #expect(json["device_name"] == nil)
        #expect(json["capabilities"] == nil)
        #expect(json.count == 4)
    }

    /// The shape the server accepts, as the crate's `#[serde(flatten)]` produces.
    @Test func registerRequestFlattensDeviceProfile() throws {
        let json = try encodedFields(
            device: DeviceProfileFields(
                deviceName: "iPhone 16 Pro",
                deviceOsName: "iOS",
                deviceOsVersion: "26.0",
                capabilities: ["runtime:llama_cpp"]
            )
        )
        #expect(json["device"] == nil)
        #expect(json["device_name"] as? String == "iPhone 16 Pro")
        #expect(json["device_os_name"] as? String == "iOS")
        #expect(json["capabilities"] as? [String] == ["runtime:llama_cpp"])
        #expect(json["public_key"] as? String == String(repeating: "a3f8", count: 16))
    }

    @Test func registerRequestKeepsTheCallersClientDetails() throws {
        let json = try encodedFields(device: profileUpdate(device: deviceInfoFixture(), capabilities: []).device)
        #expect(json["client_details"] as? String == "iPhone")
    }

    // MARK: - Credentials

    @Test func preauthKeyTrimAndRedaction() throws {
        let key = try ClientPreauthKey("  preauth_id.secret\n")
        #expect(key.value == "preauth_id.secret", "surrounding whitespace trimmed")
        #expect(String(reflecting: key) == "ClientPreauthKey(<redacted>)",
                "a dump must not leak the key")
        #expect("\(key)" == "<redacted>",
                "interpolation must not leak it either, so no caller reaches for .value")

        for empty in ["", "   ", "\n\t"] {
            #expect((try? ClientPreauthKey(empty)) == nil,
                    "\(empty.debugDescription) should be rejected as empty")
        }
    }

    @Test func preauthKeyFromStrMessage() {
        #expect(EmptyClientPreauthKeyError().errorDescription == "pre-auth key must not be empty")
    }

    /// Any 32 bytes are a valid Ed25519 seed.
    private static let privateKeyHex =
        "aa11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899"

    @Test func privateKeyNormalizationAndRedaction() throws {
        let key = try PrivateKey("  \(Self.privateKeyHex)\n")
        #expect(key.value == Self.privateKeyHex, "surrounding whitespace trimmed")
        #expect(try PrivateKey(Self.privateKeyHex.uppercased()) == key,
                "hex is case-insensitive, so the same key compares equal however it was written")
        #expect(String(reflecting: key) == "PrivateKey(<redacted>)",
                "a dump must not leak the key a `ClientIdentity` would carry")
        #expect("\(key)" == "<redacted>", "interpolation must not leak it either")
    }

    /// Holding a `PrivateKey` must mean it can sign; either check alone (hex,
    /// length) admits one that cannot.
    @Test(arguments: [
        "zz11bb22cc33dd44ee55ff6600778899aa11bb22cc33dd44ee55ff6600778899x",
        "aab",
        "aabb",
        String(repeating: "aa11bb22cc33dd44ee55ff6600778899", count: 4),
        "",
    ])
    func privateKeyRefusesWhatCannotSign(raw: String) {
        #expect((try? PrivateKey(raw)) == nil, "\(raw.debugDescription) should be refused")
    }

    // MARK: - The server

    /// URL assembly is just `"\(server)\(path)"`, so no trailing slash, and the
    /// path prefix survives because the signature covers it.
    @Test(arguments: [
        ("https://mgmt.example.com", "https://mgmt.example.com"),
        ("https://mgmt.example.com/", "https://mgmt.example.com"),
        ("  https://mgmt.example.com/\n", "https://mgmt.example.com"),
        ("https://mgmt.example.com/api/", "https://mgmt.example.com/api"),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
    ])
    func serverUrlNormalizesToAJoinableBase(raw: String, expected: String) throws {
        let server = try ServerUrl(raw)
        #expect(server.value == expected)
        #expect("\(server)/clients/me" == "\(expected)/clients/me")
    }

    /// As `url::Url::parse` does, so a record written with either spelling
    /// still reads as the same server.
    @Test func serverUrlLowercasesTheSchemeAndHost() throws {
        #expect(try ServerUrl("HTTPS://Collector.Example.COM").value
            == "https://collector.example.com")
    }

    /// A query or fragment, appended to, sends every request to `/` with no
    /// error to say why. Messages name the URL, for operators running several
    /// deployments. The scheme is checked first, so `mailto:` and `file:` are
    /// refused for their scheme.
    @Test(arguments: [
        ("mgmt.example.com", "`mgmt.example.com`: relative URL without a base"),
        ("https://mgmt.example.com?tenant=a",
         "`https://mgmt.example.com?tenant=a` must not carry a query or fragment"),
        ("https://mgmt.example.com#section",
         "`https://mgmt.example.com#section` must not carry a query or fragment"),
        ("mailto:ops@example.com", "`mailto:ops@example.com` must use http or https"),
        ("file:///tmp/mgmt", "`file:///tmp/mgmt` must use http or https"),
    ])
    func serverUrlRejectsWhatIsNotAJoinableHttpBase(raw: String, expected: String) {
        do {
            _ = try ServerUrl(raw)
            Issue.record("expected a rejection of \(raw)")
        } catch {
            let message = error.errorDescription ?? ""
            #expect(message.contains(expected), "unexpected rejection: \(message)")
        }
    }

    // MARK: - Clients

    @Test func updateRequestCarriesClientDetailsBesideTheFlattenedProfile() throws {
        let request = UpdateClientRequest(
            clientDetails: "iPhone",
            device: DeviceProfileFields(deviceName: "iPhone 16 Pro")
        )
        let json = try #require(
            JSONSerialization.jsonObject(with: try JSONEncoder().encode(request)) as? [String: Any]
        )
        #expect(json["client_details"] as? String == "iPhone")
        #expect(json["device_name"] as? String == "iPhone 16 Pro")
        #expect(json.count == 2)
    }

    /// Fields an older server omits default, so `auth me` works against it.
    @Test(arguments: [
        (
            #"{"client_id":"ev1_abc","organization":"o","client_details":"d","contact_email":"a@b.com","status":"approved"}"#,
            [String](), false, [String]()
        ),
        (
            #"{"client_id":"ev1_abc","organization":"o","client_details":"d","contact_email":"a@b.com","status":"approved","tags":["team/mobile/ios","batch/2026-q3"],"reindex_pending":true,"capabilities":["runtime:llama_cpp"]}"#,
            ["team/mobile/ios", "batch/2026-q3"], true, ["runtime:llama_cpp"]
        ),
    ])
    func clientProfileForwardCompat(
        body: String, expectedTags: [String], expectedReindex: Bool, expectedCaps: [String]
    ) throws {
        let profile = try Coding.decoder.decode(ClientProfile.self, from: Data(body.utf8))
        #expect(profile.tags == expectedTags)
        #expect(profile.reindexPending == expectedReindex)
        #expect(profile.capabilities == expectedCaps)
        #expect(profile.status == .approved)
        #expect(profile.contactEmail == (try ContactEmail("a@b.com")))
    }

    // MARK: - Plans

    /// Without a job id there is nothing to fail against. The unmodeled server
    /// label and unknown field must not fail the decode either.
    @Test(arguments: [true, false])
    func anUnreadableSpecStillYieldsAReportableJob(specPresent: Bool) throws {
        var body: [String: Any] = [
            "job_id": "job-1",
            "benchmark_id": "prefill_throughput_256",
            "time_window": "PT10M",
            "model_name": "m",
            "future_field": 42,
        ]
        if specPresent {
            body["spec"] = ["benchmark": "prefill_throughput_256", "model": "nope"]
        }
        let job = try Coding.decoder.decode(
            ClaimedJob.self, from: try JSONSerialization.data(withJSONObject: body))
        #expect(job.jobId == "job-1")
        #expect((job.spec != nil) == specPresent)

        let failure = FailureSubmission.fromClaim(
            job, failureReason: "[ts] bad spec", retriable: false,
            clientVersion: "9.9.9 (build t)")
        let wire = try #require(
            JSONSerialization.jsonObject(with: try JSONEncoder().encode(failure)) as? [String: Any])
        #expect(wire["message_type"] as? String == "failure")
        #expect(wire["job_id"] as? String == "job-1")
        #expect(wire["benchmark_id"] as? String == "prefill_throughput_256")
        #expect(wire["failure_reason"] as? String == "[ts] bad spec")
        #expect(wire["retriable"] as? Bool == false)
        // The server cannot recover which build reported this from the job body.
        #expect(wire["client_version"] as? String == "9.9.9 (build t)")
        #expect(wire["device_time_ms"] == nil)
    }

    // MARK: - Responses

    @Test func batchSubmitResponseDefaultsToNoResults() throws {
        let response = try Coding.decoder.decode(
            BatchSubmitResponse.self, from: Data("{}".utf8))
        #expect(response.results.isEmpty)
    }

    /// Unknown parameters included, because sync persists the raw entry.
    @Test func aBenchmarkSummaryCarriesTheEntryItWasSent() throws {
        let body = """
            [{"benchmark_id":"p","benchmark_type":"prefill_throughput","parameter_prefill_tokens":512}]
            """
        let summaries = try Coding.decoder.decode(
            [PipetteMgmtClient.BenchmarkSummary].self, from: Data(body.utf8))
        let summary = try #require(summaries.first)
        #expect(summary.benchmarkId == "p")
        #expect(summary.benchmarkType == "prefill_throughput")
        let entry = try #require(
            JSONSerialization.jsonObject(with: summary.rawJson) as? [String: Any])
        #expect(entry["parameter_prefill_tokens"] as? Int == 512)
    }

    @Test func clientProfileDefaultsMissingOptionalFields() throws {
        let json = Data("""
        {
          "client_id": "ev1_x",
          "organization": "o",
          "client_details": "d",
          "contact_email": "a@b.com",
          "status": "approved"
        }
        """.utf8)
        let profile = try JSONDecoder().decode(ClientProfile.self, from: json)
        #expect(profile.tags.isEmpty)
        #expect(profile.reindexPending == false)
        #expect(profile.capabilities.isEmpty)
        #expect(profile.status == .approved)
    }

    @Test func updateClientRequestOmitsUnsetFields() throws {
        let update = DeviceProfileFields(
            deviceName: "iPhone 16",
            deviceFormFactor: .phone,
            deviceOsName: "iOS",
            deviceOsVersion: "18.0",
            deviceCpuArch: "arm64",
            deviceOsFamily: "iOS",
            deviceChipModel: "A18",
            deviceRamBytes: 8_000_000_000,
            capabilities: ["runtime:llama_cpp"]
        )
        let data = try JSONEncoder().encode(update)
        let obj = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(obj["device_name"] as? String == "iPhone 16")
        #expect(obj["device_cpu_arch"] as? String == "arm64")
        #expect(obj["device_os_family"] as? String == "iOS")
        #expect(obj["capabilities"] as? [String] == ["runtime:llama_cpp"])
        #expect(obj["device_gpu_model"] == nil)
    }
}
