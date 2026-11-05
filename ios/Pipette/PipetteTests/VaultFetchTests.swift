import CryptoKit
import Foundation
import Observation
import Testing

@testable import Pipette

// A vault fetch over a stubbed server, mirroring `pipette-artifacts`' `vault.rs` tests.
// Each test serves its own host through `RoutedURLProtocol`, so the suites run in parallel.

/// A stub vault: each resource's record and its bytes, at the routes mgmt serves them on.
nonisolated struct StubVault {
    /// Resource id to the bytes its record pins.
    var resources: [String: Data]
    /// A status to answer a path with instead of its route, for a refusal.
    var refusals: [String: Int] = [:]
    /// Bytes to serve on a content route in place of the pinned ones, for drift.
    var replaced: [String: Data] = [:]
    /// A status to answer a request with, by its position among the requests from 0, for a
    /// failure that a later attempt does not meet.
    var failing: [Int: Int] = [:]
    /// Request positions, from 0, whose body stops one byte short, for a body that breaks
    /// part-way on one attempt only.
    var cutting: Set<Int> = []
    /// Whole replies, by request position from 0. A reply can cut a body at a byte of the
    /// test's choice, or send a range answer that the route never gives. `failing` wins
    /// over `scripted`, and `cutting` applies only to the routed answer.
    var scripted: [Int: StubResponse] = [:]
    /// Called with each request's path before the stub answers it.
    var onRequest: (@Sendable (String) -> Void)?

    static func recordPath(_ id: String) -> String { "/vault/resources/\(id)" }
    static func contentPath(_ id: String) -> String { "/vault/resources/\(id)/content" }

    func serve() throws -> (http: HttpClient, server: ServerUrl, seen: StubRecorder) {
        try RoutedURLProtocol.serve { request, index in
            onRequest?(request.url?.path ?? "")
            if let status = failing[index] {
                return StubResponse(status: status, body: Data("unavailable".utf8), headers: [:])
            }
            if let reply = scripted[index] { return reply }
            let full = answer(request)
            guard cutting.contains(index) else { return full }
            return StubResponse(status: full.status, body: full.body.dropLast(),
                                headers: full.headers)
        }
    }

    private func answer(_ request: URLRequest) -> StubResponse {
        let path = request.url?.path ?? ""
        if let status = refusals[path] {
            return StubResponse(status: status, body: Data("refused".utf8),
                                headers: status / 100 == 3 ? ["Location": "/elsewhere"] : [:])
        }
        for (id, pinned) in resources {
            if path == Self.recordPath(id) {
                let record = #"{"resource_id":"\#(id)","name":"\#(id).gguf","sha256":"\#(sha256Hex(pinned))","bytes":\#(pinned.count),"owner":{"type":"org","org_id":"org_1"},"created_at":"2026-10-02T00:00:00Z"}"#
                return StubResponse(status: 200, body: Data(record.utf8),
                                    headers: ["Content-Type": "application/json"])
            }
            if path == Self.contentPath(id) {
                return content(replaced[id] ?? pinned, pin: "\"\(sha256Hex(pinned))\"",
                               request: request)
            }
        }
        return StubResponse(status: 404, body: Data("no route for \(path)".utf8), headers: [:])
    }

    /// The content route as mgmt serves it. The route answers `Range: bytes=N-` with a
    /// `206` from byte `N` if `If-Range` is absent or equals the pin. It answers with a
    /// `416` if `N` is at or past the end. Any other request gets the whole object.
    private func content(_ served: Data, pin: String, request: URLRequest) -> StubResponse {
        let first = request.value(forHTTPHeaderField: "Range")
            .flatMap { $0.hasPrefix("bytes=") && $0.hasSuffix("-") ? $0 : nil }
            .flatMap { UInt64($0.dropFirst("bytes=".count).dropLast()) }
        let ifRange = request.value(forHTTPHeaderField: "If-Range")
        guard let first, ifRange == nil || ifRange == pin else {
            return StubResponse(status: 200, body: served, headers: [
                "Content-Type": "application/octet-stream",
                "Content-Length": "\(served.count)",
                "ETag": pin,
            ])
        }
        guard first < served.count else {
            return StubResponse(status: 416, body: Data("past the end".utf8),
                                headers: ["Content-Range": "bytes */\(served.count)"])
        }
        let rest = Data(served.dropFirst(Int(first)))
        return StubResponse(status: 206, body: rest, headers: [
            "Content-Type": "application/octet-stream",
            "Content-Length": "\(rest.count)",
            "Content-Range": "bytes \(first)-\(served.count - 1)/\(served.count)",
            "ETag": pin,
        ])
    }
}

nonisolated func sha256Hex(_ data: Data) -> String {
    Data(SHA256.hash(data: data)).hexEncodedString()
}

nonisolated func vaultPath(_ id: String) throws -> VaultPath {
    try VaultPath(StubVault.contentPath(id))
}

nonisolated func vaultLabel(_ label: String?) throws -> CustomLabel {
    CustomLabel(try label.map { try NonEmptyTrimmedString($0) })
}

nonisolated func vaultTextSpec(
    _ id: String, pin: Sha256? = nil, label: String? = nil
) throws -> Model {
    .ggufText(GgufText(source: .vaultUrl(
        url: try vaultPath(id), sha256: pin, principalId: try VaultPrincipalId("org_1"),
        customLabel: try vaultLabel(label))))
}

nonisolated func vaultSpeculativeSpec(
    _ model: String, _ mmproj: String, draft: String?, label: String? = nil
) throws -> Model {
    .ggufVisionSpeculative(GgufVisionSpeculative(source: .vaultUrl(
        model: try vaultPath(model), modelSha256: nil, mmproj: try vaultPath(mmproj),
        mmprojSha256: nil, draft: try draft.map(vaultPath), draftSha256: nil,
        principalId: try VaultPrincipalId("org_1"), customLabel: try vaultLabel(label))))
}

nonisolated func vaultVisionSpec(
    _ model: String, _ mmproj: String, label: String? = nil
) throws -> Model {
    .ggufVision(GgufVision(source: .vaultUrl(
        model: try vaultPath(model), modelSha256: nil, mmproj: try vaultPath(mmproj),
        mmprojSha256: nil, principalId: try VaultPrincipalId("org_1"),
        customLabel: try vaultLabel(label))))
}

/// An MLX directory model held as one vault archive.
nonisolated func vaultArchiveSpec(_ id: String, label: String? = nil) throws -> Model {
    .mlx(Mlx(source: .vaultArchive(
        url: try vaultPath(id), sha256: nil, principalId: try VaultPrincipalId("org_1"),
        customLabel: try vaultLabel(label))))
}

/// The same archive as an ONNX Runtime GenAI package.
nonisolated func vaultOnnxGenaiSpec(_ id: String, label: String? = nil) throws -> Model {
    .onnxGenai(OnnxGenai(source: .vaultArchive(
        url: try vaultPath(id), sha256: nil, principalId: try VaultPrincipalId("org_1"),
        customLabel: try vaultLabel(label))))
}

/// Every vault arm this client fetches, without a label and with one.
nonisolated func vaultSpecPairs() throws -> [(plain: Model, labelled: Model)] {
    let label = "acme LFM3 q4"
    return try [
        (vaultTextSpec("vault_1"), vaultTextSpec("vault_1", label: label)),
        (vaultVisionSpec("vault_a", "vault_b"), vaultVisionSpec("vault_a", "vault_b", label: label)),
        (vaultSpeculativeSpec("vault_a", "vault_b", draft: "vault_c"),
         vaultSpeculativeSpec("vault_a", "vault_b", draft: "vault_c", label: label)),
        (vaultArchiveSpec("vault_1"), vaultArchiveSpec("vault_1", label: label)),
        (vaultOnnxGenaiSpec("vault_1"), vaultOnnxGenaiSpec("vault_1", label: label)),
    ]
}

/// The `custom_label` a result filed for `model` reports. Read off the descriptor, because
/// `==` ignores the label.
nonisolated func reportedLabel(_ model: Model) throws -> String? {
    let descriptor = Data(try SubmissionRef.model(model).utf8)
    let object = try JSONSerialization.jsonObject(with: descriptor) as? [String: Any]
    return object?["custom_label"] as? String
}

/// The record route, then the content route, as one fetch does them.
struct VaultFetchTests {
    private static let id = "vault_0195c3e2"
    private static let body = Data("abc".utf8)

    private func identity() throws -> (ClientIdentity, Curve25519.Signing.PrivateKey) {
        let key = Curve25519.Signing.PrivateKey()
        let auth = ClientIdentity(
            clientId: try ClientId("ev1_a3f8"),
            privateKeyHex: try PrivateKey(key.rawRepresentation.hexEncodedString()))
        return (auth, key)
    }

    private func scratchFile() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("VaultFetchTests-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent("\(Self.id).gguf")
    }

    private func fetch(
        _ vault: StubVault, pinned: Sha256? = nil, dest: URL
    ) async throws -> StubRecorder {
        let (http, server, seen) = try vault.serve()
        let access = VaultAccess(server: server, identity: try identity().0)
        let path = try vaultPath(Self.id)
        let record = try await VaultFetch.readRecord(
            access, http: http, path: path, pinned: pinned, jobId: nil)
        var partial: VaultPartial?
        try await VaultFetch.download(
            record, access: access, http: http, to: dest, partial: &partial, jobId: nil) { _ in }
        return seen
    }

    /// The fetch's failure, which every case here expects, and the file that it keeps for
    /// another attempt.
    private func failure(
        _ vault: StubVault, pinned: Sha256? = nil, dest: URL
    ) async throws -> (VaultFetchError, StubRecorder, VaultPartial?) {
        let (http, server, seen) = try vault.serve()
        let access = VaultAccess(server: server, identity: try identity().0)
        let path = try vaultPath(Self.id)
        var partial: VaultPartial?
        do throws(VaultFetchError) {
            let record = try await VaultFetch.readRecord(
                access, http: http, path: path, pinned: pinned, jobId: nil)
            try await VaultFetch.download(
                record, access: access, http: http, to: dest, partial: &partial,
                jobId: nil) { _ in }
        } catch {
            return (error, seen, partial)
        }
        throw VaultFetchTestError.fetchSucceeded
    }

    /// Both routes carry the `v1` device signature over their own target, which is the
    /// whole credential: the plan holds no token.
    @Test func aFetchReadsTheRecordThenStreamsAndVerifies() async throws {
        let (auth, key) = try identity()
        let (http, server, seen) = try StubVault(resources: [Self.id: Self.body]).serve()
        let access = VaultAccess(server: server, identity: auth)
        let dest = scratchFile()
        defer { try? FileManager.default.removeItem(at: dest.deletingLastPathComponent()) }

        let record = try await VaultFetch.readRecord(
            access, http: http, path: try vaultPath(Self.id), pinned: nil, jobId: nil)
        var partial: VaultPartial?
        try await VaultFetch.download(
            record, access: access, http: http, to: dest, partial: &partial, jobId: nil) { _ in }

        #expect(try Data(contentsOf: dest) == Self.body)
        #expect(partial == nil)
        #expect(seen.methodsAndPaths == [
            "GET \(StubVault.recordPath(Self.id))", "GET \(StubVault.contentPath(Self.id))",
        ])
        for request in seen.requests {
            let header = { (name: String) in request.value(forHTTPHeaderField: name) }
            #expect(header("X-Client-Id") == "ev1_a3f8")
            let timestamp = try #require(header("X-Timestamp"))
            let nonce = try #require(header("X-Nonce"))
            let signature = try #require(header("X-Signature").flatMap(Data.init(hexEncoded:)))
            let target = try #require(request.url?.path)
            let payload = "v1\nGET\n\(target)\n\(timestamp)\nev1_a3f8\n\(nonce)"
            #expect(key.publicKey.isValidSignature(signature, for: Data(payload.utf8)),
                    "the signature does not cover \(target)")
        }
    }

    /// A redirect would carry the signature to another target, so it is surfaced, and
    /// it is permanent: every device would be sent the same way.
    @Test func aRedirectIsRefusedAndNotFollowed() async throws {
        let vault = StubVault(resources: [Self.id: Self.body],
                              refusals: [StubVault.recordPath(Self.id): 308])
        let (error, seen, _) = try await failure(vault, dest: scratchFile())

        guard case let .mgmt(source) = error else {
            Issue.record("expected the client's own error, got \(error)"); return
        }
        #expect(source.httpStatusCode == 308)
        #expect(error.isPermanent)
        #expect(seen.requests.count == 1, "the redirect target must never be reached")
    }

    /// `403` and `404` on either route are refusals every device with this owner would
    /// meet. `503` is the server's to recover from, and `401` is this device's signature.
    /// Only a content failure that this process retries keeps the file for the retry.
    @Test(arguments: [
        (403, false, true), (404, false, true), (503, false, false), (401, false, false),
        (403, true, true), (404, true, true), (503, true, false), (401, true, false),
    ])
    func aRecordRefusalMapsToItsDisposition(status: Int, onContent: Bool, permanent: Bool)
        async throws {
        let path = onContent ? StubVault.contentPath(Self.id) : StubVault.recordPath(Self.id)
        let dest = scratchFile()
        defer { try? FileManager.default.removeItem(at: dest.deletingLastPathComponent()) }
        let (error, _, _) = try await failure(
            StubVault(resources: [Self.id: Self.body], refusals: [path: status]), dest: dest)

        #expect(error.isPermanent == permanent)
        if status == 403 || status == 404 {
            guard case let .refused(resourceId, refusedStatus, _) = error else {
                Issue.record("expected a refusal, got \(error)"); return
            }
            #expect(resourceId.value == Self.id)
            #expect(refusedStatus == status)
            #expect(error.localizedDescription.contains(Self.id))
            #expect(!error.localizedDescription.contains("Hugging Face"))
        }
        #expect(FileManager.default.fileExists(atPath: dest.path)
            == (onContent && error.disposition != .permanent))
    }

    @Test func aPinThatDisagreesFetchesNothing() async throws {
        let other = try Sha256(sha256Hex(Data("other".utf8)))
        let (error, seen, _) = try await failure(
            StubVault(resources: [Self.id: Self.body]), pinned: other, dest: scratchFile())

        guard case let .pinDisagrees(_, plan, record) = error else {
            Issue.record("expected pinDisagrees, got \(error)"); return
        }
        #expect(plan == other.value)
        #expect(record == sha256Hex(Self.body))
        #expect(error.isPermanent)
        #expect(seen.methodsAndPaths == ["GET \(StubVault.recordPath(Self.id))"])
    }

    @Test func aPinThatAgreesFetches() async throws {
        let dest = scratchFile()
        defer { try? FileManager.default.removeItem(at: dest.deletingLastPathComponent()) }
        _ = try await fetch(StubVault(resources: [Self.id: Self.body]),
                            pinned: try Sha256(sha256Hex(Self.body)), dest: dest)
        #expect(try Data(contentsOf: dest) == Self.body)
    }

    @Test func driftedBytesAreRefusedPermanently() async throws {
        let dest = scratchFile()
        let vault = StubVault(resources: [Self.id: Self.body],
                              replaced: [Self.id: Data("abd".utf8)])
        let (error, _, _) = try await failure(vault, dest: dest)

        guard case let .drift(_, expected, actual) = error else {
            Issue.record("expected drift, got \(error)"); return
        }
        #expect(expected == sha256Hex(Self.body))
        #expect(actual == sha256Hex(Data("abd".utf8)))
        #expect(error.isPermanent)
        #expect(!FileManager.default.fileExists(atPath: dest.path))
    }

    /// A cancel never leaves the transfer waiting on a request that never ran.
    @Test func aCancelledPullReturnsWithoutItsContentRequest() async throws {
        let (http, server, seen) = try StubVault(resources: [Self.id: Self.body]).serve()
        let access = VaultAccess(server: server, identity: try identity().0)
        let record = try await VaultFetch.readRecord(
            access, http: http, path: try vaultPath(Self.id), pinned: nil, jobId: nil)
        let dest = scratchFile()
        defer { try? FileManager.default.removeItem(at: dest.deletingLastPathComponent()) }

        let pull = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            var partial: VaultPartial?
            try await VaultFetch.download(
                record, access: access, http: http, to: dest, partial: &partial,
                jobId: nil) { _ in }
        }
        await #expect(throws: VaultFetchError.self) { try await pull.value }
        #expect(!seen.methodsAndPaths.contains("GET \(StubVault.contentPath(Self.id))"))
        #expect(!FileManager.default.fileExists(atPath: dest.path))
    }

    /// A body that ends short of the record's size, with no transport error, is this
    /// device's failure: retriable, and not drift. The file stays for the next attempt to
    /// continue.
    @Test func aShortBodyIsRetriableNotDrift() async throws {
        let dest = scratchFile()
        defer { try? FileManager.default.removeItem(at: dest.deletingLastPathComponent()) }
        let vault = StubVault(resources: [Self.id: Self.body],
                              replaced: [Self.id: Self.body.dropLast()])
        let (error, _, partial) = try await failure(vault, dest: dest)

        guard case .truncated = error else {
            Issue.record("expected a short body, got \(error)"); return
        }
        #expect(!error.isPermanent)
        #expect(error.disposition == .transient(retryAfter: nil))
        #expect(partial?.written == 2)
        #expect(try Data(contentsOf: dest) == Self.body.dropLast())
    }

    /// A 256-byte object, long enough to cut part-way.
    fileprivate static let long = Data((0...255).map { UInt8($0) })
    /// The pin of ``long`` as its strong `ETag`.
    fileprivate static let longEtag = "\"\(sha256Hex(long))\""

    private static func partialReply(
        _ body: Data, etag: String, contentRange: String
    ) -> StubResponse {
        StubResponse(status: 206, body: body,
                     headers: ["ETag": etag, "Content-Range": contentRange])
    }

    /// The `206` that continues ``long`` after its first 100 bytes.
    private static let restFrom100 = partialReply(
        Data(long.dropFirst(100)), etag: longEtag, contentRange: "bytes 100-255/256")

    /// A `200` that sends the first 100 bytes of ``long`` under its pin, then ends.
    fileprivate static let cutAfter100 = StubResponse(
        status: 200, body: Data(long.prefix(100)), headers: ["ETag": longEtag])

    /// ``long`` whole, under its pin.
    private static let wholeLong = StubResponse(status: 200, body: long,
                                                headers: ["ETag": longEtag])

    /// The position of the request that continues the cut: the record is request 0, and
    /// the cut body is request 1.
    private static let resumeRequest = 2

    private struct Fetched {
        let file: Data
        let seen: StubRecorder
        let progress: [VaultProgress]
        let sleeps: [TimeInterval]
    }

    /// Read the record of ``long``, then stream its bytes under the download policy with
    /// no waits. One `partial` carries the file across the attempts, as
    /// `DownloadCoordinator.fetchVaultFile` does. Request 1 is a body cut after 100 bytes,
    /// and `tail` answers the requests after it.
    private func fetchAfterCut(_ tail: [StubResponse]) async throws -> Fetched {
        let scripted = Dictionary(uniqueKeysWithValues: zip(1..., [Self.cutAfter100] + tail))
        let (http, server, seen) = try StubVault(resources: [Self.id: Self.long],
                                                 scripted: scripted).serve()
        let access = VaultAccess(server: server, identity: try identity().0)
        let record = try await VaultFetch.readRecord(
            access, http: http, path: try vaultPath(Self.id), pinned: nil, jobId: nil)
        let dest = scratchFile()
        defer { try? FileManager.default.removeItem(at: dest.deletingLastPathComponent()) }
        let reports = ProgressReports()
        let retries = RetryRecorder()
        var partial: VaultPartial?
        try await retrying(
            url: StubVault.contentPath(Self.id),
            attempt: { _ in
                try await VaultFetch.download(
                    record, access: access, http: http, to: dest, partial: &partial,
                    jobId: nil) { reports.append($0) }
            },
            classify: VaultFetch.disposition(of:), hooks: retries.hooks())
        return Fetched(file: try Data(contentsOf: dest), seen: seen, progress: reports.all,
                       sleeps: retries.sleeps)
    }

    /// A cut body continues from byte 100 only on a `206` for the same object; any other
    /// answer restarts from zero. The crate's
    /// `a_broken_vault_body_resumes_only_where_the_server_continues_it` has the same cases.
    @Test(arguments: [
        ("resumed", [restFrom100]),
        ("whole file sent", [wholeLong]),
        ("range refused", [.reply(416), wholeLong]),
        ("another etag", [
            partialReply(Data(long.dropFirst(100)), etag: "\"other\"",
                         contentRange: "bytes 100-255/256"),
            wholeLong,
        ]),
        ("another offset", [
            partialReply(long, etag: longEtag, contentRange: "bytes 0-255/256"), wholeLong,
        ]),
        ("another total", [
            partialReply(Data(long.dropFirst(100)), etag: longEtag,
                         contentRange: "bytes 100-255/300"),
            wholeLong,
        ]),
        ("weak etag", [
            partialReply(Data(long.dropFirst(100)), etag: "W/\(longEtag)",
                         contentRange: "bytes 100-255/256"),
            wholeLong,
        ]),
        ("no content range", [
            StubResponse(status: 206, body: Data(long.dropFirst(100)),
                         headers: ["ETag": longEtag]),
            wholeLong,
        ]),
        ("unknown total", [
            partialReply(Data(long.dropFirst(100)), etag: longEtag,
                         contentRange: "bytes 100-255/*"),
            wholeLong,
        ]),
        // The `206` passes the resume checks and does not continue the object. The resumed
        // file fails its pin, and the next attempt downloads the object from zero.
        ("resumed then fails its pin", [
            partialReply(Data(count: 156), etag: longEtag, contentRange: "bytes 100-255/256"),
            wholeLong,
        ]),
    ] as [(String, [StubResponse])])
    func aBrokenVaultBodyResumesOnlyWhereTheServerContinuesIt(
        _ name: String, tail: [StubResponse]
    ) async throws {
        let fetched = try await fetchAfterCut(tail)

        #expect(fetched.file == Self.long)
        let requests = fetched.seen.requests
        #expect(requests.count == 2 + tail.count, "\(fetched.seen.methodsAndPaths)")
        let ranges = requests.map { $0.value(forHTTPHeaderField: "Range") }
        let ifRanges = requests.map { $0.value(forHTTPHeaderField: "If-Range") }
        #expect(ranges.enumerated().allSatisfy { n, range in
            range == (n == Self.resumeRequest ? "bytes=100-" : nil)
        }, "\(ranges)")
        #expect(ifRanges.enumerated().allSatisfy { n, ifRange in
            ifRange == (n == Self.resumeRequest ? Self.longEtag : nil)
        }, "\(ifRanges)")
        #expect(fetched.progress.last?.written == 256)
        #expect(fetched.progress.allSatisfy { $0.written <= 256 })
    }

    /// A `206` that sends all of its range and stops before the end of the file is kept.
    /// The same attempt asks for the bytes after it, so the only wait is the one after the
    /// cut.
    @Test func aRangeThatStopsShortContinuesInTheSameAttempt() async throws {
        let fetched = try await fetchAfterCut([
            Self.partialReply(Data(Self.long[100..<200]), etag: Self.longEtag,
                              contentRange: "bytes 100-199/256"),
            Self.partialReply(Data(Self.long.dropFirst(200)), etag: Self.longEtag,
                              contentRange: "bytes 200-255/256"),
        ])

        #expect(fetched.file == Self.long)
        let ranges = fetched.seen.requests.map { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == [nil, nil, "bytes=100-", "bytes=200-"], "\(ranges)")
        #expect(fetched.sleeps.count == 1, "\(fetched.sleeps)")
        #expect(fetched.progress.allSatisfy { $0.restarts == 0 })
        #expect(fetched.progress.last?.written == 256)
    }

    /// A range request that fails before the server judges the range keeps the file.
    @Test func aFailedRangeRequestAsksForTheRangeAgain() async throws {
        let fetched = try await fetchAfterCut([.reply(503), Self.restFrom100])

        #expect(fetched.file == Self.long)
        let ranged = fetched.seen.requests.enumerated()
            .filter { $0.element.value(forHTTPHeaderField: "Range") == "bytes=100-" }
            .map(\.offset)
        #expect(ranged == [2, 3])
        #expect(fetched.progress.allSatisfy { $0.restarts == 0 })
    }

    /// Only a failure that can pass in this process is retried, as the crate's
    /// `a_failure_is_retried_only_when_it_can_pass` states. A `401` is retriable for the
    /// worker, but this process signs with the same clock on every attempt.
    @Test(arguments: [
        (VaultFetchError.mgmt(.transport(method: "GET", url: "/v", message: "reset")), true),
        (.mgmt(.httpStatus(method: "GET", url: "/v", status: 503, body: "")), true),
        (.mgmt(.httpStatus(method: "GET", url: "/v", status: 429, body: "")), true),
        (.truncated(dest: "m.gguf", received: 1, expected: 3), true),
        (.resumedDrift(resourceId: VaultResourceId(validated: "vault_0195c3e2"), resumedAt: 1,
                       expected: "a", actual: "b"), true),
        (.mgmt(.httpStatus(method: "GET", url: "/v", status: 401, body: "")), false),
        (.mgmt(.httpStatus(method: "GET", url: "/v", status: 400, body: "")), false),
        (.io(dest: "m.gguf", source: "disk full"), false),
        (.unregistered(url: "/v"), false),
    ])
    func aFailureIsRetriedOnlyWhenItCanPass(error: VaultFetchError, transient: Bool) {
        #expect((error.disposition == .transient(retryAfter: nil)) == transient)
    }

    /// A device with no registration has nothing to sign as, on every retry.
    @Test func anUnregisteredWorkspaceIsNamed() throws {
        let path = try vaultPath(Self.id)
        let error = #expect(throws: VaultFetchError.self) {
            try VaultAccess.of({ throw IdentityError.registrationMissing }, path: path)
        }
        #expect(error?.isPermanent == true)
    }
}

private enum VaultFetchTestError: Error { case fetchSucceeded }

/// The coordinator a stub handler cancels on, set after the stub is made.
private final class CoordinatorHolder: @unchecked Sendable {
    @MainActor var coordinator: DownloadCoordinator?
    /// The bytes that the row showed when a test looked at it.
    @MainActor var observed: [Int64] = []
    /// The bytes that the row showed before each change to `downloads`, after
    /// `recordEachChange()`.
    @MainActor var published: [Int64] = []

    /// Append the bytes of the row to `published` before each change to `downloads`. The
    /// coordinator changes `downloads` only on the main actor, so no value is missed.
    @MainActor func recordEachChange() {
        withObservationTracking {
            _ = coordinator?.downloads
        } onChange: { [self] in
            MainActor.assumeIsolated {
                if let row = coordinator?.downloads.values.first,
                   published.last != row.bytesDownloaded {
                    published.append(row.bytesDownloaded)
                }
                recordEachChange()
            }
        }
    }
}

/// A vault model fetched through `ensureModel`: sized from its records, streamed, and
/// installed into one store entry.
@MainActor
struct VaultEnsureTests {
    private func coordinator(
        _ ws: PipetteWorkspace, http: HttpClient, server: ServerUrl,
        retries: RetryRecorder? = nil
    ) throws -> DownloadCoordinator {
        let retries = retries ?? RetryRecorder()
        let auth = try testIdentity()
        let session = MgmtSession(
            registration: IdentityRegistration(
                clientId: auth.clientId, status: "approved", serverUrl: server,
                organization: "test", contactEmail: "test@example.com",
                registeredAt: "2026-10-02T00:00:00Z"),
            auth: auth)
        return DownloadCoordinator(
            ws: ws, downloadHttpClient: UnreachableURLProtocol.httpClient,
            foregroundSession: true, reachability: FixedReachability(isConnected: true),
            vaultHttpClient: http, vaultSession: { session },
            retryHooks: retries.hooks())
    }

    @Test func aVaultTextModelInstallsAndBinds() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let weights = Data("weights".utf8)
        let (http, server, seen) = try StubVault(resources: [id: weights]).serve()
        let spec = try vaultTextSpec(id, pin: try Sha256(sha256Hex(weights)))
        let coordinator = try coordinator(ws, http: http, server: server)

        let bound = try await ensureModel(
            spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))

        let paths = try #require(bound.boundPaths)
        #expect(URL(fileURLWithPath: paths.payload).lastPathComponent == "\(id).gguf")
        #expect(try Data(contentsOf: URL(fileURLWithPath: paths.payload)) == weights)
        // Sized by the record at admission, then read again before the bytes move.
        #expect(seen.methodsAndPaths == [
            "GET \(StubVault.recordPath(id))", "GET \(StubVault.recordPath(id))",
            "GET \(StubVault.contentPath(id))",
        ])
        #expect(!coordinator.hasTransfer(for: spec))
    }

    @Test func aVaultVisionModelFetchesBothFilesIntoOneEntry() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let (model, mmproj) = ("vault_model", "vault_mmproj")
        let resources = [model: Data("weights".utf8), mmproj: Data("projector".utf8)]
        let (http, server, _) = try StubVault(resources: resources).serve()
        let spec = try vaultVisionSpec(model, mmproj)
        let coordinator = try coordinator(ws, http: http, server: server)

        let bound = try await ensureModel(
            spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))

        let paths = try #require(bound.boundPaths)
        let mmprojPath = try #require(paths.mmproj)
        #expect(try Data(contentsOf: URL(fileURLWithPath: paths.payload)) == resources[model])
        #expect(try Data(contentsOf: URL(fileURLWithPath: mmprojPath)) == resources[mmproj])
        #expect(URL(fileURLWithPath: paths.payload).deletingLastPathComponent()
            == URL(fileURLWithPath: mmprojPath).deletingLastPathComponent())
    }

    /// A failed fetch leaves its row behind. The next `ensureModel` retires that row and
    /// fetches again, so a failure that the worker can retry is not reported a second
    /// time. A `401` fails the fetch at once: this process does not retry it.
    @Test func aFetchAfterARetriableFailureFetchesAgain() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let weights = Data("weights".utf8)
        // Request 2 is the first content request: two record reads come before it.
        let vault = StubVault(resources: [id: weights], failing: [2: 401])
        let (http, server, seen) = try vault.serve()
        let spec = try vaultTextSpec(id)
        let coordinator = try coordinator(ws, http: http, server: server)

        do {
            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))
            Issue.record("a fetch that met a 401 bound a model")
        } catch let error as VaultFetchError {
            #expect(classifyRunError(error))
        }
        #expect(coordinator.failureReason(for: spec) != nil)

        let bound = try await ensureModel(
            spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))

        let paths = try #require(bound.boundPaths)
        #expect(try Data(contentsOf: URL(fileURLWithPath: paths.payload)) == weights)
        #expect(seen.methodsAndPaths.count == 6)
        #expect(!coordinator.hasTransfer(for: spec))
    }

    /// One transient failure, then the bytes: the fetch retries within the one call and
    /// installs the model. Request 0 is the record read at admission, request 1 the record
    /// read of the pull, and request 2 the content.
    @Test(arguments: [
        ([0: 503], []),
        ([1: 503], []),
        ([2: 503], []),
        ([2: 429], []),
        ([:], [2]),
    ] as [([Int: Int], Set<Int>)])
    func aTransientFailureRetriesTheWholeFetch(
        failing: [Int: Int], cutting: Set<Int>
    ) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let weights = Data("weights".utf8)
        let vault = StubVault(resources: [id: weights], failing: failing, cutting: cutting)
        let (http, server, seen) = try vault.serve()
        let spec = try vaultTextSpec(id)
        let retries = RetryRecorder()
        let coordinator = try coordinator(ws, http: http, server: server, retries: retries)

        let bound = try await ensureModel(
            spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))

        let paths = try #require(bound.boundPaths)
        #expect(try Data(contentsOf: URL(fileURLWithPath: paths.payload)) == weights)
        #expect(seen.methodsAndPaths.count == 4)
        #expect(retries.sleeps == [1])
        let url = failing.keys.contains { $0 < 2 }
            ? StubVault.recordPath(id) : StubVault.contentPath(id)
        #expect(retries.events.count == 2)
        #expect(retries.events.last == .recovered(url: url, attempts: 2))
    }

    /// A body cut part-way continues on the retry with `Range`, and the model installs.
    /// The row keeps the bytes that it already counted: the retry does not move the
    /// progress back to the start of the file. Request 2 is the cut content, and request 3
    /// continues it.
    @Test func aBrokenBodyResumesAndTheRowKeepsItsProgress() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let weights = VaultFetchTests.long
        let holder = CoordinatorHolder()
        let cut = VaultFetchTests.cutAfter100
        let vault = StubVault(resources: [id: weights], scripted: [2: cut], onRequest: { _ in
            DispatchQueue.main.sync {
                MainActor.assumeIsolated {
                    guard let row = holder.coordinator?.downloads.values.first else { return }
                    holder.observed.append(row.bytesDownloaded)
                }
            }
        })
        let (http, server, seen) = try vault.serve()
        let spec = try vaultTextSpec(id)
        let coordinator = try coordinator(ws, http: http, server: server)
        holder.coordinator = coordinator

        let bound = try await ensureModel(
            spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))

        let paths = try #require(bound.boundPaths)
        #expect(try Data(contentsOf: URL(fileURLWithPath: paths.payload)) == weights)
        let ranges = seen.requests.map { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == [nil, nil, nil, "bytes=100-"])
        #expect(holder.observed.last == 100, "\(holder.observed)")
    }

    /// A body cut part-way asks for the rest, and the server refuses the range with a
    /// `416`. The file starts again from zero, and the row's progress goes back to zero
    /// before it climbs to the size of the file. Request 2 is the cut content, request 3
    /// is the refused range, and request 4 asks for the whole object.
    @Test func aRefusedRangeMovesTheRowBackToZero() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let weights = VaultFetchTests.long
        let holder = CoordinatorHolder()
        let cut = VaultFetchTests.cutAfter100
        let vault = StubVault(resources: [id: weights], scripted: [2: cut, 3: .reply(416)])
        let (http, server, seen) = try vault.serve()
        let spec = try vaultTextSpec(id)
        let coordinator = try coordinator(ws, http: http, server: server)
        holder.coordinator = coordinator
        holder.recordEachChange()

        let bound = try await ensureModel(
            spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))

        let paths = try #require(bound.boundPaths)
        #expect(try Data(contentsOf: URL(fileURLWithPath: paths.payload)) == weights)
        let ranges = seen.requests.map { $0.value(forHTTPHeaderField: "Range") }
        #expect(ranges == [nil, nil, nil, "bytes=100-", nil])
        let published = holder.published
        let resumedAt = try #require(published.firstIndex(of: 100), "\(published)")
        #expect(published[resumedAt...].contains(0), "\(published)")
        #expect(published.last == 256, "\(published)")
        #expect(published.allSatisfy { $0 <= 256 }, "\(published)")
    }

    /// Every content attempt fails: five attempts, four waits, and the fetch fails with a
    /// failure that the worker can retry on a later claim.
    @Test func theAttemptsRunOutAndTheFetchFails() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let vault = StubVault(resources: [id: Data("weights".utf8)],
                              failing: [2: 503, 3: 503, 4: 503, 5: 503, 6: 503])
        let (http, server, seen) = try vault.serve()
        let spec = try vaultTextSpec(id)
        let retries = RetryRecorder()
        let coordinator = try coordinator(ws, http: http, server: server, retries: retries)

        do {
            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))
            Issue.record("a fetch that met five 503s bound a model")
        } catch let error as VaultFetchError {
            #expect(classifyRunError(error))
        }
        #expect(seen.methodsAndPaths.filter { $0.hasSuffix("/content") }.count == 5)
        #expect(retries.sleeps == [1, 2, 4, 8])
        #expect(retries.events.count == 4)
    }

    /// A cancel while the second file of a pair streams removes the entry that the first
    /// file published, so no half of a model stays on disk.
    @Test func aCancelBetweenTheFilesOfAPairLeavesNoEntry() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let (model, mmproj) = ("vault_model", "vault_mmproj")
        let resources = [model: Data("weights".utf8), mmproj: Data("projector".utf8)]
        let holder = CoordinatorHolder()
        // The cancel lands when the projector's content is asked for: the weights are in
        // the entry by then, and the projector is not.
        let vault = StubVault(resources: resources, onRequest: { path in
            guard path == StubVault.contentPath(mmproj) else { return }
            DispatchQueue.main.sync {
                MainActor.assumeIsolated {
                    guard let coordinator = holder.coordinator else { return }
                    for key in coordinator.downloads.keys { coordinator.cancel(key: key) }
                }
            }
        })
        let (http, server, _) = try vault.serve()
        let spec = try vaultVisionSpec(model, mmproj)
        let coordinator = try coordinator(ws, http: http, server: server)
        holder.coordinator = coordinator

        await #expect(throws: (any Error).self) {
            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))
        }
        let entry = try #require(ws.models.entryDir(for: spec))
        #expect(!FileManager.default.fileExists(atPath: entry.path))
        #expect(!coordinator.hasTransfer(for: spec))
    }

    /// A grant from a declared size reads no record. When the records weigh more than the
    /// quota allows, the pull is refused before its content request.
    @Test func aRecordHeavierThanItsDeclaredGrantIsReadmitted() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        try ws.setStorageQuotaBytes(1_500)
        let id = "vault_0195c3e2"
        let (http, server, seen) = try StubVault(resources: [id: Data(count: 2_000)]).serve()
        let spec = try vaultTextSpec(id)
        let coordinator = try coordinator(ws, http: http, server: server)

        await #expect(throws: (any Error).self) {
            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator, declaredSizeBytes: 1,
                stallTimeout: .seconds(30))
        }
        #expect(seen.methodsAndPaths == ["GET \(StubVault.recordPath(id))"])
        #expect(await coordinator.admission.outstandingBytes == 0)
    }

    /// A vision pair that names one resource for both files has no store form. The fetch
    /// is refused at once, and no content request goes out.
    @Test func aVisionPairOnOneResourceIsRefusedBeforeItsBytes() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_model"
        let (http, server, seen) = try StubVault(resources: [id: Data("weights".utf8)]).serve()
        let spec = try vaultVisionSpec(id, id)
        let coordinator = try coordinator(ws, http: http, server: server)

        let started = ContinuousClock.now
        await #expect(throws: ModelStoreError.self) {
            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))
        }
        #expect(ContinuousClock.now - started < .seconds(10))
        #expect(!seen.methodsAndPaths.contains("GET \(StubVault.contentPath(id))"))
        #expect(!coordinator.hasTransfer(for: spec))
    }

    /// A refusal reaches the caller in its type, at admission or at the download, so the
    /// worker can read whether every device would meet it.
    @Test(arguments: [true, false])
    func aVaultRefusalReachesTheCallerTyped(onContent: Bool) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let path = onContent ? StubVault.contentPath(id) : StubVault.recordPath(id)
        let vault = StubVault(resources: [id: Data("weights".utf8)], refusals: [path: 403])
        let (http, server, _) = try vault.serve()
        let spec = try vaultTextSpec(id)
        let coordinator = try coordinator(ws, http: http, server: server)

        do {
            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))
            Issue.record("a refused fetch bound a model")
        } catch let error as VaultFetchError {
            guard case let .refused(_, status, _) = error else {
                Issue.record("expected a refusal, got \(error)"); return
            }
            #expect(status == 403)
            #expect(!classifyRunError(error))
        }
    }

    // MARK: Vault archives

    /// A directory model's archive, as `ensureModel` sees it.
    private func archiveSpec(_ id: String, type: String, pin: Sha256?) throws -> Model {
        let source = ModelSource.vaultArchive(
            url: try vaultPath(id), sha256: pin, principalId: try VaultPrincipalId("org_1"),
            customLabel: CustomLabel(nil))
        return type == "mlx" ? .mlx(Mlx(source: source)) : .onnxGenai(OnnxGenai(source: source))
    }

    /// The files of a complete model of `type`.
    private func modelFiles(_ type: String) throws -> [String: Data] {
        guard type == "onnx_genai" else {
            return [
                "config.json": Data(#"{"model_type":"lfm2"}"#.utf8),
                "model.safetensors": Data(repeating: 0x5a, count: 2_000),
                "tokenizer.json": Data("{}".utf8),
            ]
        }
        let config: [String: Any] = ["model": ["decoder": ["filename": "model.onnx"]]]
        return [
            "genai_config.json": try JSONSerialization.data(withJSONObject: config),
            "model.onnx": onnxGraphNaming("model.onnx_data"),
            "model.onnx_data": Data(repeating: 0x77, count: 512),
            "tokenizer_config.json": Data("{}".utf8),
            "tokenizer.json": Data("{}".utf8),
        ]
    }

    /// What is left under `.staging`, which a finished or failed pull empties.
    private func staged(_ ws: PipetteWorkspace) -> [String] {
        (try? FileManager.default.contentsOfDirectory(
            atPath: ws.models.stagingRoot.path)) ?? []
    }

    @Test(arguments: ["mlx", "onnx_genai"])
    func aVaultArchiveInstallsAndBinds(type: String) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let files = try modelFiles(type)
        let archive = ArchiveFixture.deterministicTar(files)
        let (http, server, seen) = try StubVault(resources: [id: archive]).serve()
        let spec = try archiveSpec(id, type: type, pin: try Sha256(sha256Hex(archive)))
        let coordinator = try coordinator(ws, http: http, server: server)

        let bound = try await ensureModel(
            spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))

        let dir = URL(fileURLWithPath: try #require(bound.boundPaths).payload)
        #expect(dir == ws.models.blobsDir(for: spec)?.standardizedFileURL)
        for (name, contents) in files {
            #expect(try Data(contentsOf: dir.appendingPathComponent(name)) == contents)
        }
        #expect(!FileManager.default.fileExists(atPath: dir.appendingPathComponent(".vault-archive").path))
        #expect(seen.methodsAndPaths.last == "GET \(StubVault.contentPath(id))")
        #expect(staged(ws).isEmpty)
        #expect(!coordinator.hasTransfer(for: spec))
    }

    /// An archive and its unpacked tree are on disk together, so the grant is twice the
    /// record. A quota that holds the archive once and not twice refuses the pull before
    /// its content request.
    @Test func aVaultArchiveIsAdmittedAtTwiceItsRecord() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let archive = ArchiveFixture.deterministicTar(try modelFiles("mlx"))
        try ws.setStorageQuotaBytes(Int64(archive.count) * 3 / 2)
        let (http, server, seen) = try StubVault(resources: [id: archive]).serve()
        let spec = try archiveSpec(id, type: "mlx", pin: nil)
        let coordinator = try coordinator(ws, http: http, server: server)

        await #expect(throws: ModelStoreError.self) {
            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))
        }
        #expect(seen.methodsAndPaths == ["GET \(StubVault.recordPath(id))"])
    }

    /// An archive that does not unpack, or that unpacks to no complete model, is refused
    /// for good: the bytes are pinned, so every device unpacks the same tree.
    @Test(arguments: ["zip", "no model"])
    func aVaultArchiveThatYieldsNoModelIsPermanent(kind: String) async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let archive = kind == "zip"
            ? Data("PK\u{03}\u{04} a zipped model".utf8)
            : ArchiveFixture.deterministicTar(["config.json": Data("{}".utf8)])
        let (http, server, _) = try StubVault(resources: [id: archive]).serve()
        let spec = try archiveSpec(id, type: "mlx", pin: nil)
        let coordinator = try coordinator(ws, http: http, server: server)

        do {
            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))
            Issue.record("an archive with no model bound a model")
        } catch let error as VaultFetchError {
            switch (kind, error) {
            case ("zip", .archive(_, .zipUnsupported)), ("no model", .unusable): break
            default: Issue.record("unexpected refusal for \(kind): \(error)")
            }
            #expect(error.isPermanent)
            #expect(!classifyRunError(error))
        }
        let entry = try #require(ws.models.entryDir(for: spec))
        #expect(!FileManager.default.fileExists(atPath: entry.path))
        #expect(staged(ws).isEmpty)
    }

    @Test func aCancelDuringAnArchivePullLeavesNothing() async throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let id = "vault_0195c3e2"
        let archive = ArchiveFixture.deterministicTar(try modelFiles("mlx"))
        let holder = CoordinatorHolder()
        let vault = StubVault(resources: [id: archive], onRequest: { path in
            guard path == StubVault.contentPath(id) else { return }
            DispatchQueue.main.sync {
                MainActor.assumeIsolated {
                    guard let coordinator = holder.coordinator else { return }
                    for key in coordinator.downloads.keys { coordinator.cancel(key: key) }
                }
            }
        })
        let (http, server, _) = try vault.serve()
        let spec = try archiveSpec(id, type: "mlx", pin: nil)
        let coordinator = try coordinator(ws, http: http, server: server)
        holder.coordinator = coordinator

        await #expect(throws: (any Error).self) {
            _ = try await ensureModel(
                spec, ws: ws, coordinator: coordinator, stallTimeout: .seconds(30))
        }
        let entry = try #require(ws.models.entryDir(for: spec))
        #expect(!FileManager.default.fileExists(atPath: entry.path))
        #expect(staged(ws).isEmpty)
        #expect(!coordinator.hasTransfer(for: spec))
    }
}

/// Every progress report of a download, in the order that the delegate queue sent them.
private nonisolated final class ProgressReports: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [VaultProgress] = []

    func append(_ report: VaultProgress) {
        lock.withLock { stored.append(report) }
    }

    var all: [VaultProgress] {
        lock.withLock { stored }
    }
}
