import CryptoKit
import Foundation

// Fetching a vault resource, after `crates/pipette-artifacts/src/vault.rs`: a private
// artifact mgmt serves only to the devices of the principal that owns it. The record
// gives the digest and size, then the bytes are pulled and checked against it. The
// device signature is the whole credential; the plan holds no token.

nonisolated struct VaultAccess: Sendable {
    let server: ServerUrl
    let identity: ClientIdentity

    static func of(_ session: () throws -> MgmtSession, path: VaultPath)
        throws(VaultFetchError) -> VaultAccess {
        do {
            let session = try session()
            return VaultAccess(server: session.serverUrl, identity: session.auth)
        } catch {
            throw .unregistered(url: path.value)
        }
    }
}

/// The crate's `VaultFetchError`.
nonisolated enum VaultFetchError: Error, LocalizedError {
    case unregistered(url: String)
    /// `403` (the device's owner does not own it, or the device is unapproved or ownerless)
    /// or `404` (no such resource, or its object is gone).
    case refused(resourceId: VaultResourceId, status: Int, detail: String)
    /// Neither pin is trusted over the other: the plan holds the submitter's digest, the
    /// record the digest of what the server sends. They must agree before a byte moves.
    case pinDisagrees(resourceId: VaultResourceId, plan: String, record: String)
    /// The object was replaced after the record was made; an operator must re-register it.
    case drift(resourceId: VaultResourceId, expected: String, actual: String)
    /// The bytes may have broken in transit, so the next attempt starts from zero.
    case resumedDrift(
        resourceId: VaultResourceId, resumedAt: Int64, expected: String, actual: String)
    /// Every other mgmt failure: transport, `5xx`, `401`, `3xx`, an unreadable record.
    case mgmt(PipetteMgmtClient.Error)
    case io(dest: String, source: String)
    /// The body ended short with no transport error: this device's connection, not the
    /// object, so another attempt can fetch it whole.
    case truncated(dest: String, received: Int64, expected: Int64)
    /// The bytes match the pin but the unpack refused them or failed.
    case archive(resourceId: VaultResourceId, source: ArchiveError)
    /// Unpacked, but not a complete model of the declared format.
    case unusable(resourceId: VaultResourceId, reason: String)

    /// The worker's disposition: a failure every device with this owner would meet
    /// identically is not worth another claim.
    var isPermanent: Bool {
        switch self {
        case .unregistered, .refused, .pinDisagrees, .drift, .unusable: true
        case let .archive(_, error): error.isPermanent
        // A `401` is about this device's signature: a skewed clock, or a nonce the server
        // saw before. Another device signs its own.
        case let .mgmt(source): !source.isTransient && source.httpStatusCode != 401
        case .io, .truncated, .resumedDrift: false
        }
    }

    /// The crate's `disposition`: whether another attempt in this process can succeed. A
    /// `401` is not retried here (unlike ``isPermanent``) because this process signs with
    /// the same clock every time. The client keeps no `Retry-After`, so a `429` waits the
    /// schedule.
    var disposition: Disposition {
        switch self {
        case let .mgmt(source) where source.isTransient || source.httpStatusCode == 429:
            .transient(retryAfter: nil)
        case .truncated, .resumedDrift:
            .transient(retryAfter: nil)
        // An unpack runs after the stream, outside the retry.
        case .unregistered, .refused, .pinDisagrees, .drift, .mgmt, .io, .archive, .unusable:
            .permanent
        }
    }

    var errorDescription: String? {
        switch self {
        case let .unregistered(url):
            "`\(url)` is a vault resource. Only a registered device can fetch it, and "
                + "this device has no registered identity. Register this device, then try "
                + "again."
        case let .refused(resourceId, status, detail):
            "the management server refused vault resource \(resourceId) with \(status) "
                + "(\(Self.refusalMeaning(status))): \(detail)"
        case let .pinDisagrees(resourceId, plan, record):
            "the record for vault resource \(resourceId) pins sha256 \(record), but the "
                + "plan expects \(plan)"
        case let .drift(resourceId, expected, actual):
            "vault resource \(resourceId) hashed to \(actual) after the download, and its "
                + "record pins \(expected). Someone replaced the object after the record was "
                + "made. Register the resource again."
        case let .resumedDrift(resourceId, resumedAt, expected, actual):
            "vault resource \(resourceId) resumed at byte \(resumedAt) and hashed to "
                + "\(actual), not \(expected). The next attempt downloads it again from zero."
        case let .mgmt(source):
            source.errorDescription
        case let .io(dest, reason):
            "the app could not write \(dest): \(reason)"
        case let .truncated(dest, received, expected):
            "the download of \(dest) ended after \(received) of \(expected) bytes"
        case let .archive(resourceId, error):
            "the app could not unpack vault resource \(resourceId): "
                + (error.errorDescription ?? "\(error)")
        case let .unusable(resourceId, reason):
            "vault resource \(resourceId) unpacked, but it is not a complete model: \(reason)"
        }
    }

    /// A `403` is about ownership, never a Hugging Face token: the plan holds none.
    private static func refusalMeaning(_ status: Int) -> String {
        status == 403
            ? "this device's owner does not own it, or the device is unapproved or has no owner"
            : "no such resource, or its bytes are gone"
    }
}

/// `jobId` goes into every vault log event so an operator can join a refusal to its job.
nonisolated enum VaultFetch {
    /// The crate's `resource`: a disagreeing pin is refused before a byte moves.
    static func readRecord(
        _ access: VaultAccess, http: HttpClient, path: VaultPath, pinned: Sha256?,
        jobId: JobId?
    ) async throws(VaultFetchError) -> VaultResource {
        let resourceId = path.resourceId
        let record: VaultResource
        do {
            record = try await PipetteMgmtClient.ClientApi.getVaultResource(
                server: access.server, http: http, auth: access.identity,
                resourceId: resourceId)
        } catch {
            throw refusal(resourceId, error, jobId: jobId)
        }
        if let pinned, pinned != record.sha256 {
            AppLog.vault.warning(
                "vault record pin disagrees resource_id=\(resourceId) "
                    + "job_id=\(jobId?.value ?? "none") "
                    + "plan=\(pinned) record=\(record.sha256)")
            throw .pinDisagrees(
                resourceId: resourceId, plan: pinned.value, record: record.sha256.value)
        }
        return record
    }

    /// Each read retries transient failures on the download policy, as the crate's
    /// `resource` does.
    static func records(
        _ files: [ModelFetch.VaultFile], access: VaultAccess, http: HttpClient, jobId: JobId?,
        hooks: RetryHooks = .live
    ) async throws -> [VaultResource] {
        var records: [VaultResource] = []
        for file in files {
            records.append(try await retrying(
                url: "/vault/resources/\(file.path.resourceId.value)",
                attempt: { _ in
                    try await readRecord(
                        access, http: http, path: file.path, pinned: file.pin, jobId: jobId)
                },
                classify: Self.disposition(of:), hooks: hooks))
        }
        return records
    }

    static func disposition(of error: any Error) -> Disposition {
        (error as? VaultFetchError)?.disposition ?? .permanent
    }

    /// Saturating: the figures are the server's.
    static func sizeBytes(_ records: [VaultResource]) -> Int64 {
        records.reduce(Int64(0)) { ModelFetch.saturating($0, Int64(clamping: $1.bytes)) }
    }

    /// A directory model's archive and unpacked tree coexist until the unpack deletes the
    /// archive, so it needs twice the record. A compressed archive needs more than that.
    static func admissionBytes(_ declared: Model, records: [VaultResource]) -> Int64 {
        let packed = sizeBytes(records)
        switch declared {
        case .mlx, .onnxGenai: return ModelFetch.saturating(packed, packed)
        case .ggufText, .ggufVision, .ggufVisionSpeculative, .appleFoundationText: return packed
        }
    }

    /// One attempt of the crate's `fetch_to_file` download half, hashing as it writes.
    /// Pass the same `partial` (initially `nil`) to each attempt: a retryable failure
    /// keeps the file there and the next attempt resumes it with `Range`/`If-Range`. A
    /// resumed file that fails the pin is `resumedDrift` and is emptied; a from-zero one
    /// is `drift`. Any other failure, or a cancel, removes `dest` and clears `partial`.
    ///
    /// The file reports to the `progress` closure of the attempt that created it.
    static func download(
        _ record: VaultResource,
        access: VaultAccess,
        http: HttpClient,
        to dest: URL,
        partial: inout VaultPartial?,
        jobId: JobId?,
        progress: @escaping @Sendable (VaultProgress) -> Void
    ) async throws(VaultFetchError) {
        let resourceId = record.resourceId
        let job = jobId?.value ?? "none"
        AppLog.vault.info(
            "vault fetch started resource_id=\(resourceId) job_id=\(job) bytes=\(record.bytes) "
                + "offset=\(partial?.written ?? 0)")
        let file: VaultPartial
        if let kept = partial {
            file = kept
        } else {
            do {
                try FileManager.default.createDirectory(
                    at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
                file = try HashingFileSink(file: dest, progress: progress)
            } catch {
                throw .io(dest: dest.path, source: error.localizedDescription)
            }
        }
        partial = nil
        func drop() {
            file.close()
            try? FileManager.default.removeItem(at: dest)
        }
        let actual: String
        do throws(VaultFetchError) {
            try await stream(record, access: access, http: http, into: file, dest: dest,
                             jobId: jobId)
            guard file.written >= Int64(clamping: record.bytes) else {
                throw VaultFetchError.truncated(
                    dest: dest.path, received: file.written,
                    expected: Int64(clamping: record.bytes))
            }
            do {
                actual = try file.digest()
            } catch {
                throw VaultFetchError.io(dest: dest.path, source: error.localizedDescription)
            }
        } catch {
            if !Task.isCancelled, error.disposition != .permanent {
                partial = file
            } else {
                drop()
            }
            throw error
        }
        guard actual == record.sha256.value else {
            guard let resumedAt = file.resumedAt else {
                drop()
                AppLog.vault.error(
                    "vault content drifted from its record resource_id=\(resourceId) "
                        + "job_id=\(job) expected=\(record.sha256) actual=\(actual)")
                throw .drift(
                    resourceId: resourceId, expected: record.sha256.value, actual: actual)
            }
            AppLog.vault.warning(
                "vault content failed its pin after a resume resource_id=\(resourceId) "
                    + "job_id=\(job) resumed_at=\(resumedAt) expected=\(record.sha256) "
                    + "actual=\(actual)")
            guard file.reset() else {
                drop()
                throw .io(dest: dest.path, source: file.failure ?? "the file did not empty")
            }
            partial = file
            throw .resumedDrift(
                resourceId: resourceId, resumedAt: resumedAt, expected: record.sha256.value,
                actual: actual)
        }
        file.close()
        AppLog.vault.info(
            "vault content verified resource_id=\(resourceId) job_id=\(job) "
                + "bytes=\(file.written) "
                + "resumed_at=\(file.resumedAt.map(String.init) ?? "none")")
    }

    /// The unpack half of the crate's `fetch_and_unpack`. Synchronous and takes seconds
    /// for a large model: call it off the main actor.
    static func unpack(
        _ record: VaultResource, archive: URL, into dest: URL, jobId: JobId?
    ) throws(VaultFetchError) {
        let resourceId = record.resourceId
        let job = jobId?.value ?? "none"
        let unpacked: Archive.Unpacked
        do throws(ArchiveError) {
            unpacked = try Archive.unpack(archive, into: dest)
        } catch {
            // Only a permanent error is a fault in the pinned bytes.
            if error == .cancelled {
                AppLog.vault.info(
                    "vault archive unpack cancelled resource_id=\(resourceId) job_id=\(job)")
            } else if error.isPermanent {
                AppLog.vault.warning(
                    "vault archive refused resource_id=\(resourceId) job_id=\(job) "
                        + "reason=\(error.localizedDescription)")
            } else {
                AppLog.vault.warning(
                    "vault archive unpack failed on this device resource_id=\(resourceId) "
                        + "job_id=\(job) reason=\(error.localizedDescription)")
            }
            throw .archive(resourceId: resourceId, source: error)
        }
        AppLog.vault.info(
            "vault archive unpacked resource_id=\(resourceId) job_id=\(job) "
                + "format=\(unpacked.kind.rawValue) entries=\(unpacked.entries) "
                + "skipped=\(unpacked.skipped)")
        // The caller removes the scratch directory with the archive anyway, so a failed
        // delete does not fail the install.
        do {
            try FileManager.default.removeItem(at: archive)
        } catch {
            AppLog.vault.warning(
                "vault archive delete failed after the unpack resource_id=\(resourceId) "
                    + "job_id=\(job) reason=\(error.localizedDescription)")
        }
    }

    /// The crate's `resume::open` and `resume::attempt`. A `416` or a non-continuing
    /// response restarts from zero; a `206` that ends before the file does is kept and the
    /// rest requested. A failure before the server judges the range keeps the file as is.
    private static func stream(
        _ record: VaultResource,
        access: VaultAccess,
        http: HttpClient,
        into file: VaultPartial,
        dest: URL,
        jobId: JobId?
    ) async throws(VaultFetchError) {
        let resourceId = record.resourceId
        let job = jobId?.value ?? "none"
        var point = ResumePoint.of(file, record: record)
        while true {
            let sink = RangeSink(file: file, resume: point, resourceId: resourceId,
                                 jobId: job)
            let failure: PipetteMgmtClient.Error
            do {
                _ = try await PipetteMgmtClient.ClientApi.getVaultContent(
                    server: access.server, http: http, auth: access.identity,
                    resourceId: resourceId, resume: point?.request, into: sink)
                // A `206` can send less than the rest of the file.
                let written = UInt64(file.written)
                guard let last = sink.pieceEnd, last < .max, written == last + 1,
                      written < record.bytes else { return }
                AppLog.vault.info(
                    "range answer stopped short of the end resource_id=\(resourceId) "
                        + "job_id=\(job) offset=\(written)")
                point = ResumePoint.of(file, record: record)
                continue
            } catch {
                failure = error
            }
            let reason: String
            if sink.refused {
                reason = "not_a_continuation"
            } else if point != nil, failure.httpStatusCode == 416 {
                reason = "range_refused"
            } else {
                // A failed write surfaces as a transport error; report the write.
                if let cause = file.failure { throw .io(dest: dest.path, source: cause) }
                throw refusal(resourceId, failure, jobId: jobId)
            }
            guard file.reset() else {
                throw .io(dest: dest.path, source: file.failure ?? "the file did not empty")
            }
            point = nil
            AppLog.vault.info(
                "restarting download from zero resource_id=\(resourceId) job_id=\(job) "
                    + "reason=\(reason)")
        }
    }

    /// The crate's `refusal`: only `403` and `404` become `.refused`.
    private static func refusal(
        _ resourceId: VaultResourceId, _ error: PipetteMgmtClient.Error, jobId: JobId?
    ) -> VaultFetchError {
        guard let status = error.httpStatusCode, status == 403 || status == 404 else {
            return .mgmt(error)
        }
        AppLog.vault.warning(
            "vault fetch refused resource_id=\(resourceId) job_id=\(jobId?.value ?? "none") "
                + "status=\(status)")
        return .refused(
            resourceId: resourceId, status: status,
            detail: error.errorDescription ?? "status \(status)")
    }
}

/// `written` can go down when `restarts` goes up.
nonisolated struct VaultProgress: Sendable {
    let written: Int64
    let restarts: Int
}

/// The crate's `Partial`.
typealias VaultPartial = HashingFileSink

/// The crate's `ResumePoint`. The `ETag` is the record's pin.
private nonisolated struct ResumePoint {
    let offset: UInt64
    let etag: EntityTag
    let total: UInt64

    static func of(_ file: HashingFileSink, record: VaultResource) -> ResumePoint? {
        guard file.written > 0,
              let etag = try? EntityTag("\"\(record.sha256.value)\"") else { return nil }
        return ResumePoint(offset: UInt64(file.written), etag: etag, total: record.bytes)
    }

    var request: PipetteMgmtClient.ResumeFrom {
        PipetteMgmtClient.ResumeFrom(offset: offset, ifRange: etag)
    }

    /// Compares the `ETag` itself rather than trusting the server's `If-Range` handling.
    func continued(by head: PipetteMgmtClient.HttpResponse) -> Bool {
        head.status == 206
            && head.strongEtag == etag.value
            && head.contentRange.map { $0.first == offset && $0.total == total } == true
    }
}

/// `@unchecked Sendable` for the reason ``HashingFileSink`` gives; `refused` and
/// `pieceEnd` are read only after the exchange completes.
private nonisolated final class RangeSink: ByteSink, @unchecked Sendable {
    private let file: HashingFileSink
    private let resume: ResumePoint?
    private let resourceId: VaultResourceId
    private let jobId: String
    /// `begin` refused a 2xx that does not continue the file.
    private(set) var refused = false
    /// The last byte of a continuing `206`'s range.
    private(set) var pieceEnd: UInt64?

    init(file: HashingFileSink, resume: ResumePoint?, resourceId: VaultResourceId,
         jobId: String) {
        self.file = file
        self.resume = resume
        self.resourceId = resourceId
        self.jobId = jobId
    }

    func begin(_ head: PipetteMgmtClient.HttpResponse) -> Bool {
        guard let resume else { return true }
        if resume.continued(by: head) {
            file.markResumed(at: Int64(clamping: resume.offset))
            pieceEnd = head.contentRange?.last
            AppLog.vault.info(
                "resuming download resource_id=\(resourceId) job_id=\(jobId) "
                    + "offset=\(resume.offset)")
            return true
        }
        guard head.status == 200 else {
            refused = true
            return false
        }
        guard file.reset() else { return false }
        AppLog.vault.info(
            "restarting download from zero resource_id=\(resourceId) job_id=\(jobId) "
                + "reason=whole_file")
        return true
    }

    func write(_ chunk: Data) -> Bool {
        file.write(chunk)
    }
}

/// The crate's `HashingWriter` over a `File`. Only bytes the file took are hashed, so
/// the digest never runs ahead of the file.
///
/// `@unchecked Sendable`: `URLSession` calls `write` serially from its delegate queue, and
/// the fetch reads the state only after the exchange completes.
nonisolated final class HashingFileSink: @unchecked Sendable {
    private let handle: FileHandle
    private let progress: @Sendable (VaultProgress) -> Void
    private var hasher = SHA256()
    private(set) var written: Int64 = 0
    private(set) var restarts = 0
    /// `nil` while the file holds only bytes from a download that started at zero.
    private(set) var resumedAt: Int64?
    private(set) var failure: String?

    init(file: URL, progress: @escaping @Sendable (VaultProgress) -> Void) throws {
        guard FileManager.default.createFile(atPath: file.path, contents: nil) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: file.path])
        }
        handle = try FileHandle(forWritingTo: file)
        self.progress = progress
    }

    func write(_ chunk: Data) -> Bool {
        do {
            try handle.write(contentsOf: chunk)
        } catch {
            failure = error.localizedDescription
            return false
        }
        hasher.update(data: chunk)
        written += Int64(chunk.count)
        progress(VaultProgress(written: written, restarts: restarts))
        return true
    }

    /// On `false`, `failure` holds the reason.
    func reset() -> Bool {
        do {
            try handle.truncate(atOffset: 0)
        } catch {
            failure = error.localizedDescription
            return false
        }
        hasher = SHA256()
        written = 0
        restarts += 1
        resumedAt = nil
        progress(VaultProgress(written: 0, restarts: restarts))
        return true
    }

    /// The first resume since the last restart is the one that counts.
    func markResumed(at offset: Int64) {
        resumedAt = resumedAt ?? offset
    }

    /// Flushes to disk; the file stays open.
    func digest() throws -> String {
        try handle.synchronize()
        return Data(hasher.finalize()).hexEncodedString()
    }

    func close() {
        try? handle.close()
    }
}
