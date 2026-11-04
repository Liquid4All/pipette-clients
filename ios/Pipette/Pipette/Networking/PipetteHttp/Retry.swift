import Foundation

// Retries after a transient failure: the `pipette-http` crate's `retry.rs`.
//
// `HttpClient` never retries on its own: a benchmark request sent twice would change the
// measurement. Only callers that ask retry (downloads, vault reads). The worker's claim,
// submit and idle waits share the backoff and jitter here.
//
// An attempt is the whole transfer (request, body and write); the caller defines it and
// classifies its error. A GGUF transfer is a `URLSession` task whose failure arrives at a
// delegate, so `DownloadCoordinator` calls `RetryPolicy.next(after:_:jitter:)` and
// schedules the wait itself instead of calling `retrying`.

nonisolated struct RetryPolicy: Sendable, Equatable {
    /// Attempts in total, the first included.
    let attempts: Int
    /// The wait after the first failed attempt, before jitter.
    let first: TimeInterval
    /// The longest wait the schedule reaches, before jitter.
    let cap: TimeInterval
    /// A server that asks for a longer `Retry-After` gets no retry: the job fails,
    /// and the worker claims again later.
    let retryAfterLimit: TimeInterval

    static let download = RetryPolicy(attempts: 5, first: 1, cap: 30, retryAfterLimit: 120)

    enum Next: Sendable, Equatable {
        case retry(in: TimeInterval)
        /// The failure is permanent, or the attempts ran out.
        case giveUp
        /// The server asked for a longer wait than `retryAfterLimit`.
        case refuseRetryAfter(TimeInterval)
    }

    /// `attempt` counts from one. A transient failure on the last attempt gives up,
    /// whatever the server asked for. A `Retry-After` replaces this attempt's wait only.
    func next(after attempt: Int, _ disposition: Disposition,
              jitter: (TimeInterval) -> TimeInterval) -> Next {
        guard case let .transient(retryAfter) = disposition, attempt < attempts else {
            return .giveUp
        }
        guard let asked = retryAfter else {
            return .retry(in: jitter(exponentialBackoff(attempt: attempt, first: first, cap: cap)))
        }
        return asked > retryAfterLimit ? .refuseRetryAfter(asked) : .retry(in: asked)
    }
}

/// Whether another attempt can succeed where this one failed.
nonisolated enum Disposition: Sendable, Equatable {
    /// Every attempt fails the same way: a `404`, a digest mismatch, a full disk.
    case permanent
    /// `retryAfter` is the wait the server asked for.
    case transient(retryAfter: TimeInterval?)

    /// Transient when no answer arrived or the body broke part-way, and for every
    /// offline code (``isOffline(_:)``).
    init(of code: URLError.Code) {
        if isOffline(code) {
            self = .transient(retryAfter: nil)
            return
        }
        switch code {
        case .timedOut, .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed,
             .networkConnectionLost, .secureConnectionFailed:
            self = .transient(retryAfter: nil)
        default:
            self = .permanent
        }
    }

    /// Every status but a `5xx` and a `429` is permanent, for example a `401` for a
    /// gated model.
    init(status: Int, retryAfter: TimeInterval?) {
        self = (500...599).contains(status) || status == 429
            ? .transient(retryAfter: retryAfter) : .permanent
    }
}

/// The wait after `attempt` consecutive transient failures: `first`, doubling, capped at
/// `cap`.
nonisolated func exponentialBackoff(attempt: Int, first: TimeInterval,
                                    cap: TimeInterval) -> TimeInterval {
    // Thirty-one doublings dwarf any cap; the clamp keeps the shift in range for a
    // session left retrying for days.
    let doublings = min(max(attempt - 1, 0), 31)
    return min(cap, first * TimeInterval(1 << doublings))
}

/// A wait in `0...wait`, picked by `roll` to the millisecond. Full jitter spreads the
/// retries of clients that fail together.
nonisolated func fullJitter(_ wait: TimeInterval, roll: UInt64) -> TimeInterval {
    // Capped at 1e15 ms (about 30 000 years), well below `UInt64.max`, so the conversion
    // cannot trap and the `+ 1` cannot overflow.
    let maxMillis = UInt64(min(max(wait, 0) * 1000, 1e15))
    return TimeInterval(roll % (maxMillis + 1)) / 1000
}

nonisolated func jittered(_ wait: TimeInterval) -> TimeInterval {
    fullJitter(wait, roll: UInt64.random(in: .min ... .max))
}

/// `error` and each underlying error, joined with `": "`. Descriptions only: an
/// `NSError`'s `userInfo` can hold resume data or a signed CDN URL, which no log line
/// may contain.
nonisolated func errorChain(_ error: any Error) -> String {
    sequence(first: error as NSError) { $0.userInfo[NSUnderlyingErrorKey] as? NSError }
        .map(\.localizedDescription)
        .joined(separator: ": ")
}

/// Run `attempt` (numbered from one) until it succeeds, fails permanently, or runs out,
/// and throw the last attempt's error unchanged. A cancelled task stops at once with
/// `CancellationError`, whatever its attempt failed with.
nonisolated func retrying<T>(
    _ policy: RetryPolicy = .download,
    url: String,
    attempt: (Int) async throws -> T,
    classify: (any Error) -> Disposition,
    hooks: RetryHooks = .live
) async throws -> T {
    var n = 1
    while true {
        do {
            let value = try await attempt(n)
            if n > 1 { hooks.report(.recovered(url: url, attempts: n)) }
            return value
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // A cancelled task's attempt can fail with another error, such as the
            // `URLError(.cancelled)` that `HubApi` wraps in `.downloadError`.
            try Task.checkCancellation()
            let next = policy.next(after: n, classify(error), jitter: hooks.jitter)
            if let event = RetryEvent(after: next, url: url, attempt: n, policy: policy,
                                      error: error) {
                hooks.report(event)
            }
            guard case let .retry(wait) = next else { throw error }
            try await hooks.sleep(wait)
            n += 1
        }
    }
}

// MARK: - iOS: the test seam, offline codes, `Retry-After`, and the log

/// A test injects these to read the waits and warnings with no real delay.
nonisolated struct RetryHooks {
    var jitter: (TimeInterval) -> TimeInterval = jittered
    var sleep: (TimeInterval) async throws -> Void = { try await Task.sleep(for: .seconds($0)) }
    var report: (RetryEvent) -> Void = { $0.log() }

    static var live: RetryHooks { RetryHooks() }
}

/// `DownloadCoordinator` checks this first and pauses a GGUF transfer until the network
/// comes back; a directory pull retries these codes on the schedule.
nonisolated func isOffline(_ code: URLError.Code) -> Bool {
    switch code {
    case .notConnectedToInternet, .dataNotAllowed, .internationalRoamingOff: true
    default: false
    }
}

/// The delta-seconds form only: an HTTP-date is ignored and the retry uses its schedule.
nonisolated func retryAfter(_ value: String?) -> TimeInterval? {
    guard let value,
          let seconds = UInt64(value.trimmingCharacters(in: .whitespaces)) else { return nil }
    return TimeInterval(seconds)
}

nonisolated func retryAfter(of response: HTTPURLResponse) -> TimeInterval? {
    retryAfter(response.value(forHTTPHeaderField: "Retry-After"))
}

/// A step of a retried download that the log records.
nonisolated enum RetryEvent: Sendable, Equatable {
    case attemptFailed(url: String, attempt: Int, wait: TimeInterval, error: String)
    case retryAfterRefused(url: String, attempt: Int, asked: TimeInterval,
                           limit: TimeInterval, error: String)
    case recovered(url: String, attempts: Int)

    /// `nil` on `.giveUp`: a permanent failure or the last attempt logs nothing here.
    init?(after next: RetryPolicy.Next, url: String, attempt: Int, policy: RetryPolicy,
          error: any Error) {
        switch next {
        case .giveUp:
            return nil
        case let .refuseRetryAfter(asked):
            self = .retryAfterRefused(url: url, attempt: attempt, asked: asked,
                                      limit: policy.retryAfterLimit, error: errorChain(error))
        case let .retry(wait):
            self = .attemptFailed(url: url, attempt: attempt, wait: wait,
                                  error: errorChain(error))
        }
    }

    var message: String {
        switch self {
        case let .attemptFailed(url, attempt, wait, error):
            "download attempt failed; retrying url=\(url) attempt=\(attempt) "
                + "retry_in=\(Self.seconds(wait)) error=\(error)"
        case let .retryAfterRefused(url, attempt, asked, limit, error):
            "download not retried: Retry-After exceeds the limit url=\(url) "
                + "attempt=\(attempt) retry_after=\(Self.seconds(asked)) "
                + "limit=\(Self.seconds(limit)) error=\(error)"
        case let .recovered(url, attempts):
            "download succeeded after retries url=\(url) attempts=\(attempts)"
        }
    }

    func log() {
        switch self {
        case .attemptFailed, .retryAfterRefused: AppLog.storage.warning(message)
        case .recovered: AppLog.storage.info(message)
        }
    }

    /// One fixed form on every device: a `.` before the fraction, no grouping.
    private static func seconds(_ value: TimeInterval) -> String {
        value.formatted(.number.precision(.fractionLength(0...3)).grouping(.never)
            .locale(Locale(identifier: "en_US_POSIX"))) + "s"
    }
}
