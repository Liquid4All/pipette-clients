import Foundation
import Testing

@testable import Pipette

/// The `pipette-http` crate's `retry.rs` tests.
struct RetryTests {
    /// The app's schedules: the worker's claim (1s to 60s), its submit and a download (1s to
    /// 30s). The last row checks the shift does not overflow.
    @Test(arguments: [
        (attempt: 1, cap: 60.0, expected: 1.0),
        (attempt: 2, cap: 60, expected: 2),
        (attempt: 6, cap: 60, expected: 32),
        (attempt: 7, cap: 60, expected: 60),
        (attempt: 8, cap: 60, expected: 60),
        (attempt: 5, cap: 30, expected: 16),
        (attempt: 6, cap: 30, expected: 30),
        (attempt: 4_000_000_000, cap: 60, expected: 60),
    ])
    func exponentialBackoffDoublesFromTheFirstWaitToTheCap(
        attempt: Int, cap: TimeInterval, expected: TimeInterval
    ) {
        #expect(exponentialBackoff(attempt: attempt, first: 1, cap: cap) == expected)
    }

    /// The roll picks a wait from zero to the whole wait, both ends included.
    @Test(arguments: [
        (roll: UInt64(0), expected: 0.0),
        (roll: 1_500, expected: 1.5),
        (roll: 4_000, expected: 4),
        (roll: 4_001, expected: 0),
        (roll: .max, expected: TimeInterval(UInt64.max % 4_001) / 1000),
    ])
    func fullJitterStaysWithinTheWait(roll: UInt64, expected: TimeInterval) {
        #expect(fullJitter(4, roll: roll) == expected)
    }

    @Test func fullJitterOfNoWaitIsNoWait() {
        #expect(fullJitter(0, roll: .max) == 0)
    }

    struct Failed: Error, LocalizedError {
        let attempt: Int
        var errorDescription: String? { "connection reset" }
    }

    /// What one ``retrying(_:url:attempt:classify:hooks:)`` run did.
    struct Run {
        var result: Result<Int, any Error>
        var tries: [Int] = []
        var sleeps: [TimeInterval] = []
        var events: [RetryEvent] = []
    }

    /// Drive the retries with no jitter: attempt `n` fails as `outcome(n)` says, or
    /// succeeds when it says `nil`.
    static func drive(_ outcome: (Int) -> Disposition?) async -> Run {
        var tries: [Int] = []
        var sleeps: [TimeInterval] = []
        var events: [RetryEvent] = []
        let result: Result<Int, any Error>
        do {
            result = .success(try await retrying(
                .download, url: "https://example.test/model.gguf",
                attempt: { n in
                    tries.append(n)
                    guard outcome(n) == nil else { throw Failed(attempt: n) }
                    return n
                },
                classify: { error in
                    (error as? Failed).flatMap { outcome($0.attempt) } ?? .permanent
                },
                hooks: RetryHooks(jitter: { $0 }, sleep: { sleeps.append($0) },
                                   report: { events.append($0) })))
        } catch {
            result = .failure(error)
        }
        return Run(result: result, tries: tries, sleeps: sleeps, events: events)
    }

    @Test func aTransientFailureIsRetriedOnTheSchedule() async throws {
        let run = await Self.drive { $0 < 3 ? .transient(retryAfter: nil) : nil }
        #expect(try run.result.get() == 3)
        #expect(run.tries == [1, 2, 3])
        #expect(run.sleeps == [1, 2])
    }

    /// A wait between attempts and none after the last; the last error is thrown.
    @Test func theAttemptsRunOut() async {
        let run = await Self.drive { _ in .transient(retryAfter: nil) }
        #expect((run.result.failure as? Failed)?.attempt == 5)
        #expect(run.tries == [1, 2, 3, 4, 5])
        #expect(run.sleeps == [1, 2, 4, 8])
    }

    @Test func aPermanentFailureIsNotRetried() async {
        let run = await Self.drive { _ in .permanent }
        #expect((run.result.failure as? Failed)?.attempt == 1)
        #expect(run.tries == [1])
        #expect(run.sleeps.isEmpty)
        #expect(run.events.isEmpty)
    }

    /// Up to the limit; past it the retry stops.
    @Test(arguments: [
        (asked: 2.0, tries: [1, 2], sleeps: [2.0]),
        (asked: 7, tries: [1, 2], sleeps: [7]),
        (asked: 120, tries: [1, 2], sleeps: [120]),
        (asked: 121, tries: [1], sleeps: []),
    ])
    func retryAfterReplacesTheWait(asked: TimeInterval, tries: [Int], sleeps: [TimeInterval]) async {
        let run = await Self.drive { $0 < 2 ? .transient(retryAfter: asked) : nil }
        #expect(run.tries == tries)
        #expect(run.sleeps == sleeps)
    }

    @Test func aRefusedRetryAfterIsLogged() async {
        let run = await Self.drive { _ in .transient(retryAfter: 121) }
        #expect(run.events == [.retryAfterRefused(
            url: "https://example.test/model.gguf", attempt: 1, asked: 121, limit: 120,
            error: "connection reset")])
    }

    @Test func eachRetryIsLogged() async {
        let run = await Self.drive { $0 < 4 ? .transient(retryAfter: nil) : nil }
        let url = "https://example.test/model.gguf"
        #expect(run.events == [
            .attemptFailed(url: url, attempt: 1, wait: 1, error: "connection reset"),
            .attemptFailed(url: url, attempt: 2, wait: 2, error: "connection reset"),
            .attemptFailed(url: url, attempt: 3, wait: 4, error: "connection reset"),
            .recovered(url: url, attempts: 4),
        ])
        #expect(run.events.first?.message
            == "download attempt failed; retrying url=\(url) attempt=1 retry_in=1s "
            + "error=connection reset")
        #expect(run.events.last?.message == "download succeeded after retries url=\(url) attempts=4")
    }

    /// The waits log in one form on every device, whatever its locale.
    @Test func theWaitsLogInAFixedForm() {
        let url = "https://example.test/model.gguf"
        #expect(RetryEvent.attemptFailed(url: url, attempt: 1, wait: 0.457, error: "x").message
            .contains(" retry_in=0.457s "))
        #expect(RetryEvent.retryAfterRefused(url: url, attempt: 1, asked: 4_294_967_296,
                                              limit: 120, error: "x").message
            .contains(" retry_after=4294967296s limit=120s "))
    }

    @Test func aFirstTimeSuccessIsNotLogged() async {
        let run = await Self.drive { _ in nil }
        #expect(run.events.isEmpty)
    }

    @Test func cancellationIsNotRetried() async {
        var tries = 0
        await #expect(throws: CancellationError.self) {
            try await retrying(
                .download, url: "https://example.test/model.gguf",
                attempt: { _ -> Int in
                    tries += 1
                    throw CancellationError()
                },
                classify: { _ in .transient(retryAfter: nil) },
                hooks: RetryHooks(report: { _ in }))
        }
        #expect(tries == 1)
    }

    /// Nor reported, though the error classifies as transient.
    @Test func aFailureInACancelledTaskIsNotRetried() async {
        let run = await Task { () -> (tries: Int, events: [RetryEvent], error: (any Error)?) in
            var tries = 0
            var events: [RetryEvent] = []
            do {
                _ = try await retrying(
                    .download, url: "https://example.test/model.gguf",
                    attempt: { _ -> Int in
                        tries += 1
                        withUnsafeCurrentTask { $0?.cancel() }
                        throw URLError(.cancelled)
                    },
                    classify: { _ in .transient(retryAfter: nil) },
                    hooks: RetryHooks(sleep: { _ in }, report: { events.append($0) }))
                return (tries, events, nil)
            } catch {
                return (tries, events, error)
            }
        }.value
        #expect(run.error is CancellationError)
        #expect(run.tries == 1)
        #expect(run.events.isEmpty)
    }

    /// The codes that mean no answer arrived, or the body broke, can pass; a malformed
    /// answer or a refused certificate cannot.
    @Test(arguments: [
        (URLError.Code.timedOut, true),
        (.cannotFindHost, true),
        (.cannotConnectToHost, true),
        (.dnsLookupFailed, true),
        (.networkConnectionLost, true),
        (.secureConnectionFailed, true),
        (.notConnectedToInternet, true),
        (.badServerResponse, false),
        (.serverCertificateUntrusted, false),
        (.userAuthenticationRequired, false),
        (.badURL, false),
    ])
    func aTransportErrorIsTransientOnlyWhenItCanPass(code: URLError.Code, transient: Bool) {
        #expect((Disposition(of: code) != .permanent) == transient)
    }

    @Test(arguments: [
        (status: 500, transient: true),
        (status: 503, transient: true),
        (status: 429, transient: true),
        (status: 401, transient: false),
        (status: 403, transient: false),
        (status: 404, transient: false),
        (status: 416, transient: false),
    ])
    func anHttpStatusIsTransientOnlyForAServerErrorOrARateLimit(status: Int, transient: Bool) {
        #expect((Disposition(status: status, retryAfter: nil) != .permanent) == transient)
    }

    @Test func aTransientStatusCarriesItsRetryAfter() {
        #expect(Disposition(status: 503, retryAfter: 3) == .transient(retryAfter: 3))
    }

    /// Delta-seconds only: an HTTP-date leaves the retry on its own schedule.
    @Test(arguments: [
        (raw: "2", seconds: 2.0 as TimeInterval?),
        (raw: " 7 ", seconds: 7),
        (raw: "4294967296", seconds: 4_294_967_296),
        (raw: "Wed, 21 Oct 2015 07:28:00 GMT", seconds: nil),
        (raw: "-1", seconds: nil),
        (raw: "", seconds: nil),
    ])
    func retryAfterReadsDeltaSeconds(raw: String, seconds: TimeInterval?) {
        #expect(retryAfter(raw) == seconds)
    }

    @Test func theChainNamesEveryCause() {
        let error = NSError(domain: "Pipette", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "the download failed",
            NSUnderlyingErrorKey: URLError(.networkConnectionLost),
        ])
        #expect(errorChain(error)
            == "the download failed: \(URLError(.networkConnectionLost).localizedDescription)")
    }
}

private extension Result {
    var failure: Failure? {
        if case let .failure(error) = self { error } else { nil }
    }
}
