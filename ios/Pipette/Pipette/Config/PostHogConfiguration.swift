import Foundation

/// PostHog project configuration, the iOS twin of the Android client's `PostHogConfiguration`.
///
/// The project token is public by design: a `phc_…` key is write-only, able to ingest events but
/// never to read them, so it lives in `Info.plist` alongside `SentryDSN` and the Clerk publishable
/// key rather than being treated as a secret. `PostHogHost` is the ingest host, not the dashboard
/// host.
///
/// `isComplete` guards initialization: without a resolved project the SDK never starts, so an
/// unconfigured build runs analytics-free instead of pushing at an endpoint that would reject it.
enum PostHogConfiguration {
    /// Resolved once: the token is fixed for the lifetime of the process, matching how
    /// `SentryConfiguration.dsn` avoids re-reading Info.plist on every access.
    static let projectToken: String? = Bundle.main.normalizedInfoString("PostHogProjectToken")

    static let host: String? = Bundle.main.normalizedInfoString("PostHogHost")

    static var isComplete: Bool {
        isCompletePostHogConfig(projectToken, host)
    }
}

/// Pure predicate behind ``PostHogConfiguration/isComplete``, kept free of `Bundle` so it is
/// directly unit-testable. Mirrors Android's `isCompletePostHogConfig`.
///
/// The `phc_` prefix check is the security-relevant part: a PostHog **personal** API key (`phx_…`)
/// is a read-write credential that must never ship in a client, so refusing to initialize with one
/// turns a leaked-secret incident into a silently disabled feature. `normalizedInfoString` has
/// already rejected blanks and unresolved `$(…)` placeholders by the time values arrive here.
nonisolated func isCompletePostHogConfig(_ projectToken: String?, _ host: String?) -> Bool {
    guard let projectToken, let host else { return false }
    return projectToken.hasPrefix("phc_") && host.hasPrefix("https://")
}
