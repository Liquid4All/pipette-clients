import Foundation

/// PostHog project configuration, the twin of Android's `PostHogConfiguration`.
///
/// The project token is not a secret: a `phc_` key can ingest events but never read them.
/// `PostHogHost` is the ingest host, not the dashboard host. Without both the SDK never
/// starts, so an unconfigured build runs analytics-free.
enum PostHogConfiguration {
    static let projectToken: String? = Bundle.main.normalizedInfoString("PostHogProjectToken")

    static let host: String? = Bundle.main.normalizedInfoString("PostHogHost")

    static var isComplete: Bool {
        isCompletePostHogConfig(projectToken, host)
    }
}

/// Free of `Bundle` so it is unit-testable; mirrors Android's `isCompletePostHogConfig`.
///
/// The `phc_` check is the security part: a personal API key (`phx_`) is a read-write
/// credential, so refusing it turns a leaked secret into a disabled feature.
nonisolated func isCompletePostHogConfig(_ projectToken: String?, _ host: String?) -> Bool {
    guard let projectToken, let host else { return false }
    return projectToken.hasPrefix("phc_") && host.hasPrefix("https://")
}
