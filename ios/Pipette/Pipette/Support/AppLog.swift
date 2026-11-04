import Foundation
import os
import Sentry

/// Diagnostic logging. Each call goes to unified logging (subsystem = bundle id, one
/// `category` per area), a Sentry breadcrumb when Sentry is running, and stderr in
/// headless runs (see `mirrorHeadless`).
///
/// Messages are logged `public`, so keep anything user-identifying out of them: the
/// whole message is visible in Console.app and in Sentry.
///
/// `nonisolated` so the data layer and detached executor tasks can log without hopping
/// to the main actor; `os.Logger` and the Sentry SDK are thread-safe.
nonisolated struct AppLog {
    private let logger: Logger
    private let category: String

    private init(_ category: String) {
        self.logger = Logger(subsystem: Self.subsystem, category: category)
        self.category = category
    }

    /// The fallback is for hosts with no bundle id (some test runners).
    private static let subsystem = Bundle.main.bundleIdentifier ?? "ai.liquid.liquid-pipette"

    /// Job run lifecycle, per-cell outcomes and the ops a cell runs.
    static let jobRun = AppLog("jobRun")

    static let mlx = AppLog("mlx")

    /// `max_memory_usage` diagnostics: the settled floor before a sample and its peak.
    static let memory = AppLog("memory")

    /// Benchmark definition sync and pruning.
    static let benchmarkSync = AppLog("benchmarkSync")

    /// A `vl_spec_decode` corpus: its download, digest verdict and lease cleanup.
    static let corpus = AppLog("corpus")

    /// On-disk storage, downloads and manifest decoding.
    static let storage = AppLog("storage")

    /// Local results: payload parsing, display, CSV export and refresh.
    static let results = AppLog("results")

    static let resultUploader = AppLog("resultUploader")

    /// Device-profile and capability reporting, which the planner matches jobs against.
    static let profile = AppLog("profile")

    /// Sign-in, client registration and identity export/import.
    static let auth = AppLog("auth")

    /// Apple Foundation Models runtime: prompt shaping and per-sample timings.
    static let afm = AppLog("afm")

    /// Vault fetches: the signed record and content download for a model a plan names
    /// by vault path.
    static let vault = AppLog("vault")

    func debug(_ message: String) {
        logger.debug("\(message, privacy: .public)")
        breadcrumb(message, .debug)
        mirrorHeadless(message, "debug", verboseOnly: true)
    }

    func info(_ message: String) {
        logger.info("\(message, privacy: .public)")
        breadcrumb(message, .info)
        mirrorHeadless(message, "info")
    }

    func warning(_ message: String) {
        logger.warning("\(message, privacy: .public)")
        breadcrumb(message, .warning)
        mirrorHeadless(message, "warning")
    }

    func error(_ message: String) {
        logger.error("\(message, privacy: .public)")
        breadcrumb(message, .error)
        mirrorHeadless(message, "error")
    }

    /// `devicectl --console` forwards no unified logging, so a CLI run sees only stdout
    /// and stderr. stderr because stdout is parsed (the `BENCH_DONE` sentinel, `results
    /// show` payloads for `jq`).
    ///
    /// `debug` waits for `-v`: unified logging drops it unless collected, but mirrored it
    /// would always print. `pipette-ios-proxy` reads the level back (`app_log_level` in
    /// `tools/pipette-ios-proxy/src/app/console.rs`); a changed prefix demotes every line
    /// to debug.
    private func mirrorHeadless(_ message: String, _ level: String, verboseOnly: Bool = false) {
        guard HeadlessRunner.isHeadless else { return }
        guard !verboseOnly || HeadlessRunner.isVerbose else { return }
        HeadlessRunner.logDiagnostic("\(level) \(category) \(message)")
    }

    /// Breadcrumbs only surface with a captured event, so logging never sends anything
    /// on its own.
    private func breadcrumb(_ message: String, _ level: SentryLevel) {
        guard SentrySDK.isEnabled else { return }
        let crumb = Breadcrumb(level: level, category: category)
        crumb.message = message
        SentrySDK.addBreadcrumb(crumb)
    }
}
