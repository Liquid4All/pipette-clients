import Foundation
import os
import Sentry

/// Centralized diagnostic logging. Each call fans out to three sinks:
///
/// 1. Apple's unified logging (`os.Logger`) — one subsystem (the bundle id) with
///    a per-area `category`, so logs filter in Console.app / `log stream` by
///    `subsystem:ai.liquid.liquid-pipette category:<area>` and by level. Replaces
///    the old `print("[Tag] …")` convention, which only reached the Xcode console.
/// 2. Sentry — a breadcrumb (when the SDK is running) so the same diagnostic
///    trail rides along with any crash report. No-op when Sentry is disabled
///    (no DSN), so dev builds pay nothing.
/// 3. **stderr, headless runs only** — see `mirrorHeadless`.
///
/// Messages are plain interpolated `String`s, logged `public` (these are
/// non-sensitive diagnostics — benchmark ids, model names, counts, error
/// descriptions — matching the previous `print` visibility). Keep anything
/// user-identifying out of them: the whole message is visible in Console.app and
/// in Sentry. `HeadlessRunner` keeps `print("[HEADLESS] …")` for the CLI's own
/// records: that stdout is a `devicectl --console` contract, not diagnostics.
///
/// `nonisolated` (and therefore `Sendable` — it holds only a `Logger` + `String`):
/// logging must be callable from the nonisolated data layer and the detached executor
/// tasks without hopping to the main actor. `os.Logger` and the Sentry SDK are both
/// thread-safe.
nonisolated struct AppLog {
    private let logger: Logger
    private let category: String

    private init(_ category: String) {
        self.logger = Logger(subsystem: Self.subsystem, category: category)
        self.category = category
    }

    /// The unified-logging subsystem — the bundle id, with a literal fallback for
    /// hosts where the Info.plist value is unavailable (e.g. some test runners).
    private static let subsystem = Bundle.main.bundleIdentifier ?? "ai.liquid.liquid-pipette"

    /// Job run lifecycle and per-cell outcomes (`JobExecutor`).
    static let jobRun = AppLog("jobRun")

    /// MLX runtime memory instrumentation (`MLXRuntime`).
    static let mlx = AppLog("mlx")

    /// `max_memory_usage` measurement diagnostics shared by both runtimes — the
    /// settled floor before a sample and the peak it reached (`ProcessMemory`,
    /// `LlamaBenchmark.maxMemory`, `MLXRuntime.maxMemory`).
    static let memory = AppLog("memory")

    /// Benchmark definition sync and pruning (`BenchmarkSync`, `BenchmarkStore`,
    /// `RegistrationService`, settings pull-to-refresh).
    static let benchmarkSync = AppLog("benchmarkSync")

    /// Local on-disk storage maintenance and manifest decoding (`LocalStorage`).
    static let storage = AppLog("storage")

    /// Result payload/metric parsing for display and CSV export
    /// (`CompletedResultsCSVExporter`, `CellDetailView`).
    static let results = AppLog("results")

    /// Result submission/upload outcomes (`ResultUploader`).
    static let resultUploader = AppLog("resultUploader")

    /// Device-profile and capability reporting to the management server
    /// (`ProfileReporter`) — the input the planner matches jobs against.
    static let profile = AppLog("profile")

    /// Sign-in states the gate can't act on (`EmailAuthModel`) — the strategies
    /// behind an unanswerable challenge, which never reach the screen.
    static let auth = AppLog("auth")

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

    /// Repeat the line on stderr when the process was launched with `headlessrun`.
    ///
    /// A CLI run sees stdout and stderr and nothing else — `devicectl --console`
    /// forwards no unified logging — so without this the per-cell results and the
    /// thermal-gate progress are unreachable from the one place a CLI user is looking.
    ///
    /// stderr rather than stdout because stdout is parsed: `pipette-plan` scans it
    /// for `BENCH_DONE` and `results show` writes a payload there for `jq`. Its
    /// `run_streaming_scanning` pipes stderr to the operator's terminal while
    /// scanning stdout alone, so diagnostics reach a human without entering the
    /// contract.
    ///
    /// `verboseOnly` holds `debug` back until `-v`: unified logging drops it unless
    /// someone is collecting, and mirroring makes it unconditional output instead.
    /// The level is carried into the line because an error and a debug are otherwise
    /// indistinguishable once they are both plain stderr text.
    private func mirrorHeadless(_ message: String, _ level: String, verboseOnly: Bool = false) {
        guard HeadlessRunner.isHeadless else { return }
        guard !verboseOnly || HeadlessRunner.isVerbose else { return }
        HeadlessRunner.logDiagnostic("\(level) \(category) \(message)")
    }

    /// Leave a Sentry breadcrumb so the trail is attached to the next event /
    /// crash. Gated on `SentrySDK.isEnabled` so nothing is allocated when Sentry
    /// is off; breadcrumbs only surface if an event is captured, so logging never
    /// *sends* anything on its own.
    private func breadcrumb(_ message: String, _ level: SentryLevel) {
        guard SentrySDK.isEnabled else { return }
        let crumb = Breadcrumb(level: level, category: category)
        crumb.message = message
        SentrySDK.addBreadcrumb(crumb)
    }
}
