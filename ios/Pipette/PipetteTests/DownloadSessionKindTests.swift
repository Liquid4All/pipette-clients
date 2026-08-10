import Foundation
import Testing

@testable import Pipette

/// Which `URLSession` a download runs on, which decides its throughput.
///
/// A background session is scheduled discretionarily by iOS and throttles hard
/// on a phone warm from benchmarking — measured 373 KB/s at 73°C against 23 MB/s
/// available on the same link. Picking the wrong one is invisible until a cell
/// fails on "took too long", so both halves of the choice are pinned here.
@Suite("download session kind")
struct DownloadSessionKindTests {
    @Test("a one-shot headless cell downloads on a foreground session", arguments: [
        ["Pipette", "headlessrun", "benchmarks", "run", "benchmark=decode_throughput_256"],
        ["Pipette", "headlessrun", "bench", "benchmarks=decode_throughput_256"],
        ["Pipette", "headlessrun", "models", "pull", "model={}"],
        ["Pipette", "headlessrun", "sync"],
    ])
    func oneShotUsesForeground(_ argv: [String]) {
        #expect(DownloadCoordinator.usesForegroundSession(argv))
    }

    /// `settings run` leaves the app resident until killed, so it can be
    /// suspended between claims — the one headless verb that still needs a
    /// background session to outlive suspension.
    @Test("the resident planner worker keeps the background session")
    func plannerWorkerStaysBackground() {
        #expect(!DownloadCoordinator.usesForegroundSession(
            ["Pipette", "headlessrun", "settings", "run"]))
    }

    /// `settings` alone is not the worker, and `run` after another verb is a
    /// different command — neither should be mistaken for it.
    @Test("only settings-run is excluded, not every verb containing run")
    func excludesOnlyTheWorkerVerb() {
        #expect(DownloadCoordinator.usesForegroundSession(
            ["Pipette", "headlessrun", "settings", "show"]))
        #expect(DownloadCoordinator.usesForegroundSession(
            ["Pipette", "headlessrun", "benchmarks", "run"]))
    }

    /// No `headlessrun` means the interactive app or a deep link — someone is
    /// holding the phone and may background it mid-download.
    @Test("an interactive launch keeps the background session")
    func interactiveStaysBackground() {
        #expect(!DownloadCoordinator.usesForegroundSession(["Pipette"]))
        #expect(!DownloadCoordinator.usesForegroundSession(["Pipette", "bench", "benchmarks=x"]))
    }

    /// The identifier is what makes a session background: with one, iOS hands
    /// the transfer to `nsurlsessiond`; without, it runs in-process.
    @Test("the foreground configuration is not a background session")
    func foregroundConfigurationHasNoIdentifier() {
        #expect(DownloadCoordinator.sessionConfiguration(foreground: true).identifier == nil)
        #expect(DownloadCoordinator.sessionConfiguration(foreground: false).identifier != nil)
    }

    /// `sessionSendsLaunchEvents` asks the system to relaunch the app to deliver
    /// completions — meaningful only for a background session.
    @Test("launch events are requested only for the background session")
    func launchEventsOnlyWhenBackground() {
        #expect(!DownloadCoordinator.sessionConfiguration(foreground: true).sessionSendsLaunchEvents)
        #expect(DownloadCoordinator.sessionConfiguration(foreground: false).sessionSendsLaunchEvents)
    }

    /// Both kinds keep the settings the transfer needs regardless of scheduling.
    @Test("shared settings survive the branch", arguments: [true, false])
    func sharedSettingsApplyToBoth(_ foreground: Bool) {
        let config = DownloadCoordinator.sessionConfiguration(foreground: foreground)
        #expect(config.allowsCellularAccess)
        #expect(config.waitsForConnectivity)
    }
}
