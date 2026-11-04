import ClerkKit
import SwiftUI
import UIKit

/// For the system callback SwiftUI scenes don't expose: the background download session's
/// completion handler, which the coordinator calls from `urlSessionDidFinishEvents`.
final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        MainActor.assumeIsolated {
            DownloadCoordinator.shared.backgroundCompletionHandler = completionHandler
        }
    }
}

@main
struct PipetteApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @Environment(\.scenePhase) private var scenePhase
    private let ws: PipetteWorkspace
    /// One app-wide runner, so UI jobs and `PlannerWorker` share the busy gate.
    private let jobRunner = JobRunner()
    @State private var isRegistered: Bool
    @State private var jobStore: JobStore
    @State private var modelStore: ModelStore
    @State private var deepLinkRouter = DeepLinkRouter()
    /// So the first frame is the Pipette mark rather than a blank screen or the auth gate.
    @State private var showSplash = true

    init() {
        let ws = PipetteWorkspace.production
        self.ws = ws
        let jobStore = JobStore(ws: ws)
        _isRegistered = State(initialValue: ws.registration.isRegistered)
        _jobStore = State(initialValue: jobStore)
        _modelStore = State(initialValue: ModelStore(ws: ws))

        // First, so failures during the rest of launch are caught.
        SentryConfiguration.start()
        // After Sentry, so a failure inside PostHog's own setup is reported.
        Analytics.start()
        // Warm now so the result upload's online/offline gate is accurate by the first cell.
        NetworkReachability.shared.start()
        ws.cleanupLegacyEdgeEvalsStorage()
        ws.migrateIdentityDirectory()
        // After the migration, so the directory it creates is covered too.
        ws.excludeIdentityFromBackup()
        // After the migration, so both halves are read from where they now live.
        ws.registration.backfillRegistrationMirror()
        ws.registration.backfillPrivateKeyMirror()
        // Before recovery, which would otherwise re-read every undecodable job and log again.
        // Once per build: it decodes every manifest, and only a new build can make one undecodable.
        if LocalStorage.lastUndecodableJobSweepBuild != Bundle.main.appVersionDisplayString {
            ws.discardUndecodableJobs()
            LocalStorage.lastUndecodableJobSweepBuild = Bundle.main.appVersionDisplayString
        }
        ws.recoverInterruptedJobs()

        // Early, so this launch's events attribute to the device rather than an anonymous id.
        if let clientId = ws.registration.get()?.clientId {
            Analytics.identify(clientId.value)
        }
        Analytics.capture(AnalyticsEvents.appLaunched, [
            // Bare version, no OS name: Android sends a bare `Build.VERSION.RELEASE` under this
            // key, so `os_version` groups across both platforms.
            AnalyticsEvents.osVersion: detectOsVersion(),
            AnalyticsEvents.deviceModel: detectDeviceName(),
            AnalyticsEvents.chip: detectChipModel(),
            AnalyticsEvents.formFactor: detectFormFactor(),
        ])
        if let publishableKey = ClerkConfiguration.publishableKey {
            Clerk.configure(publishableKey: publishableKey)
            resetClerkKeychainIfPublishableKeyChanged(publishableKey)
        }
        // Headless (incl. `settings run`) shares this runner/store with the UI.
        HeadlessRunner.startIfRequested(
            ws: ws,
            jobRunner: jobRunner,
            jobStore: jobStore
        )
    }

    var body: some Scene {
        WindowGroup {
            ZStack {
                Group {
                    if !ClerkConfiguration.isComplete {
                        ClerkConfigurationErrorView()
                    } else {
                        ClerkAuthGateView(isRegistered: $isRegistered)
                            .environment(Clerk.shared)
                    }
                }

                if showSplash {
                    SplashView()
                        .transition(.opacity)
                        .zIndex(1)
                }
            }
            .task {
                try? await Task.sleep(for: .seconds(1.4))
                withAnimation(.easeOut(duration: 0.4)) { showSplash = false }
            }
            .preferredColorScheme(.light)
            // Here rather than in `ModelStore.init`, so the migrations in `init()` have run
            // before anything is read from disk. `JobStore` reads disk on first use.
            .onAppear {
                modelStore.reload()
            }
            // Downloads finish whichever screen is visible; re-scan so every model consumer sees
            // the file.
            .onChange(of: DownloadCoordinator.shared.completedVersion) { _, _ in
                modelStore.reload()
            }
            .onOpenURL { url in
                deepLinkRouter.handle(url, ws: ws, jobRunner: jobRunner, jobStore: jobStore)
            }
            .deepLinkPresentations(deepLinkRouter)
            .onChange(of: scenePhase) { _, phase in
                // iOS suspends a backgrounded app within seconds (so wall-clock timings would
                // span the suspension), and a suspended app holding gigabytes is the first
                // jetsam target. Pause instead; the manifest records why so the job page can
                // offer Resume.
                if phase == .background, jobRunner.isRunning {
                    jobRunner.cancel(reason: .background)
                }
                // Same reasons. Nothing restarts it: the toggle shows off, and a
                // `headlessrun worker` session ends there (`outcome=stopped`).
                if phase == .background, PlannerWorker.shared.isRunning {
                    PlannerWorker.shared.setEnabled(
                        false,
                        ws: ws,
                        jobRunner: jobRunner,
                        jobStore: jobStore,
                        background: true
                    )
                }
                // Cold launch and every return to the foreground re-drive stranded result
                // submissions; nothing else retries them globally. A one-shot headless command
                // does not: `pipette-ios-proxy` holds a lease as this client while it runs.
                if phase == .active, !HeadlessRunner.isOneShot {
                    Task {
                        let outcomes = await ResultUploader.shared.drainAll()
                        // The drain writes `serverJobId`s to disk; re-read so job screens see them.
                        if !outcomes.isEmpty { jobStore.reload() }
                    }
                }
            }
        }
        .environment(\.ws, ws)
        .environment(jobRunner)
        .environment(jobStore)
        .environment(modelStore)
        .environment(deepLinkRouter)
        .environment(DownloadCoordinator.shared)
    }

    private func resetClerkKeychainIfPublishableKeyChanged(_ publishableKey: String) {
        let defaultsKey = "ai.liquid.pipette.lastClerkPublishableKey"
        let defaults = UserDefaults.standard

        guard defaults.string(forKey: defaultsKey) != publishableKey else {
            return
        }

        Clerk.clearAllKeychainItems()
        defaults.set(publishableKey, forKey: defaultsKey)
    }
}
