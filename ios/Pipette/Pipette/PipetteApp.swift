import ClerkKit
import Combine
import SwiftUI
import UIKit

/// Receives system callbacks that SwiftUI scenes don't expose directly —
/// currently just `handleEventsForBackgroundURLSession`, which the system
/// invokes after waking us up to deliver events from the background download
/// session. We stash the completion handler on the coordinator, which calls
/// it from `urlSessionDidFinishEvents`.
final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication,
                     handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        MainActor.assumeIsolated {
            DownloadCoordinator.shared.backgroundCompletionHandler = completionHandler
        }
    }
}

/// Pipette iOS app entry point.
///
/// On launch, signs in with Clerk first, then checks the local device
/// registration and private key before showing the main app.
@main
struct PipetteApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @Environment(\.scenePhase) private var scenePhase
    /// The app's single storage instance, built here at the composition root and
    /// injected into the stores and the SwiftUI environment. Nothing below the
    /// root reaches `FileStorage.production` directly.
    private let storage: Storage
    /// Single app-wide runner — UI jobs and `PlannerWorker` share the busy gate.
    private let jobRunner = JobRunner()
    @State private var isRegistered: Bool
    @State private var jobStore: JobStore
    @State private var modelStore: ModelStore
    @State private var deepLinkRouter = DeepLinkRouter()
    /// Drives the cold-start branded splash. Starts visible and fades out after
    /// a brief hold so the first thing the user sees is the Pipette mark rather
    /// than a blank frame or the auth gate popping in.
    @State private var showSplash = true

    init() {
        let storage = FileStorage.production
        self.storage = storage
        let jobStore = JobStore(storage: storage)
        _isRegistered = State(initialValue: storage.identity.isRegistered)
        _jobStore = State(initialValue: jobStore)
        _modelStore = State(initialValue: ModelStore(storage: storage))

        // Start crash/error reporting first so failures during the rest of launch are caught.
        SentryConfiguration.start()
        // Product analytics. After Sentry so a failure inside PostHog's own setup is reported.
        Analytics.start()
        // Warm the reachability monitor at launch so the per-cell result upload's
        // online/offline gate is accurate before the first benchmark cell finishes.
        NetworkReachability.shared.start()
        storage.cleanupLegacyEdgeEvalsStorage()
        storage.migrateIdentityDirectory()
        // After the migration, so the record is read from where it now lives. A device that
        // registered before the mirror existed gains reinstall protection here.
        storage.identity.backfillRegistrationMirror()
        // Before recovery, which would otherwise re-read every one of them and log again.
        storage.discardUndecodableJobs()
        storage.recoverInterruptedJobs()

        // Bind analytics to the device identity as early as possible so events from this launch
        // attribute to the right device rather than to a fresh anonymous id. Only meaningful once
        // the device has registered; before that the SDK's anonymous id applies.
        if let clientId = storage.identity.getRegistration()?.clientId {
            Analytics.identify(clientId.value)
        }
        Analytics.capture(AnalyticsEvents.appLaunched, [
            // Bare version, no OS name: Android sends a bare `Build.VERSION.RELEASE` under this
            // same key, and `platform` (ios/android) plus `form_factor` (phone/tablet) already carry
            // everything the name would add. One grammar keeps `os_version` groupable across both.
            AnalyticsEvents.osVersion: DeviceProbe.detectOsVersion(),
            AnalyticsEvents.deviceModel: DeviceProbe.detectDeviceName(),
            AnalyticsEvents.chip: DeviceProbe.detectChipModel(),
            AnalyticsEvents.formFactor: DeviceProbe.detectFormFactor(),
        ])
        if let publishableKey = ClerkConfiguration.publishableKey {
            Clerk.configure(publishableKey: publishableKey)
            resetClerkKeychainIfPublishableKeyChanged(publishableKey)
        }
        // Headless (incl. `settings run`) shares this runner/store with the UI.
        HeadlessRunner.startIfRequested(
            storage: storage,
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
            // Hold the splash briefly at cold start, then fade to the app.
            .task {
                try? await Task.sleep(for: .seconds(1.4))
                withAnimation(.easeOut(duration: 0.4)) { showSplash = false }
            }
            .preferredColorScheme(.light)
            // Initial load happens here rather than in the stores' inits so
            // the storage migration and interrupted-job recovery in `init()`
            // above have already run by the time anything is read from disk.
            .onAppear {
                jobStore.reload()
                modelStore.reload()
                // Re-report the device profile + capability set the planner
                // matches jobs against. Its inputs drift between launches (an OS
                // update, a security patch, a build with a different llama.cpp
                // commit), and an unchanged resubmit is a server-side no-op.
                //
                // Only when the planner worker is off: its claim loop opens with
                // the same PATCH plus a `reindex_pending` wait, and reporting from
                // both places would fire two concurrent identical requests at
                // launch. Harmless today (the server voids queue standing only on
                // a real diff), but wasteful, and it would stop being harmless the
                // moment the payload gained a value that drifts within a session —
                // the later PATCH could then relinquish a lease the worker is
                // holding mid-job.
                //
                // Best-effort — a failure here is advisory and must never block
                // the UI or the auth gate. Devices that don't claim work are not
                // matched against this profile anyway; benchmark submissions carry
                // their own `device_*` fields.
                if !LocalStorage.plannerWorkerEnabled {
                    Task {
                        do {
                            try await ProfileReporter.refresh(storage: storage)
                        } catch {
                            AppLog.profile.warning(
                                "device profile refresh failed: \(error.localizedDescription)"
                            )
                        }
                    }
                }
                // Resume the planner claim loop if the user left it enabled.
                if LocalStorage.plannerWorkerEnabled {
                    PlannerWorker.shared.setEnabled(
                        true,
                        storage: storage,
                        jobRunner: jobRunner,
                        jobStore: jobStore
                    )
                }
            }
            // Downloads finish in the background coordinator regardless of
            // which screen is visible; re-scan so every model consumer
            // (Models tab, job wizard, Jobs empty-state gate) sees the file.
            .onChange(of: DownloadCoordinator.shared.completedVersion) { _, _ in
                modelStore.reload()
            }
            // `pipette://run/…` deep links (Shortcuts / MDM / a tapped link):
            // drive the live app's shared controllers — parse, gate on the
            // allow-list, and start work that shows up in the normal Jobs UI.
            .onOpenURL { url in
                deepLinkRouter.handle(url, storage: storage, jobRunner: jobRunner, jobStore: jobStore)
            }
            .deepLinkPresentations(deepLinkRouter)
            .onChange(of: scenePhase) { _, phase in
                // A benchmark can't survive backgrounding: iOS suspends the
                // process within seconds (freezing it mid-measurement, so
                // wall-clock timings span the suspension), and a suspended
                // app holding gigabytes is the first jetsam target. Pause at
                // the next cancellation point instead; the manifest records
                // why so the job page can explain the pause and offer Resume.
                if phase == .background, jobRunner.isRunning {
                    jobRunner.cancel(reason: .background)
                }
                // Planner worker can't run suspended (same jetsam/timing reasons
                // as a local job). Graceful stop: cancel in-flight cell and let
                // the loop submit a retriable failure before exiting; restart
                // when active if the setting is still on.
                if phase == .background, LocalStorage.plannerWorkerEnabled {
                    PlannerWorker.shared.setEnabled(
                        false,
                        storage: storage,
                        jobRunner: jobRunner,
                        jobStore: jobStore,
                        background: true
                    )
                }
                // scenePhase flips to .active on cold launch and on every
                // return to the foreground — both are the moments to re-drive
                // stranded result submissions (a past network error left
                // cells failed with payloads on disk; nothing else retries
                // them globally). The drain skips running jobs and
                // already-synced cells.
                if phase == .active {
                    if LocalStorage.plannerWorkerEnabled {
                        PlannerWorker.shared.setEnabled(
                            true,
                            storage: storage,
                            jobRunner: jobRunner,
                            jobStore: jobStore
                        )
                    }
                    Task {
                        let outcomes = await ResultUploader.shared.drainAll()
                        // The drain persists serverJobIds via LocalStorage
                        // directly; re-read so every job screen sees them.
                        if !outcomes.isEmpty { jobStore.reload() }
                    }
                }
            }
        }
        .environment(\.storage, storage)
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

/// Main tab navigation: Jobs, Models, Settings — with a custom pill tab bar.
struct MainTabView: View {
    @Binding var isRegistered: Bool
    @Environment(DeepLinkRouter.self) private var deepLinkRouter
    @State private var selectedTab: MainTab = .jobs
    @State private var hidesPillTabBar = false
    @State private var pillTabBarHeight: CGFloat = 0
    @State private var keyboardIsVisible = false

    var body: some View {
        TabView(selection: $selectedTab) {
            JobsView(selectedTab: $selectedTab)
                .tag(MainTab.jobs)
                .toolbar(.hidden, for: .tabBar)
                .environment(\.pillTabBarReservedHeight, reservedPillTabBarHeight)
            ModelsView()
                .tag(MainTab.models)
                .toolbar(.hidden, for: .tabBar)
                .environment(\.pillTabBarReservedHeight, reservedPillTabBarHeight)
            SettingsView(isRegistered: $isRegistered)
                .tag(MainTab.settings)
                .toolbar(.hidden, for: .tabBar)
                .environment(\.pillTabBarReservedHeight, reservedPillTabBarHeight)
        }
        .overlay(alignment: .bottom) {
            ZStack {
                if !hidesPillTabBar && !keyboardIsVisible {
                    PillTabBar(selected: $selectedTab)
                        .padding(.horizontal, 20)
                        .padding(.top, 8)
                        .padding(.bottom, 4)
                        .frame(maxWidth: .infinity)
                        .readPillTabBarHeight()
                        .transition(.offset(y: 100).combined(with: .opacity))
                }
            }
            .animation(.snappy(duration: 0.2, extraBounce: 0.1), value: [hidesPillTabBar, keyboardIsVisible])
        }
        .onPreferenceChange(PillTabBarHiddenPreferenceKey.self) { hidden in
            hidesPillTabBar = hidden
        }
        // A deep link asks for a tab by setting `requestedTab`; consume and clear
        // it so a later link to the same tab still fires.
        .onChange(of: deepLinkRouter.requestedTab) { _, tab in
            guard let tab else { return }
            selectedTab = tab
            deepLinkRouter.requestedTab = nil
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIResponder.keyboardWillShowNotification
        )) { _ in
            withAnimation(.easeOut(duration: 0.25)) { keyboardIsVisible = true }
        }
        .onReceive(NotificationCenter.default.publisher(
            for: UIResponder.keyboardWillHideNotification
        )) { _ in
            withAnimation(.easeOut(duration: 0.25)) { keyboardIsVisible = false }
        }
        .onPreferenceChange(PillTabBarHeightPreferenceKey.self) { height in
            pillTabBarHeight = height
        }
    }

    private var reservedPillTabBarHeight: CGFloat {
        (hidesPillTabBar || keyboardIsVisible) ? 0 : pillTabBarHeight
    }
}

private struct PillTabBarHeightPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

private extension View {
    func readPillTabBarHeight() -> some View {
        background {
            GeometryReader { proxy in
                Color.clear.preference(
                    key: PillTabBarHeightPreferenceKey.self,
                    value: proxy.size.height
                )
            }
        }
    }
}

private struct PillTabBarReservedHeightKey: EnvironmentKey {
    static let defaultValue: CGFloat = 0
}

extension EnvironmentValues {
    var pillTabBarReservedHeight: CGFloat {
        get { self[PillTabBarReservedHeightKey.self] }
        set { self[PillTabBarReservedHeightKey.self] = newValue }
    }
}

struct PillTabBarHiddenPreferenceKey: PreferenceKey {
    static var defaultValue = false

    static func reduce(value: inout Bool, nextValue: () -> Bool) {
        value = value || nextValue()
    }
}

enum MainTab: Hashable {
    case jobs
    case models
    case settings

    var iconName: String {
        switch self {
        case .jobs:     return "list.bullet.clipboard"
        case .models:   return "cube"
        case .settings: return "gearshape"
        }
    }

    var accessibilityLabel: String {
        switch self {
        case .jobs:     return "Jobs"
        case .models:   return "Models"
        case .settings: return "Settings"
        }
    }
}

/// Pill-shaped bottom tab bar with one black-filled active item.
struct PillTabBar: View {
    @Binding var selected: MainTab

    private let tabs: [MainTab] = [.jobs, .models, .settings]

    @Namespace private var selection

    var body: some View {
        HStack(spacing: 8) {
            ForEach(tabs, id: \.self) { tab in
                Button {
                    withAnimation(.snappy(duration: 0.2, extraBounce: 0.1)) {
                        selected = tab
                    }
                } label: {
                    ZStack {
                        if selected == tab {
                            Capsule()
                                .fill(Color.primary)
                                .matchedGeometryEffect(id: "selection", in: selection)
                        }
                        Image(systemName: tab.iconName)
                            .font(.system(size: 20, weight: .regular))
                            .foregroundStyle(
                                selected == tab
                                    ? Color(.systemBackground)
                                    : Color.secondary
                            )
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(tab.accessibilityLabel)
                .accessibilityAddTraits(selected == tab ? .isSelected : [])
            }
        }
        .padding(6)
        .background(
            Capsule()
                .fill(.clear)
                .glassEffect(.regular.interactive(), in: .capsule)
//                .fill(.ultraThinMaterial)
//                // A light background tint keeps the icons legible over busy
//                // content without turning the bar opaque — the glass still shows
//                // scrolling content through it.
//                .overlay {
//                    Capsule()
//                        .fill(Color(.systemBackground).opacity(0.15))
//                }
//                .overlay {
//                    Capsule()
//                        .strokeBorder(Color.primary.opacity(0.10), lineWidth: 1)
//                }
//                // Composite first so the 80% opacity applies to the whole glass
//                // element uniformly rather than compounding across the layers.
//                .compositingGroup()
//                .opacity(0.8)
//                .shadow(color: Color.black.opacity(0.08), radius: 12, y: 4)
        )
    }
}
