import ClerkKit
import SwiftUI

/// Settings view showing device registration, privacy defaults, and local
/// device maintenance actions.
struct SettingsView: View {
    /// A substitute auth layer. See `ClerkAuthGateView` for why it has no default.
    private let injectedAuth: ClerkAuthenticating?

    @Binding var isRegistered: Bool

    init(isRegistered: Binding<Bool>, auth: ClerkAuthenticating? = nil) {
        self._isRegistered = isRegistered
        self.injectedAuth = auth
    }

    @Environment(Clerk.self) private var clerk
    @Environment(\.ws) private var ws
    @Environment(\.pillTabBarReservedHeight) private var pillTabBarReservedHeight
    @State private var registration: IdentityRegistration?
    // On unless the user turns it off. The payload is performance data only.
    @AppStorage(LocalStorage.defaultContributeResultsKey) private var defaultContributeResults = true
    /// The analytics opt-out, inverted so "on" is permissive like every other toggle here.
    /// Re-seeded in `.onAppear`; nothing outside this view changes it while it is on screen.
    @State private var shareAnalytics = !Analytics.isOptedOut
    @Environment(JobRunner.self) private var jobRunner
    @Environment(JobStore.self) private var jobStore
    @Environment(ModelStore.self) private var modelStore
    @Environment(DownloadCoordinator.self) private var downloadCoordinator
    private var plannerWorker: PlannerWorker { PlannerWorker.shared }
    /// Walked on appear and after a sweep: it is filesystem work, not published state.
    @State private var storageUsedBytes: Int64 = 0
    /// State, not a store read-through, so the row re-renders the instant the limit
    /// changes and no body pass reads `settings.json`.
    @State private var storageQuotaBytes: Int64 = 0
    /// Resolved on appear: `defaultStorageQuotaBytes` stats the volume.
    @State private var defaultQuotaBytes: Int64 = 0
    @State private var showSignOutAlert = false
    /// Results the sign-out is about to delete, counted when the button is tapped.
    @State private var pendingResultsAtSignOut = 0
    @State private var showDeleteAccountAlert = false
    /// Presents `EmailCodeSignInView` for a device the gate admitted on its local identity.
    @State private var showSignIn = false
    @State private var showResetDataAlert = false
    @State private var activeAlert: SettingsAlert?
    @State private var showFeedback = false

    /// The one error alert the Settings screen can show at a time.
    private enum SettingsAlert: Identifiable {
        case resetFailed(String)
        case signOutFailed(String)
        case syncFailed(String)
        case limitFailed(String)

        var id: String {
            switch self {
            case .resetFailed: return "resetFailed"
            case .signOutFailed: return "signOutFailed"
            case .syncFailed: return "syncFailed"
            case .limitFailed: return "limitFailed"
            }
        }

        var title: String {
            switch self {
            case .resetFailed: return "Reset Failed"
            case .signOutFailed: return "Sign Out Failed"
            case .syncFailed: return "Benchmark Sync Failed"
            case .limitFailed: return "Couldn't Change Limit"
            }
        }

        var message: String {
            switch self {
            case .resetFailed(let message): return message
            case .signOutFailed(let message): return message
            case .syncFailed(let message): return "\(message) Please try again later."
            case .limitFailed(let message): return message
            }
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    Text("Settings")
                        .pageHeaderLarge()
                        .foregroundStyle(.primary)
                        .padding(.bottom, 28)

                    sectionTitle("Account")

                    accountCard
                        .padding(.top, 16)

                    // A device admitted without a session has no Clerk account to delete.
                    if isSignedIn {
                        signOutButton
                            .padding(.top, 17)

                        deleteAccountButton
                            .padding(.top, 8)
                    } else {
                        signInButton
                            .padding(.top, 17)

                        signOutButton
                            .padding(.top, 8)
                    }

                    if canSubmitResults {
                        contributionDefaultToggle
                            .padding(.top, 34)

                        plannerWorkerToggle
                            .padding(.top, 24)
                    }

                    // Not gated on `canSubmitResults`: analytics start at launch, before
                    // registration, so the control that stops them must be reachable then too.
                    if Analytics.isAvailable {
                        analyticsToggle
                            .padding(.top, canSubmitResults ? 24 : 34)
                    }

                    sectionTitle("Device")
                        .padding(.top, 46)

                    thermalStateCard
                        .padding(.top, 19)

                    storageCard
                        .padding(.top, 19)

                    freeUpSpaceButton
                        .padding(.top, 19)

                    resetDataButton
                        .padding(.top, 34)

                    sectionTitle("About")
                        .padding(.top, 46)

                    licensesCard
                        .padding(.top, 19)

                    // Only when a Sentry DSN is configured, as on Android.
                    if SentryConfiguration.isEnabled {
                        sectionTitle("Feedback")
                            .padding(.top, 46)

                        feedbackButton
                            .padding(.top, 19)
                    }

                    // Compiled out rather than hidden: the card carries the Clerk session id,
                    // client id and data-root path, which do not belong in an App Store
                    // binary. `PIPETTE_DEBUG_UI` is set by the Internal Testing Xcode Cloud
                    // workflow (docs/pipette-ios/build.md).
                    #if DEBUG || PIPETTE_DEBUG_UI
                    sectionTitle("Debugging")
                        .padding(.top, 46)

                    debugInfoCard
                        .padding(.top, 19)
                    #endif
                }
                .padding(.horizontal, 24)
                .padding(.top, 12)
                .padding(.bottom, 36 + pillTabBarReservedHeight)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(.systemBackground))
            .navigationBarHidden(true)
            .onAppear {
                registration = ws.registration.get()
                storageUsedBytes = ws.storageUsageBytes()
                storageQuotaBytes = ws.storageQuotaBytes
                defaultQuotaBytes = ws.defaultStorageQuotaBytes
                if !canSubmitResults {
                    defaultContributeResults = false
                }
                // The `@State` initializer may run before `Analytics.start()` swaps in the
                // real sink.
                shareAnalytics = !Analytics.isOptedOut
            }
            .refreshable {
                await syncBenchmarkDefinitions()
            }
            // A device with no session is not signing out of anything.
            .alert(isSignedIn ? "Sign Out?" : "Reset Device?", isPresented: $showSignOutAlert) {
                Button("Cancel", role: .cancel) {}
                Button(isSignedIn ? "Sign Out" : "Reset", role: .destructive) {
                    signOutEverywhere()
                }
            } message: {
                Text(signOutConfirmMessage)
            }
            .alert("Delete Account?", isPresented: $showDeleteAccountAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Delete Account", role: .destructive) {
                    deleteAccount()
                }
            } message: {
                Text("This signs out of Clerk and deletes the account, current device registration, private key, and your saved Hugging Face token. You will need to sign in and register this device again.")
            }
            .alert("Reset Data on This Device?", isPresented: $showResetDataAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Reset", role: .destructive) {
                    resetDataOnDevice()
                }
            } message: {
                Text("This deletes local jobs, benchmark results, and downloaded models. Your device identity is kept.")
            }
            .alert(
                activeAlert?.title ?? "",
                isPresented: Binding<Bool>(
                    get: { activeAlert != nil },
                    set: { if !$0 { activeAlert = nil } }
                ),
                presenting: activeAlert
            ) { _ in
                Button("OK", role: .cancel) {}
            } message: { alert in
                Text(alert.message)
            }
            .sheet(isPresented: $showFeedback) {
                FeedbackView(defaultEmail: accountEmail)
            }
            // Sign-in success sets `clerk.user`, which dismisses the sheet; the gate then
            // links this device's registration to the account.
            .sheet(isPresented: $showSignIn) {
                EmailCodeSignInView()
            }
            .onChange(of: clerk.user?.id) { _, newValue in
                if newValue != nil {
                    showSignIn = false
                }
            }
        }
    }

    private var accountCard: some View {
        SettingsCard(cornerRadius: 23) {
            stackedRow(title: "Email", value: displayValue(accountEmail))
            SettingsDivider()
            stackedRow(title: "Organization", value: displayValue(registration?.organization))
            SettingsDivider()
            stackedRow(title: "Collector", value: effectiveCollectorDescription)
            SettingsDivider()
            stackedRow(title: "Registered", value: registeredDateDescription)
        }
    }

    /// Links the existing registration rather than creating one, so the `clientId`, signing
    /// key and submitted results are kept.
    private var signInButton: some View {
        Button {
            showSignIn = true
        } label: {
            HStack(spacing: 13) {
                Image(systemName: "person.crop.circle.badge.plus")
                    .font(.system(size: 17, weight: .regular))
                Text("Sign in")
                    .font(.system(size: 17, weight: .regular))
            }
            .foregroundStyle(Color(.systemBackground))
            .frame(maxWidth: .infinity)
            .frame(height: 43)
            .background(Color.primary, in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Sign in")
    }

    /// `signOutEverywhere` ends the Clerk session when there is one and resets the device
    /// either way, so the label names the outcome the device actually gets.
    private var signOutButton: some View {
        Button(role: .destructive) {
            // Counted once, here: see `signOutConfirmMessage`. From disk: the store holds
            // summaries, not cells.
            pendingResultsAtSignOut = ws.results.deletableResultCount(across: ws.loadAllJobManifests())
            showSignOutAlert = true
        } label: {
            HStack(spacing: 13) {
                Image(systemName: isSignedIn
                    ? "rectangle.portrait.and.arrow.right"
                    : "arrow.counterclockwise")
                    .font(.system(size: 17, weight: .regular))
                Text(isSignedIn ? "Sign out" : "Reset device")
                    .font(.system(size: 17, weight: .regular))
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .frame(height: 43)
            .background(Color(red: 0.91, green: 0.12, blue: 0.14), in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(isSignedIn ? "Sign out" : "Reset device")
    }

    @ViewBuilder
    private var deleteAccountButton: some View {
        Button("Delete Account", role: .destructive) {
            showDeleteAccountAlert = true
        }
        .underline()
        .frame(maxWidth: .infinity)
        .font(.footnote)
        .accessibilityLabel("Delete Account")
    }

    private var contributionDefaultToggle: some View {
        HStack(alignment: .top, spacing: 16) {
            Toggle("", isOn: $defaultContributeResults)
                .labelsHidden()
                .tint(.primary)
                .scaleEffect(0.72, anchor: .leading)
                .frame(width: 38, height: 24, alignment: .leading)
                .padding(.top, 1)
                .accessibilityLabel("Auto-submit benchmark results by default")

            Text("By default, auto-submit benchmark results to the public dataset when jobs finish. Only performance metrics are shared, never personal or device data.")
                .font(.system(size: 15.5, weight: .regular))
                .foregroundStyle(Color.primary.opacity(0.75))
                .lineSpacing(6)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var analyticsToggle: some View {
        HStack(alignment: .top, spacing: 16) {
            Toggle("", isOn: $shareAnalytics)
                .labelsHidden()
                .tint(.primary)
                .scaleEffect(0.72, anchor: .leading)
                .frame(width: 38, height: 24, alignment: .leading)
                .padding(.top, 1)
                .accessibilityLabel("Share anonymous usage analytics")
                .onChange(of: shareAnalytics) { _, on in
                    Analytics.setOptedOut(!on)
                }

            Text("Share anonymous usage analytics: which app features are used and whether benchmark runs succeed. Never your results, prompts, or account details.")
                .font(.system(size: 15.5, weight: .regular))
                .foregroundStyle(Color.primary.opacity(0.75))
                .lineSpacing(6)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var plannerWorkerToggle: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 16) {
                // Not saved: the worker never starts on a launch.
                Toggle("", isOn: Binding(
                    get: { plannerWorker.isEnabled },
                    set: { on in
                        PlannerWorker.shared.setEnabled(
                            on,
                            ws: ws,
                            jobRunner: jobRunner,
                            jobStore: jobStore
                        )
                    }
                ))
                    .labelsHidden()
                    .tint(.primary)
                    .scaleEffect(0.72, anchor: .leading)
                    .frame(width: 38, height: 24, alignment: .leading)
                    .padding(.top, 1)
                    .accessibilityLabel("Planner worker")

                Text(
                    "Planner worker: claim and run jobs from the management server when this device is idle. Requires an approved registration and matching models installed locally."
                )
                .font(.system(size: 15.5, weight: .regular))
                .foregroundStyle(Color.primary.opacity(0.75))
                .lineSpacing(6)
                .fixedSize(horizontal: false, vertical: true)
            }
            if plannerWorker.isEnabled {
                Text(plannerWorker.statusText)
                    .font(.system(size: 13, weight: .regular).monospaced())
                    .foregroundStyle(Color.secondary)
                    .padding(.leading, 54)
            }
        }
    }

    private var canSubmitResults: Bool {
        ResultSubmissionFeatureGate.canSubmitResults(registration: registration)
    }

    private var thermalStateCard: some View {
        SettingsCard(cornerRadius: 23) {
            settingsRow(
                title: "Thermal state",
                value: thermalStateDescription,
                valueColor: thermalStateColor,
                labelWidth: 150,
                valueAlignment: .leading,
                rowHeight: 53
            )
        }
    }

    /// Used / limit, where the row is also the control that sets the limit.
    private var storageCard: some View {
        SettingsCard(cornerRadius: 23) {
            Menu {
                Picker("Model storage limit", selection: limitSelection) {
                    ForEach(limitOptions) { Text($0.title).tag($0.bytes) }
                }
                .pickerStyle(.inline)
            } label: {
                storageRowLabel
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Model storage limit")
            // The label alone would replace the row's contents for VoiceOver, and this is
            // the only place the app reports model storage use.
            .accessibilityValue(storageUsageSummary)

            if let overLimitNotice {
                SettingsDivider()
                overLimitRow(overLimitNotice)
            }
        }
    }

    /// `settingsRow`'s metrics plus the menu affordance; `settingsRow` stays a static
    /// label/value line.
    private var storageRowLabel: some View {
        HStack(spacing: 16) {
            Text("Model storage")
                .font(.system(size: 16.5, weight: .regular))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .frame(width: 150, alignment: .leading)

            // Both sides in binary units: decimal usage against a binary limit could read
            // "17 GB / 16 GB" for a store under its cap.
            Text(storageUsageSummary)
                .font(.system(size: 16.5, weight: .regular))
                .foregroundStyle(Color(.systemGray))
                .lineLimit(1)
                .minimumScaleFactor(0.68)
                .allowsTightening(true)
                .frame(maxWidth: .infinity, alignment: .leading)

            Image(systemName: "chevron.up.chevron.down")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(Color(.systemGray3))
        }
        .padding(.horizontal, 24)
        .frame(height: 53)
        .contentShape(Rectangle())
    }

    private func overLimitRow(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 13) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 15))
                .foregroundStyle(.orange)

            Text(message)
                .font(.system(size: 14, weight: .regular))
                .foregroundStyle(Color.primary.opacity(0.75))
                .lineSpacing(4)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 24)
        .padding(.vertical, 14)
    }

    private var storageUsageSummary: String {
        "\(ByteFormat.storageLimit(storageUsedBytes)) / \(ByteFormat.storageLimit(storageQuotaBytes))"
    }

    private var overLimitNotice: String? {
        StorageLimitOption.overLimitMessage(
            usedBytes: storageUsedBytes, limitBytes: storageQuotaBytes)
    }

    private var limitOptions: [StorageLimitOption] {
        StorageLimitOption.all(default: defaultQuotaBytes)
    }

    private var limitSelection: Binding<Int64> {
        Binding(get: { storageQuotaBytes }, set: { applyLimit($0) })
    }

    /// Sets the limit and nothing else, so lowering it never deletes a model out from
    /// under the user: enforcement stays at collection time and "Free up space".
    ///
    /// Usage is re-walked because the `.onAppear` snapshot goes stale behind a finishing
    /// download, and judging a new limit against it can hide a real overage.
    private func applyLimit(_ bytes: Int64) {
        do {
            try ws.setStorageQuotaBytes(bytes)
            storageQuotaBytes = bytes
            storageUsedBytes = ws.storageUsageBytes()
        } catch {
            // A silent failure would leave the row showing a limit that isn't persisted.
            activeAlert = .limitFailed(error.localizedDescription)
        }
    }

    /// Pins come from the coordinator, not job manifests alone: a running transfer holds
    /// an entry and a hub snapshot that job state says nothing about.
    private var freeUpSpaceButton: some View {
        settingsCapsuleButton(icon: "internaldrive", title: "Free up space") {
            ws.sweepToQuota(pinning: downloadCoordinator.sweepPins(justInstalled: nil))
            storageUsedBytes = ws.storageUsageBytes()
            modelStore.reload()
        }
    }

    private var resetDataButton: some View {
        settingsCapsuleButton(icon: "arrow.counterclockwise", title: "Reset data on this device") {
            showResetDataAlert = true
        }
    }

    private var feedbackButton: some View {
        settingsCapsuleButton(icon: "exclamationmark.bubble", title: "Submit feedback") {
            showFeedback = true
        }
    }

    /// Shared by the secondary Settings actions so their styling can't drift apart.
    private func settingsCapsuleButton(
        icon: String,
        title: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 13) {
                Image(systemName: icon)
                    .font(.system(size: 17, weight: .regular))
                    .foregroundStyle(Color(.systemGray))
                Text(title)
                    .font(.system(size: 17, weight: .regular))
                    .foregroundStyle(.primary)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 42)
            .background(Color(.systemBackground), in: Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(Color(.systemGray4), lineWidth: 1)
            )
            .shadow(color: Color.black.opacity(0.06), radius: 3, y: 1)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(title)
    }

    private var licensesCard: some View {
        SettingsCard(cornerRadius: 23) {
            NavigationLink {
                AcknowledgementsView()
            } label: {
                HStack(spacing: 16) {
                    Text("Open source licenses")
                        .font(.system(size: 16.5, weight: .regular))
                        .foregroundStyle(.primary)

                    Spacer()

                    Image(systemName: "chevron.right")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(Color(.systemGray3))
                }
                .padding(.horizontal, 24)
                .frame(height: 57)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Open source licenses")
        }
    }

    private var debugInfoCard: some View {
        SettingsCard(cornerRadius: 23) {
            ForEach(Array(debugRows.enumerated()), id: \.offset) { item in
                debugRow(title: item.element.title, value: item.element.value)

                if item.offset < debugRows.count - 1 {
                    SettingsDivider()
                }
            }
        }
    }

    private var clerkPrimaryEmail: String? {
        clerk.user?.primaryEmailAddress?.emailAddress
    }

    /// The gate admits a device on its registration and signing key alone, so this
    /// screen is reachable with no session.
    private var isSignedIn: Bool {
        clerk.user != nil
    }

    /// Reads `pendingResultsAtSignOut`, captured on tap, so the figure cannot change while
    /// the user reads a destructive prompt. It counts what the reset destroys, including the
    /// `local/` half the submit button ignores.
    private var signOutConfirmMessage: String {
        let base = isSignedIn
            ? "This signs out of Clerk and deletes this device's registration and "
                + "private key, along with every local job, benchmark result, downloaded "
                + "model, and your saved Hugging Face token. You will need to sign in and "
                + "register this device again."
            : "This deletes this device's registration and private key, along with every "
                + "local job, benchmark result, downloaded model, and your saved Hugging "
                + "Face token. You will need to sign in and register this device again."
        let pending = pendingResultsAtSignOut
        guard pending > 0 else { return base }
        let noun = pending == 1 ? "result has" : "results have"
        return base + "\n\n\(pending) \(noun) not been submitted yet and will be permanently deleted."
    }

    private var accountEmail: String? {
        clerkPrimaryEmail ?? registration?.clerkPrimaryEmail ?? registration?.contactEmail
    }

    /// The URL captured at registration, else the production endpoint. Nil only if the
    /// shipped production literal stopped being a valid ``ServerUrl``.
    private var effectiveServerUrl: ServerUrl? {
        CollectorEndpoint.effectiveServerUrl(registered: registration?.serverUrl)
    }

    /// Liquid's own collector by its setup-picker name, anything else verbatim.
    /// `isSameCollector` keeps a URL that differs only by trailing slash or scheme or
    /// host case from reading as a custom deployment.
    private var effectiveCollectorDescription: String {
        let current = effectiveServerUrl?.value ?? CollectorEndpoint.productionURL
        return CollectorEndpoint.isSameCollector(current, as: CollectorEndpoint.productionURL)
            ? CollectorEndpointOption.production.title
            : current
    }

    private var registeredDateDescription: String {
        guard let registeredAt = registration?.registeredAt,
              let registeredDate = JobDateFormat.iso8601.date(from: registeredAt)
        else {
            return displayValue(registration?.registeredAt)
        }
        return registeredDate.formatted(.dateTime.month(.wide).day().year())
    }

    private func sectionTitle(_ title: String) -> some View {
        Text(title)
            .font(.serif(21))
            .foregroundStyle(Color(.systemGray))
    }

    private func settingsRow(
        title: String,
        value: String,
        valueColor: Color = Color(.systemGray),
        labelWidth: CGFloat = 120,
        valueAlignment: Alignment = .trailing,
        rowHeight: CGFloat = 57
    ) -> some View {
        HStack(spacing: 16) {
            Text(title)
                .font(.system(size: 16.5, weight: .regular))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .frame(width: labelWidth, alignment: .leading)

            Text(value)
                .font(.system(size: 16.5, weight: .regular))
                .foregroundStyle(valueColor)
                .lineLimit(1)
                .minimumScaleFactor(0.68)
                .allowsTightening(true)
                .frame(maxWidth: .infinity, alignment: valueAlignment)
        }
        .padding(.horizontal, 24)
        .frame(height: rowHeight)
    }

    /// Caption above value, so long values like the collector URL get the full width.
    private func stackedRow(title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title.uppercased())
                .font(.footnote)
                .foregroundStyle(.secondary)

            Text(value)
                .font(.body)
                .foregroundStyle(.primary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 24)
        .padding(.vertical, 13)
    }

    private func debugRow(title: String, value: String) -> some View {
        HStack(alignment: .top, spacing: 16) {
            Text(title)
                .font(.system(size: 13.5, weight: .medium))
                .foregroundStyle(.primary)
                .lineLimit(2)
                .frame(width: 114, alignment: .leading)

            Text(value)
                .font(.system(size: 12.5, weight: .regular, design: .monospaced))
                .foregroundStyle(Color(.systemGray))
                .textSelection(.enabled)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
    }

    private var debugRows: [(title: String, value: String)] {
        [
            ("Client ID", debugDisplayValue(registration?.clientId.value)),
            // The collector's approval state (`pending` until a pre-auth key or an
            // operator admits it), not the Clerk session.
            ("Status", debugDisplayValue(registration?.status)),
            ("App Version", appVersionDescription),
            ("Thermal", BuildFlavor.thermalDescription),
            ("Memory", BuildFlavor.memoryDescription),
            ("Build Config", buildConfiguration),
            ("Bundle ID", Bundle.main.bundleIdentifier ?? "Unknown"),
            ("Device", detectDeviceName()),
            ("Chip", detectChipModel()),
            ("Form Factor", detectFormFactor().rawValue),
            ("OS", "\(detectOsName()) \(detectOsVersion())"),
            ("RAM", formattedBytes(Int64(detectRamBytes()))),
            ("Thermal", thermalStateDescription),
            ("Auto-submit", canSubmitResults ? (defaultContributeResults ? "Enabled" : "Disabled") : "Unavailable"),
            ("Planner worker", plannerWorker.isEnabled ? plannerWorker.statusText : "Off"),
            ("Jobs", "\(jobStore.jobs.count)"),
            ("Models", "\(ws.availableModels().count)"),
            ("Storage", storageUsageSummary),
            ("Job Folders", "\(directoryItemCount(at: ws.jobsDir))"),
            ("Private Key", ws.registration.getPrivateKey() == nil ? "Missing" : "Present"),
            ("HF Token", KeychainHelper.loadHfToken() == nil ? "Missing" : "Present"),
            ("Clerk User", debugDisplayValue(registration?.clerkUserId)),
            ("Clerk Session", debugDisplayValue(registration?.clerkSessionId)),
            ("Clerk Linked", debugDisplayValue(registration?.clerkLinkedAt)),
            ("Data Root", ws.root.path)
        ]
    }

    private func displayValue(_ value: String?) -> String {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty
        else {
            return "Not set"
        }
        return value
    }

    private func debugDisplayValue(_ value: String?) -> String {
        guard let value = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty
        else {
            return "Unavailable"
        }
        return value
    }

    private var thermalStateDescription: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "Normal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }

    private var thermalStateColor: Color {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return Color(red: 0.12, green: 0.75, blue: 0.32)
        case .fair: return .primary
        case .serious: return .orange
        case .critical: return .red
        @unknown default: return .secondary
        }
    }

    private var appVersionDescription: String {
        Bundle.main.appVersionDisplayString
    }

    private var buildConfiguration: String {
        #if DEBUG
        return "Debug"
        #else
        return "Release"
        #endif
    }

    private func formattedBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB]
        formatter.countStyle = .memory
        return formatter.string(fromByteCount: bytes)
    }

    private func directoryItemCount(at url: URL) -> Int {
        let contents = try? FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        return contents?.count ?? 0
    }

    /// Clear the registration, its signing key, and every Hugging Face credential. Shared by
    /// `signOutEverywhere` and `deleteAccount`. The worker is stopped here because the sign-out
    /// that follows tears down this view and its toggle.
    ///
    /// The HF tokens go too (PIP-459): leaving an account is how a shared device changes hands,
    /// and the next person must not inherit gated-repo access. Both kinds: the one typed into
    /// Settings, and the per-model ones `PlannerWorker.stashClaimCredential` stashes off a claim.
    private func deleteDeviceIdentity() {
        PlannerWorker.shared.setEnabled(false, ws: ws, jobRunner: jobRunner, jobStore: jobStore)
        ws.registration.clear()
        _ = KeychainHelper.deleteHfToken()
        let clearedModelTokens = KeychainHelper.deleteAllModelHfTokens()
        if clearedModelTokens > 0 {
            AppLog.auth.debug("cleared \(clearedModelTokens) stashed model HF token(s)")
        }
        registration = nil
        isRegistered = false
    }

    private func resetDataOnDevice() {
        do {
            try ws.resetDeviceData()
            // Re-walk, or the card would still report the deleted `models/` as usage.
            storageUsedBytes = ws.storageUsageBytes()
        } catch {
            activeAlert = .resetFailed(error.localizedDescription)
        }
    }

    /// Pull-to-refresh re-pulls the benchmark catalog. `GET /benchmarks` is public, so
    /// this works unregistered against the production URL.
    private func syncBenchmarkDefinitions() async {
        AppLog.benchmarkSync.debug("pull-to-refresh requested")
        guard let effectiveServerUrl else {
            AppLog.benchmarkSync.error("pull-to-refresh: no usable collector URL")
            activeAlert = .syncFailed("No collector configured.")
            return
        }
        do {
            _ = try await BenchmarkSyncCoordinator.shared.sync(
                serverUrl: effectiveServerUrl,
                ws: ws
            )
        } catch {
            AppLog.benchmarkSync.error("settings pull-to-refresh failed: \(error)")
            activeAlert = .syncFailed(humanizedRegistrationError(error))
            return
        }
        await resendResultsFromOtherCollectors()
    }

    /// Re-send results submitted to a different collector, limited to benchmarks this one
    /// offers, since it accepts no others. Best-effort: failures are logged, not alerted.
    /// Skips the whole-store scan once results are reconciled to the current collector.
    private func resendResultsFromOtherCollectors() async {
        guard canSubmitResults, let current = effectiveServerUrl else { return }
        // The comparison per-result matching uses, so the two cannot disagree.
        guard !CollectorEndpoint.isSameCollector(LocalStorage.lastReconciledCollector, as: current.value)
        else { return }

        let benchmarkIds = Set(storedDefinitions(store: ws.benchmarks)
            .map(\.benchmarkId.value))
        guard !benchmarkIds.isEmpty else { return }

        let outcome = await ResultUploader.shared.resendForCollectorChange(benchmarkIds: benchmarkIds)
        // Only on a clean sweep, so the next refresh retries what failed.
        if outcome.errors.isEmpty {
            LocalStorage.lastReconciledCollector = current.value
        }
        if outcome.submitted > 0 || !outcome.errors.isEmpty {
            AppLog.resultUploader.info(
                "collector-change resend: \(outcome.submitted) sent, \(outcome.errors.count) failed"
            )
        }
    }

    /// Sign out and reset the device (PIP-459).
    ///
    /// A failed session call does not abort the reset (as in Android's
    /// `ShellViewModel.signOutAndResetDevice`): refusing it offline would leave the user
    /// unable to sign out at all. The session then survives, and there is no local-only way to
    /// end it: `Clerk.clearAllKeychainItems()` leaves in-memory state, so `clerk.user` stays set
    /// and the wiped device lands on `SetupView` as that account. Logged, not alerted: this view
    /// is torn down as soon as the registration goes.
    private func signOutEverywhere() {
        Task {
            if clerk.user != nil || clerk.session != nil {
                do {
                    try await (injectedAuth ?? RealClerkAuth(auth: clerk.auth)).signOut()
                } catch {
                    AppLog.auth.error(
                        "sign-out failed, resetting the device anyway: \(error.localizedDescription)")
                }
            }
            deleteDeviceIdentity()
            await resetLocalDataForSignOut()
        }
    }

    /// Clear what the previous session left on disk: jobs, results, models, and the synced
    /// catalog (PIP-459); `settings.json` stays. Runs after `deleteDeviceIdentity`, so a
    /// failure here cannot leave a signed-out device holding a usable registration.
    ///
    /// Logged rather than alerted: the gate has already torn down this view, so an alert
    /// would never be seen. The store reloads are deferred so they run even when the delete
    /// fails: `jobStore` and `modelStore` outlive this view and would list deleted files.
    private func resetLocalDataForSignOut() async {
        // Requests, not barriers: an in-flight cell or transfer can still write below, but
        // the next one never starts.
        jobRunner.cancel()
        downloadCoordinator.cancelAll()
        // Wait for the runner to honor the cancel: deleting under a live cell lets its write
        // land after the wipe, or makes `removeItem` fail partway through.
        let deadline = Date().addingTimeInterval(Self.runnerStopTimeout)
        while jobRunner.runningJobId != nil, Date() < deadline {
            try? await Task.sleep(for: .milliseconds(100))
        }

        defer {
            jobStore.reload()
            modelStore.reload()
            storageUsedBytes = ws.storageUsageBytes()
        }
        do {
            try ws.resetDeviceData()
        } catch {
            AppLog.storage.error("sign-out reset did not complete: \(error.localizedDescription)")
        }
    }

    /// Bounded so a wedged cell cannot strand the user on a sign-out that never completes.
    private static let runnerStopTimeout: TimeInterval = 10

    private func deleteAccount() {
        Task {
            do {
                if let user = clerk.user {
                    try await user.delete()
                }
                deleteDeviceIdentity()
            } catch {
                activeAlert = .signOutFailed(error.localizedDescription)
            }
        }
    }
}

private struct SettingsCard<Content: View>: View {
    let cornerRadius: CGFloat
    let content: Content

    init(cornerRadius: CGFloat, @ViewBuilder content: () -> Content) {
        self.cornerRadius = cornerRadius
        self.content = content()
    }

    var body: some View {
        VStack(spacing: 0) {
            content
        }
        .appCard(cornerRadius: cornerRadius)
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }
}

private struct SettingsDivider: View {
    var body: some View {
        Divider()
            .background(Color(.systemGray5))
    }
}

#if DEBUG
#Preview("Settings") {
    SettingsView(isRegistered: .constant(true))
        .environment(Clerk.shared)
        .environment(ModelStore(ws: PipetteWorkspace.production))
        // "Free up space" reads the coordinator, so without this the preview traps on tap.
        .environment(DownloadCoordinator.shared)
}
#endif
