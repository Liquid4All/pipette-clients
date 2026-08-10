import ClerkKit
import SwiftUI

/// Settings view showing device registration, privacy defaults, and local
/// device maintenance actions.
struct SettingsView: View {
    /// A substitute auth layer, when something wants to supply one. See
    /// `ClerkAuthGateView` for why this is not resolved to a default here.
    private let injectedAuth: ClerkAuthenticating?

    @Binding var isRegistered: Bool

    init(isRegistered: Binding<Bool>, auth: ClerkAuthenticating? = nil) {
        self._isRegistered = isRegistered
        self.injectedAuth = auth
    }

    @Environment(Clerk.self) private var clerk
    @Environment(\.storage) private var storage
    @Environment(\.pillTabBarReservedHeight) private var pillTabBarReservedHeight
    // Loaded in `.onAppear` from the injected storage.
    @State private var registration: IdentityRegistration?
    // Opt-out default (matches `LocalStorage.defaultContributeResults`): on unless the
    // user turns it off. Performance-only payload, never personal/device data.
    @AppStorage(LocalStorage.defaultContributeResultsKey) private var defaultContributeResults = true
    /// Opt-in: pull and run jobs from the management server planner queue.
    @AppStorage(LocalStorage.plannerWorkerEnabledKey) private var plannerWorkerEnabled = false
    /// Mirror of the analytics opt-out, inverted so "on" is the permissive setting like every other
    /// toggle here. Read through the sink rather than `@AppStorage` so there is one accessor for the
    /// value on both platforms, even though ``LocalStorage/analyticsOptOut`` backs it in
    /// `UserDefaults` either way. Re-seeded in `.onAppear`, which is enough: nothing outside this
    /// view changes it while the view is on screen.
    @State private var shareAnalytics = !Analytics.isOptedOut
    @Environment(JobRunner.self) private var jobRunner
    @Environment(JobStore.self) private var jobStore
    @Environment(ModelStore.self) private var modelStore
    @Environment(DownloadCoordinator.self) private var downloadCoordinator
    /// Shared `@Observable` worker — reading `statusText` in the body tracks updates.
    private var plannerWorker: PlannerWorker { PlannerWorker.shared }
    /// Walked on appear (and after a manual sweep) rather than observed — the walk is
    /// filesystem work, not state the store publishes.
    @State private var storageUsedBytes: Int64 = 0
    /// State rather than a read-through of the store: the row and the over-limit notice
    /// have to re-render the instant the limit changes, and this also drops a
    /// `settings.json` read from every body pass.
    @State private var storageQuotaBytes: Int64 = 0
    /// Resolved on appear — `defaultStorageQuotaBytes` stats the volume.
    @State private var defaultQuotaBytes: Int64 = 0
    @State private var showSignOutAlert = false
    /// Results the sign-out is about to delete, counted when the button is tapped.
    @State private var pendingResultsAtSignOut = 0
    @State private var showDeleteAccountAlert = false
    @State private var showResetDataAlert = false
    @State private var activeAlert: SettingsAlert?
    @State private var showFeedback = false

    /// The one error alert the Settings screen can show at a time, carrying its
    /// message. Collapses the former `resetDataError`/`signOutError`/`syncError`
    /// trio of `String?` fields into a single presented value.
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

                    signOutButton
                        .padding(.top, 17)

                    deleteAccountButton
                        .padding(.top, 8)

                    if canSubmitResults {
                        contributionDefaultToggle
                            .padding(.top, 34)

                        plannerWorkerToggle
                            .padding(.top, 24)
                    }

                    // Not gated on `canSubmitResults`: analytics start at launch, before this
                    // device has registered, so the control that stops them has to be reachable
                    // then too.
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

                    // Shown only when a Sentry DSN is configured (crash reporting active) —
                    // the iOS analogue of the Android `Sentry.isEnabled()` gate.
                    if SentryConfiguration.isEnabled {
                        sectionTitle("Feedback")
                            .padding(.top, 46)

                        feedbackButton
                            .padding(.top, 19)
                    }

                    // Internal-only, and compiled out rather than hidden: this card
                    // carries the Clerk session id, the client id, and the data-root
                    // path, none of which belong in an App Store build even behind a
                    // false branch. `DEBUG` covers local work; `PIPETTE_DEBUG_UI` is
                    // set by the Internal Testing Xcode Cloud workflow (see
                    // docs/pipette-ios/build.md). An App Store archive defines
                    // neither, so the section does not exist in that binary.
                    //
                    // This is the iOS analogue of Android's `if (state.isDebug)` gate
                    // in `SettingsScreen.kt`, which is what kept the same block out of
                    // its release builds.
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
                registration = storage.identity.getRegistration()
                storageUsedBytes = storage.storageUsageBytes()
                storageQuotaBytes = storage.storageQuotaBytes
                defaultQuotaBytes = storage.defaultStorageQuotaBytes
                if !canSubmitResults {
                    defaultContributeResults = false
                }
                // The `@State` initializer only runs the first time this view appears, and
                // `Analytics.start()` may not have swapped in the real sink by then. Re-read so the
                // row shows the SDK's actual state rather than a stale default.
                shareAnalytics = !Analytics.isOptedOut
            }
            .refreshable {
                await syncBenchmarkDefinitions()
            }
            .alert("Sign Out?", isPresented: $showSignOutAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Sign Out", role: .destructive) {
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

    private var signOutButton: some View {
        Button(role: .destructive) {
            // Counted once, here: see `signOutConfirmMessage`.
            pendingResultsAtSignOut = storage.results.deletableResultCount(across: jobStore.jobs)
            showSignOutAlert = true
        } label: {
            HStack(spacing: 13) {
                Image(systemName: "rectangle.portrait.and.arrow.right")
                    .font(.system(size: 17, weight: .regular))
                Text("Sign out")
                    .font(.system(size: 17, weight: .regular))
            }
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity)
            .frame(height: 43)
            .background(Color(red: 0.91, green: 0.12, blue: 0.14), in: Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Sign out")
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
                Toggle("", isOn: $plannerWorkerEnabled)
                    .labelsHidden()
                    .tint(.primary)
                    .scaleEffect(0.72, anchor: .leading)
                    .frame(width: 38, height: 24, alignment: .leading)
                    .padding(.top, 1)
                    .accessibilityLabel("Planner worker")
                    .onChange(of: plannerWorkerEnabled) { _, on in
                        PlannerWorker.shared.setEnabled(
                            on,
                            storage: storage,
                            jobRunner: jobRunner,
                            jobStore: jobStore
                        )
                    }

                Text(
                    "Planner worker: claim and run jobs from the management server when this device is idle. Requires an approved registration and matching models installed locally."
                )
                .font(.system(size: 15.5, weight: .regular))
                .foregroundStyle(Color.primary.opacity(0.75))
                .lineSpacing(6)
                .fixedSize(horizontal: false, vertical: true)
            }
            if plannerWorkerEnabled {
                // Read through the shared `@Observable` instance so status
                // updates invalidate this view as the claim loop progresses.
                Text(plannerWorker.statusText)
                    .font(.system(size: 13, weight: .regular).monospaced())
                    .foregroundStyle(Color.secondary)
                    .padding(.leading, 54)
            }
        }
        .onAppear {
            // Re-sync the daemon if the toggle was already on from a prior launch.
            if plannerWorkerEnabled {
                PlannerWorker.shared.setEnabled(
                    true,
                    storage: storage,
                    jobRunner: jobRunner,
                    jobStore: jobStore
                )
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

    /// Used / limit, where the row is also the control that sets the limit — one row,
    /// one number, no duplicate readout.
    private var storageCard: some View {
        SettingsCard(cornerRadius: 23) {
            Menu {
                // Inline picker inside the menu: the section header names what is being
                // chosen, and the current limit gets a checkmark for free.
                Picker("Model storage limit", selection: limitSelection) {
                    ForEach(limitOptions) { Text($0.title).tag($0.bytes) }
                }
                .pickerStyle(.inline)
            } label: {
                storageRowLabel
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Model storage limit")
            // The label alone would replace the row's contents for VoiceOver, and this
            // row is the only place the app reports how much storage models occupy.
            .accessibilityValue(storageUsageSummary)

            if let overLimitNotice {
                SettingsDivider()
                overLimitRow(overLimitNotice)
            }
        }
    }

    /// `settingsRow`'s metrics, plus the menu's affordance. Hand-rolled rather than
    /// given to `settingsRow`, which stays a static label/value line for its other
    /// callers (the `licensesCard` precedent).
    private var storageRowLabel: some View {
        HStack(spacing: 16) {
            Text("Model storage")
                .font(.system(size: 16.5, weight: .regular))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .frame(width: 150, alignment: .leading)

            // Both sides in binary units. Mixing `fileSize` here with the binary limit
            // rendered a store 180 MB *under* a 16 GiB cap as "17 GB / 16 GB".
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

    /// What the row reports, and the string VoiceOver reads as the row's value.
    private var storageUsageSummary: String {
        "\(ByteFormat.storageLimit(storageUsedBytes)) / \(ByteFormat.storageLimit(storageQuotaBytes))"
    }

    /// Pure in its `@State` inputs, so it re-evaluates on the same pass that changes
    /// them — no disk work to make the disclosure appear or clear.
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

    /// Sets the limit and nothing else. Enforcement stays at collection time and
    /// "Free up space" stays the explicit reclaim, so lowering the limit can never
    /// delete a model out from under the user — it only makes the notice appear.
    ///
    /// Usage is re-walked even though no bytes moved: the *comparison* changed, and the
    /// snapshot from `.onAppear` goes stale while the screen sits open behind a
    /// finishing download. Judging a new limit against it can hide a real overage.
    private func applyLimit(_ bytes: Int64) {
        do {
            try storage.setStorageQuotaBytes(bytes)
            storageQuotaBytes = bytes
            storageUsedBytes = storage.storageUsageBytes()
        } catch {
            // A silent failure would leave the row showing a limit that isn't persisted.
            activeAlert = .limitFailed(error.localizedDescription)
        }
    }

    /// Reclaim now rather than waiting for the next download's sweep. The pin set
    /// comes from the coordinator, not from job manifests alone: a transfer running
    /// behind this screen holds an entry and a hub snapshot that job state says
    /// nothing about, and reclaiming either would destroy a live download.
    private var freeUpSpaceButton: some View {
        settingsCapsuleButton(icon: "internaldrive", title: "Free up space") {
            storage.sweepToQuota(pinning: downloadCoordinator.sweepPins(justInstalled: nil))
            storageUsedBytes = storage.storageUsageBytes()
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

    /// A full-width, hairline-bordered capsule action button (icon + label). Shared by the
    /// secondary Settings actions so their styling can't drift apart.
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

    /// Sign-out confirmation copy. The reset is described unconditionally; the count of
    /// results about to be lost is appended only when there are any, so a device with
    /// nothing pending doesn't read a sentence about zero.
    ///
    /// Reads `pendingResultsAtSignOut`, captured when the button is tapped, rather than counting here,
    /// so the figure is settled before the alert appears and cannot change while the user is reading a
    /// destructive prompt.
    ///
    /// What is captured is what the reset *destroys*, not what could still be submitted:
    /// `resetDeviceData` removes the whole `results/` tree, including the `local/` half the submit
    /// button rightly ignores.
    private var signOutConfirmMessage: String {
        let base = "This signs out of Clerk and deletes this device's registration and "
            + "private key, along with every local job, benchmark result, downloaded "
            + "model, and your saved Hugging Face token. You will need to sign in and "
            + "register this device again."
        let pending = pendingResultsAtSignOut
        guard pending > 0 else { return base }
        let noun = pending == 1 ? "result has" : "results have"
        return base + "\n\n\(pending) \(noun) not been submitted yet and will be permanently deleted."
    }

    private var accountEmail: String? {
        clerkPrimaryEmail ?? registration?.clerkPrimaryEmail ?? registration?.contactEmail
    }

    /// The collector this device actually reports to: the URL captured at
    /// registration, falling back to the production endpoint when unregistered.
    private var effectiveServerURL: ServerURL {
        registration?.serverUrl ?? ServerURL(CollectorEndpoint.productionURL)
    }

    /// How the effective collector reads on the Settings screen: Liquid's own
    /// collector by the name the setup picker used to choose it, anything else
    /// verbatim. `https://collector.pipette.liquid.ai` is an implementation
    /// detail to everyone who never typed it, and every device that left the
    /// picker alone reports there — so the URL was the one value on this card
    /// that named nothing the user had done.
    ///
    /// The title comes from `CollectorEndpointOption.production` rather than a
    /// literal, so this row and the picker cannot drift apart. Matching goes
    /// through `isSameCollector`, so a stored URL that differs only by trailing
    /// slash, scheme case, or host case still reads as Liquid AI instead of
    /// masquerading as a custom deployment.
    private var effectiveCollectorDescription: String {
        CollectorEndpoint.isSameCollector(effectiveServerURL.value, as: CollectorEndpoint.productionURL)
            ? CollectorEndpointOption.production.title
            : effectiveServerURL.value
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

    /// A stacked field cell: a small uppercased caption label above a
    /// body-weight value (the Contacts/grouped-list idiom). The value gets the
    /// full row width, so long values like the collector URL aren't squeezed.
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
            // The collector's approval state for this device (`pending` until a
            // pre-auth key or an operator admits it) — not the Clerk session, which
            // the "Authentication Status" label on the account card implied. Sits
            // next to Client ID because that is the pair it describes, and because
            // Android's debug block orders it exactly here.
            ("Status", debugDisplayValue(registration?.status)),
            ("App Version", appVersionDescription),
            ("Thermal", BuildFlavor.thermalDescription),
            ("Build Config", buildConfiguration),
            ("Bundle ID", Bundle.main.bundleIdentifier ?? "Unknown"),
            ("Device", DeviceProbe.detectDeviceName()),
            ("Chip", DeviceProbe.detectChipModel()),
            ("Form Factor", DeviceProbe.detectFormFactor().rawValue),
            ("OS", "\(DeviceProbe.detectOsName()) \(DeviceProbe.detectOsVersion())"),
            ("RAM", formattedBytes(Int64(DeviceProbe.detectRamBytes()))),
            ("Thermal", thermalStateDescription),
            ("Auto-submit", canSubmitResults ? (defaultContributeResults ? "Enabled" : "Disabled") : "Unavailable"),
            ("Planner worker", plannerWorkerEnabled ? plannerWorker.statusText : "Off"),
            ("Jobs", "\(storage.loadAllJobManifests().count)"),
            ("Models", "\(storage.availableModels().count)"),
            ("Storage", storageUsageSummary),
            ("Job Folders", "\(directoryItemCount(at: storage.jobsDir))"),
            ("Private Key", storage.identity.getPrivateKey() == nil ? "Missing" : "Present"),
            ("HF Token", KeychainHelper.loadHfToken() == nil ? "Missing" : "Present"),
            ("Clerk User", debugDisplayValue(registration?.clerkUserId)),
            ("Clerk Session", debugDisplayValue(registration?.clerkSessionId)),
            ("Clerk Linked", debugDisplayValue(registration?.clerkLinkedAt)),
            ("Data Root", storage.dataRoot.path)
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

    /// Clear the registration, the key that signed submissions under it, and every Hugging Face credential the
    /// session could spend. Shared by `signOutEverywhere` and `deleteAccount`, which is why the planner worker and
    /// the credentials are dealt with here rather than in either of them.
    ///
    /// Switched off, not merely stopped: the flag lives in `UserDefaults`, which `resetDeviceData` does not touch,
    /// and `PipetteApp` starts the worker on launch and on foreground whenever it is set. Left on, the next account
    /// to register this device would silently begin claiming and running server jobs it never opted into. The
    /// toggle's own `onChange` cannot do it, since this view is torn down by the sign-out that follows.
    ///
    /// The Hugging Face tokens go with the session that could use them (PIP-459), and the sign-out confirmation says
    /// so. They are not part of the device identity, but both exits from an account are how a shared device changes
    /// hands, and gated-repo access under the previous account must not be what the next person inherits. Deleting
    /// the account is the more final of the two, so it is the odder one to leave them behind on.
    ///
    /// Both kinds, because there are two. `deleteHfToken` drops the one the user typed into Settings;
    /// `deleteAllModelHfTokens` drops the per-model `hf_token:<reference>` entries the planner stashes off a claim
    /// (`PlannerWorker.stashClaimCredential`), which are just as usable and outnumber it. `AuthCommands.reset` clears
    /// the per-model ones for the same reason, though not the typed one. Android needs only the first: it has a
    /// single slot.
    private func deleteDeviceIdentity() {
        LocalStorage.plannerWorkerEnabled = false
        PlannerWorker.shared.setEnabled(false, storage: storage, jobRunner: jobRunner, jobStore: jobStore)
        storage.identity.clearRegistrationMaterial()
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
            try storage.resetDeviceData()
            // The reset drops `models/` but keeps `settings.json`, so re-walk: a stale
            // usage figure would leave the card claiming an over-limit store that is
            // now empty.
            storageUsedBytes = storage.storageUsageBytes()
        } catch {
            activeAlert = .resetFailed(error.localizedDescription)
        }
    }

    /// Pull-to-refresh re-pulls the server benchmark-definition catalog. The
    /// `GET /benchmarks` endpoints are public (no client id or signature), so this
    /// works unregistered — it falls back to the production server URL.
    /// `.refreshable` drives its own progress indicator; a failure surfaces as an
    /// alert when the continuation resumes on the main actor.
    private func syncBenchmarkDefinitions() async {
        AppLog.benchmarkSync.debug("pull-to-refresh requested")
        do {
            _ = try await BenchmarkSyncCoordinator.shared.sync(
                serverUrl: effectiveServerURL,
                storage: storage
            )
        } catch {
            AppLog.benchmarkSync.error("settings pull-to-refresh failed: \(error)")
            activeAlert = .syncFailed(humanizedRegistrationError(error))
            return
        }
        await resendResultsFromOtherCollectors()
    }

    /// After refreshing the catalog, migrate any results submitted to a
    /// different collector onto the one this device now reports to — restricted
    /// to benchmarks this collector actually offers, since those are the only
    /// ones it will accept. Best-effort and quiet: a failure is logged, not
    /// surfaced, so pull-to-refresh doesn't turn into an error prompt.
    ///
    /// Skips the whole-store scan once results are reconciled to the current
    /// collector, so a stable configuration doesn't re-scan on every refresh.
    private func resendResultsFromOtherCollectors() async {
        guard canSubmitResults else { return }
        let current = effectiveServerURL
        // Same normalized comparison the per-result matching uses, so the skip
        // can't disagree with it over a trailing slash or host case.
        guard !CollectorEndpoint.isSameCollector(LocalStorage.lastReconciledCollector, as: current.value)
        else { return }

        let benchmarkIds = Set(BenchmarkSync.storedDefinitions(store: storage.benchmarks).map(\.benchmarkId))
        guard !benchmarkIds.isEmpty else { return }

        let outcome = await ResultUploader.shared.resendForCollectorChange(benchmarkIds: benchmarkIds)
        // Advance the memo only on a clean sweep; leaving it unset on error means
        // the next refresh retries the results that failed to migrate.
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
    /// A failed session call does not abort the reset: the local wipe is what the confirmation promised,
    /// and refusing it offline would leave the user unable to sign out at all. Android's
    /// `ShellViewModel.signOutAndResetDevice` makes the same choice.
    ///
    /// The session then survives, and there is no local-only way to end it: `Clerk.clearAllKeychainItems()`
    /// is documented as leaving in-memory state, so `clerk.user` stays set and the wiped device lands on
    /// `SetupView` as the account it was cleared to leave, until the next launch reads the empty keychain.
    /// Logged rather than alerted for the same reason `resetLocalDataForSignOut` is: this view is torn down
    /// as soon as the registration goes.
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

    /// Clear what the previous session left on disk: jobs, results, downloaded models, and the synced
    /// benchmark catalog (PIP-459). `settings.json` stays, as it does for `resetDataOnDevice`. The
    /// identity is already gone by the time this runs, via `deleteDeviceIdentity`.
    ///
    /// Ordered after the identity so a failure here cannot leave a signed-out device still
    /// holding a usable registration.
    ///
    /// Logged rather than alerted, unlike `resetDataOnDevice`. By this point the Clerk sign-out
    /// has landed, so `ClerkAuthGateView` has swapped its content for the sign-in view and taken
    /// this view (and the `@State` an alert would be presented from) with it. An `activeAlert`
    /// set here would never be seen, and the sign-out cannot be undone to report against.
    ///
    /// The store reloads still matter for the same reason the alert does not: `jobStore` and
    /// `modelStore` outlive this view, so entries left in them would resurface on the next
    /// sign-in without an app restart. Deferred so they run even when the delete fails, since a
    /// partial reset is exactly the case that leaves them listing files that are already gone.
    private func resetLocalDataForSignOut() async {
        // Both cancels are requests, not barriers: `JobRunner.cancel()` raises the cancel flag and
        // `DownloadCoordinator.cancelAll()` issues async task cancels, so a cell or a transfer already in
        // flight can still write into the tree below. They stop the *next* one, which is what keeps a
        // multi-hour job or a multi-GB pull from rebuilding what this deletes.
        jobRunner.cancel()
        downloadCoordinator.cancelAll()
        // The Hugging Face credentials are already gone by the time this runs, via `deleteDeviceIdentity`, which is
        // where they belong: `deleteAccount` needs them cleared too and does not come through here.
        // The cancel above is the request; this is the part that waits for it to be honored. An
        // in-flight cell keeps going until it next checks the flag, and deleting `results/` and
        // `models/` under a live run lets that write land after the wipe, or makes `removeItem` itself
        // fail partway through. Bounded, so a wedged cell cannot strand the sign-out.
        let deadline = Date().addingTimeInterval(Self.runnerStopTimeout)
        while jobRunner.runningJobId != nil, Date() < deadline {
            try? await Task.sleep(for: .milliseconds(100))
        }

        defer {
            jobStore.reload()
            modelStore.reload()
            storageUsedBytes = storage.storageUsageBytes()
        }
        do {
            try storage.resetDeviceData()
        } catch {
            AppLog.storage.error("sign-out reset did not complete: \(error.localizedDescription)")
        }
    }

    /// How long to wait for the runner to park before deleting under it. Bounded so a wedged
    /// cell cannot strand the user on a sign-out that never completes.
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
        .environment(ModelStore(storage: FileStorage.production))
        // "Free up space" reads the coordinator for its pin set, so without this the
        // preview renders but traps on tap.
        .environment(DownloadCoordinator.shared)
}
#endif
