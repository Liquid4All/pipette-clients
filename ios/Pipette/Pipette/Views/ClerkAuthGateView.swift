import ClerkKit
import SwiftUI

/// Renders the screen `reduceAuthGate` picks (`AuthGateScreen.swift`), and owns the Clerk
/// and identity reads it is given.
struct ClerkAuthGateView: View {
    /// A substitute auth layer. Nil in the app, where `signOut` builds a `RealClerkAuth`
    /// over the environment's client at the moment it is needed.
    ///
    /// Not defaulted in the initializer: a `RealClerkAuth` built there would use the shared
    /// client, so a preview or test with its own `Clerk` would talk to a different one.
    private let injectedAuth: ClerkAuthenticating?

    @Binding var isRegistered: Bool

    init(isRegistered: Binding<Bool>, auth: ClerkAuthenticating? = nil) {
        self._isRegistered = isRegistered
        self.injectedAuth = auth
    }

    @Environment(Clerk.self) private var clerk
    @Environment(\.ws) private var ws
    @State private var registration: IdentityRegistration?
    /// Cached so routing does not read the Keychain on every body pass.
    @State private var hasPrivateKey = false
    @State private var activeAlert: GateAlert?
    @State private var showDeleteIdentityAlert = false

    /// The one error alert this gate can show at a time.
    private enum GateAlert: Identifiable {
        case linkFailed(String)
        /// The session ended but the device is still pinned to the old account. Apart from
        /// `signOutFailed` because the sign-out succeeded, and saying otherwise would send
        /// the user to retry the wrong thing.
        case unlinkFailed(String)
        case signOutFailed(String)

        var id: String {
            switch self {
            case .linkFailed: return "linkFailed"
            case .unlinkFailed: return "unlinkFailed"
            case .signOutFailed: return "signOutFailed"
            }
        }

        var title: String {
            switch self {
            case .linkFailed: return "Clerk Link Failed"
            case .unlinkFailed: return "Sign Out Incomplete"
            case .signOutFailed: return "Sign Out Failed"
            }
        }

        var message: String {
            switch self {
            case .linkFailed(let message): return message
            case .unlinkFailed(let message):
                return "You are signed out, but this device is still linked to the previous "
                    + "account, because the change could not be saved (\(message)). Signing in "
                    + "with a different account will report a mismatch."
            case .signOutFailed(let message): return message
            }
        }
    }

    var body: some View {
        // `registration` and `hasPrivateKey` load in `.onAppear`, so the first pass routes as
        // if the device held no identity. The 1.4-second `SplashView` (`PipetteApp`) hides it.
        Group {
            switch reduceAuthGate(
                isHeadless: HeadlessRunner.isHeadless,
                clerkLoaded: clerk.isLoaded,
                clerkUserId: clerk.user?.id,
                hasRegistration: registration != nil,
                hasPrivateKey: hasPrivateKey,
                linkedClerkUserId: registration?.clerkUserId
            ) {
            case .headless:
                HeadlessModeView(clientId: registration?.clientId.value)
            case .loading:
                ClerkLoadingView()
            case .signIn:
                EmailCodeSignInView()
            case .setup:
                setupContent
            case .keyMissing:
                keyMissingContent
            case .mismatch:
                mismatchContent
            case .app:
                appContent
            }
        }
        .onAppear(perform: refreshRegistration)
        .onChange(of: isRegistered) { _, _ in
            refreshRegistration()
        }
        .onChange(of: clerk.user?.id) { _, _ in
            refreshRegistration()
        }
        .alert("Delete Device Identity?", isPresented: $showDeleteIdentityAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                deleteDeviceIdentity()
            }
        } message: {
            Text("This deletes the current registration and private key. You will need to register this device again.")
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
    }

    /// A registration with no signing key. The one action is destructive by necessity:
    /// clearing the record un-attributes every result submitted under this `client_id`,
    /// which `ensureEmpty()` refuses to let setup do quietly, so it sits behind a confirmation.
    ///
    /// `headlessrun auth import` is the non-destructive repair that keeps the `client_id`,
    /// but it needs a cable and the exported bundle, so it cannot be the only way out.
    @ViewBuilder
    private var keyMissingContent: some View {
        if let registration {
            SigningKeyMissingView(
                clientId: registration.clientId.value,
                onDeleteIdentity: { showDeleteIdentityAlert = true }
            )
            .task(id: registration.clientId.value) {
                // Once per admission, not per body pass. Nothing else records that this
                // device can no longer submit.
                AppLog.auth.warning("gate found a registration with no signing key "
                    + "client_id=\(registration.clientId.value)")
            }
        }
    }

    /// The device is pinned to one account and a different account is signed in.
    private var mismatchContent: some View {
        ClerkAccountMismatchView(
            linkedEmail: registration?.clerkPrimaryEmail,
            currentEmail: clerk.user?.primaryEmailAddress?.emailAddress,
            onSignOut: signOut,
            onDeleteIdentity: { showDeleteIdentityAlert = true }
        )
    }

    /// `SetupView` records the Clerk link on the registration it creates. `reduceAuthGate`
    /// reaches this arm only with a user; the `if let` is how a `ViewBuilder` says so.
    @ViewBuilder
    private var setupContent: some View {
        if let user = clerk.user {
            SetupView(
                isRegistered: $isRegistered,
                clerkUserId: user.id,
                clerkSessionId: clerk.session?.id,
                clerkPrimaryEmail: user.primaryEmailAddress?.emailAddress,
                onSignOut: signOut
            )
        }
    }

    /// The task is keyed on the Clerk user id, so signing in from Settings links a device
    /// the gate admitted without a session.
    ///
    /// No catalog pull on appear: as on the CLI, a sync is something the user asks for.
    /// Tied to a view appearing, it could fetch and write to disk mid-benchmark.
    private var appContent: some View {
        MainTabView(isRegistered: $isRegistered)
            .task(id: clerk.user?.id) {
                linkRegistrationIfNeeded(
                    userId: clerk.user?.id,
                    sessionId: clerk.session?.id,
                    primaryEmail: clerk.user?.primaryEmailAddress?.emailAddress
                )
            }
    }

    private func refreshRegistration() {
        registration = ws.registration.get()
        hasPrivateKey = ws.registration.getPrivateKey() != nil
        isRegistered = registration != nil
    }

    private func linkRegistrationIfNeeded(
        userId: String?,
        sessionId: String?,
        primaryEmail: String?
    ) {
        guard let userId else {
            // Access granted without a session is worth a record, and this runs once per
            // admission rather than once per body pass.
            AppLog.auth.info("gate admitted a device with no Clerk session "
                + "client_id=\(registration?.clientId.value ?? "none")")
            return
        }

        guard let current = ws.registration.get() else {
            refreshRegistration()
            return
        }

        guard current.clerkUserId == nil || current.clerkUserId == userId else {
            registration = current
            return
        }

        guard current.clerkUserId != userId ||
              current.clerkSessionId != sessionId ||
              current.clerkPrimaryEmail != primaryEmail
        else {
            registration = current
            return
        }

        let linked = current.withClerkLink(
            userId: userId,
            sessionId: sessionId,
            primaryEmail: primaryEmail
        )

        do {
            try ws.registration.put(linked)
            registration = linked
            isRegistered = true
            // The client id, never the Clerk user id or the email: `AppLog` is `os.Logger`,
            // which persists to the device log store.
            AppLog.auth.info("linked the registration to the signed-in Clerk account "
                + "client_id=\(linked.clientId.value)")
        } catch {
            activeAlert = .linkFailed(error.localizedDescription)
        }
    }

    private func signOut() {
        Task {
            do {
                try await (injectedAuth ?? RealClerkAuth(auth: clerk.auth)).signOut()
            } catch {
                // Report and stop: unlinking while still signed in would fall through to
                // `MainTabView`, whose `linkRegistrationIfNeeded` re-links the account the user
                // asked to leave. `Clerk.clearAllKeychainItems()` is no way out either: it is
                // documented as leaving in-memory state, so `clerk.user` would survive it.
                activeAlert = .signOutFailed(error.localizedDescription)
                return
            }
            unlinkRegistration()
        }
    }

    /// Un-pin this device from its account, keeping the registration (`clientId`, signing key,
    /// submitted results). Without it the link outlives the session and the next account lands
    /// on `ClerkAccountMismatchView`, whose only exit deletes the identity.
    ///
    /// Runs after the sign-out: clearing the link while the session is live would flash the
    /// app on the way out. A failed write is surfaced, since the sign-out already happened and
    /// a silent failure would reappear as a mismatch at the next sign-in.
    ///
    /// Stays on the main actor, disk write included. The mismatch screen also offers "Delete
    /// device identity", and main-actor confinement is what serializes the two: interleaved,
    /// this read-modify-write would write the record back with no signing key. Android needs
    /// `ShellViewModel.registrationMutex` because its equivalents run off the main thread.
    /// Moving this off the main actor needs an actor around the store, not just an `await`.
    private func unlinkRegistration() {
        guard let current = ws.registration.get(),
              current.clerkUserId != nil
        else {
            refreshRegistration()
            return
        }

        let unlinked = current.withoutClerkLink()
        do {
            try ws.registration.put(unlinked)
            registration = unlinked
        } catch {
            activeAlert = .unlinkFailed(error.localizedDescription)
        }
    }

    private func deleteDeviceIdentity() {
        ws.registration.clear()
        registration = nil
        hasPrivateKey = false
        isRegistered = false
    }

}

/// The `client_id`, selectable and monospaced because an operator reads it back or pastes
/// it when reporting a device. The caller supplies the padding.
private struct ClientIdReadout: View {
    let clientId: String

    var body: some View {
        VStack(spacing: 5) {
            Text("Client ID")
                .font(.system(size: 12, weight: .regular))
                .foregroundStyle(Color(.systemGray))

            Text(clientId)
                .font(.system(size: 13, weight: .regular, design: .monospaced))
                .foregroundStyle(.primary)
                .multilineTextAlignment(.center)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// The screen a `headlessrun` process shows for the life of the run.
///
/// A readout with no controls: an operator drives the device from a terminal, so a tap
/// would act outside the session they can see, and a stray one could delete the registration.
private struct HeadlessModeView: View {
    /// Nil on a device that holds no registration.
    let clientId: String?

    var body: some View {
        VStack(spacing: 18) {
            PipetteLogoMark()

            Text("HEADLESS MODE")
                .font(.system(size: 20, weight: .semibold, design: .monospaced))
                .kerning(2)
                .multilineTextAlignment(.center)

            Text("This device runs from the command line.")
                .font(.system(size: 16, weight: .regular))
                .multilineTextAlignment(.center)
                .foregroundStyle(Color(.systemGray))
                .fixedSize(horizontal: false, vertical: true)

            ClientIdReadout(clientId: clientId ?? "unregistered")
                .padding(.top, 6)
        }
        .padding(.horizontal, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }
}

struct ClerkConfigurationErrorView: View {
    var body: some View {
        VStack(spacing: 18) {
            PipetteLogoMark()

            Text("Clerk is not configured")
                .font(.serif(26))
                .multilineTextAlignment(.center)

            Text("Set CLERK_PUBLISHABLE_KEY and CLERK_FRONTEND_API_DOMAIN in the iOS xcconfig files.")
                .font(.system(size: 16, weight: .regular))
                .multilineTextAlignment(.center)
                .foregroundStyle(Color(.systemGray))
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }
}

/// Shown until Clerk has both an environment and a client, with a retry past a deadline.
///
/// ClerkKit's startup fetch retries three times and then only logs, leaving `isLoaded`
/// false for the rest of the launch (typically a first run with no network). Android
/// shows the same state as `AuthGate.InitError`; here the SDK won't retry on its own.
private struct ClerkLoadingView: View {
    @Environment(Clerk.self) private var clerk
    @State private var failureMessage: String?
    @State private var isRetrying = false

    /// Long enough to cover the SDK's own three attempts and their backoff
    /// (~4s worst case offline), short enough not to read as a hang.
    private static let deadline = Duration.seconds(10)
    private static let unreachable = "Couldn't reach the sign-in service. Check your connection and try again."

    #if DEBUG
    /// `previewFailure` starts the view in its failed state.
    init(previewFailure: String? = nil) {
        self._failureMessage = State(initialValue: previewFailure)
    }
    #endif

    var body: some View {
        VStack(spacing: 18) {
            PipetteLogoMark()

            if let failureMessage {
                Text("Sign-in unavailable")
                    .font(.serif(26))
                    .multilineTextAlignment(.center)

                Text(failureMessage)
                    .font(.system(size: 16, weight: .regular))
                    .multilineTextAlignment(.center)
                    .foregroundStyle(Color(.systemGray))
                    .fixedSize(horizontal: false, vertical: true)

                AuthPrimaryButton(
                    title: "Try again",
                    isEnabled: true,
                    isLoading: isRetrying,
                    action: retry
                )
                .padding(.top, 6)
            } else {
                ProgressView()
                    .controlSize(.large)
            }
        }
        .padding(.horizontal, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
        .task(waitForDeadline)
    }

    /// The view is torn down when `isLoaded` flips, which cancels the sleep.
    private func waitForDeadline() async {
        try? await Task.sleep(for: Self.deadline)
        guard !Task.isCancelled, !clerk.isLoaded, failureMessage == nil else { return }
        failureMessage = Self.unreachable
    }

    private func retry() {
        guard !isRetrying else { return }
        isRetrying = true
        Task {
            do {
                // `isLoaded` needs both; concurrent so a slow link pays once.
                async let environment = clerk.refreshEnvironment()
                async let client = clerk.refreshClient()
                _ = try await environment
                _ = try await client
                // On success `isLoaded` flips and the gate replaces this view.
                failureMessage = clerk.isLoaded ? nil : Self.unreachable
            } catch {
                failureMessage = error.clerkDisplayMessage
            }
            isRetrying = false
        }
    }
}

/// The screen for a registration with no signing key. Shows the `client_id`: an operator
/// needs it to repair the device over a cable, and it is lost by the destructive exit.
private struct SigningKeyMissingView: View {
    let clientId: String
    let onDeleteIdentity: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            PipetteLogoMark()
                .padding(.bottom, 14)

            Text("Signing key missing")
                .font(.serif(26))
                .multilineTextAlignment(.center)
                .padding(.bottom, 14)

            Text("This device is registered, but the key that signs its results is gone, so "
                + "it cannot submit work. To keep this client, restore the key from an "
                + "identity bundle. To start over, delete the identity and register again.")
                .font(.system(size: 16, weight: .regular))
                .foregroundStyle(Color(.systemGray))
                .multilineTextAlignment(.center)
                .lineSpacing(5)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.bottom, 22)

            ClientIdReadout(clientId: clientId)
                .padding(.bottom, 28)

            Button(role: .destructive, action: onDeleteIdentity) {
                Text("Delete device identity")
                    .font(.system(size: 17, weight: .regular))
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }
}

private struct ClerkAccountMismatchView: View {
    let linkedEmail: String?
    let currentEmail: String?
    let onSignOut: () -> Void
    let onDeleteIdentity: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            PipetteLogoMark()
                .padding(.bottom, 14)

            Text("Account mismatch")
                .font(.serif(26))
                .multilineTextAlignment(.center)
                .padding(.bottom, 14)

            Text(message)
                .font(.system(size: 16, weight: .regular))
                .foregroundStyle(Color(.systemGray))
                .multilineTextAlignment(.center)
                .lineSpacing(5)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.bottom, 28)

            Button(action: onSignOut) {
                Text("Sign out of Clerk")
                    .font(.system(size: 17, weight: .regular))
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .foregroundStyle(Color(.systemBackground))
                    .background(Color.primary, in: Capsule())
            }
            .buttonStyle(.plain)

            Button(role: .destructive, action: onDeleteIdentity) {
                Text("Delete device identity")
                    .font(.system(size: 17, weight: .regular))
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .foregroundStyle(.red)
            }
            .buttonStyle(.plain)
            .padding(.top, 10)
        }
        .padding(.horizontal, 32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
    }

    private var message: String {
        let linked = linkedEmail ?? "another Clerk account"
        let current = currentEmail ?? "the current Clerk account"
        return "This device identity is linked to \(linked), but you are signed in as \(current)."
    }
}

#if DEBUG
#Preview("Clerk Gate · Signed Out") {
    ClerkAuthGateView(isRegistered: .constant(false))
        .environment(Clerk.preview { preview in
            preview.isSignedIn = false
        })
}

#Preview("Clerk Gate · Setup") {
    ClerkAuthGateView(isRegistered: .constant(false))
        .environment(Clerk.preview { preview in
            preview.isSignedIn = true
        })
}

#Preview("Headless Mode") {
    HeadlessModeView(clientId: "01J8Z5S9QW0000000000000000")
}

#Preview("Headless Mode · Unregistered") {
    HeadlessModeView(clientId: nil)
}

#Preview("Clerk Configuration Error") {
    ClerkConfigurationErrorView()
}

#Preview("Clerk Loading") {
    ClerkLoadingView()
        .environment(Clerk.preview { preview in
            preview.isSignedIn = false
        })
}

#Preview("Clerk Unreachable") {
    ClerkLoadingView(previewFailure: "The Internet connection appears to be offline.")
        .environment(Clerk.preview { preview in
            preview.isSignedIn = false
        })
}

#Preview("Signing Key Missing") {
    SigningKeyMissingView(
        clientId: "ev1_74fe0632cf0c3c3453f6b7ab03f3e7977a28207f4418681503d5b8ae88db2861",
        onDeleteIdentity: {}
    )
}

#Preview("Clerk Account Mismatch") {
    ClerkAccountMismatchView(
        linkedEmail: "first@example.com",
        currentEmail: "second@example.com",
        onSignOut: {},
        onDeleteIdentity: {}
    )
}
#endif
