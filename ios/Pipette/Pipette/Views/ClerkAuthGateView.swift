import ClerkKit
import SwiftUI

struct ClerkAuthGateView: View {
    /// A substitute auth layer, when something wants to supply one. Nil in the
    /// app, where `signOut` builds a `RealClerkAuth` over the environment's
    /// client at the moment it is needed.
    ///
    /// This makes sign-out substitutable, which is as far as it goes today: the
    /// logic acting on a failure is a private method mutating private `@State`,
    /// so no test can drive it yet. See `FakeClerkAuth.signOutError`.
    ///
    /// Deliberately not resolved to a default here. A `RealClerkAuth` built in
    /// this initializer would read the *shared* client, where `main` read the
    /// injected environment one at the moment of the call, so a preview or a test
    /// with its own `Clerk` would silently be talking to a different client.
    ///
    /// Constructing one would be harmless in itself: `RealClerkAuth` resolves
    /// `Clerk.shared` lazily precisely so that it can be built before the SDK is
    /// configured. It is which client it ends up on that matters here.
    private let injectedAuth: ClerkAuthenticating?

    @Binding var isRegistered: Bool

    init(isRegistered: Binding<Bool>, auth: ClerkAuthenticating? = nil) {
        self._isRegistered = isRegistered
        self.injectedAuth = auth
    }

    @Environment(Clerk.self) private var clerk
    @Environment(\.storage) private var storage
    // Loaded in `refreshRegistration` (.onAppear) from the injected storage.
    @State private var registration: IdentityRegistration?
    @State private var activeAlert: GateAlert?
    @State private var showDeleteIdentityAlert = false

    /// The one error alert this gate can show at a time, carrying its message.
    /// Collapses the former `linkError`/`signOutError` `String?` pair into a
    /// single presented value.
    private enum GateAlert: Identifiable {
        case linkFailed(String)
        /// The session ended but the device is still pinned to the account it was linked
        /// to, so signing in with a different one would land back on the mismatch screen.
        /// Kept apart from `signOutFailed` because the sign-out itself succeeded, and
        /// telling the user it failed would send them to retry the wrong thing.
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
        Group {
            if !clerk.isLoaded {
                ClerkLoadingView()
            } else if let user = clerk.user {
                signedInContent(userId: user.id)
            } else {
                EmailCodeSignInView()
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

    @ViewBuilder
    private func signedInContent(userId: String) -> some View {
        let sessionId = clerk.session?.id
        let primaryEmail = clerk.user?.primaryEmailAddress?.emailAddress

        if let registration,
           let linkedUserId = registration.clerkUserId,
           linkedUserId != userId {
            ClerkAccountMismatchView(
                linkedEmail: registration.clerkPrimaryEmail,
                currentEmail: primaryEmail,
                onSignOut: signOut,
                onDeleteIdentity: { showDeleteIdentityAlert = true }
            )
        } else if registration != nil,
                  storage.identity.getPrivateKey() != nil {
            // No catalog pull on appear. A sync is something the user asks for — the
            // Settings refresh, the New Job screen, `headlessrun sync` — as it is on the
            // CLI, where `commands::sync` is the only caller of `pull_remote_benchmarks`.
            // Tying it to a view appearing meant a network fetch and a disk write at a
            // moment nothing chose, including partway through a measured benchmark.
            MainTabView(isRegistered: $isRegistered)
                .task(id: userId) {
                    linkRegistrationIfNeeded(
                        userId: userId,
                        sessionId: sessionId,
                        primaryEmail: primaryEmail
                    )
                }
        } else {
            SetupView(
                isRegistered: $isRegistered,
                clerkUserId: userId,
                clerkSessionId: sessionId,
                clerkPrimaryEmail: primaryEmail,
                onSignOut: signOut
            )
        }
    }

    private func refreshRegistration() {
        registration = storage.identity.getRegistration()
        isRegistered = registration != nil
    }

    private func linkRegistrationIfNeeded(
        userId: String,
        sessionId: String?,
        primaryEmail: String?
    ) {
        guard let current = storage.identity.getRegistration() else {
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
            try storage.identity.putRegistration(linked)
            registration = linked
            isRegistered = true
        } catch {
            activeAlert = .linkFailed(error.localizedDescription)
        }
    }

    private func signOut() {
        Task {
            do {
                try await (injectedAuth ?? RealClerkAuth(auth: clerk.auth)).signOut()
            } catch {
                // Report and stop. The unlink below is only safe once the session is gone: a device that
                // dropped its link while still signed in as the old account no longer matches the
                // mismatch branch, falls through to `MainTabView`, and its `linkRegistrationIfNeeded`
                // re-links to the very account the user asked to leave. There is no local-only way out
                // either: `Clerk.clearAllKeychainItems()` is documented as leaving in-memory state, so
                // `clerk.user` would survive it and reach exactly that re-link.
                activeAlert = .signOutFailed(error.localizedDescription)
                return
            }
            unlinkRegistration()
        }
    }

    /// Un-pin this device from the account it was linked to, keeping the registration itself
    /// (and so the `clientId`, the signing key, and every result already submitted under it).
    /// The counterpart to `linkRegistrationIfNeeded`, and the reason a second account can sign
    /// in on a device the first one used: without it the link outlives the session, and the
    /// next user lands on `ClerkAccountMismatchView` whose only exit deletes the identity.
    ///
    /// Ordered *after* the sign-out on purpose. Clearing the link while the session is still
    /// live would leave a signed-in user with nothing to mismatch against, and the gate would
    /// flash the app itself on the way out.
    ///
    /// A failed write is surfaced rather than swallowed: the sign-out itself already happened,
    /// so a silent failure here reads as success and reappears as a mismatch at the next
    /// sign-in, a screen away from the action that caused it.
    ///
    /// Stays on the main actor, disk write included, and not for want of noticing. The mismatch
    /// screen offers this beside "Delete device identity", so the read-modify-write here races
    /// that delete: interleaved, the load straddles the delete and writes the record back with no
    /// signing key to go with it, leaving a device that reads as registered and fails every
    /// submission. Main-actor confinement is what serializes the two, so hopping the I/O to a
    /// background executor to save a frame would buy that race. Android needs an explicit
    /// `Mutex` for the same pair (see `ShellViewModel.registrationMutex`) precisely because its
    /// equivalents already run off the main thread. The cost being avoided is one small atomic
    /// file write plus a Keychain update, on a transition that has just awaited a network
    /// sign-out; if this ever does need to move, it needs an actor around the store, not just an
    /// `await`.
    private func unlinkRegistration() {
        guard let current = storage.identity.getRegistration(),
              current.clerkUserId != nil
        else {
            refreshRegistration()
            return
        }

        let unlinked = current.withoutClerkLink()
        do {
            try storage.identity.putRegistration(unlinked)
            registration = unlinked
        } catch {
            activeAlert = .unlinkFailed(error.localizedDescription)
        }
    }

    private func deleteDeviceIdentity() {
        storage.identity.clearRegistrationMaterial()
        registration = nil
        isRegistered = false
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

/// Shown until Clerk has both an environment and a client — and, past a
/// deadline, instead of forever.
///
/// ClerkKit's startup fetch is fire-and-forget: it retries three times and then
/// only logs, leaving `isLoaded` false for the rest of the launch. A first run
/// with no network is the usual way in, and without this the app is a spinner on
/// a blank screen with nothing to tap. Android surfaces the same state as
/// `AuthGate.InitError`; here it also offers the retry, since the SDK won't try
/// again on its own.
private struct ClerkLoadingView: View {
    @Environment(Clerk.self) private var clerk
    @State private var failureMessage: String?
    @State private var isRetrying = false

    /// Long enough to cover the SDK's own three attempts and their backoff
    /// (~4s worst case offline), short enough not to read as a hang.
    private static let deadline = Duration.seconds(10)
    private static let unreachable = "Couldn't reach the sign-in service. Check your connection and try again."

    #if DEBUG
    /// `previewFailure` starts the view in its failed state; reaching it for real
    /// means a device that can't reach Clerk for ten seconds.
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

    /// Give the SDK's own startup attempt its full run before calling it: this
    /// view is torn down the moment `isLoaded` flips, which cancels the sleep.
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
                // Both, because `isLoaded` needs both. Concurrently, so a slow
                // link doesn't pay for them one after the other.
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

#Preview("Clerk Account Mismatch") {
    ClerkAccountMismatchView(
        linkedEmail: "first@example.com",
        currentEmail: "second@example.com",
        onSignOut: {},
        onDeleteIdentity: {}
    )
}
#endif
