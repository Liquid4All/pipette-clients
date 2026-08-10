import AuthenticationServices
import ClerkKit
import Foundation

extension Error {
    /// What to show the user for a failed Clerk call.
    ///
    /// Clerk's API errors carry their own user-facing copy (`longMessage` is the
    /// sentence, `message` the terse form), and that is always better than
    /// anything written here: it knows whether the code expired, the password
    /// was wrong, or the instance is rate limiting. Everything else (transport,
    /// decoding, cancellation) localizes itself well enough: a `URLError`
    /// offline reads "The Internet connection appears to be offline."
    var clerkDisplayMessage: String {
        guard let apiError = self as? ClerkAPIError else { return localizedDescription }
        return apiError.longMessage ?? apiError.message ?? "Something went wrong. Try again."
    }
}

extension SecondFactor {
    var mfaType: SignIn.MfaType {
        switch self {
        case .emailCode: .emailCode
        case .phoneCode: .phoneCode
        case .totp: .totp
        case .backupCode: .backupCode
        }
    }

    /// The factors this screen can answer, each paired with the Clerk `Factor`
    /// that describes it, in the same stable order as `offered(strategies:)`.
    ///
    /// The `Factor` has to be kept, not just its strategy: sending an email or
    /// phone code needs that resource's id, and `sendMfa*Code`'s no-argument
    /// fallback cannot supply it (see `sendSecondFactorCode`).
    ///
    /// One entry per strategy. An account with two phones offers two
    /// `phone_code` factors; the first is the one Clerk's own UI would use.
    static func offered(by signIn: SignIn) -> [(factor: SecondFactor, source: Factor)] {
        let supported = signIn.supportedSecondFactors ?? []
        return allCases.compactMap { candidate in
            supported
                .first { $0.strategy.rawValue == candidate.rawValue }
                .map { (candidate, $0) }
        }
    }
}

extension Clerk {
    /// The social providers enabled on the instance, one button each. Empty until
    /// the backend enables OAuth and until the environment has loaded.
    /// `notSelectable` hides link-only providers. Unlike Android there is no
    /// unknown-strategy filter: `OAuthProvider(strategy:)` maps anything it
    /// doesn't know to `.custom`, so every entry is one `signInWithOAuth` can run.
    /// Sorted by name because the environment holds them in a dictionary.
    ///
    /// On `Clerk` rather than behind `ClerkAuthenticating`, though it reads Clerk
    /// data, because it is an `@Observable` read and not a call. A view reaching
    /// it through the seam would stop re-rendering when the environment finishes
    /// loading, and the buttons would never appear.
    var oauthProviders: [OAuthProviderInfo] {
        guard let social = environment?.userSettings.social else { return [] }
        return social.values
            .filter { $0.enabled && $0.authenticatable && !$0.notSelectable }
            .map { OAuthProviderInfo(strategy: $0.strategy, name: $0.name) }
            .sorted { $0.name < $1.name }
    }
}

/// The real thing: `ClerkAuthenticating` over ClerkKit. Owns the in-progress
/// attempt, so no `SignIn` or `SignUp` ever reaches the model. Mirrors Android's
/// `RealClerkAuth`.
@MainActor
final class RealClerkAuth: ClerkAuthenticating {
    private let injectedAuth: Auth?

    /// The shared client's, unless one was injected. Injectable so a test that
    /// wants the real implementation over a stubbed `Auth` can have one, though
    /// the fake is the usual route.
    ///
    /// Resolved on each access rather than captured in the initializer, for two
    /// reasons. `Clerk.shared` calls `assertionFailure` when the SDK has not been
    /// configured, and this type is constructed by view initializers that run
    /// before an `#Preview` gets to install a configured client, so capturing at
    /// init traps every preview in the app. And `Clerk.auth` is a computed facade
    /// rebuilt from `Clerk.dependencies`, which `configure` replaces wholesale,
    /// so a captured value would go stale if the SDK is ever reconfigured.
    private var auth: Auth { injectedAuth ?? Clerk.shared.auth }

    init(auth: Auth? = nil) {
        self.injectedAuth = auth
    }

    /// The in-progress attempt between a send and its verify: a `SignIn`
    /// (returning address) or a `SignUp` (new address).
    private var pending: Pending?

    private enum Pending {
        case returningUser(SignIn)
        case newUser(SignUp)
        /// A first factor was accepted and Clerk answered with a code challenge:
        /// the sign-in the second factor completes, why it was asked for, and the
        /// Clerk `Factor` behind each offered option, which carries the resource
        /// id needed to send its code.
        ///
        /// The reason is carried here as well as in `EmailAuthState`, because
        /// `verifyCode` needs it to word a failure and this case is the thing that
        /// knows a challenge is outstanding. They cannot drift: one status feeds
        /// both, since `completeSignIn` reads the reason off it, stores it here,
        /// and reports it in the result the state is built from. Android does the
        /// same on its `Pending.SecondFactorRequired`.
        case secondFactor(SignIn, reason: SecondFactorReason, sources: [SecondFactor: Factor])
        /// A password reset partway through: the sign-in the emailed reset code
        /// answers, and afterwards the one `resetPassword` sets the new value on.
        ///
        /// Its own case rather than reusing `returningUser`, so a reset code
        /// cannot be submitted through `verifyCode` as an ordinary first factor
        /// and vice versa. The two look identical from the outside (both are a
        /// `SignIn` waiting on six digits) and are prepared with different
        /// strategies, so the type is the only thing keeping them apart.
        case passwordReset(SignIn)
    }

    var hasSecondFactorChallenge: Bool {
        if case .secondFactor = pending { return true }
        return false
    }

    func abandonAttempt() {
        pending = nil
    }

    // MARK: - First factor

    func sendEmailCode(to email: String) async -> AuthActionResult {
        // Drop any prior attempt up front: a failed send must not leave a stale
        // `pending` that a later verify could accidentally complete.
        pending = nil
        do {
            // Creating the sign-in with the email-code strategy also emails the code.
            pending = .returningUser(try await auth.signInWithEmailCode(emailAddress: email))
            return .success
        } catch {
            // An unknown address fails with `form_identifier_not_found`, which is the
            // signal to register it instead of surfacing an error.
            guard isIdentifierNotFound(error) else { return .failure(error.clerkDisplayMessage) }
            return await signUpEmailCode(email)
        }
    }

    /// New-address path: create the sign-up, then send the code as a separate
    /// step (a sign-up doesn't send one on its own).
    private func signUpEmailCode(_ email: String) async -> AuthActionResult {
        do {
            // `legalAccepted` is required, not optional: the instance has legal
            // consent enabled, so without it the sign-up comes back stuck at
            // `missingRequirements` and no code is ever sent. What makes sending it
            // true is the notice under the button. See `EmailAuthState.termsURL`.
            let created = try await auth.signUp(emailAddress: email, legalAccepted: true)
            pending = .newUser(try await created.sendEmailCode())
            return .success
        } catch {
            return .failure(error.clerkDisplayMessage)
        }
    }

    func signInWithPassword(email: String, password: String) async -> AuthActionResult {
        pending = nil
        do {
            return completeSignIn(try await auth.signInWithPassword(identifier: email, password: password))
        } catch {
            // `strategy_for_user_invalid` means the account has no password at
            // all, which Clerk words as a verification strategy not being valid
            // for the account. That reads as an account defect, and it sent a
            // real user looking for one; what it means is that nothing they could
            // have typed would work, and that the reset step is the way in.
            guard isNoPasswordOnAccount(error) else { return .failure(error.clerkDisplayMessage) }
            // Our sentence replaces Clerk's, so Clerk's goes to the log instead of
            // being lost with it.
            AppLog.auth.warning("password sign-in, no password on the account: \(error.clerkDisplayMessage)")
            return .failure(EmailAuthModel.noPasswordOnAccount)
        }
    }

    func verifyCode(_ code: String) async -> AuthActionResult {
        do {
            // A verify call returning without throwing only means the HTTP round-trip
            // worked. The result can still be non-terminal (a second factor, missing
            // sign-up requirements), in which case no session is created and the gate
            // never flips. Both tails decide that, rather than this call site.
            switch pending {
            case .returningUser(let signIn):
                return completeSignIn(try await signIn.verifyCode(code))
            case .newUser(let signUp):
                return completeSignUp(try await signUp.verifyEmailCode(code))
            case .secondFactor(_, let reason, _):
                // The code step is behind us; a code typed here belongs to the
                // challenge, not the first factor.
                switch reason {
                case .mfa: return .failure("Enter your two-step verification code.")
                case .deviceVerification: return .failure("Enter the code that confirms this device.")
                }
            case .passwordReset:
                // Only reachable if the step on screen and the parked attempt
                // disagree: the reset code has its own step and its own call.
                return .failure("Enter the code that lets you set a password.")
            case nil:
                return .failure("No verification in progress. Request a new code.")
            }
        } catch {
            return .failure(error.clerkDisplayMessage)
        }
    }

    func createPassword(_ password: String) async -> AuthActionResult {
        guard case .newUser(let signUp) = pending else {
            return .failure("No sign-up in progress. Request a new code.")
        }
        do {
            // Clerk owns the policy. Length, strength, and the breach corpus are
            // checked server-side, so a rejection arrives carrying its own wording
            // rather than a rule guessed at here.
            return completeSignUp(try await signUp.update(password: password))
        } catch {
            return .failure(error.clerkDisplayMessage)
        }
    }

    // MARK: - Password reset

    /// Clerk's strategy name for the emailed reset code. Named once: it picks the
    /// factor the id is read off, and the send below prepares that same strategy.
    private static let resetPasswordEmailCode = "reset_password_email_code"

    /// The parked reset's sign-in, if that is what is parked. The three calls
    /// below all need exactly this, and a second factor or a sign-up parked
    /// instead must not be mistaken for one.
    private var parkedReset: SignIn? {
        if case .passwordReset(let signIn) = pending { return signIn }
        return nil
    }

    func sendPasswordResetCode(to email: String) async -> AuthActionResult {
        // Held for the one failure worth undoing. The resend on the reset-code
        // step makes this reachable while a reset is already parked and its code
        // is sitting in the user's inbox, and a resend that never left the device
        // must not be what kills that code.
        let liveReset = parkedReset
        // Dropped up front all the same, so a failure that leaves nothing usable
        // cannot leave a stale attempt a later verify could complete.
        pending = nil

        let signIn: SignIn
        do {
            // Two calls, unlike `signInWithEmailCode`. The create carries nothing
            // but the identifier: no strategy, no code. The send below is what
            // picks `reset_password_email_code` and mails one.
            signIn = try await auth.signIn(email)
        } catch {
            // Nothing was created, so the old attempt is still the client's live
            // sign-in and its code is still answerable. This is the offline
            // resend, and the only failure here that can be undone.
            pending = liveReset.map(Pending.passwordReset)
            return .failure(error.clerkDisplayMessage)
        }

        do {
            // The address id is passed rather than left to the SDK's fallback,
            // which is `package` and unreachable from here in any case. It
            // resolves the id by matching `factor.safeIdentifier == identifier`,
            // and a probe against the instance confirms both that the reset
            // factor is on a freshly created sign-in and that its
            // `safeIdentifier` is the full address, so the fallback would in fact
            // resolve. Passing it explicitly costs nothing, since the sign-in is
            // already in hand, and does not rest on a comparison of two optionals
            // that would match `nil` to `nil`.
            let addressId = signIn.supportedFirstFactors?
                .first { $0.strategy.rawValue == Self.resetPasswordEmailCode }?
                .emailAddressId
            pending = .passwordReset(try await signIn.sendResetPasswordEmailCode(emailAddressId: addressId))
            return .success
        } catch {
            // Deliberately *not* restored here, unlike above. The create
            // succeeded, and a Clerk client holds one sign-in, so the attempt this
            // replaced is no longer the one the client is on: keeping it would
            // answer the next verify with Clerk's own not-found wording instead of
            // this app's, and the code it belongs to is dead either way. That
            // leaves nothing parked while the user may still be looking at a code
            // step, so the message has to say so.
            return .failure(EmailAuthModel.resetSendFailed(error.clerkDisplayMessage))
        }
    }

    func verifyPasswordResetCode(_ code: String) async -> AuthActionResult {
        guard let signIn = parkedReset else {
            return .failure(EmailAuthModel.noPasswordResetInProgress)
        }
        do {
            // The same call the email-code path uses: it dispatches on the
            // attempt's prepared strategy, which the send left as
            // `reset_password_email_code`, so the code goes out as a reset attempt
            // rather than a sign-in one. A cleared code lands on
            // `needs_new_password`, which `completeSignIn` turns into the step
            // that collects the new value.
            return completeSignIn(try await signIn.verifyCode(code))
        } catch {
            return .failure(error.clerkDisplayMessage)
        }
    }

    func resetPassword(_ password: String) async -> AuthActionResult {
        guard let signIn = parkedReset else {
            return .failure(EmailAuthModel.noPasswordResetInProgress)
        }
        do {
            // `signOutOfOtherSessions` stays at its `false` default. The flow's
            // ordinary cause is an account that never had a password, not a
            // compromised one, and the user reaching it is usually signed in on
            // the web; ending those sessions is not what this screen was asked to
            // do. Same call and same choice as Android's.
            //
            // Clerk owns the password policy here exactly as it does in
            // `createPassword`, so a rejection arrives carrying its own wording.
            return completeSignIn(try await signIn.resetPassword(newPassword: password))
        } catch {
            return .failure(error.clerkDisplayMessage)
        }
    }

    // MARK: - Second factor

    /// Sends the chosen factor's code.
    ///
    /// The resource id is passed explicitly, as Clerk's own UI does. Its
    /// no-argument fallback resolves the id by matching
    /// `factor.safeIdentifier == identifier`, and the sign-in identifier here is
    /// always an email address, so a phone factor can never match it and the
    /// send would go out with no phone number attached.
    ///
    /// Whether a code is *needed* is the model's call, not this one's: it is what
    /// knows the step was re-entered and what the user has already been sent. This
    /// only sends when asked.
    func sendSecondFactorCode(_ factor: SecondFactor) async -> AuthActionResult {
        guard case .secondFactor(let signIn, let reason, let sources) = pending else {
            return .failure(EmailAuthModel.noSecondFactorInProgress)
        }
        let source = sources[factor]
        do {
            let prepared: SignIn
            switch factor {
            case .emailCode:
                prepared = try await signIn.sendMfaEmailCode(emailAddressId: source?.emailAddressId)
            case .phoneCode:
                prepared = try await signIn.sendMfaPhoneCode(phoneNumberId: source?.phoneNumberId)
            case .totp, .backupCode:
                // Gated out by `needsSending` at the call site; nothing to request.
                prepared = signIn
            }
            // The reason carries over unchanged: a delivery does not change why
            // the code was asked for.
            pending = .secondFactor(prepared, reason: reason, sources: sources)
            return .success
        } catch {
            return .failure(error.clerkDisplayMessage)
        }
    }

    func verifySecondFactor(_ code: String, factor: SecondFactor) async -> AuthActionResult {
        guard case .secondFactor(let signIn, _, _) = pending else {
            return .failure(EmailAuthModel.noSecondFactorInProgress)
        }
        do {
            return completeSignIn(try await signIn.verifyMfaCode(code, type: factor.mfaType))
        } catch {
            return .failure(error.clerkDisplayMessage)
        }
    }

    // MARK: - OAuth and sign-out

    func signInWithOAuth(strategy: String) async -> AuthActionResult {
        // Any in-progress email-code attempt is abandoned when the user picks a
        // social provider instead.
        pending = nil
        do {
            let result = try await auth.signInWithOAuth(provider: OAuthProvider(strategy: strategy))
            // Clerk handles the sign-in ↔ sign-up transfer internally, so exactly
            // one of the two comes back. Each goes through the same tail as the
            // rest of the flow: an OAuth sign-in that answers with a challenge is
            // as answerable as a password one, and reading the status here instead
            // is what made it look like a flat failure.
            return switch result {
            case .signIn(let signIn): completeSignIn(signIn)
            case .signUp(let signUp): completeSignUp(signUp)
            }
        } catch {
            // Dismissing the provider sheet is a decision, not a failure. Reported
            // as its own case so the model can clear the spinner and leave the
            // screen exactly as it was, rather than showing a cancellation as an
            // error.
            guard !isUserCancellation(error) else { return .cancelled }
            return .failure(error.clerkDisplayMessage)
        }
    }

    func signOut() async throws {
        try await auth.signOut()
    }

    // MARK: - Tails

    /// Shared tail for every call that returns a `SignIn`: the emailed code, the
    /// password, an OAuth round-trip, a second-factor attempt: complete it, park
    /// a code challenge for the second-factor step, or name the status.
    ///
    /// One place, because three call sites reading the same statuses is how an
    /// OAuth sign-in that answered `needs_second_factor` ended up reported as a
    /// flat failure while the other two routed it correctly. Mirrors
    /// `completeSignUp`, and Android's `completeSignIn`.
    ///
    /// Three non-terminal statuses have a step behind them, and the two
    /// challenges only when they name a factor the step can drive; anything else
    /// is a dead end worth naming rather than looping on.
    ///
    /// `needs_new_password` is one of the three, and handling it here rather than
    /// only on the reset route is deliberate: it is the status the reset code's
    /// verify lands on, *and* what a correct password gets back when the instance
    /// has forced a reset on the account. One branch serves both, because Clerk
    /// finishes both with the same `resetPassword` call. It is answered before
    /// `secondFactorReason(for:)` is consulted, which is what keeps that
    /// function's `default` from swallowing it.
    ///
    /// `needs_client_trust` is the second of those, and it used to fall through
    /// to "needs another sign-in step that isn't available here" even though the
    /// account offered `email_code` and this screen can drive exactly that. It is
    /// resolved through the same calls as MFA, which the SDK settles rather than
    /// leaving to inference: `AuthNavigation.setToStepForStatus` reads
    /// `startingSecondFactor` for both statuses, and
    /// `SignInFactorCodeView.FactorMode.clientTrust.usesSecondFactorAPI` is true,
    /// so Clerk's own UI prepares with `sendMfaEmailCode` and verifies with
    /// `verifyMfaCode` here too. So all this function owes it is to be let
    /// through. What differs is the wording, which `SecondFactorReason` carries.
    ///
    /// Internal rather than private so the tests can drive it with a `SignIn`
    /// decoded off the wire. Pinning `secondFactorReason(for:)` alone would leave
    /// the wiring free to stop calling it.
    func completeSignIn(_ signIn: SignIn) -> AuthActionResult {
        if signIn.status == .complete {
            pending = nil
            return .success
        }
        if signIn.status == .needsNewPassword {
            // Park this exact sign-in: `resetPassword` needs it, and it is the
            // only thing carrying the authorization the accepted code (or the
            // accepted password) just earned.
            pending = .passwordReset(signIn)
            return .needsNewPassword
        }
        guard let reason = Self.secondFactorReason(for: signIn.status) else {
            pending = nil
            return .failure(Self.incompleteSignIn(signIn))
        }
        // Read off the sign-in for both statuses. Clerk populates
        // `supportedSecondFactors` for a client-trust challenge too, with the
        // delivery strategies the instance has enabled for it, so there is no
        // separate list to consult.
        let offered = SecondFactor.offered(by: signIn)
        guard !offered.isEmpty else {
            // The strategies are the only clue to what this challenge wants, and
            // nobody can read them off the screen.
            let strategies = (signIn.supportedSecondFactors ?? []).map(\.strategy.rawValue)
            AppLog.auth.warning("unsupported second factors (\(signIn.status.rawValue)): \(strategies.joined(separator: ", "))")
            pending = nil
            return .failure(EmailAuthModel.unanswerableChallenge(reason))
        }
        pending = .secondFactor(
            signIn,
            reason: reason,
            sources: Dictionary(offered.map { ($0.factor, $0.source) }, uniquingKeysWith: { first, _ in first })
        )
        return .needsSecondFactor(offered.map(\.factor), reason)
    }

    /// What a sign-up that came back from the server means for the screen.
    /// Mirrors `completeSignIn`, and Android's `completeSignUp`.
    ///
    /// The password step is offered only when a password is the *entire*
    /// shortfall, because it is the only field this UI can supply. Anything
    /// else missing would land the user on a screen that can't finish the job.
    private func completeSignUp(_ signUp: SignUp) -> AuthActionResult {
        if signUp.status == .complete {
            pending = nil
            return .success
        }
        if signUp.missingFields == [.password] {
            pending = .newUser(signUp)
            return .needsPassword
        }
        pending = nil
        return .failure(Self.incompleteSignUp(signUp))
    }

    // MARK: - Status and message builders

    /// Which of the two answerable challenges a status is, or nil when the status
    /// has no step behind it.
    ///
    /// Split out from `completeSignIn` because this mapping *is* the fix, and the
    /// rest of that function needs a live `SignIn` decoded off the wire to
    /// exercise. Kept as the single place the two statuses are named, so the pair
    /// cannot drift apart.
    ///
    /// Note the `default`: unlike the switches over `AuthActionResult`, a new
    /// status the app should answer will fall through here silently rather than
    /// failing to compile. Anything added to Clerk's status list needs looking at
    /// here on purpose. `needsNewPassword` is the one that already did: it has a
    /// step behind it but is not a challenge, so it is answered in
    /// `completeSignIn` above this call rather than by widening this mapping.
    static func secondFactorReason(for status: SignIn.Status) -> SecondFactorReason? {
        switch status {
        case .needsSecondFactor: .mfa
        case .needsClientTrust: .deviceVerification
        default: nil
        }
    }

    /// Why a sign-in stopped short of a session. The three statuses with a step
    /// behind them are handled before this; every other one names itself, since
    /// without that a status we have no route for and a challenge we dropped are
    /// indistinguishable, and the only way to tell is a debugger.
    ///
    /// `needs_new_password` used to be the example here, and is now one of the
    /// three answered above, so it can no longer reach this message. Same reasoning
    /// as `incompleteSignUpMessage`.
    static func incompleteSignIn(_ signIn: SignIn) -> String {
        EmailAuthModel.incompleteSignInMessage(status: signIn.status.rawValue)
    }

    private static let signUpIncomplete = "Your account needs more details to finish signing up."

    /// Why a sign-up stopped short of an account, naming what Clerk is still
    /// waiting for. The generic sentence sent people to support with nothing to
    /// report; `missingFields` is the difference between "more details" and
    /// "a password", and today it is reliably the latter, since the instance requires
    /// one at sign-up and this flow does not yet collect it.
    private static func incompleteSignUp(_ signUp: SignUp) -> String {
        incompleteSignUpMessage(missing: signUp.missingFields)
    }

    static func incompleteSignUpMessage(missing fields: [SignUp.Field]) -> String {
        let named = fields.map(label(for:))
        guard !named.isEmpty else { return signUpIncomplete }
        return "Your account needs more to finish signing up: \(named.joined(separator: ", "))."
    }

    private static func label(for field: SignUp.Field) -> String {
        switch field {
        case .password: "a password"
        case .emailAddress: "an email address"
        case .phoneNumber: "a phone number"
        case .username: "a username"
        case .firstName: "a first name"
        case .lastName: "a last name"
        case .legalAccepted: "accepting the terms"
        // Forward compatibility: a field the SDK gained after this was written
        // reads as its wire name rather than vanishing from the list.
        default: field.rawValue.replacingOccurrences(of: "_", with: " ")
        }
    }

    // MARK: - Error classification

    /// True when a sign-in failed only because the address isn't a known account
    /// (so we should register it instead).
    private func isIdentifierNotFound(_ error: Error) -> Bool {
        (error as? ClerkAPIError)?.code == "form_identifier_not_found"
    }

    /// True when a password sign-in failed because the account has no password
    /// strategy on offer, rather than because the password was wrong.
    private func isNoPasswordOnAccount(_ error: Error) -> Bool {
        (error as? ClerkAPIError)?.code == "strategy_for_user_invalid"
    }

    /// True when the user dismissed the OAuth browser sheet themselves.
    private func isUserCancellation(_ error: Error) -> Bool {
        (error as? ASWebAuthenticationSessionError)?.code == .canceledLogin
    }
}
