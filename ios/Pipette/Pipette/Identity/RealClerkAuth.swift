import AuthenticationServices
import ClerkKit
import Foundation

extension Error {
    /// Clerk's own copy for API errors (it knows whether the code expired or the instance
    /// is rate limiting); everything else localizes itself well enough.
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

    /// Like `offered(strategies:)`, but keeps each Clerk `Factor`: sending an email or phone
    /// code needs its resource id (see `sendSecondFactorCode`). One entry per strategy; with
    /// two phones, the first is the one Clerk's own UI would use.
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
    /// Empty until the environment has loaded. `notSelectable` hides link-only providers.
    /// No unknown-strategy filter, unlike Android: `OAuthProvider(strategy:)` maps anything
    /// unknown to `.custom`, which `signInWithOAuth` can still run. Sorted because the
    /// environment holds them in a dictionary.
    ///
    /// Not behind `ClerkAuthenticating`: it is an `@Observable` read, and through the seam
    /// the buttons would never appear once the environment loads.
    var oauthProviders: [OAuthProviderInfo] {
        guard let social = environment?.userSettings.social else { return [] }
        return social.values
            .filter { $0.enabled && $0.authenticatable && !$0.notSelectable }
            .map { OAuthProviderInfo(strategy: $0.strategy, name: $0.name) }
            .sorted { $0.name < $1.name }
    }
}

/// `ClerkAuthenticating` over ClerkKit. Owns the in-progress attempt, so no `SignIn` or
/// `SignUp` reaches the model. Mirrors Android's `RealClerkAuth`.
@MainActor
final class RealClerkAuth: ClerkAuthenticating {
    private let injectedAuth: Auth?

    /// Resolved on each access, not captured at init: `Clerk.shared` asserts when the SDK
    /// is unconfigured, and view initializers build this before an `#Preview` configures
    /// it. `Clerk.auth` is also rebuilt whenever `configure` runs.
    private var auth: Auth { injectedAuth ?? Clerk.shared.auth }

    init(auth: Auth? = nil) {
        self.injectedAuth = auth
    }

    /// The attempt between a send and its verify.
    private var pending: Pending?

    private enum Pending {
        case returningUser(SignIn)
        case newUser(SignUp)
        /// `sources` holds the Clerk `Factor` behind each option, which carries the resource
        /// id needed to send its code. The reason is kept here too because `verifyCode`
        /// words a failure with it; `completeSignIn` sets both from one status.
        case secondFactor(SignIn, reason: SecondFactorReason, sources: [SecondFactor: Factor])
        /// Not `returningUser`: both are a `SignIn` waiting on six digits but prepared with
        /// different strategies, and only the type stops a reset code going through
        /// `verifyCode` as a first factor.
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
        // Up front, so a failed send cannot leave a stale attempt a later verify completes.
        pending = nil
        do {
            pending = .returningUser(try await auth.signInWithEmailCode(emailAddress: email))
            return .success
        } catch {
            guard isIdentifierNotFound(error) else { return .failure(error.clerkDisplayMessage) }
            return await signUpEmailCode(email)
        }
    }

    /// A sign-up does not send a code on its own, so this sends it separately.
    private func signUpEmailCode(_ email: String) async -> AuthActionResult {
        do {
            // The instance has legal consent enabled: without `legalAccepted` the sign-up
            // stays at `missingRequirements`. The notice under the button makes it true;
            // see `EmailAuthState.termsURL`.
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
            // The account has no password at all; Clerk's wording for this reads as an
            // account defect, so point at the reset step instead and log Clerk's.
            guard isNoPasswordOnAccount(error) else { return .failure(error.clerkDisplayMessage) }
            AppLog.auth.warning("password sign-in, no password on the account: \(error.clerkDisplayMessage)")
            return .failure(EmailAuthModel.noPasswordOnAccount)
        }
    }

    func verifyCode(_ code: String) async -> AuthActionResult {
        do {
            // Not throwing only means the round-trip worked; the result can still be
            // non-terminal, which the two tails decide.
            switch pending {
            case .returningUser(let signIn):
                return completeSignIn(try await signIn.verifyCode(code))
            case .newUser(let signUp):
                return completeSignUp(try await signUp.verifyEmailCode(code))
            case .secondFactor(_, let reason, _):
                switch reason {
                case .mfa: return .failure("Enter your two-step verification code.")
                case .deviceVerification: return .failure("Enter the code that confirms this device.")
                }
            case .passwordReset:
                // Only if the step on screen and the parked attempt disagree.
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
            // Clerk checks the password policy server-side and words its own rejection.
            return completeSignUp(try await signUp.update(password: password))
        } catch {
            return .failure(error.clerkDisplayMessage)
        }
    }

    // MARK: - Password reset

    private static let resetPasswordEmailCode = "reset_password_email_code"

    private var parkedReset: SignIn? {
        if case .passwordReset(let signIn) = pending { return signIn }
        return nil
    }

    func sendPasswordResetCode(to email: String) async -> AuthActionResult {
        // Kept so a resend that never reached Clerk does not kill the code already sent.
        let liveReset = parkedReset
        pending = nil

        let signIn: SignIn
        do {
            // Unlike `signInWithEmailCode`, the create sends nothing; the send below does.
            signIn = try await auth.signIn(email)
        } catch {
            // Nothing was created, so the old attempt and its code are still live.
            pending = liveReset.map(Pending.passwordReset)
            return .failure(error.clerkDisplayMessage)
        }

        do {
            // Passed explicitly rather than left to the SDK's fallback, which matches
            // `factor.safeIdentifier == identifier` and would match `nil` to `nil`.
            let addressId = signIn.supportedFirstFactors?
                .first { $0.strategy.rawValue == Self.resetPasswordEmailCode }?
                .emailAddressId
            pending = .passwordReset(try await signIn.sendResetPasswordEmailCode(emailAddressId: addressId))
            return .success
        } catch {
            // Not restored: the create replaced the client's one sign-in, so the old code is
            // dead. Nothing is parked, and the message says so.
            return .failure(EmailAuthModel.resetSendFailed(error.clerkDisplayMessage))
        }
    }

    func verifyPasswordResetCode(_ code: String) async -> AuthActionResult {
        guard let signIn = parkedReset else {
            return .failure(EmailAuthModel.noPasswordResetInProgress)
        }
        do {
            // Dispatches on the prepared strategy (`reset_password_email_code`); a cleared
            // code lands on `needs_new_password`.
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
            // `signOutOfOtherSessions` stays `false`, as on Android: the usual cause is an
            // account that never had a password, not a compromised one.
            return completeSignIn(try await signIn.resetPassword(newPassword: password))
        } catch {
            return .failure(error.clerkDisplayMessage)
        }
    }

    // MARK: - Second factor

    /// The resource id is passed explicitly, as Clerk's own UI does: the no-argument
    /// fallback matches `factor.safeIdentifier` against the sign-in identifier, an email
    /// address, so a phone factor would never match. Whether to send is the model's call.
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
                // Gated out by `needsSending` at the call site.
                prepared = signIn
            }
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
        pending = nil
        do {
            let result = try await auth.signInWithOAuth(provider: OAuthProvider(strategy: strategy))
            // Through the same tails as the rest, so an OAuth challenge is answerable too.
            return switch result {
            case .signIn(let signIn): completeSignIn(signIn)
            case .signUp(let signUp): completeSignUp(signUp)
            }
        } catch {
            guard !isUserCancellation(error) else { return .cancelled }
            return .failure(error.clerkDisplayMessage)
        }
    }

    func signOut() async throws {
        try await auth.signOut()
    }

    // MARK: - Tails

    /// The one tail for every call that returns a `SignIn`: complete it, park a step, or
    /// name the status. Mirrors Android's `completeSignIn`.
    ///
    /// `needs_new_password` is both the reset code's result and what a correct password
    /// gets when the instance forced a reset; Clerk finishes both with `resetPassword`. It is
    /// checked before `secondFactorReason(for:)`, whose `default` would swallow it.
    ///
    /// `needs_client_trust` uses the MFA calls, as Clerk's own UI does
    /// (`SignInFactorCodeView.FactorMode.clientTrust.usesSecondFactorAPI`).
    ///
    /// Internal so tests can drive it with a `SignIn` decoded off the wire.
    func completeSignIn(_ signIn: SignIn) -> AuthActionResult {
        if signIn.status == .complete {
            pending = nil
            return .success
        }
        if signIn.status == .needsNewPassword {
            // This exact sign-in carries the authorization `resetPassword` needs.
            pending = .passwordReset(signIn)
            return .needsNewPassword
        }
        guard let reason = Self.secondFactorReason(for: signIn.status) else {
            pending = nil
            return .failure(Self.incompleteSignIn(signIn))
        }
        // Clerk fills `supportedSecondFactors` for a client-trust challenge too.
        let offered = SecondFactor.offered(by: signIn)
        guard !offered.isEmpty else {
            // Logged: the strategies are the only clue, and the screen does not show them.
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

    /// Mirrors Android's `completeSignUp`. The password step is offered only when a password
    /// is the entire shortfall: it is the only field this UI can supply.
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

    /// Nil when the status is not an answerable challenge.
    ///
    /// The `default` means a new Clerk status falls through silently rather than failing to
    /// compile; check here when Clerk adds one.
    static func secondFactorReason(for status: SignIn.Status) -> SecondFactorReason? {
        switch status {
        case .needsSecondFactor: .mfa
        case .needsClientTrust: .deviceVerification
        default: nil
        }
    }

    /// Names the status, so an unrouted status and a dropped challenge can be told apart.
    static func incompleteSignIn(_ signIn: SignIn) -> String {
        EmailAuthModel.incompleteSignInMessage(status: signIn.status.rawValue)
    }

    private static let signUpIncomplete = "Your account needs more details to finish signing up."

    /// Names what Clerk is still waiting for, so the user has something to report.
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
        // A field the SDK adds later reads as its wire name rather than vanishing.
        default: field.rawValue.replacingOccurrences(of: "_", with: " ")
        }
    }

    // MARK: - Error classification

    /// The address is not a known account, so register it instead.
    private func isIdentifierNotFound(_ error: Error) -> Bool {
        (error as? ClerkAPIError)?.code == "form_identifier_not_found"
    }

    /// The account has no password strategy, as opposed to a wrong password.
    private func isNoPasswordOnAccount(_ error: Error) -> Bool {
        (error as? ClerkAPIError)?.code == "strategy_for_user_invalid"
    }

    private func isUserCancellation(_ error: Error) -> Bool {
        (error as? ASWebAuthenticationSessionError)?.code == .canceledLogin
    }
}
