import Foundation

/// Outcome of a one-shot auth action. Mirrors Android's `AuthActionResult` (`ClerkAuth.kt`).
/// Free of ClerkKit types so `FakeClerkAuth` can exist; see `ClerkAuthenticating`.
enum AuthActionResult: Equatable {
    case success
    /// The user dismissed the OAuth sheet: a decision, not a failure, so nothing is said.
    /// Its own case so every switch has to decide what a cancellation does.
    case cancelled
    /// Clerk wants a code before issuing a session. The factors are the ones this screen
    /// can drive, never empty (an unanswerable challenge is a failure instead); the reason
    /// picks the step's wording.
    ///
    /// The reason has no default, unlike Android's, so a reason not threaded through fails
    /// to compile rather than showing MFA copy.
    case needsSecondFactor([SecondFactor], SecondFactorReason)
    /// The address is verified but the instance requires a password at sign-up, so the
    /// account doesn't exist yet. Answered by `submitNewPassword`.
    case needsPassword
    /// Clerk wants a new password on an existing account, with no session yet. Answered by
    /// `submitResetPassword`.
    ///
    /// Reached after a reset code, or when Clerk demands a reset on an otherwise good
    /// sign-in (password, first-factor code, OAuth or second factor). The result cannot say
    /// which, so the flow records it in `resetWasRequested`.
    case needsNewPassword
    case failure(String)
}

/// Why Clerk is asking for a second code. Both are answered with the MFA prepare and verify
/// calls; only the wording differs. Mirrors Android's `SecondFactorReason`.
enum SecondFactorReason: Equatable {
    /// `needs_second_factor`: the account has two-step verification on.
    case mfa

    /// `needs_client_trust`: Clerk's Client Trust does not recognize this device. Unrelated
    /// to the account's settings, so it must not be called two-step verification. Clerk's
    /// docs say it skips accounts with MFA on, but the SDK does not enforce that, so the
    /// two are handled as independently reachable.
    case deviceVerification

    /// Not "Two-step verification" for a device check: the account may never have turned
    /// that on, and the step would read as a dead end.
    var title: String {
        switch self {
        case .mfa: "Two-step verification"
        case .deviceVerification: "Confirm this device"
        }
    }

    /// `source` is the chosen factor's sentence, or the chooser's. Client Trust arrives
    /// unannounced on a password sign-in, so its sentence leads: without it the screen
    /// looks like the password was wrong.
    func prompt(_ source: String) -> String {
        switch self {
        case .mfa: source
        case .deviceVerification: "This is the first sign-in on this device, so it needs confirming. \(source)"
        }
    }
}

/// A second factor from Clerk's `supportedSecondFactors`. Mirrors Android's `SecondFactor`.
enum SecondFactor: String, CaseIterable, Identifiable, Equatable {
    case emailCode = "email_code"
    case phoneCode = "phone_code"
    case totp
    case backupCode = "backup_code"

    var id: String { rawValue }

    var needsSending: Bool {
        self == .emailCode || self == .phoneCode
    }

    /// Nil for backup codes, which are alphanumeric and vary in length, so they get a plain
    /// field rather than the segmented one.
    var digitCount: Int? {
        self == .backupCode ? nil : 6
    }

    var title: String {
        switch self {
        case .emailCode: "Email code"
        case .phoneCode: "Text message"
        case .totp: "Authenticator app"
        case .backupCode: "Backup code"
        }
    }

    var prompt: String {
        switch self {
        case .emailCode: "We sent a 6-digit code to your email."
        case .phoneCode: "We sent a 6-digit code to your phone."
        case .totp: "Enter the 6-digit code from your authenticator app."
        // No "when you turned on two-step verification": this also appears under a device
        // check, on accounts that may never have enabled it.
        case .backupCode: "Enter one of your saved backup codes."
        }
    }

    /// The factors this screen can answer, in a stable order; anything else (a passkey, a
    /// hardware key) is dropped. Empty means unanswerable here. The Clerk-typed
    /// counterpart, `offered(by:)`, lives in `RealClerkAuth.swift`.
    static func offered(strategies: [String]) -> [SecondFactor] {
        let offered = Set(strategies)
        return allCases.filter { offered.contains($0.rawValue) }
    }
}

/// A social sign-in provider the Clerk instance has enabled (from the environment config,
/// not hard-coded). `strategy` is Clerk's identifier, e.g. `oauth_google`, passed back to
/// `signInWithOAuth`.
struct OAuthProviderInfo: Equatable, Identifiable {
    let strategy: String
    let name: String

    var id: String { strategy }
}

/// What the sign-in screen is showing. The text fields stay local `@State` in the views
/// (`rememberSaveable` on Android).
struct EmailAuthState: Equatable {
    enum Step {
        case email
        case password
        case code
        /// Finishing a new account after the emailed code verified the address.
        case createPassword
        /// Either `needs_second_factor` or `needs_client_trust`: answered identically, so
        /// one step, worded by `secondFactorReason`.
        case secondFactor
        /// Not a mode on `code`: answered by a different call against a differently
        /// prepared attempt.
        case resetCode
        /// Finishing a reset on an existing account.
        case resetPassword
    }

    var step: Step = .email
    /// The address the code was sent to, shown on the code step.
    var email: String = ""
    /// Disables the submit button, shows its spinner, and guards against double submits.
    var submitting: Bool = false
    var error: String?
    /// Only populated on the `secondFactor` step.
    var secondFactorOptions: [SecondFactor] = []
    /// Nil while the user is still choosing; a single-option challenge is pre-chosen.
    var chosenSecondFactor: SecondFactor?
    /// Meaningless off the `secondFactor` step; not optional because the copy needs an
    /// answer either way.
    var secondFactorReason: SecondFactorReason = .mfa
    /// Bumped on entering or re-entering the step and by an accepted resend, so the view
    /// clears the code field even when factor and reason are unchanged. Otherwise a fresh
    /// challenge would show the spent digits as if they were still good.
    var challengeGeneration: Int = 0
    /// The same for the reset code's field: bumped by every send Clerk accepted. Separate
    /// because it guards a different field on a different step.
    var resetCodeGeneration: Int = 0
    /// Set by `afterSendResetCode` once Clerk has sent a code; cleared only by rebuilding
    /// the state (`changeEmail`, success).
    ///
    /// Recorded, not derived from the step being left: the reset step is reachable from
    /// the reset code, a correct password, a first-factor code, OAuth, and a second-factor
    /// step in front of any of them, so the prior step does not say who asked. Android
    /// derives it in `ShellViewModel.kt` and so words a 2FA account's requested reset as
    /// demanded.
    var resetWasRequested: Bool = false

    /// A forced reset lands on a correct password, so without saying so the step reads as
    /// a rejection. Meaningless off the `resetPassword` step.
    var resetWasForced: Bool { !resetWasRequested }
}

extension EmailAuthState {
    /// The clickwrap links under the email step's button. Required: the Clerk instance has
    /// legal consent enabled, so `signUp` sends `legal_accepted`, which is only honest while
    /// no one can register without seeing these. Android pairs the same links with the same
    /// flag in `AuthGateScreen.kt`; change both together.
    static let termsURL = "https://pipette.liquid.ai/terms"
    static let privacyPolicyURL = "https://www.liquid.ai/privacy-policy"

    /// The Clerk instance's password floor, matching Android's `MIN_PASSWORD_LENGTH`. Here
    /// so the copy that states it and the check that enforces it read one number.
    static let minimumPasswordLength = 12

    /// Shared by both email-step actions (register, and the password path).
    static func canSubmitEmail(_ email: String, submitting: Bool) -> Bool {
        !email.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !submitting
    }

    func afterSendCode(_ result: AuthActionResult) -> EmailAuthState {
        switch result {
        case .success:
            var next = self
            next.step = .code
            next.submitting = false
            next.error = nil
            return next
        // A send only returns success or failure; the rest are routed anyway rather than
        // dropped, so an unexpected result cannot hang the spinner.
        case .needsSecondFactor(let options, let reason):
            return challenged(with: options, reason: reason)
        case .needsPassword:
            return askingToCreatePassword()
        case .needsNewPassword:
            return askingToResetPassword()
        case .cancelled:
            return settled()
        case .failure(let message):
            return failed(message)
        }
    }

    /// Success resets to the initial state: the signed-in user flips the gate, and no
    /// step should linger behind it.
    func afterAuthentication(_ result: AuthActionResult) -> EmailAuthState {
        switch result {
        case .success:
            return EmailAuthState()
        case .needsSecondFactor(let options, let reason):
            return challenged(with: options, reason: reason)
        case .needsPassword:
            return askingToCreatePassword()
        case .needsNewPassword:
            return askingToResetPassword()
        case .cancelled:
            // An error here would read as the provider having rejected them.
            return settled()
        case .failure(let message):
            return failed(message)
        }
    }

    /// Clear the spinner and change nothing else.
    private func settled() -> EmailAuthState {
        var next = self
        next.submitting = false
        return next
    }

    /// For the first send and every resend. The generation moves only on a send Clerk
    /// accepted, so a failed resend leaves the digits alone: after a failed create they are
    /// still the live code's, and after a failed send `resetSendFailed` says to start again.
    func afterSendResetCode(_ result: AuthActionResult) -> EmailAuthState {
        switch result {
        case .success:
            var next = self
            next.step = .resetCode
            next.submitting = false
            next.error = nil
            next.resetCodeGeneration = resetCodeGeneration + 1
            // Only on success: a failed send leaves the user where they were, and a
            // correct password they then submit could hit a forced reset.
            next.resetWasRequested = true
            next.clearSecondFactorChallenge()
            return next
        // Not produced by a send; routed rather than dropped, as in `afterSendCode`.
        case .needsNewPassword:
            return askingToResetPassword()
        case .needsSecondFactor(let options, let reason):
            return challenged(with: options, reason: reason)
        case .needsPassword:
            return askingToCreatePassword()
        case .cancelled:
            return settled()
        case .failure(let message):
            return failed(message)
        }
    }

    /// For an account that does not exist yet; `askingToResetPassword` is for one that does.
    private func askingToCreatePassword() -> EmailAuthState {
        var next = self
        next.step = .createPassword
        next.submitting = false
        next.error = nil
        next.clearSecondFactorChallenge()
        return next
    }

    /// Who asked is not decided here: `resetWasRequested` carries over from when the reset
    /// started. The error is kept on re-entry, as in `challenged`.
    private func askingToResetPassword() -> EmailAuthState {
        var next = self
        next.step = .resetPassword
        next.submitting = false
        next.error = step == .resetPassword ? error : nil

        next.clearSecondFactorChallenge()
        return next
    }

    /// On the way to any other step, so the second-factor step can never render against a
    /// challenge that no longer exists. Defensive: every caller has already dropped the
    /// attempt behind the seam.
    mutating func clearSecondFactorChallenge() {
        secondFactorOptions = []
        chosenSecondFactor = nil
        secondFactorReason = .mfa
    }

    /// Re-entry keeps the error and the picked factor (while still offered): Clerk can
    /// answer an attempt with another challenge, and clearing them would look like the step
    /// reset itself. The reason is always taken fresh, since a cleared client-trust check
    /// can be followed by a real MFA one.
    private func challenged(with options: [SecondFactor], reason: SecondFactorReason) -> EmailAuthState {
        let reentering = step == .secondFactor
        var next = self
        next.step = .secondFactor
        next.submitting = false
        next.error = reentering ? error : nil
        next.secondFactorOptions = options
        next.chosenSecondFactor = Self.factorToAnswer(
            keeping: reentering ? chosenSecondFactor : nil,
            from: options
        )
        next.secondFactorReason = reason
        next.challengeGeneration = challengeGeneration + 1
        return next
    }

    /// The one already picked while still offered, else a lone option, else nil.
    static func factorToAnswer(keeping current: SecondFactor?, from options: [SecondFactor]) -> SecondFactor? {
        if let current, options.contains(current) { return current }
        return options.count == 1 ? options.first : nil
    }

    private func failed(_ message: String) -> EmailAuthState {
        var next = self
        next.submitting = false
        next.error = message
        return next
    }
}

/// Drives the custom email-code sign-in, replacing the SDK's prebuilt `AuthView`. Pure
/// transitions live on `EmailAuthState`; this picks the auth call and applies the result.
@MainActor
@Observable
final class EmailAuthModel {
    private(set) var state = EmailAuthState()

    private let auth: ClerkAuthenticating

    /// Resolved in the body rather than as a default argument: a default argument is
    /// evaluated in a nonisolated context, and `RealClerkAuth()` is main-actor isolated.
    init(auth: ClerkAuthenticating? = nil) {
        self.auth = auth ?? RealClerkAuth()
    }

    /// Factors whose code has been sent, so a re-render or re-pick doesn't send another.
    /// Kept here, not behind the seam, because it records what the user has been shown.
    private var sentFactors: Set<SecondFactor> = []

    // MARK: - Steps

    func submitEmail(_ email: String) async {
        guard !state.submitting else { return }
        let address = email.trimmingCharacters(in: .whitespacesAndNewlines)
        state.email = address
        state.submitting = true
        state.error = nil
        apply(await auth.sendEmailCode(to: address), through: EmailAuthState.afterSendCode)
        // No `sendChosenSecondFactorIfNeeded` tail: a code send cannot produce a challenge.
    }

    /// Email step to password step, keeping the typed address. Nothing is sent.
    func usePassword(_ email: String) {
        guard !state.submitting else { return }
        auth.abandonAttempt()
        state.email = email.trimmingCharacters(in: .whitespacesAndNewlines)
        state.step = .password
        state.error = nil
        state.clearSecondFactorChallenge()
    }

    func submitPassword(email: String, password: String) async {
        guard !state.submitting else { return }
        let address = email.trimmingCharacters(in: .whitespacesAndNewlines)
        state.email = address
        state.submitting = true
        state.error = nil
        apply(
            await auth.signInWithPassword(email: address, password: password),
            through: EmailAuthState.afterAuthentication
        )
        // A password can land on a challenge whose lone email/SMS factor is pre-chosen,
        // with no picker tap to send its code.
        await sendChosenSecondFactorIfNeeded()
    }

    func submitCode(_ code: String) async {
        guard !state.submitting else { return }
        state.submitting = true
        state.error = nil
        apply(await auth.verifyCode(code), through: EmailAuthState.afterAuthentication)
        await sendChosenSecondFactorIfNeeded()
    }

    func submitNewPassword(_ password: String) async {
        guard !state.submitting else { return }
        state.submitting = true
        state.error = nil
        apply(await auth.createPassword(password), through: EmailAuthState.afterAuthentication)
    }

    /// Apply `transition` and keep `sentFactors` in step: a session or a fresh challenge
    /// clears it (nothing has been sent against a new challenge). The rest leave it, so a
    /// failed send stays distinguishable from one that never happened.
    private func apply(
        _ result: AuthActionResult,
        through transition: (EmailAuthState) -> (AuthActionResult) -> EmailAuthState
    ) {
        switch result {
        case .success, .needsSecondFactor:
            sentFactors.removeAll()
        case .needsPassword, .needsNewPassword, .cancelled, .failure:
            // Every read of the set is guarded on the second-factor step, so leaving it is
            // harmless for the password results.
            break
        }
        state = transition(state)(result)
    }

    // MARK: - Password reset

    /// Also the way in for an account that has no password at all.
    func startPasswordReset(_ email: String) async {
        guard !state.submitting else { return }
        let address = email.trimmingCharacters(in: .whitespacesAndNewlines)
        state.email = address
        state.submitting = true
        state.error = nil
        // `resetWasRequested` is set by `afterSendResetCode` on success only, not here: a
        // failed send would leave it set, and a forced reset would then read as requested.
        apply(await auth.sendPasswordResetCode(to: address), through: EmailAuthState.afterSendResetCode)
    }

    /// A resend is the same two calls (a fresh sign-in each time), so this delegates.
    func resendPasswordResetCode() async {
        await startPasswordReset(state.email)
    }

    /// Success is not a session but the right to set the password.
    func submitResetCode(_ code: String) async {
        guard !state.submitting else { return }
        state.submitting = true
        state.error = nil
        apply(await auth.verifyPasswordResetCode(code), through: EmailAuthState.afterAuthentication)
        // An account with two-step verification is challenged here before the password.
        await sendChosenSecondFactorIfNeeded()
    }

    func submitResetPassword(_ password: String) async {
        guard !state.submitting else { return }
        state.submitting = true
        state.error = nil
        apply(await auth.resetPassword(password), through: EmailAuthState.afterAuthentication)
        await sendChosenSecondFactorIfNeeded()
    }

    /// Email and phone codes are sent on the pick, not on a further tap.
    func chooseSecondFactor(_ factor: SecondFactor) async {
        guard !state.submitting, state.secondFactorOptions.contains(factor) else { return }
        state.chosenSecondFactor = factor
        state.error = nil
        await sendChosenSecondFactorIfNeeded()
    }

    func changeSecondFactor() {
        guard state.secondFactorOptions.count > 1 else { return }
        state.chosenSecondFactor = nil
        state.error = nil
    }

    /// The one place that bypasses `sentFactors`: the code may have expired or its send
    /// failed, and a lone `email_code` challenge (how client trust arrives) has no other
    /// retry. The generation moves only once Clerk has sent, so a failed resend leaves the
    /// working digits in the field.
    func resendSecondFactorCode() async {
        guard !state.submitting, let factor = state.chosenSecondFactor, factor.needsSending else { return }
        // Reachable: the auth layer drops the attempt for a status it has no step for, while
        // the step stays up. Said only when the screen is otherwise silent, since whatever
        // ended the challenge usually left a more specific message.
        guard auth.hasSecondFactorChallenge else {
            if state.error == nil {
                state.error = Self.noSecondFactorInProgress
            }
            return
        }
        let alreadySent = sentFactors.contains(factor)
        sentFactors.remove(factor)
        state.error = nil
        await sendChosenSecondFactorIfNeeded()
        if sentFactors.contains(factor) {
            state.challengeGeneration += 1
        } else if alreadySent {
            // A failed resend sent nothing, so the user's code is still live. Without the guard
            // the next verify's tail would silently send a replacement.
            sentFactors.insert(factor)
        }
    }

    func submitSecondFactorCode(_ code: String) async {
        guard !state.submitting, let factor = state.chosenSecondFactor else { return }
        state.submitting = true
        state.error = nil
        apply(
            await auth.verifySecondFactor(code, factor: factor),
            through: EmailAuthState.afterAuthentication
        )
        // A chained challenge clears `sentFactors` in `apply`, so this sends its code, even
        // for a repeat identical in factor and reason (which Android leaves to Resend). A
        // no-op on success and on an ordinary failure; it retries a send that had failed.
        await sendChosenSecondFactorIfNeeded()
    }

    /// Sends the chosen factor's code, once. Every path that can land on the second-factor
    /// step ends with this, or the step would ask for a code nobody sent.
    private func sendChosenSecondFactorIfNeeded() async {
        guard state.step == .secondFactor,
              let factor = state.chosenSecondFactor,
              factor.needsSending,
              !sentFactors.contains(factor),
              auth.hasSecondFactorChallenge
        else { return }

        state.submitting = true
        switch await auth.sendSecondFactorCode(factor) {
        case .success:
            sentFactors.insert(factor)
            state.submitting = false
        case .failure(let message):
            state.submitting = false
            state.error = message
        case .needsSecondFactor, .needsPassword, .needsNewPassword, .cancelled:
            // Not produced by a send; handled so the step cannot be left spinning.
            state.submitting = false
        }
    }

    /// `strategy` comes from `Clerk.oauthProviders`.
    func signInWithOAuth(strategy: String) async {
        guard !state.submitting else { return }
        auth.abandonAttempt()
        state.submitting = true
        state.error = nil
        apply(await auth.signInWithOAuth(strategy: strategy), through: EmailAuthState.afterAuthentication)
        await sendChosenSecondFactorIfNeeded()
    }

    /// "Use a different email". The address is kept, since the usual reason is a typo in it.
    func changeEmail() {
        auth.abandonAttempt()
        sentFactors.removeAll()
        state = EmailAuthState(step: .email, email: state.email)
    }

    func clearError() {
        state.error = nil
    }

    // MARK: - Errors

    private static let signInIncomplete = "This account needs another sign-in step that isn't available here."

    /// The step is up but the challenge behind it is gone (`RealClerkAuth.completeSignIn`
    /// drops the attempt for a status it has no step for).
    static let noSecondFactorInProgress = "No verification in progress. Sign in again."

    /// For a status this UI has no step for. Here rather than in `RealClerkAuth` so the
    /// sentence is testable without a Clerk value.
    static func incompleteSignInMessage(status: String) -> String {
        "\(signInIncomplete) (\(status))"
    }

    /// The link, the step title and `noPasswordOnAccount` share this so a message quoting
    /// the label cannot go stale. Not "Set a new password": the main case is an account
    /// that never had one. Same as Android's `SET_PASSWORD_ACTION`.
    static let setPasswordAction = "Set a password"

    /// Replaces Clerk's wording for `strategy_for_user_invalid`, which names a strategy the
    /// user never chose.
    static let noPasswordOnAccount = "This account has no password yet. Tap \"\(setPasswordAction)\" to choose one."

    /// What a send failing after its create leaves behind. Names no control because the two
    /// steps it shows on offer different ones. Unlike Android's, it does not say "go back":
    /// `ClerkAuthGateView` presents this screen with no navigation stack, so there is no back.
    static let noPasswordResetInProgress = "That password reset is no longer in progress. Start it again."

    /// A reset code send that failed after its sign-in was created, so no code is coming.
    /// Clerk's reason leads; the second sentence stops the step inviting a code that can
    /// never be answered.
    static func resetSendFailed(_ reason: String) -> String {
        let trimmed = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        // Clerk's short message is a fragment ("Too many requests"), so punctuate before
        // appending our sentence.
        guard let last = trimmed.last else { return resetMustRestart }
        let sentence = ".!?".contains(last) ? trimmed : "\(trimmed)."
        return "\(sentence) \(resetMustRestart)"
    }

    private static let resetMustRestart = "Start the reset again."

    /// " for <address>", or "" with no address: a forced reset reached through OAuth never
    /// sets `state.email`. Ported from Android's `forEmail`.
    static func forEmail(_ email: String) -> String {
        let trimmed = email.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "" : " for \(trimmed)"
    }

    /// A forced reset explains itself, or it reads as a rejection of a correct password.
    /// Neither sentence says the account has no password, and the forced one names no
    /// credential, since it is also reached through an emailed code or OAuth. Same as
    /// Android's `resetPasswordPrompt`.
    static func resetPasswordPrompt(email: String, wasForced: Bool) -> String {
        let choose = "Choose a password\(forEmail(email)), and you'll be signed in with it."
        guard wasForced else { return choose }
        return "A password has to be set on this account before you can sign in. \(choose)"
    }

    /// A challenge whose delivery methods this screen can't drive (Client Trust can use an
    /// email link). Split by reason so a device check does not read as an account problem.
    static func unanswerableChallenge(_ reason: SecondFactorReason) -> String {
        switch reason {
        case .mfa:
            "This account's two-step verification isn't supported here. Use a method you can complete on another device."
        case .deviceVerification:
            "This device needs to be confirmed, and the way this account does that isn't supported here."
        }
    }

    #if DEBUG
    /// Preview-only: `state` is settable only from this file.
    static func previewAtCodeStep(email: String) -> EmailAuthModel {
        let model = EmailAuthModel()
        model.state = EmailAuthState(step: .code, email: email)
        return model
    }
    #endif
}
