import Foundation

/// Outcome of a one-shot auth action. Mirrors Android's `AuthActionResult`
/// (`ClerkAuth.kt`) so both clients' sign-in flows read the same.
///
/// No ClerkKit type appears here, which is what lets `FakeClerkAuth` exist. See
/// `ClerkAuthenticating`.
enum AuthActionResult: Equatable {
    case success
    /// The user dismissed the OAuth sheet. A decision, not a failure: the screen
    /// stays exactly as it was and nothing is said.
    ///
    /// Its own case rather than a `failure` the call site filters out, because
    /// the filtering used to live in `signInWithOAuth` as an early return, where
    /// it was invisible to the transitions. As a case, every switch over this
    /// type has to say what it does with a cancellation.
    case cancelled
    /// The first factor was accepted but Clerk wants a code before it will issue
    /// a session, so there is no session yet. Carries the factors this screen can
    /// actually drive, already filtered, and never empty, since a challenge we
    /// can offer no answer to is a failure instead.
    ///
    /// The reason says which of the two situations this is. They are answered
    /// through the same pair of calls but read completely differently to the
    /// user, so it is what the step's wording keys on.
    ///
    /// Unlike Android's `AuthActionResult.NeedsSecondFactor`, which defaults the
    /// reason to MFA, this one requires it: a Swift enum case cannot carry a
    /// default at all. The cost is small here, since `completeSignIn` is the only
    /// thing that builds one, and it means a reason that failed to be threaded
    /// through fails to compile rather than quietly showing MFA copy.
    case needsSecondFactor([SecondFactor], SecondFactorReason)
    /// The address is verified but the instance requires a password at sign-up,
    /// so the account doesn't exist yet. Answered by `submitNewPassword`.
    case needsPassword
    /// Clerk is waiting for a new password on an *existing* account, and no
    /// session exists yet. Answered by `submitResetPassword`.
    ///
    /// One case rather than several, because every call that finishes with a
    /// `SignIn` can raise it: `completeSignIn` is the only thing that builds it.
    /// The ordinary route is `verifyPasswordResetCode`, where the emailed reset
    /// code was accepted and that is what earns the right to set the password.
    /// The rest are Clerk demanding one on a sign-in that was otherwise good, and
    /// `signInWithPassword` is only the most obvious: a first-factor code, an
    /// OAuth round-trip, and a second-factor verify can all land here too.
    ///
    /// Distinct from `needsPassword`, which finishes a registration that has no
    /// account behind it yet. Which of the two situations this is cannot be read
    /// off the result, which is why the flow records it (`resetWasRequested`).
    case needsNewPassword
    case failure(String)
}

/// Why Clerk is asking for a second code. Both arrive as a non-terminal `SignIn`
/// carrying `supportedSecondFactors`, and both are answered with the MFA prepare
/// and verify calls, so the difference is entirely in what the user is told.
/// Mirrors Android's `SecondFactorReason`.
enum SecondFactorReason: Equatable {
    /// `needs_second_factor`: the account has two-step verification switched on,
    /// and this is that second step.
    case mfa

    /// `needs_client_trust`: Clerk's Client Trust does not recognize this device,
    /// so it wants the sign-in confirmed out of band before issuing a session.
    /// Nothing to do with the account's own settings, and the user may never have
    /// set up two-step verification, which is why calling it that would be wrong.
    /// It fires only on password sign-ins, and Clerk's docs say not at all once an
    /// account has MFA on, since then `mfa` covers the same ground. Nothing in the
    /// SDK enforces that, so the two are handled as independently reachable.
    case deviceVerification

    /// The step's heading.
    ///
    /// Not "Two-step verification" for a device check: Client Trust fires on
    /// accounts that have never switched that on, and telling someone to enter a
    /// code from a feature they do not use is how a solvable step reads as a dead
    /// end.
    var title: String {
        switch self {
        case .mfa: "Two-step verification"
        case .deviceVerification: "Confirm this device"
        }
    }

    /// Where the code is coming from, and for device verification why it is being
    /// asked for at all. `source` is the chosen factor's own sentence, or the
    /// chooser's when no factor has been picked yet.
    ///
    /// The MFA wording can stay terse because the user set two-step verification
    /// up and is expecting it. Client Trust arrives unannounced on an ordinary
    /// password sign-in, so the device sentence leads: without it the screen looks
    /// like the password was wrong.
    ///
    /// The `source` sentences stay iOS's own rather than being aligned word for
    /// word with Android's. It is the structure that mirrors, not the strings.
    func prompt(_ source: String) -> String {
        switch self {
        case .mfa: source
        case .deviceVerification: "This is the first sign-in on this device, so it needs confirming. \(source)"
        }
    }
}

/// A second factor an account can be challenged with, mapped from Clerk's
/// `supportedSecondFactors` strategies. Mirrors Android's `SecondFactor`.
enum SecondFactor: String, CaseIterable, Identifiable, Equatable {
    case emailCode = "email_code"
    case phoneCode = "phone_code"
    case totp
    case backupCode = "backup_code"

    var id: String { rawValue }

    /// Email and phone codes have to be *sent* before they can be verified;
    /// TOTP and backup codes are read off something the user already holds, so
    /// they go straight to entry.
    var needsSending: Bool {
        self == .emailCode || self == .phoneCode
    }

    /// How many digits the code has, or nil when it isn't a fixed-width numeric
    /// code — backup codes are alphanumeric and vary, so they get a plain field
    /// rather than the segmented one.
    var digitCount: Int? {
        self == .backupCode ? nil : 6
    }

    /// Label for the picker, when the account offers more than one.
    var title: String {
        switch self {
        case .emailCode: "Email code"
        case .phoneCode: "Text message"
        case .totp: "Authenticator app"
        case .backupCode: "Backup code"
        }
    }

    /// What to say above the field once this factor is the one being answered.
    var prompt: String {
        switch self {
        case .emailCode: "We sent a 6-digit code to your email."
        case .phoneCode: "We sent a 6-digit code to your phone."
        case .totp: "Enter the 6-digit code from your authenticator app."
        // Deliberately does not say "when you turned on two-step verification".
        // This sentence composes under `SecondFactorReason.prompt`, and a device
        // check would then blame a feature the account may never have enabled,
        // which is the exact confusion that reason exists to prevent.
        case .backupCode: "Enter one of your saved backup codes."
        }
    }

    /// The factors from a challenge that this screen can answer, in a stable
    /// order — anything Clerk offers that isn't one of these (a passkey, a
    /// hardware key) is dropped, which is what keeps the step honest about what
    /// it can finish. An empty result means the challenge is unanswerable here.
    ///
    /// The Clerk-typed counterpart, `offered(by:)`, lives with `RealClerkAuth`:
    /// it needs the `Factor` behind each strategy, and that must not cross the
    /// seam.
    static func offered(strategies: [String]) -> [SecondFactor] {
        let offered = Set(strategies)
        return allCases.filter { offered.contains($0.rawValue) }
    }
}

/// A social sign-in provider the Clerk instance has enabled, surfaced to the
/// sign-in screen as one button. Sourced from the Clerk *environment config*
/// (the dashboard), not hard-coded — which buttons appear is driven entirely by
/// what the backend has turned on. `strategy` is Clerk's stable identifier
/// (e.g. `oauth_google`) and is the token passed back to `signInWithOAuth`.
struct OAuthProviderInfo: Equatable, Identifiable {
    let strategy: String
    let name: String

    var id: String { strategy }
}

/// What the sign-in screen is showing. The text fields themselves stay local
/// `@State` in the views (as they are local `rememberSaveable` state on
/// Android); only the step, the address a code was sent to, and the in-flight /
/// error status live here.
struct EmailAuthState: Equatable {
    enum Step {
        case email
        case password
        case code
        /// Setting the password that finishes a new account, after the emailed
        /// code verified the address.
        case createPassword
        /// A code challenge after a first factor was accepted, from either
        /// `needs_second_factor` (the account has two-step verification on) or
        /// `needs_client_trust` (Clerk does not recognize this device). One step
        /// for both, because they are answered identically; `secondFactorReason`
        /// is what makes it say which.
        case secondFactor
        /// Answering the emailed `reset_password_email_code`. Its own step rather
        /// than a mode on `code`: the two are answered by different calls against
        /// differently prepared attempts, and only this one has a password step
        /// waiting behind it.
        case resetCode
        /// Choosing the password that finishes a reset, on an account that already
        /// exists. `createPassword` is the counterpart for one that does not.
        case resetPassword
    }

    var step: Step = .email
    /// The address the code was sent to — shown on the code step.
    var email: String = ""
    /// A request is in flight: disables the submit button, shows its spinner,
    /// and guards against double submits.
    var submitting: Bool = false
    /// Last failure surfaced to the user (nil when clear).
    var error: String?
    /// The factors the account can be challenged with. Only ever populated on
    /// the `secondFactor` step.
    var secondFactorOptions: [SecondFactor] = []
    /// The factor being answered, or nil while the user is still choosing. A
    /// single-option challenge skips the choice and lands here directly.
    var chosenSecondFactor: SecondFactor?
    /// Why the `secondFactor` step is being shown, which is the whole of what the
    /// step's wording keys on. Meaningless off that step, and left at `.mfa`
    /// there rather than made optional, since the step's copy needs an answer
    /// either way.
    var secondFactorReason: SecondFactorReason = .mfa
    /// Counts the codes this step has asked for, so the view can tell two
    /// challenges apart when the factor and the reason are both unchanged. Bumped
    /// on entering or re-entering the step, and by a resend that Clerk accepted,
    /// which is the one send with no step change behind it.
    ///
    /// Clerk can answer an attempt with a fresh challenge rather than a failure,
    /// and `submitSecondFactorCode` then emails a new code. Without this the code
    /// field would keep the six digits just spent, looking filled in and correct
    /// while the code that actually works sits unread in the user's inbox.
    var challengeGeneration: Int = 0
    /// The same idea as `challengeGeneration`, for the reset code's field: bumped
    /// by every send Clerk accepted, so a resend clears the digits belonging to
    /// the code it just replaced. Separate rather than shared, because the two
    /// counters guard two different fields on two different steps and the
    /// invariants around this one are the resend's alone.
    var resetCodeGeneration: Int = 0
    /// Whether the user asked for a password reset, set by `afterSendResetCode`
    /// once Clerk has actually sent a code, and cleared by every transition that
    /// rebuilds this state from scratch (`changeEmail`, and the reset on success).
    ///
    /// A recorded fact rather than something derived from the step being left,
    /// which is what two earlier attempts did and neither could get right. The
    /// reset step is reachable from the reset code, from a correct password, from
    /// a first-factor code, from OAuth, and from the second-factor step standing
    /// in front of any of them, so the step behind it does not determine who
    /// asked. Android derives it (`ShellViewModel.kt:279`) and so words a 2FA
    /// account's requested reset as demanded; preserving the derivation instead,
    /// as this did a moment ago, inverts the error and words a demanded reset as
    /// requested. Only the flow itself knows, so the flow records it.
    var resetWasRequested: Bool = false

    /// Whether Clerk demanded the reset rather than the user asking for it, which
    /// is the whole of what the reset step's wording keys on.
    ///
    /// A forced reset lands on a *correct* password, so without this the step
    /// reads as a rejection of what was just typed. Meaningless off the
    /// `resetPassword` step.
    var resetWasForced: Bool { !resetWasRequested }
}

extension EmailAuthState {
    /// The clickwrap targets shown under the email step's button.
    ///
    /// Load-bearing rather than decorative: the Clerk instance has legal consent
    /// enabled, so `signUp` has to send `legal_accepted` or the sign-up never
    /// leaves `missingRequirements`, and sending it is only honest while a user
    /// cannot register without having been shown this. Android pairs the same
    /// two links with the same flag in `AuthGateScreen.kt` — change one side and
    /// the other has to follow.
    static let termsURL = "https://pipette.liquid.ai/terms"
    static let privacyPolicyURL = "https://www.liquid.ai/privacy-policy"

    /// The instance's password floor, matching Android's `MIN_PASSWORD_LENGTH`.
    /// Stated up front on both password steps rather than as a post-submit error,
    /// because the rule is knowable before the user commits to anything, and
    /// understating it would invite a password Clerk then rejects.
    ///
    /// On the state rather than in the view because it is instance policy rather
    /// than layout, and because the copy that states it and the rule that
    /// enforces it have to read the same number. `SetPasswordStep` is the single
    /// reader today, since both password steps share it.
    static let minimumPasswordLength = 12

    /// Whether the email step's actions can fire. Both of them — registering and
    /// taking the password path — need an address and no request in flight, so
    /// they share one rule rather than drifting apart.
    static func canSubmitEmail(_ email: String, submitting: Bool) -> Bool {
        !email.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !submitting
    }

    /// Transition after a code send: advance to the code step on success, else
    /// clear the spinner and surface the error while staying on the email step.
    func afterSendCode(_ result: AuthActionResult) -> EmailAuthState {
        switch result {
        case .success:
            var next = self
            next.step = .code
            next.submitting = false
            next.error = nil
            return next
        case .needsSecondFactor(let options, let reason):
            // Sending a code can't produce a challenge, but the result type is
            // shared; treat it the same rather than dropping it on the floor.
            return challenged(with: options, reason: reason)
        case .needsPassword:
            // Likewise: Clerk asks for the password after the address is
            // verified, not when the code goes out.
            return askingToCreatePassword()
        case .needsNewPassword:
            // Likewise again: a first-factor code send cannot land on a forced
            // reset. Routed rather than dropped, for the same reason.
            return askingToResetPassword()
        case .cancelled:
            // Only OAuth can be cancelled, and it does not come through here.
            // Answered anyway, and answered the same way, so a route that ever
            // did would clear its spinner rather than hang on one.
            return settled()
        case .failure(let message):
            return failed(message)
        }
    }

    /// Transition after anything that completes a sign-in — verifying a code,
    /// a password sign-in, an OAuth round-trip. Success resets to the initial
    /// state: the signed-in user flips the gate, so no code-step UI should
    /// linger behind it. A failure just surfaces on the current step.
    func afterAuthentication(_ result: AuthActionResult) -> EmailAuthState {
        switch result {
        case .success:
            return EmailAuthState()
        case .needsSecondFactor(let options, let reason):
            return challenged(with: options, reason: reason)
        case .needsPassword:
            return askingToCreatePassword()
        case .needsNewPassword:
            // The reset code cleared, or Clerk demanded a password on a sign-in
            // that was otherwise good: a correct password, a first-factor code,
            // or an OAuth round-trip. One step answers all of them, and
            // `askingToResetPassword` is what tells the asked-for case from the
            // rest.
            return askingToResetPassword()
        case .cancelled:
            // The user dismissed the OAuth sheet. Leave the screen exactly as it
            // was and say nothing: they chose this, and an error here would read
            // as the provider having rejected them.
            return settled()
        case .failure(let message):
            return failed(message)
        }
    }

    /// Clear the spinner and change nothing else. For an outcome that is neither
    /// progress nor a problem, so the step, the error already on screen, and any
    /// challenge all stay as they were.
    private func settled() -> EmailAuthState {
        var next = self
        next.submitting = false
        return next
    }

    /// Transition after a reset code send, whether it is the first or a resend.
    /// The generation moves only on a send Clerk accepted, so a failed resend
    /// leaves the digits already in the field alone: they still belong to the
    /// live code.
    func afterSendResetCode(_ result: AuthActionResult) -> EmailAuthState {
        switch result {
        case .success:
            var next = self
            next.step = .resetCode
            next.submitting = false
            next.error = nil
            next.resetCodeGeneration = resetCodeGeneration + 1
            // Recorded here rather than where the reset was requested, because
            // this is the first point at which it is true: Clerk has sent a reset
            // code, so the user is in a reset. A send that failed leaves them
            // wherever they were, and must not leave this behind. `.resetCode` is
            // only reachable through this branch, so nothing downstream can see
            // the step without the record.
            next.resetWasRequested = true
            next.clearSecondFactorChallenge()
            return next
        case .needsNewPassword:
            // Sending a code cannot skip to the password itself, but the result
            // type is shared; route it rather than drop it.
            return askingToResetPassword()
        case .needsSecondFactor(let options, let reason):
            return challenged(with: options, reason: reason)
        case .needsPassword:
            return askingToCreatePassword()
        case .cancelled:
            // Only OAuth can be cancelled, and it does not come through here.
            return settled()
        case .failure(let message):
            return failed(message)
        }
    }

    /// Move to the create-password step. The address is already verified, so
    /// this is the last thing between the user and an account.
    ///
    /// Named for *which* password choice it is: this one builds an account that
    /// does not exist yet, where `askingToResetPassword` replaces the password on
    /// one that does. They were one name and one step apart, which is exactly the
    /// pair worth keeping distinguishable.
    private func askingToCreatePassword() -> EmailAuthState {
        var next = self
        next.step = .createPassword
        next.submitting = false
        next.error = nil
        next.clearSecondFactorChallenge()
        return next
    }

    /// Move to the reset-password step, or stay on it.
    ///
    /// Nothing about who asked is decided here. `resetWasRequested` was recorded
    /// when the reset started and is carried by `var next = self` like every
    /// other field, which is what makes this correct from all five routes into
    /// the step rather than only the two a step-based rule can tell apart.
    ///
    /// The error is kept across a re-entry for the same reason `challenged` keeps
    /// it: the message that explained why would otherwise be the only thing on
    /// screen accounting for the step not having moved. Defensive in practice,
    /// since every caller clears the error before awaiting.
    private func askingToResetPassword() -> EmailAuthState {
        var next = self
        next.step = .resetPassword
        next.submitting = false
        next.error = step == .resetPassword ? error : nil

        next.clearSecondFactorChallenge()
        return next
    }

    /// Forget an outstanding challenge on the way to a step that is not the
    /// second-factor one. Leaving these set would let that step render against a
    /// challenge that no longer exists.
    ///
    /// No caller reaches it with a challenge still live today, since each has
    /// already replaced or dropped the attempt behind the seam, by its own call
    /// or through `abandonAttempt`. Kept because every one
    /// of them is a step change away from the challenge and a divergence here
    /// would be silent. Android's
    /// `enterCreatePassword` clears the options and the selection; its `copy()`
    /// carries the reason forward, which is the same latent staleness one field
    /// short.
    mutating func clearSecondFactorChallenge() {
        secondFactorOptions = []
        chosenSecondFactor = nil
        secondFactorReason = .mfa
    }

    /// Move to the second-factor step, or stay on it.
    ///
    /// Re-entry is not a reset. Clerk can answer an attempt with another
    /// `needs_second_factor` rather than a failure, and clearing the choice there
    /// drops the user back to the chooser — with the message that explained why
    /// cleared too, so the step looks like it reset itself for no reason. Keep
    /// both, and keep the picked factor as long as the account still offers it.
    ///
    /// The reason is the exception: it is taken from the result every time rather
    /// than kept across a re-entry. Both challenges are answered by the same call,
    /// so Clerk can follow a cleared client-trust attempt with a real MFA one, and
    /// a step still headed "Confirm this device" would be describing the wrong
    /// challenge.
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

    /// Which factor the step should be answering: the one already picked when it
    /// is still on offer, otherwise a lone option (a chooser with one entry is
    /// just a speed bump), otherwise nothing and let the user choose.
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

/// Drives the custom email-code sign-in, replacing the SDK's prebuilt `AuthView`.
/// The pure transitions live on `EmailAuthState`; this type decides which auth
/// call to make and what the result means for the screen.
///
/// It no longer touches ClerkKit, and no longer holds the attempt in progress.
/// Both moved behind `ClerkAuthenticating` to `RealClerkAuth`, which is what
/// lets these flows be driven by a fake. The one piece of cross-call state left
/// here is `sentFactors`, which is about what the user has been shown rather
/// than about Clerk.
@MainActor
@Observable
final class EmailAuthModel {
    private(set) var state = EmailAuthState()

    /// The Clerk-touching half, behind a protocol so tests can drive this whole
    /// flow. The in-progress attempt lives there, not here: see
    /// `ClerkAuthenticating` for why nothing of Clerk's crosses back.
    private let auth: ClerkAuthenticating

    /// Defaults to the real implementation, so `EmailAuthModel()` in a view is
    /// unchanged and only tests and previews pass anything.
    ///
    /// Resolved in the body rather than as a default argument: a default
    /// argument expression is evaluated in a *nonisolated* context even when the
    /// type is `@MainActor`, and `RealClerkAuth()` is main-actor isolated.
    init(auth: ClerkAuthenticating? = nil) {
        self.auth = auth ?? RealClerkAuth()
    }

    /// Factors whose code has already been sent, so re-rendering the step or
    /// re-picking the same factor doesn't email a second code.
    ///
    /// Stays on this side of the seam although it is about sending. What it
    /// actually records is what the *user* has been shown, and the questions it
    /// answers (was the step re-entered, has this person already been sent
    /// something) are the model's. Splitting it across the boundary is how
    /// `resendSecondFactorCode`'s failure handling would quietly come apart.
    private var sentFactors: Set<SecondFactor> = []

    // MARK: - Steps

    /// Email step, passwordless path: email a code for `email`, then move to the
    /// code step.
    func submitEmail(_ email: String) async {
        guard !state.submitting else { return }
        let address = email.trimmingCharacters(in: .whitespacesAndNewlines)
        state.email = address
        state.submitting = true
        state.error = nil
        apply(await auth.sendEmailCode(to: address), through: EmailAuthState.afterSendCode)
        // No `sendChosenSecondFactorIfNeeded` tail here, unlike the paths that can
        // actually be challenged: sending a first-factor code returns only
        // `.success` or `.failure`, so the challenge branch in `afterSendCode` is
        // defensive and the helper's own guard would decline anyway.
    }

    /// Email step → password step, carrying the typed address so the next screen
    /// doesn't ask for it again. Nothing is sent: this only changes what is on
    /// screen.
    func usePassword(_ email: String) {
        guard !state.submitting else { return }
        auth.abandonAttempt()
        state.email = email.trimmingCharacters(in: .whitespacesAndNewlines)
        state.step = .password
        state.error = nil
        state.clearSecondFactorChallenge()
    }

    /// Password step: sign in with the password first factor. No code is sent and
    /// the code step is never reached.
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
        // A password can clear the first factor and land on a challenge just as an
        // emailed code can, and a lone email/SMS factor is pre-chosen with no
        // picker tap to send it — so this tail is not optional here.
        await sendChosenSecondFactorIfNeeded()
    }

    /// Code step: verify the emailed code and, on success, land the session.
    func submitCode(_ code: String) async {
        guard !state.submitting else { return }
        state.submitting = true
        state.error = nil
        apply(await auth.verifyCode(code), through: EmailAuthState.afterAuthentication)
        await sendChosenSecondFactorIfNeeded()
    }

    /// Create-password step: set the password the instance requires at sign-up.
    /// On success the account exists and the session is active.
    func submitNewPassword(_ password: String) async {
        guard !state.submitting else { return }
        state.submitting = true
        state.error = nil
        apply(await auth.createPassword(password), through: EmailAuthState.afterAuthentication)
    }

    /// Move the state on by `transition`, and keep `sentFactors` in step with the
    /// same result.
    ///
    /// The bookkeeping is here because `completeSignIn` used to do it, back when
    /// it held the attempt and this set in one place. Two results clear it and
    /// they are named rather than defaulted: a session ends the challenge
    /// outright, and a fresh challenge is a fresh attempt with nothing sent
    /// against it yet. The rest leave the record alone, which is what lets a
    /// failed send stay distinguishable from one that never happened.
    private func apply(
        _ result: AuthActionResult,
        through transition: (EmailAuthState) -> (AuthActionResult) -> EmailAuthState
    ) {
        switch result {
        case .success, .needsSecondFactor:
            sentFactors.removeAll()
        case .needsPassword, .needsNewPassword, .cancelled, .failure:
            // The two password results leave the second-factor step behind
            // entirely, so what it recorded stops being asked: every read of this
            // set is guarded on that step. Left alone rather than cleared, to keep
            // the rule above to the two cases that genuinely turn on it.
            break
        }
        state = transition(state)(result)
    }

    // MARK: - Password reset

    /// Password step → reset code step: email a `reset_password_email_code` for
    /// `email`. This is the way in for an account that has no password at all, as
    /// much as for a forgotten one.
    func startPasswordReset(_ email: String) async {
        guard !state.submitting else { return }
        let address = email.trimmingCharacters(in: .whitespacesAndNewlines)
        state.email = address
        state.submitting = true
        state.error = nil
        // Note what is *not* here: the record that the user asked for this.
        // `afterSendResetCode` sets it, on the success branch only. Setting it up
        // front looks equivalent and is not: a send that fails leaves the user on
        // the password step with the flag set, and the correct password they then
        // submit can come back `needs_new_password`, which would be worded as a
        // reset they asked for. The flag exists to prevent exactly that.
        apply(await auth.sendPasswordResetCode(to: address), through: EmailAuthState.afterSendResetCode)
    }

    /// Ask for another reset code, from the step already showing one.
    ///
    /// Delegates rather than duplicating, because a resend *is* the same two
    /// calls: the auth layer creates a fresh sign-in each time, so unlike
    /// `resendSecondFactorCode` there is no already-sent guard here to defeat and
    /// no parked attempt this could find missing. What the two share is the rule
    /// that matters: the generation moves only once Clerk has sent, so a resend
    /// that fails leaves the digits on screen alone. They are still the live
    /// code's when the create failed, and when the send failed the error says the
    /// reset has to start again.
    func resendPasswordResetCode() async {
        await startPasswordReset(state.email)
    }

    /// Reset code step: answer the emailed code. Success is not a session. It is
    /// the right to set the password, which lands on the reset-password step.
    func submitResetCode(_ code: String) async {
        guard !state.submitting else { return }
        state.submitting = true
        state.error = nil
        apply(await auth.verifyPasswordResetCode(code), through: EmailAuthState.afterAuthentication)
        // A cleared reset code can be answered with a challenge rather than the
        // password step: an account with two-step verification is asked for its
        // second factor here, before Clerk will let the password be set. So this
        // tail is not optional, and the step it lands on needs a code sent.
        await sendChosenSecondFactorIfNeeded()
    }

    /// Reset-password step: set the new password. On success the session is
    /// active, unless the account answers with a second factor first.
    func submitResetPassword(_ password: String) async {
        guard !state.submitting else { return }
        state.submitting = true
        state.error = nil
        apply(await auth.resetPassword(password), through: EmailAuthState.afterAuthentication)
        await sendChosenSecondFactorIfNeeded()
    }

    /// Second-factor step: the user picked which factor to answer. Email and
    /// phone codes are sent here rather than on a further tap.
    func chooseSecondFactor(_ factor: SecondFactor) async {
        guard !state.submitting, state.secondFactorOptions.contains(factor) else { return }
        state.chosenSecondFactor = factor
        state.error = nil
        await sendChosenSecondFactorIfNeeded()
    }

    /// Back to the list of factors, for an account that offers more than one.
    func changeSecondFactor() {
        guard state.secondFactorOptions.count > 1 else { return }
        state.chosenSecondFactor = nil
        state.error = nil
    }

    /// Ask Clerk for another code for the factor already chosen.
    ///
    /// `sentFactors` is what keeps a re-render or a re-pick of the same factor
    /// from emailing a second code, and this is the one place that has to defeat
    /// it: the code on screen may have expired, or the send it recorded may have
    /// failed. Without this a challenge offering `email_code` alone, which is the
    /// shape a client-trust check arrives in, has no retry at all: the chooser is
    /// hidden when there is nothing to switch to, and the only other way out
    /// restarts from the address. Android's `Resend code` covers the same case.
    ///
    /// The generation moves only once Clerk has sent something, which
    /// `sentFactors` records. Bumping it up front would clear a field whose
    /// digits are still the working code when the resend is what fails.
    func resendSecondFactorCode() async {
        guard !state.submitting, let factor = state.chosenSecondFactor, factor.needsSending else { return }
        // Nothing to ask a code against. The auth layer drops the attempt for a status it has no step for, while
        // `failed` leaves this step standing, so this is reachable. Said only when the screen is otherwise silent:
        // whatever ended the challenge left a message that names the status, which is more than this sentence can
        // manage. What must not happen is the send-nothing-say-nothing case below, which is why this branch exists.
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
            // Put the guard back, because a resend that failed sent nothing: the code the user already has is still
            // the live one. Left off, the auto-send tail on the next verify would email a replacement without saying
            // so, and without the generation bump that clears the digits the replaced code left in the field.
            sentFactors.insert(factor)
        }
    }

    /// Second-factor step: answer the challenge and land the session.
    func submitSecondFactorCode(_ code: String) async {
        guard !state.submitting, let factor = state.chosenSecondFactor else { return }
        state.submitting = true
        state.error = nil
        apply(
            await auth.verifySecondFactor(code, factor: factor),
            through: EmailAuthState.afterAuthentication
        )
        // Answering a challenge can produce another one, and `apply`
        // clears `sentFactors` when it does. Without this tail that second
        // step would ask for a code nobody sent, which is the dead end every
        // other entry point ends with this call to avoid. A no-op on success,
        // since the step is gone, and on the ordinary failure, since the factor
        // is still in `sentFactors` from the code being answered.
        //
        // It does send on the failure whose challenge never got a code out:
        // `sentFactors` records the send, not the challenge, so a send that failed
        // leaves it empty and this retries it. Deliberate, since the step needs a
        // code either way, and the digits it leaves in the field are ones the
        // error above just told the user were wrong.
        //
        // Android arrives at the same place from the other side. Its
        // `secondFactorToDeliver` fires on entering the step and on a challenge
        // whose reason or factor moved, so a chained one gets its code there too;
        // what it cannot recognize is a repeat identical in both, which it leaves
        // to its Resend button. Here that case needs nothing of its own:
        // `apply` clears `sentFactors` for every challenge, so an identical
        // repeat sends like any other, and `challengeGeneration` clears the
        // digits the retired code left in the field.
        await sendChosenSecondFactorIfNeeded()
    }

    /// Sends the chosen factor's code, once. A step that asks for a code it never
    /// sent is a dead end, so every path that can land on the second-factor step
    /// ends by calling this.
    ///
    /// Whether to send is decided here and the sending is done behind the seam,
    /// which is the split that matters: `sentFactors` is what this reads, and it
    /// is about what the user has been shown rather than about Clerk.
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
            // A send reports only success or failure. Answered rather than
            // ignored so the step cannot be left spinning by a result this was
            // not written for.
            state.submitting = false
        }
    }

    /// Run the redirect-based OAuth flow for `strategy` (a value from
    /// `Clerk.oauthProviders`). Success lands the session exactly as verifying a
    /// code does. There is no separate verify step.
    func signInWithOAuth(strategy: String) async {
        guard !state.submitting else { return }
        // Any in-progress email-code attempt is abandoned when the user picks a
        // social provider instead.
        auth.abandonAttempt()
        state.submitting = true
        state.error = nil
        apply(await auth.signInWithOAuth(strategy: strategy), through: EmailAuthState.afterAuthentication)
        await sendChosenSecondFactorIfNeeded()
    }

    /// Abandon the in-progress attempt and go back to the email step ("Use a
    /// different email"). The address is kept so the field comes back filled in
    /// — the usual reason to come back here is a typo in it, and Android's step
    /// restores it the same way.
    func changeEmail() {
        auth.abandonAttempt()
        sentFactors.removeAll()
        state = EmailAuthState(step: .email, email: state.email)
    }

    /// Clear a surfaced error as soon as the user edits the field it referred to.
    func clearError() {
        state.error = nil
    }

    // MARK: - Errors

    private static let signInIncomplete = "This account needs another sign-in step that isn't available here."

    /// The step is still up but the challenge behind it is gone, which
    /// `RealClerkAuth.completeSignIn` can leave when it drops the attempt for a
    /// status it has no step for. Shared by the two actions that can be taken
    /// from there, and by the auth layer, so all of them say the same thing about
    /// the same state.
    static let noSecondFactorInProgress = "No verification in progress. Sign in again."

    /// Why a sign-in stopped short of a session, for a status this UI has no step
    /// for. `RealClerkAuth.incompleteSignIn` is the caller and carries the
    /// reasoning, since it is the side that knows which statuses are answered
    /// before this is reached.
    ///
    /// The copy stays here while the `SignIn`-typed caller
    /// (`RealClerkAuth.incompleteSignIn`) sits behind the seam, so the sentence
    /// is testable without a Clerk value.
    static func incompleteSignInMessage(status: String) -> String {
        "\(signInIncomplete) (\(status))"
    }

    /// Names the reset route wherever the user meets it: the link on the password
    /// step, the title of the step that finishes it, and `noPasswordOnAccount`,
    /// the message that points a passwordless account at it. One string for all
    /// three, because a message quoting a label is wrong the moment the label
    /// changes and nothing else would catch it.
    ///
    /// Not "Set a new password". The flow's primary case is an account that has
    /// never had one, and the message beside the link says exactly that, so "new"
    /// would contradict the sentence pointing at it. Under-describing the
    /// forgotten-password case is the cheaper of the two faults. Shared with
    /// Android's `SET_PASSWORD_ACTION`, word for word.
    static let setPasswordAction = "Set a password"

    /// Answers the one failure the password step has that is not about the
    /// password: the account has none. Names the situation and the way out of it,
    /// since Clerk's own wording for `strategy_for_user_invalid` describes a
    /// strategy the user never chose and cannot see.
    ///
    /// `setPasswordAction` is the only control named. The step's other exit is
    /// labelled "Use a different email", which is not what to tell someone who
    /// typed the right address.
    static let noPasswordOnAccount = "This account has no password yet. Tap \"\(setPasswordAction)\" to choose one."

    /// The reset is gone but a step that belongs to it is still up, which is what
    /// a send failing after its create leaves behind.
    ///
    /// Names no control, because it surfaces on two steps offering different ones:
    /// the code step, which has a resend, and the password step, whose only way
    /// out is back to the address.
    ///
    /// Android's counterpart says to go *back*, which is right there and wrong
    /// here: it has a system back button, and this screen has none. It is
    /// presented bare by `ClerkAuthGateView`, with no navigation stack behind it,
    /// so telling an iOS user to go back names a gesture that does not exist on
    /// the two steps where they are already stuck.
    static let noPasswordResetInProgress = "That password reset is no longer in progress. Start it again."

    /// A reset code send that failed after its sign-in was created, so nothing is
    /// parked and there is no code coming.
    ///
    /// Clerk's reason leads, since it is the half that explains why (a rate limit,
    /// or the connection dropping between the two calls); the sentence after it is
    /// the half only this app knows, and without it the step would sit there
    /// inviting a code that can never be answered. Named no control, for the same
    /// reason `noPasswordResetInProgress` does not.
    static func resetSendFailed(_ reason: String) -> String {
        let trimmed = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        // Clerk's text is not reliably a sentence: the display message prefers the
        // long form but falls back to the short one, which is a fragment ("Too
        // many requests"). Joining onto that unpunctuated would read as one
        // run-on line, and this is the only place in the flow where prose of
        // Clerk's is joined to prose of ours rather than passed through whole.
        guard let last = trimmed.last else { return resetMustRestart }
        let sentence = ".!?".contains(last) ? trimmed : "\(trimmed)."
        return "\(sentence) \(resetMustRestart)"
    }

    private static let resetMustRestart = "Start the reset again."

    /// " for <address>", or nothing at all when there is no address to name.
    ///
    /// The guard is load-bearing on the reset step and defensive on the
    /// create-password one. A forced reset is reachable through OAuth, which
    /// never writes `state.email` (the email step's field is local to the view),
    /// so that step can be reached with nothing to interpolate and would
    /// otherwise open with a leading space. Every route to the create step runs
    /// through the email field, so it always has one; it shares this anyway,
    /// because two steps asking the same thing should not word it two ways.
    /// Ported from Android's `forEmail`.
    static func forEmail(_ email: String) -> String {
        let trimmed = email.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "" : " for \(trimmed)"
    }

    /// What the reset-password step says, which turns on whether the user asked
    /// for the reset or Clerk demanded it.
    ///
    /// The requested route needs no cause at all: the user tapped the link,
    /// answered the code, and is expecting this screen. The demanded route needs
    /// one badly, since the password just typed was *correct*, and a screen that
    /// asks for another without a word reads as a rejection.
    ///
    /// Neither sentence claims the account has no password, and neither says
    /// "new". It usually does have one: only one of the ways in (an account that
    /// never had one) fits that description, and the others would be flatly
    /// contradicted by it. The forced sentence also names no credential, because
    /// that route is reached with a password submitted (the password step) and
    /// without one (a first-factor emailed code, or OAuth), so mentioning what
    /// the user just presented would be naming nothing on two of the three.
    /// Same division as Android's `resetPasswordPrompt`, for the same reasons.
    static func resetPasswordPrompt(email: String, wasForced: Bool) -> String {
        let choose = "Choose a password\(forEmail(email)), and you'll be signed in with it."
        guard wasForced else { return choose }
        return "A password has to be set on this account before you can sign in. \(choose)"
    }

    /// A challenge whose delivery methods this screen can't drive, so there is no
    /// step to send the user to.
    ///
    /// Split by reason because blaming the account for a device check sends the
    /// user off to re-read their password. Client Trust can be configured to use
    /// an email *link*, which is not a code anyone can type, and naming the device
    /// is what stops that reading as an account problem.
    static func unanswerableChallenge(_ reason: SecondFactorReason) -> String {
        switch reason {
        case .mfa:
            "This account's two-step verification isn't supported here. Use a method you can complete on another device."
        case .deviceVerification:
            "This device needs to be confirmed, and the way this account does that isn't supported here."
        }
    }

    // The sign-up shortfall copy, the two error classifiers, and the status
    // mapping moved to `RealClerkAuth`: each one reads a type this file must not
    // import. `ClerkKit` for three of them, and `AuthenticationServices` for
    // `isUserCancellation`, which classifies an `ASWebAuthenticationSessionError`
    // rather than a Clerk one.

    #if DEBUG
    /// Preview-only shortcut onto the code step — `state` is settable only from
    /// this file, so the previews in `EmailCodeSignInView` come through here
    /// rather than round-tripping the network to reach that screen.
    static func previewAtCodeStep(email: String) -> EmailAuthModel {
        let model = EmailAuthModel()
        model.state = EmailAuthState(step: .code, email: email)
        return model
    }
    #endif
}
