import Foundation

/// The Clerk-touching half of the sign-in flow, behind a protocol so it can be
/// substituted in tests. Mirrors Android's `ClerkAuth` interface (`ClerkAuth.kt`),
/// including the choice that makes the whole thing testable: no ClerkKit type
/// crosses this boundary in either direction.
///
/// That is the property to protect when adding a method here. A protocol that
/// handed a `SignIn` back would leave a fake having to build one, and `SignIn` is
/// a `Decodable` wire type, so the tests would become JSON fixtures wearing a
/// fake's clothes. So the in-progress attempt is owned by the implementation
/// (see `RealClerkAuth.Pending`), and callers get `AuthActionResult`, which is
/// the app's own vocabulary.
///
/// The boundary is drawn at *calls*, not at observed state. `clerk.user`,
/// `clerk.session`, and `clerk.environment` are `@Observable` reads that SwiftUI
/// re-renders on, so the views keep reading them directly; routing those through
/// here would drop the observation and the screen would stop updating. Only the
/// things that go to Clerk's API live behind this.
@MainActor
protocol ClerkAuthenticating {
    /// Email a first-factor code to `email`, registering the address instead if
    /// Clerk does not know it. The sign-up fallback is inside rather than at the
    /// call site because `form_identifier_not_found` is a Clerk detail.
    func sendEmailCode(to email: String) async -> AuthActionResult

    /// Sign in with the password first factor. No code is sent.
    func signInWithPassword(email: String, password: String) async -> AuthActionResult

    /// Answer the outstanding first-factor code, whether it belongs to a returning
    /// user's sign-in or a new address's sign-up. Which one it is is the
    /// implementation's business, since it is the thing holding the attempt.
    func verifyCode(_ code: String) async -> AuthActionResult

    /// Set the password that finishes a registration.
    func createPassword(_ password: String) async -> AuthActionResult

    /// Start a password reset for `email`: create a sign-in and email a 6-digit
    /// `reset_password_email_code`. Answer it with `verifyPasswordResetCode`,
    /// then set the value with `resetPassword`.
    ///
    /// This is how an account with no password at all gets one, as much as it is
    /// the route for a forgotten one. Such an account does not offer the
    /// `password` strategy, so `signInWithPassword` can only ever fail on it
    /// (`strategy_for_user_invalid`) whatever is typed. Registrations through
    /// this app do set a password (`createPassword`), so an account in that state
    /// came from somewhere else: a social sign-in, a dashboard invite, or a
    /// sign-up from before the instance required one. It never registers, unlike
    /// `sendEmailCode`: an address Clerk does not know is an error here, since
    /// there is no password to reset.
    ///
    /// Reachable twice over, from the password step and as the reset code step's
    /// resend, which is why the implementation is careful about what a failure
    /// leaves behind.
    func sendPasswordResetCode(to email: String) async -> AuthActionResult

    /// Answer the emailed reset code. `.needsNewPassword` is the usual success
    /// case: the code cleared and Clerk is now waiting for the password itself.
    /// No session exists yet.
    ///
    /// An account with two-step verification answers `.needsSecondFactor`
    /// instead, and reaches the password only after that is cleared, so the reset
    /// is not always two steps from here.
    func verifyPasswordResetCode(_ code: String) async -> AuthActionResult

    /// Set the password on the account whose reset the code just authorized, or
    /// which Clerk asked to replace its password on sign-in. On `.success` the
    /// session is active; an account with two-step verification answers with
    /// `.needsSecondFactor` first.
    func resetPassword(_ password: String) async -> AuthActionResult

    /// Send the code for `factor` on the outstanding challenge. Separate from
    /// `sendEmailCode` because it answers a second factor and needs the Clerk
    /// resource id behind the factor, which is exactly the sort of detail this
    /// boundary exists to swallow.
    func sendSecondFactorCode(_ factor: SecondFactor) async -> AuthActionResult

    /// Answer the outstanding challenge.
    func verifySecondFactor(_ code: String, factor: SecondFactor) async -> AuthActionResult

    /// Run the redirect-based OAuth flow. Clerk settles the sign-in/sign-up
    /// transfer internally, so this reports the same result type as the rest.
    func signInWithOAuth(strategy: String) async -> AuthActionResult

    /// True when a second-factor challenge is parked and can still be answered.
    /// The one piece of the attempt's shape the model needs to see, because
    /// `resendSecondFactorCode` has to tell "the challenge ended" apart from
    /// "the send failed" and word them differently.
    var hasSecondFactorChallenge: Bool { get }

    /// Drop any in-progress attempt, so a later verify cannot complete something
    /// the user has walked away from.
    func abandonAttempt()

    /// End the session.
    ///
    /// Throwing rather than reporting an `AuthActionResult`, unlike everything
    /// above, because the two call sites act on a failure differently and both
    /// need the error itself: the auth gate alerts and stops before unlinking the
    /// registration, while Settings logs and wipes the device regardless. See
    /// their comments for why each is right.
    func signOut() async throws
}
