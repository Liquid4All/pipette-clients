import Foundation

/// The Clerk-touching half of the sign-in flow, so tests can substitute a fake. Mirrors
/// Android's `ClerkAuth` interface (`ClerkAuth.kt`).
///
/// No ClerkKit type crosses this boundary in either direction: `SignIn` is a `Decodable`
/// wire type, so a fake that had to build one would be a JSON fixture. The in-progress
/// attempt stays in the implementation (`RealClerkAuth.Pending`).
///
/// Only calls go through here. `clerk.user`, `clerk.session` and `clerk.environment` are
/// `@Observable` reads, and routing them through this protocol would drop the observation.
@MainActor
protocol ClerkAuthenticating {
    /// Email a first-factor code, registering the address instead if Clerk does not know
    /// it (`form_identifier_not_found` is a Clerk detail, so the fallback lives here).
    func sendEmailCode(to email: String) async -> AuthActionResult

    func signInWithPassword(email: String, password: String) async -> AuthActionResult

    /// Answer the outstanding first-factor code, whether a sign-in's or a sign-up's.
    func verifyCode(_ code: String) async -> AuthActionResult

    /// Set the password that finishes a registration.
    func createPassword(_ password: String) async -> AuthActionResult

    /// Create a sign-in and email a `reset_password_email_code`, answered by
    /// `verifyPasswordResetCode` then `resetPassword`.
    ///
    /// Also how an account with no password gets one (from a social sign-in or a dashboard
    /// invite): it does not offer the `password` strategy, so `signInWithPassword` always
    /// fails on it with `strategy_for_user_invalid`. Unlike `sendEmailCode` it never
    /// registers an unknown address.
    ///
    /// Also the reset code step's resend, so a failure that never reached Clerk leaves the
    /// code already sent live.
    func sendPasswordResetCode(to email: String) async -> AuthActionResult

    /// Usually answers `.needsNewPassword`, with no session yet. An account with two-step
    /// verification answers `.needsSecondFactor` first.
    func verifyPasswordResetCode(_ code: String) async -> AuthActionResult

    /// Set the password after a reset code, or when Clerk demanded a new one on sign-in.
    /// An account with two-step verification answers `.needsSecondFactor` before `.success`.
    func resetPassword(_ password: String) async -> AuthActionResult

    /// Send the code for `factor` on the outstanding challenge. Needs the Clerk resource id
    /// behind the factor, which is why it is not `sendEmailCode`.
    func sendSecondFactorCode(_ factor: SecondFactor) async -> AuthActionResult

    func verifySecondFactor(_ code: String, factor: SecondFactor) async -> AuthActionResult

    /// Clerk settles the sign-in/sign-up transfer internally.
    func signInWithOAuth(strategy: String) async -> AuthActionResult

    /// Exposed so `resendSecondFactorCode` can tell "the challenge ended" from "the send
    /// failed".
    var hasSecondFactorChallenge: Bool { get }

    /// Drop any in-progress attempt, so a later verify cannot complete something the user
    /// walked away from.
    func abandonAttempt()

    /// Throws rather than returning an `AuthActionResult` because the two callers handle a
    /// failure differently: the auth gate alerts and stops before unlinking, while Settings
    /// logs and wipes the device anyway.
    func signOut() async throws
}
