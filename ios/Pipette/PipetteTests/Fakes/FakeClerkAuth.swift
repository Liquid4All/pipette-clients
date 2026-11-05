import Foundation

@testable import Pipette

/// Mirrors Android's `FakeClerkAuth`.
///
/// No `import ClerkKit`, on purpose: the moment a protocol method needs a Clerk value, this
/// file stops compiling, a better alarm than a test that quietly builds `SignIn` fixtures.
///
/// A method with an empty result queue returns `defaultResult`, so a test only scripts the
/// calls it cares about.
@MainActor
final class FakeClerkAuth: ClerkAuthenticating {
    enum Call: Equatable {
        case sendEmailCode(email: String)
        case signInWithPassword(email: String, password: String)
        case verifyCode(code: String)
        case createPassword(password: String)
        case sendPasswordResetCode(email: String)
        case verifyPasswordResetCode(code: String)
        case resetPassword(password: String)
        case sendSecondFactorCode(factor: SecondFactor)
        case verifySecondFactor(code: String, factor: SecondFactor)
        case signInWithOAuth(strategy: String)
        case abandonAttempt
        case signOut
    }

    private(set) var calls: [Call] = []

    /// `.success`, so a test about the second factor need not script the first.
    var defaultResult: AuthActionResult = .success

    var sendEmailCodeResults: [AuthActionResult] = []
    var signInWithPasswordResults: [AuthActionResult] = []
    var verifyCodeResults: [AuthActionResult] = []
    var createPasswordResults: [AuthActionResult] = []
    var sendPasswordResetCodeResults: [AuthActionResult] = []
    var verifyPasswordResetCodeResults: [AuthActionResult] = []
    var resetPasswordResults: [AuthActionResult] = []
    var sendSecondFactorCodeResults: [AuthActionResult] = []
    var verifySecondFactorResults: [AuthActionResult] = []
    var signInWithOAuthResults: [AuthActionResult] = []

    /// Derived from the results handed out, as `RealClerkAuth.completeSignIn` parks a
    /// challenge exactly when it reports `.needsSecondFactor`. Settable for tests that need
    /// the challenge to end underneath a step that is still on screen.
    var hasSecondFactorChallenge = false

    /// Thrown by `signOut` when set. No test uses it yet: the code that handles the failure
    /// is private view state (`ClerkAuthGateView.signOut`, `SettingsView.signOutEverywhere`)
    /// that a test cannot reach until a view model is extracted. Kept for that change: the
    /// gate must not unlink the registration while the session is still live.
    var signOutError: Error?

    // MARK: - ClerkAuthenticating

    func sendEmailCode(to email: String) async -> AuthActionResult {
        calls.append(.sendEmailCode(email: email))
        return startingOver(&sendEmailCodeResults)
    }

    func signInWithPassword(email: String, password: String) async -> AuthActionResult {
        calls.append(.signInWithPassword(email: email, password: password))
        return startingOver(&signInWithPasswordResults)
    }

    func verifyCode(_ code: String) async -> AuthActionResult {
        calls.append(.verifyCode(code: code))
        return next(&verifyCodeResults)
    }

    func createPassword(_ password: String) async -> AuthActionResult {
        calls.append(.createPassword(password: password))
        return next(&createPasswordResults)
    }

    func sendPasswordResetCode(to email: String) async -> AuthActionResult {
        calls.append(.sendPasswordResetCode(email: email))
        return startingOver(&sendPasswordResetCodeResults)
    }

    func verifyPasswordResetCode(_ code: String) async -> AuthActionResult {
        calls.append(.verifyPasswordResetCode(code: code))
        return next(&verifyPasswordResetCodeResults)
    }

    func resetPassword(_ password: String) async -> AuthActionResult {
        calls.append(.resetPassword(password: password))
        return next(&resetPasswordResults)
    }

    func sendSecondFactorCode(_ factor: SecondFactor) async -> AuthActionResult {
        calls.append(.sendSecondFactorCode(factor: factor))
        // Leaves the parked challenge alone: a delivery re-parks the same attempt, even when
        // the send fails.
        return sendSecondFactorCodeResults.isEmpty ? defaultResult : sendSecondFactorCodeResults.removeFirst()
    }

    func verifySecondFactor(_ code: String, factor: SecondFactor) async -> AuthActionResult {
        calls.append(.verifySecondFactor(code: code, factor: factor))
        return next(&verifySecondFactorResults)
    }

    func signInWithOAuth(strategy: String) async -> AuthActionResult {
        calls.append(.signInWithOAuth(strategy: strategy))
        return startingOver(&signInWithOAuthResults)
    }

    func abandonAttempt() {
        calls.append(.abandonAttempt)
        hasSecondFactorChallenge = false
    }

    func signOut() async throws {
        calls.append(.signOut)
        if let signOutError { throw signOutError }
    }

    // MARK: - Helpers

    /// For the calls that answer something already parked (a verify, or setting a
    /// password). A `.failure` leaves the challenge alone: in the real one it is a throw out
    /// of `verifyMfaCode` or `verifyCode`, which never touches `pending`, and the challenge
    /// must survive a mistyped code or Resend has nothing to resend against.
    private func next(_ queue: inout [AuthActionResult]) -> AuthActionResult {
        let result = queue.isEmpty ? defaultResult : queue.removeFirst()
        switch result {
        case .needsSecondFactor:
            hasSecondFactorChallenge = true
        case .success, .needsPassword, .needsNewPassword, .cancelled:
            hasSecondFactorChallenge = false
        case .failure:
            // A status `completeSignIn` has no step for does drop the attempt in the real
            // one; a test that wants that case sets the flag itself.
            break
        }
        return result
    }

    /// For the calls that begin a fresh attempt: each sets `pending = nil` first in
    /// `RealClerkAuth`, so whatever was parked is gone, failure included.
    /// (`sendPasswordResetCode` can put back a reset, never a challenge.)
    private func startingOver(_ queue: inout [AuthActionResult]) -> AuthActionResult {
        let result = next(&queue)
        if case .failure = result { hasSecondFactorChallenge = false }
        return result
    }

    func count(of predicate: (Call) -> Bool) -> Int {
        calls.filter(predicate).count
    }
}
