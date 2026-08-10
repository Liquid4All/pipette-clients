import Foundation

@testable import Pipette

/// Scripted `ClerkAuthenticating` for the flow tests. Mirrors Android's
/// `FakeClerkAuth`.
///
/// Note what is missing: `import ClerkKit`. That absence is the point of the
/// seam, and it is worth keeping: the moment a protocol method needs a Clerk
/// value, this file stops compiling, which is a better alarm than a test that
/// quietly starts building `SignIn` fixtures.
///
/// Results are queued per call. A method with an empty queue returns
/// `defaultResult` rather than trapping, so a test only scripts the calls it
/// cares about, and `calls` records everything in order for the assertions that
/// are about *what was asked of Clerk* rather than about the state that came
/// back.
@MainActor
final class FakeClerkAuth: ClerkAuthenticating {
    /// Every call made, in order, with the arguments that matter to a test.
    /// Equatable so a whole expected sequence can be compared in one assertion.
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

    /// What an unscripted call returns. `.success` keeps the common case short:
    /// a test about the second-factor step should not have to script its way
    /// through the first factor.
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

    /// Derived from the results handed out, not set by tests, because that is how
    /// the real one behaves: `RealClerkAuth.completeSignIn` parks a challenge
    /// exactly when it reports `.needsSecondFactor`, and drops the attempt
    /// otherwise.
    ///
    /// Settable all the same, for the two tests that need the challenge to end
    /// underneath a step that is still on screen. An earlier version of this fake
    /// left it purely manual and it lied: `signInWithOAuth` calls
    /// `abandonAttempt` before its result comes back, so a challenge scripted for
    /// OAuth was cleared before the model could act on it, and the auto-send
    /// looked broken when it was not.
    var hasSecondFactorChallenge = false

    /// Thrown by `signOut` when set.
    ///
    /// Not exercised by any test yet, and worth being straight about why rather
    /// than leaving it looking like coverage. Both call sites now hold a
    /// `ClerkAuthenticating` and can be handed this fake, but the logic that acts
    /// on the failure is a private method of a SwiftUI view mutating private
    /// `@State` (`ClerkAuthGateView.signOut`, `SettingsView.signOutEverywhere`),
    /// so there is nothing a test can call or observe. Reaching it needs a view
    /// model extracted from those two views, which is its own change.
    ///
    /// Kept rather than deleted because the injection is what that change would
    /// build on, and because the ordering it guards is real: the gate must not
    /// unlink the registration when the session is still live.
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
        // `startingOver`, like the other calls that begin an attempt:
        // `RealClerkAuth.sendPasswordResetCode` clears `pending` before it does
        // anything, and the only thing it can put back is a reset, never a
        // challenge.
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
        // The one call that leaves the parked challenge alone: a delivery re-parks
        // the same attempt rather than replacing it, so whether a challenge is
        // outstanding does not change here even when the send fails.
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

    /// Hand out the next scripted result, and park or drop the challenge to match
    /// it, which is the bookkeeping `RealClerkAuth.completeSignIn` does around the
    /// same outcomes.
    ///
    /// For the calls that answer something already parked: a verify, or setting a
    /// password. A `.failure` here leaves the challenge alone, because in the real
    /// one that is a *throw* out of `verifyMfaCode` or `verifyCode`, which never
    /// touches `pending`. That is the commonest path there is, a mistyped code,
    /// and the challenge has to survive it or Resend has nothing to resend
    /// against.
    ///
    /// An earlier version cleared on `.failure` here too, and it was wrong in a
    /// way that hid itself: `resendSecondFactorCode` short-circuits on
    /// `hasSecondFactorChallenge`, so a test written to prove the `sentFactors`
    /// guard is restored after a failed resend passed whether or not the
    /// restoration existed.
    private func next(_ queue: inout [AuthActionResult]) -> AuthActionResult {
        let result = queue.isEmpty ? defaultResult : queue.removeFirst()
        switch result {
        case .needsSecondFactor:
            hasSecondFactorChallenge = true
        case .success, .needsPassword, .needsNewPassword, .cancelled:
            hasSecondFactorChallenge = false
        case .failure:
            // Left as it was, per the note above. The other reading of `.failure`,
            // a status `completeSignIn` has no step for, does drop the attempt in
            // the real one; a test that wants that case sets the flag itself.
            break
        }
        return result
    }

    /// The same, for the calls that begin a fresh attempt: `sendEmailCode`,
    /// `signInWithPassword`, `signInWithOAuth`, and `sendPasswordResetCode`. Each
    /// one sets `pending = nil` before it does anything else in `RealClerkAuth`,
    /// so whatever was parked is gone by the time the result comes back, failure
    /// included.
    ///
    /// `sendPasswordResetCode` is the one with a caveat: it can put a *reset*
    /// back when its create half fails. Never a challenge, though, which is all
    /// this flag tracks, so it belongs here with the rest.
    private func startingOver(_ queue: inout [AuthActionResult]) -> AuthActionResult {
        let result = next(&queue)
        if case .failure = result { hasSecondFactorChallenge = false }
        return result
    }

    /// Calls of one kind, for assertions about how many times something was
    /// asked for. The resend tests care that a second send happened, not where
    /// it fell in the sequence.
    func count(of predicate: (Call) -> Bool) -> Int {
        calls.filter(predicate).count
    }
}
