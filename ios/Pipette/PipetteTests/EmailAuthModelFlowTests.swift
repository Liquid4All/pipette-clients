import Foundation
import Testing

@testable import Pipette

/// The sign-in flow driven end to end through `FakeClerkAuth`.
///
/// These are what the seam was introduced for. `EmailAuthFlowTests` covers the
/// pure transitions and the message builders, which never needed a substitute;
/// everything here needs one, because it is about the model *calling* the auth
/// layer and acting on what comes back.
///
/// Written to pin behavior that already existed before the refactor, so a
/// mistake in relocating the attempt or the `sentFactors` bookkeeping shows up
/// as a failure rather than as new coverage of a new bug.
@MainActor
struct EmailAuthModelFlowTests {
    /// A model wired to a fake, plus the fake, since almost every test needs both.
    private func makeModel(
        configure: (FakeClerkAuth) -> Void = { _ in }
    ) -> (EmailAuthModel, FakeClerkAuth) {
        let auth = FakeClerkAuth()
        configure(auth)
        return (EmailAuthModel(auth: auth), auth)
    }

    // MARK: - First factor

    @Test func aSentCodeAdvancesToTheCodeStep() async {
        let (model, auth) = makeModel { $0.sendEmailCodeResults = [.success] }

        await model.submitEmail("  user@example.test  ")

        #expect(model.state.step == .code)
        #expect(model.state.submitting == false)
        #expect(model.state.error == nil)
        // Trimmed on the way out, not just on the way into state: the address is
        // what Clerk matches the account on.
        #expect(auth.calls == [.sendEmailCode(email: "user@example.test")])
        #expect(model.state.email == "user@example.test")
    }

    @Test func aFailedSendStaysOnTheEmailStep() async {
        let (model, _) = makeModel { $0.sendEmailCodeResults = [.failure("no such instance")] }

        await model.submitEmail("user@example.test")

        #expect(model.state.step == .email)
        #expect(model.state.error == "no such instance")
        #expect(model.state.submitting == false)
    }

    @Test func takingThePasswordPathAbandonsTheAttempt() async {
        let (model, auth) = makeModel()

        model.usePassword("  user@example.test ")

        // The parked attempt has to go: a code verified after this would complete
        // a sign-in the user walked away from.
        #expect(auth.calls == [.abandonAttempt])
        #expect(model.state.step == .password)
        #expect(model.state.email == "user@example.test")
    }

    @Test func aCompletedPasswordSignInResetsTheScreen() async {
        let (model, auth) = makeModel { $0.signInWithPasswordResults = [.success] }

        await model.submitPassword(email: "user@example.test", password: "hunter2")

        // Success is the signed-in state flipping the gate, so nothing of the
        // sign-in UI should be left standing behind it.
        #expect(model.state == EmailAuthState())
        #expect(auth.calls == [.signInWithPassword(email: "user@example.test", password: "hunter2")])
    }

    @Test func aVerifiedCodeGoesThroughTheAuthLayer() async {
        let (model, auth) = makeModel {
            $0.sendEmailCodeResults = [.success]
            $0.verifyCodeResults = [.failure("that code has expired")]
        }
        // Driven onto the code step first, so the assertion below is about a
        // failure keeping *that* step rather than about the initial state.
        await model.submitEmail("user@example.test")

        await model.submitCode("123456")

        #expect(auth.calls == [.sendEmailCode(email: "user@example.test"), .verifyCode(code: "123456")])
        #expect(model.state.error == "that code has expired")
        // Still on the step it was invoked from, with the field's digits intact.
        #expect(model.state.step == .code)
    }

    @Test func aRegistrationAsksForAPassword() async {
        let (model, auth) = makeModel {
            $0.verifyCodeResults = [.needsPassword]
            $0.createPasswordResults = [.success]
        }

        await model.submitCode("123456")
        #expect(model.state.step == .createPassword)

        await model.submitNewPassword("hunter2")
        #expect(model.state == EmailAuthState())
        #expect(auth.calls.contains(.createPassword(password: "hunter2")))
    }

    // MARK: - Second factor

    @Test func aLoneEmailFactorIsChosenAndSentWithoutATap() async {
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .mfa)]
        }

        await model.submitPassword(email: "user@example.test", password: "hunter2")

        #expect(model.state.step == .secondFactor)
        #expect(model.state.chosenSecondFactor == .emailCode)
        #expect(model.state.secondFactorReason == .mfa)
        // A step that asks for a code nobody sent is a dead end. With one option
        // there is no picker tap to trigger the send, so the tail must.
        #expect(auth.calls.last == .sendSecondFactorCode(factor: .emailCode))
    }

    @Test func aCodelessFactorIsNeverSentFor() async {
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.totp], .mfa)]
        }

        await model.submitPassword(email: "user@example.test", password: "hunter2")

        #expect(model.state.chosenSecondFactor == .totp)
        // TOTP is read off a device the user already holds; there is nothing to
        // request, and asking would be a wasted round-trip at best.
        #expect(auth.count { $0 == .sendSecondFactorCode(factor: .totp) } == 0)
    }

    @Test func aChoiceIsNotMadeForAnAccountWithSeveral() async {
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode, .totp], .mfa)]
        }

        await model.submitPassword(email: "user@example.test", password: "hunter2")

        #expect(model.state.chosenSecondFactor == nil)
        #expect(model.state.secondFactorOptions == [.emailCode, .totp])
        // Nothing is sent until the user picks, or a code goes to an address they
        // did not choose.
        #expect(auth.count { if case .sendSecondFactorCode = $0 { return true } else { return false } } == 0)

        await model.chooseSecondFactor(.emailCode)
        #expect(auth.calls.last == .sendSecondFactorCode(factor: .emailCode))
    }

    @Test func aDeviceCheckSaysSoRatherThanBlamingTheAccount() async {
        let (model, _) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .deviceVerification)]
        }

        await model.submitPassword(email: "user@example.test", password: "hunter2")

        #expect(model.state.secondFactorReason == .deviceVerification)
        // The reason is only worth carrying for what it makes the screen say, so
        // assert that too: a device check must not be headed as two-step
        // verification, which the account may never have switched on.
        let reason = model.state.secondFactorReason
        #expect(reason.title == "Confirm this device")
        #expect(reason.prompt(SecondFactor.emailCode.prompt).contains("first sign-in on this device"))
    }

    @Test func aFailedDeliveryLeavesTheStepUpAndSaysWhy() async {
        let (model, _) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .mfa)]
            $0.sendSecondFactorCodeResults = [.failure("rate limited")]
        }

        await model.submitPassword(email: "user@example.test", password: "hunter2")

        #expect(model.state.step == .secondFactor)
        #expect(model.state.error == "rate limited")
        // The spinner must not be left running on a step the user can still act on.
        #expect(model.state.submitting == false)
    }

    // MARK: - Resend

    /// The half of the resend that worked, which is the only one that may clear
    /// the field.
    @Test func anAcceptedResendAsksAgainAndRetiresTheOldCode() async {
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .mfa)]
        }
        await model.submitPassword(email: "user@example.test", password: "hunter2")
        let generationBeforeResend = model.state.challengeGeneration

        await model.resendSecondFactorCode()

        #expect(auth.count { $0 == .sendSecondFactorCode(factor: .emailCode) } == 2)
        // Bumped only because Clerk accepted: the digits on screen belong to a
        // code that has now been replaced.
        #expect(model.state.challengeGeneration == generationBeforeResend + 1)
        #expect(model.state.error == nil)
    }

    /// The bug this guards against is silent: a resend that never left the device
    /// must not retire the code sitting in the user's inbox.
    @Test func aFailedResendKeepsTheLiveCodeAnswerable() async {
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .mfa)]
            // The first send (the automatic one) succeeds; the resend does not.
            $0.sendSecondFactorCodeResults = [.success, .failure("offline")]
            // A *failed* verify below, deliberately: a success would reset the
            // screen and the send tail would decline on the step check, so the
            // assertion would hold even with the guard restoration broken. This
            // keeps the step up so the tail really is asked.
            $0.verifySecondFactorResults = [.failure("wrong code")]
        }
        await model.submitPassword(email: "user@example.test", password: "hunter2")
        let generationBeforeResend = model.state.challengeGeneration

        await model.resendSecondFactorCode()

        #expect(model.state.error == "offline")
        // Not bumped: the field still holds the digits of a code that still works.
        #expect(model.state.challengeGeneration == generationBeforeResend)

        // And the guard went back on, so the next thing through the auto-send tail
        // does not quietly email a replacement for the code the user still has.
        let sendsSoFar = auth.count { $0 == .sendSecondFactorCode(factor: .emailCode) }
        await model.submitSecondFactorCode("000000")
        #expect(auth.count { $0 == .sendSecondFactorCode(factor: .emailCode) } == sendsSoFar)
    }

    @Test func aResendWithNothingParkedSaysSoRatherThanSilentlyDoingNothing() async {
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .mfa)]
        }
        await model.submitPassword(email: "user@example.test", password: "hunter2")
        // Whatever ended the challenge is gone, but the step is still standing.
        auth.hasSecondFactorChallenge = false
        model.clearError()
        let sendsSoFar = auth.count { $0 == .sendSecondFactorCode(factor: .emailCode) }

        await model.resendSecondFactorCode()

        #expect(model.state.error == EmailAuthModel.noSecondFactorInProgress)
        #expect(auth.count { $0 == .sendSecondFactorCode(factor: .emailCode) } == sendsSoFar)
    }

    /// A message that names the status is more use than the generic sentence, so
    /// the resend must not overwrite one that is already on screen.
    @Test func aResendDoesNotTalkOverAMoreSpecificError() async {
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .mfa)]
        }
        await model.submitPassword(email: "user@example.test", password: "hunter2")
        auth.hasSecondFactorChallenge = false
        auth.verifySecondFactorResults = [.failure("sign-in isn't finished (needs_new_password)")]
        await model.submitSecondFactorCode("123456")

        await model.resendSecondFactorCode()

        #expect(model.state.error == "sign-in isn't finished (needs_new_password)")
    }

    /// A challenge answered with another challenge has to get a code of its own,
    /// which is what `sentFactors` being cleared on every challenge buys.
    @Test func aChainedChallengeGetsItsOwnCode() async {
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .deviceVerification)]
            $0.verifySecondFactorResults = [.needsSecondFactor([.emailCode], .mfa)]
        }
        await model.submitPassword(email: "user@example.test", password: "hunter2")
        let sendsBefore = auth.count { $0 == .sendSecondFactorCode(factor: .emailCode) }

        await model.submitSecondFactorCode("123456")

        #expect(model.state.secondFactorReason == .mfa)
        #expect(auth.count { $0 == .sendSecondFactorCode(factor: .emailCode) } == sendsBefore + 1)
    }

    // MARK: - OAuth

    @Test func aDismissedProviderSheetIsNotAnError() async {
        let (model, _) = makeModel { $0.signInWithOAuthResults = [.cancelled] }
        model.usePassword("user@example.test")

        await model.signInWithOAuth(strategy: "oauth_google")

        // The user chose to back out. Saying anything would read as the provider
        // having turned them away.
        #expect(model.state.error == nil)
        #expect(model.state.submitting == false)
        #expect(model.state.step == .password)
    }

    @Test func aFailedOAuthRoundTripSurfaces() async {
        let (model, _) = makeModel { $0.signInWithOAuthResults = [.failure("provider unavailable")] }

        await model.signInWithOAuth(strategy: "oauth_google")

        #expect(model.state.error == "provider unavailable")
        #expect(model.state.submitting == false)
    }

    @Test func anOAuthChallengeIsAsAnswerableAsAPasswordOne() async {
        let (model, auth) = makeModel {
            $0.signInWithOAuthResults = [.needsSecondFactor([.emailCode], .mfa)]
        }

        await model.signInWithOAuth(strategy: "oauth_google")

        // The bug this replaced reported a challenged OAuth sign-in as a flat
        // failure while the other two entry points routed it correctly.
        #expect(model.state.step == .secondFactor)
        #expect(auth.calls.last == .sendSecondFactorCode(factor: .emailCode))
    }

    // MARK: - Password reset

    @Test func theResetRunsFromThePasswordStepToASession() async {
        let (model, auth) = makeModel {
            $0.sendPasswordResetCodeResults = [.success]
            $0.verifyPasswordResetCodeResults = [.needsNewPassword]
            $0.resetPasswordResults = [.success]
        }
        model.usePassword("user@example.test")

        await model.startPasswordReset("  user@example.test ")
        #expect(model.state.step == .resetCode)

        await model.submitResetCode("123456")
        // Not a session yet: the code buys the right to set the password.
        #expect(model.state.step == .resetPassword)
        #expect(!model.state.resetWasForced)

        await model.submitResetPassword("correct horse battery staple")

        // Success resets the screen; the signed-in user flips the gate behind it.
        #expect(model.state == EmailAuthState())
        #expect(
            auth.calls == [
                .abandonAttempt,
                .sendPasswordResetCode(email: "user@example.test"),
                .verifyPasswordResetCode(code: "123456"),
                .resetPassword(password: "correct horse battery staple"),
            ]
        )
    }

    @Test func aPasswordlessAccountReachesTheResetFromTheError() async {
        // The reported bug, from the user's side: the password step is where they
        // land, and the reset has to be reachable from it.
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.failure(EmailAuthModel.noPasswordOnAccount)]
            $0.sendPasswordResetCodeResults = [.success]
        }
        model.usePassword("user@example.test")

        await model.submitPassword(email: "user@example.test", password: "hunter2")
        #expect(model.state.step == .password)
        #expect(model.state.error == EmailAuthModel.noPasswordOnAccount)

        await model.startPasswordReset("user@example.test")

        #expect(model.state.step == .resetCode)
        #expect(model.state.error == nil)
        #expect(auth.calls.last == .sendPasswordResetCode(email: "user@example.test"))
    }

    @Test func aForcedResetArrivesOffACorrectPassword() async {
        let (model, _) = makeModel { $0.signInWithPasswordResults = [.needsNewPassword] }

        await model.submitPassword(email: "user@example.test", password: "hunter2")

        #expect(model.state.step == .resetPassword)
        // The password was right. The step has to say so, or it reads as a
        // rejection of what was just typed.
        #expect(model.state.resetWasForced)
        #expect(model.state.error == nil)
    }

    @Test func aWrongResetCodeKeepsTheStepAnswerable() async {
        let (model, _) = makeModel {
            $0.sendPasswordResetCodeResults = [.success]
            $0.verifyPasswordResetCodeResults = [.failure("Incorrect code.")]
        }
        await model.startPasswordReset("user@example.test")

        await model.submitResetCode("000000")

        #expect(model.state.step == .resetCode)
        #expect(model.state.error == "Incorrect code.")
        #expect(!model.state.submitting)
    }

    @Test func aRejectedPasswordStaysOnTheResetStep() async {
        let (model, _) = makeModel {
            $0.sendPasswordResetCodeResults = [.success]
            $0.verifyPasswordResetCodeResults = [.needsNewPassword]
            $0.resetPasswordResults = [.failure("That password has been found in a breach.")]
        }
        // Entered the way the flow does, so the reset is recorded as requested and
        // the step is reached from `.resetCode` rather than from the initial state.
        await model.startPasswordReset("user@example.test")
        await model.submitResetCode("123456")
        #expect(!model.state.resetWasForced)

        await model.submitResetPassword("password123456")

        // Clerk owns the policy, so its wording is what surfaces, and the step
        // stays up so the next attempt has somewhere to go.
        #expect(model.state.step == .resetPassword)
        #expect(model.state.error == "That password has been found in a breach.")
        #expect(!model.state.submitting)
    }

    @Test func aChallengeAfterAResetIsAnsweredLikeAnyOther() async {
        let (model, auth) = makeModel {
            $0.sendPasswordResetCodeResults = [.success]
            $0.verifyPasswordResetCodeResults = [.needsNewPassword]
            $0.resetPasswordResults = [.needsSecondFactor([.emailCode], .mfa)]
        }
        await model.startPasswordReset("user@example.test")
        await model.submitResetCode("123456")

        await model.submitResetPassword("correct horse battery staple")

        // An account with two-step verification is asked for it after the reset,
        // and a lone delivered factor has to have its code sent without a tap,
        // the same dead end every other entry point ends with this call to avoid.
        #expect(model.state.step == .secondFactor)
        #expect(auth.calls.last == .sendSecondFactorCode(factor: .emailCode))
    }

    @Test func anAcceptedResendClearsTheField() async {
        let (model, auth) = makeModel { $0.sendPasswordResetCodeResults = [.success, .success] }
        await model.startPasswordReset("user@example.test")
        let afterFirstSend = model.state.resetCodeGeneration

        await model.resendPasswordResetCode()

        // The generation is what the field's identity is keyed on, so moving it is
        // what clears digits belonging to the code just replaced.
        #expect(model.state.resetCodeGeneration == afterFirstSend + 1)
        #expect(model.state.step == .resetCode)
        // Resent to the address already on the step, with nothing retyped.
        #expect(auth.count { $0 == .sendPasswordResetCode(email: "user@example.test") } == 2)
    }

    @Test func aFailedResendLeavesTheLiveCodeAnswerable() async {
        let (model, _) = makeModel {
            $0.sendPasswordResetCodeResults = [.success, .failure("Too many requests. Start the reset again.")]
        }
        await model.startPasswordReset("user@example.test")
        let afterFirstSend = model.state.resetCodeGeneration

        await model.resendPasswordResetCode()

        // Nothing was sent, so the digits already in the field are still the live
        // code's and must survive. This is the half of #1201's bug pair that a
        // generation bumped up front would have reintroduced.
        #expect(model.state.resetCodeGeneration == afterFirstSend)
        #expect(model.state.step == .resetCode)
        #expect(model.state.error == "Too many requests. Start the reset again.")
        #expect(!model.state.submitting)
    }

    /// The bug two derivation attempts produced, driven through the model rather
    /// than the transitions: a requested reset that Clerk interrupts with a
    /// challenge must still read as requested when it lands on the password, and a
    /// demanded one must still read as demanded.
    @Test func aResetKeepsTrackOfWhoAskedAcrossAChallenge() async {
        let (requested, _) = makeModel {
            $0.sendPasswordResetCodeResults = [.success]
            // A 2FA account: the reset code clears into a challenge, not a password.
            $0.verifyPasswordResetCodeResults = [.needsSecondFactor([.totp], .mfa)]
            $0.verifySecondFactorResults = [.needsNewPassword]
        }
        await requested.startPasswordReset("user@example.test")
        await requested.submitResetCode("123456")
        #expect(requested.state.step == .secondFactor)
        await requested.submitSecondFactorCode("654321")
        #expect(requested.state.step == .resetPassword)
        #expect(!requested.state.resetWasForced)

        let (demanded, _) = makeModel {
            // A correct password on a device Clerk does not recognize, then the
            // reset it demands once the device is confirmed.
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .deviceVerification)]
            $0.verifySecondFactorResults = [.needsNewPassword]
        }
        await demanded.submitPassword(email: "user@example.test", password: "hunter2")
        #expect(demanded.state.step == .secondFactor)
        await demanded.submitSecondFactorCode("654321")
        #expect(demanded.state.step == .resetPassword)
        #expect(demanded.state.resetWasForced)
    }

    /// A reset the user asked for but never got: the send failed, so they are
    /// still on the password step. If the request were recorded before the call
    /// rather than on the send Clerk accepted, the correct password they submit
    /// next would come back demanding a reset and be worded as one they asked
    /// for, with no explanation of why their right password was not enough.
    @Test func aResetThatNeverSentDoesNotCountAsAsking() async {
        let (model, _) = makeModel {
            $0.sendPasswordResetCodeResults = [.failure("Too many requests. Start the reset again.")]
            $0.signInWithPasswordResults = [.needsNewPassword]
        }
        model.usePassword("user@example.test")

        await model.startPasswordReset("user@example.test")
        // Nothing was sent, so the user is left where they were.
        #expect(model.state.step == .password)
        #expect(!model.state.resetWasRequested)

        await model.submitPassword(email: "user@example.test", password: "hunter2")

        #expect(model.state.step == .resetPassword)
        #expect(model.state.resetWasForced)
    }

    // MARK: - Leaving

    @Test func changingEmailClearsEverythingButTheAddress() async {
        let (model, auth) = makeModel {
            $0.signInWithPasswordResults = [.needsSecondFactor([.emailCode], .mfa)]
        }
        await model.submitPassword(email: "user@example.test", password: "hunter2")

        model.changeEmail()

        #expect(model.state == EmailAuthState(step: .email, email: "user@example.test"))
        #expect(auth.calls.last == .abandonAttempt)
    }
}
