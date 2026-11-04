import ClerkKit
import Foundation
import Testing
@testable import Pipette

/// The pure transitions behind the custom sign-in screen. Android's `EmailAuthFlowTest`
/// covers the same ground.
@MainActor
struct EmailAuthFlowTests {
    @Test func sendSuccessAdvancesToCodeStepKeepingEmail() {
        let start = EmailAuthState(email: "user@example.test", submitting: true)
        let next = start.afterSendCode(.success)
        #expect(next.step == .code)
        #expect(next.email == "user@example.test")
        #expect(!next.submitting)
        #expect(next.error == nil)
    }

    @Test func sendFailureStaysOnEmailAndSurfacesMessage() {
        let start = EmailAuthState(email: "user@example.test", submitting: true)
        let next = start.afterSendCode(.failure("That email looks invalid"))
        #expect(next.step == .email)
        #expect(!next.submitting)
        #expect(next.error == "That email looks invalid")
    }

    @Test func authenticationSuccessResetsToInitialState() {
        let start = EmailAuthState(step: .code, email: "user@example.test", submitting: true)
        // The signed-in user flips the gate, so no code-step UI is left behind the app.
        #expect(start.afterAuthentication(.success) == EmailAuthState())
    }

    @Test func authenticationFailureKeepsStepAndEmail() {
        let start = EmailAuthState(step: .code, email: "user@example.test", submitting: true)
        let next = start.afterAuthentication(.failure("Incorrect code"))
        #expect(next.step == .code)
        #expect(next.email == "user@example.test")
        #expect(!next.submitting)
        #expect(next.error == "Incorrect code")
    }

    /// Sign-up sends `legal_accepted`, which is only defensible while the user was shown
    /// these links first. Android's `AuthGateScreen.kt` pins the same pair.
    @Test func theClickwrapPointsAtThePublishedDocuments() {
        #expect(EmailAuthState.termsURL == "https://pipette.liquid.ai/terms")
        #expect(EmailAuthState.privacyPolicyURL == "https://www.liquid.ai/privacy-policy")
        // An unparseable URL renders the notice as plain text with nothing to tap.
        #expect(URL(string: EmailAuthState.termsURL) != nil)
        #expect(URL(string: EmailAuthState.privacyPolicyURL) != nil)
    }

    /// The instance requires a password at sign-up, so without this transition a new user
    /// dead-ends on "needs a password".
    @Test func aSignUpMissingItsPasswordLandsOnTheCreateStep() {
        let verifying = EmailAuthState(step: .code, email: "new@example.test", submitting: true)
        let next = verifying.afterAuthentication(.needsPassword)
        #expect(next.step == .createPassword)
        #expect(next.email == "new@example.test")
        #expect(!next.submitting)
        #expect(next.error == nil)
    }

    @Test func aRejectedPasswordKeepsTheCreateStepAndSaysWhy() {
        let creating = EmailAuthState(step: .createPassword, email: "new@example.test", submitting: true)
        // Clerk owns the policy, so its wording is what shows.
        let next = creating.afterAuthentication(.failure("Password has been found in an online data breach."))
        #expect(next.step == .createPassword)
        #expect(!next.submitting)
        #expect(next.error == "Password has been found in an online data breach.")
    }

    @Test func settingThePasswordFinishesTheFlow() {
        let creating = EmailAuthState(step: .createPassword, email: "new@example.test", submitting: true)
        // The account now exists and the session is live, so the gate takes over.
        #expect(creating.afterAuthentication(.success) == EmailAuthState())
    }

    @Test func anAddressIsEnoughToActOn() {
        #expect(EmailAuthState.canSubmitEmail("user@example.test", submitting: false))
    }

    @Test func aBlankAddressCannotBeSubmitted() {
        #expect(!EmailAuthState.canSubmitEmail("", submitting: false))
        #expect(!EmailAuthState.canSubmitEmail("   ", submitting: false))
    }

    @Test func aRequestInFlightBlocksBothActions() {
        #expect(!EmailAuthState.canSubmitEmail("user@example.test", submitting: true))
    }
}

/// The second-factor challenge: which factors this client can answer, and how the step
/// it lands on is set up.
@MainActor
struct SecondFactorTests {
    @Test func strategiesMapToClerksWireValues() {
        // These strings are Clerk's: a typo silently drops a factor the account offers.
        #expect(SecondFactor.emailCode.rawValue == "email_code")
        #expect(SecondFactor.phoneCode.rawValue == "phone_code")
        #expect(SecondFactor.totp.rawValue == "totp")
        #expect(SecondFactor.backupCode.rawValue == "backup_code")
    }

    @Test func onlyEmailAndPhoneCodesHaveToBeSent() {
        #expect(SecondFactor.emailCode.needsSending)
        #expect(SecondFactor.phoneCode.needsSending)
        #expect(!SecondFactor.totp.needsSending)
        #expect(!SecondFactor.backupCode.needsSending)
    }

    @Test func onlyBackupCodesAreNotSixDigits() {
        #expect(SecondFactor.emailCode.digitCount == 6)
        #expect(SecondFactor.phoneCode.digitCount == 6)
        #expect(SecondFactor.totp.digitCount == 6)
        // Alphanumeric and variable length, so it gets a plain field.
        #expect(SecondFactor.backupCode.digitCount == nil)
    }

    @Test func unanswerableFactorsAreDropped() {
        // A passkey challenge is real but this screen cannot drive it.
        #expect(SecondFactor.offered(strategies: ["passkey"]).isEmpty)
        #expect(SecondFactor.offered(strategies: ["totp", "passkey"]) == [.totp])
    }

    @Test func offeredFactorsKeepAStableOrderRegardlessOfClerksOrdering() {
        let a = SecondFactor.offered(strategies: ["totp", "email_code", "backup_code"])
        let b = SecondFactor.offered(strategies: ["backup_code", "totp", "email_code"])
        #expect(a == [.emailCode, .totp, .backupCode])
        #expect(a == b)
    }

    /// The Clerk resource id has to survive into the challenge: the SDK's no-argument
    /// `sendMfaPhoneCode()` resolves it by matching `factor.safeIdentifier == identifier`,
    /// and the identifier here is always an email, so a phone send would go out with no
    /// number attached.
    @Test func aChallengeCarriesTheResourceIdNeededToSendEachCode() throws {
        let signIn = try signIn(
            identifier: "user@example.test",
            factors: [
                ["strategy": "phone_code", "phone_number_id": "idp_phone_1", "safe_identifier": "+1 ••• ••1234"],
                ["strategy": "email_code", "email_address_id": "idn_email_1", "safe_identifier": "u••@example.test"],
            ]
        )
        let offered = SecondFactor.offered(by: signIn)
        let phone = try #require(offered.first { $0.factor == .phoneCode })
        let email = try #require(offered.first { $0.factor == .emailCode })
        #expect(phone.source.phoneNumberId == "idp_phone_1")
        #expect(email.source.emailAddressId == "idn_email_1")
        // This mismatch is what defeats the SDK fallback.
        #expect(phone.source.safeIdentifier != signIn.identifier)
    }

    @Test func aSecondPhoneOnTheAccountDoesNotDuplicateTheOption() throws {
        let signIn = try signIn(
            identifier: "user@example.test",
            factors: [
                ["strategy": "phone_code", "phone_number_id": "idp_first"],
                ["strategy": "phone_code", "phone_number_id": "idp_second"],
            ]
        )
        let offered = SecondFactor.offered(by: signIn)
        #expect(offered.map(\.factor) == [.phoneCode])
        // First match wins, which is the one Clerk's own UI would send to.
        #expect(offered.first?.source.phoneNumberId == "idp_first")
    }

    @Test func theTwoWaysOfListingOfferedFactorsAgree() throws {
        let strategies = ["totp", "phone_code", "passkey"]
        let signIn = try signIn(
            identifier: "user@example.test",
            factors: strategies.map { ["strategy": $0] }
        )
        #expect(SecondFactor.offered(by: signIn).map(\.factor) == SecondFactor.offered(strategies: strategies))
    }

    /// Decoded the way the SDK does it: `SignIn` has no public initializer.
    private func signIn(
        identifier: String,
        factors: [[String: String]],
        status: String = "needs_second_factor"
    ) throws -> SignIn {
        let payload: [String: Any] = [
            "id": "sia_test",
            "status": status,
            "identifier": identifier,
            "supported_second_factors": factors,
        ]
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(SignIn.self, from: JSONSerialization.data(withJSONObject: payload))
    }

    @Test func aLoneFactorIsPreChosenSoThereIsNothingToTap() {
        let start = EmailAuthState(step: .code, email: "user@example.test", submitting: true)
        let next = start.afterAuthentication(.needsSecondFactor([.totp], .mfa))
        #expect(next.step == .secondFactor)
        #expect(next.chosenSecondFactor == .totp)
        #expect(next.secondFactorOptions == [.totp])
        #expect(!next.submitting)
        #expect(next.error == nil)
    }

    @Test func severalFactorsLeaveTheChoiceOpen() {
        let start = EmailAuthState(step: .code, email: "user@example.test", submitting: true)
        let next = start.afterAuthentication(.needsSecondFactor([.emailCode, .totp], .mfa))
        #expect(next.step == .secondFactor)
        #expect(next.chosenSecondFactor == nil)
        #expect(next.secondFactorOptions == [.emailCode, .totp])
    }

    @Test func answeringTheChallengeResetsTheWholeFlow() {
        let challenged = EmailAuthState(
            step: .secondFactor,
            email: "user@example.test",
            submitting: true,
            secondFactorOptions: [.emailCode, .totp],
            chosenSecondFactor: .totp
        )
        #expect(challenged.afterAuthentication(.success) == EmailAuthState())
    }

    /// Clerk can answer an attempt with another `needs_second_factor` instead of a failure;
    /// treating that as a fresh challenge would make the step look like it reset itself.
    @Test func reEnteringTheChallengeKeepsTheChosenFactorAndTheMessage() {
        let answering = EmailAuthState(
            step: .secondFactor,
            email: "user@example.test",
            submitting: true,
            error: "Incorrect code",
            secondFactorOptions: [.emailCode, .totp],
            chosenSecondFactor: .totp
        )
        let next = answering.afterAuthentication(.needsSecondFactor([.emailCode, .totp], .mfa))
        #expect(next.chosenSecondFactor == .totp)
        #expect(next.error == "Incorrect code")
        #expect(!next.submitting)
    }

    @Test func arrivingAtTheChallengeFreshDoesNotInheritAnOlderError() {
        let verifying = EmailAuthState(step: .code, email: "user@example.test", submitting: true, error: "stale")
        let next = verifying.afterAuthentication(.needsSecondFactor([.emailCode, .totp], .mfa))
        #expect(next.error == nil)
        #expect(next.chosenSecondFactor == nil)
    }

    @Test func aChosenFactorTheAccountNoLongerOffersIsDropped() {
        #expect(EmailAuthState.factorToAnswer(keeping: .totp, from: [.emailCode, .totp]) == .totp)
        // A factor no longer on offer falls back to the same rule as a fresh challenge.
        #expect(EmailAuthState.factorToAnswer(keeping: .phoneCode, from: [.emailCode, .totp]) == nil)
        #expect(EmailAuthState.factorToAnswer(keeping: .phoneCode, from: [.totp]) == .totp)
        #expect(EmailAuthState.factorToAnswer(keeping: nil, from: [.emailCode, .totp]) == nil)
    }

    @Test func aDeadEndStatusNamesItself() {
        // A shared sentence would make every status with no step behind it
        // indistinguishable without a debugger. This status gets no step: nothing
        // in this UI can supply a first factor the account did not offer.
        let message = EmailAuthModel.incompleteSignInMessage(status: "needs_first_factor")
        #expect(message.contains("needs_first_factor"))
    }

    /// The whole path from a decoded `needs_client_trust` sign-in to the result the screen
    /// acts on. The mapping test below alone would still pass if `completeSignIn` stopped
    /// calling the mapping.
    @Test func aClientTrustSignInIsAdmittedAsADeviceCheck() throws {
        let challenge = try signIn(
            identifier: "user@example.test",
            factors: [["strategy": "email_code", "email_address_id": "idn_email_1"]],
            status: "needs_client_trust"
        )
        #expect(RealClerkAuth().completeSignIn(challenge) == .needsSecondFactor([.emailCode], .deviceVerification))
    }

    /// The same sign-in shape under `needs_second_factor`, so the test above pins the
    /// admission rather than the decoding.
    @Test func anMfaSignInStillArrivesAsMfa() throws {
        let challenge = try signIn(
            identifier: "user@example.test",
            factors: [["strategy": "totp"]],
            status: "needs_second_factor"
        )
        #expect(RealClerkAuth().completeSignIn(challenge) == .needsSecondFactor([.totp], .mfa))
    }

    /// A sign-in carrying both the status and an answerable factor has to route on the
    /// status: the second-factor step would ask for a code Clerk is not waiting for.
    @Test func aNewPasswordSignInRoutesOnItsStatusNotItsFactors() throws {
        let forced = try signIn(
            identifier: "user@example.test",
            factors: [["strategy": "email_code", "email_address_id": "idn_email_1"]],
            status: "needs_new_password"
        )
        #expect(RealClerkAuth().completeSignIn(forced) == .needsNewPassword)
    }

    /// The device sentence has to lead: on its own, the factor's sentence makes an
    /// unannounced challenge look like the password was wrong.
    @Test func onlyTheDeviceCheckExplainsWhyItIsAsking() {
        let source = SecondFactor.emailCode.prompt
        #expect(SecondFactorReason.mfa.prompt(source) == source)

        let device = SecondFactorReason.deviceVerification.prompt(source)
        #expect(device.hasPrefix("This is the first sign-in on this device"))
        #expect(device.hasSuffix(source))
    }

    /// Client Trust fires on accounts that never turned two-step verification on.
    @Test func theDeviceCheckIsNotHeadedTwoStepVerification() {
        #expect(SecondFactorReason.deviceVerification.title == "Confirm this device")
        #expect(SecondFactorReason.mfa.title == "Two-step verification")
    }

    /// Every factor prompt also composes under a device check, which is not two-step
    /// verification.
    @Test func noFactorPromptBlamesTwoStepVerification() {
        for factor in SecondFactor.allCases {
            #expect(!factor.prompt.lowercased().contains("two-step verification"), "\(factor) prompt names MFA")
        }
    }

    @Test func clientTrustIsAnAnswerableChallengeAndADeadEndIsNot() {
        #expect(RealClerkAuth.secondFactorReason(for: .needsClientTrust) == .deviceVerification)
        #expect(RealClerkAuth.secondFactorReason(for: .needsSecondFactor) == .mfa)
        // Decoded the way the SDK decodes it, so a rename on the wire is caught.
        #expect(RealClerkAuth.secondFactorReason(for: SignIn.Status(rawValue: "needs_client_trust")) == .deviceVerification)
        #expect(RealClerkAuth.secondFactorReason(for: .needsNewPassword) == nil)
        #expect(RealClerkAuth.secondFactorReason(for: .needsFirstFactor) == nil)
        #expect(RealClerkAuth.secondFactorReason(for: .complete) == nil)
        #expect(RealClerkAuth.secondFactorReason(for: .unknown("needs_something_new")) == nil)
    }

    /// Mirrors Android's `needsClientTrustAdvancesToTheSameStepAsDeviceVerification`.
    @Test func clientTrustReachesTheSameStepAsADeviceCheck() {
        let start = EmailAuthState(step: .password, email: "user@example.test", submitting: true)
        let next = start.afterAuthentication(.needsSecondFactor([.emailCode], .deviceVerification))
        #expect(next.step == .secondFactor)
        #expect(next.secondFactorReason == .deviceVerification)
        #expect(next.chosenSecondFactor == .emailCode)
        #expect(next.error == nil)
    }

    /// Both statuses are answered by the same call, so Clerk can return an MFA challenge
    /// right after the device check is cleared.
    @Test func aLaterMfaChallengeStopsClaimingToBeADeviceCheck() {
        let onDeviceStep = EmailAuthState(
            step: .secondFactor,
            email: "user@example.test",
            secondFactorOptions: [.emailCode],
            chosenSecondFactor: .emailCode,
            secondFactorReason: .deviceVerification
        )
        // Two options, so the lone-option rule cannot supply the answer.
        let next = onDeviceStep.afterAuthentication(.needsSecondFactor([.totp, .backupCode], .mfa))
        #expect(next.secondFactorReason == .mfa)
        #expect(next.chosenSecondFactor == nil)
    }

    /// `SecondFactorAnswer` keys its `.id` on this count, and a re-challenge sends a fresh
    /// code, so without the bump the field would resubmit digits spent on the old one.
    @Test func eachChallengeIsCountedEvenWhenItLooksIdentical() {
        let challenged = EmailAuthState(
            step: .secondFactor,
            email: "user@example.test",
            secondFactorOptions: [.emailCode],
            chosenSecondFactor: .emailCode,
            secondFactorReason: .mfa
        )
        let next = challenged.afterAuthentication(.needsSecondFactor([.emailCode], .mfa))
        #expect(next.secondFactorReason == challenged.secondFactorReason)
        #expect(next.chosenSecondFactor == challenged.chosenSecondFactor)
        #expect(next.challengeGeneration == challenged.challengeGeneration + 1)
    }

    @Test func aWrongSecondFactorCodeKeepsTheChallengeOnScreen() {
        let challenged = EmailAuthState(
            step: .secondFactor,
            email: "user@example.test",
            submitting: true,
            secondFactorOptions: [.emailCode, .totp],
            chosenSecondFactor: .totp
        )
        let next = challenged.afterAuthentication(.failure("Incorrect code"))
        #expect(next.step == .secondFactor)
        #expect(next.chosenSecondFactor == .totp)
        #expect(next.secondFactorOptions == [.emailCode, .totp])
        #expect(!next.submitting)
        #expect(next.error == "Incorrect code")
    }
}

/// What a failed sign-in puts on screen: Clerk's own copy is preferred, and nothing falls
/// through to a raw Swift error description.
@MainActor
struct ClerkErrorMessageTests {
    /// Client Trust can be configured to send an email link, which this screen cannot drive.
    /// Naming the account would send the user off to re-check their password.
    @Test func anUnanswerableDeviceCheckDoesNotBlameTheAccount() {
        let device = EmailAuthModel.unanswerableChallenge(.deviceVerification)
        #expect(device.contains("device"))
        #expect(!device.contains("two-step verification"))
        #expect(EmailAuthModel.unanswerableChallenge(.mfa).contains("two-step verification"))
    }

    @Test func theDetailedApiMessageWins() throws {
        let error = try apiError(
            code: "form_code_incorrect",
            message: "is incorrect",
            longMessage: "Incorrect code. Try again."
        )
        #expect(error.clerkDisplayMessage == "Incorrect code. Try again.")
    }

    @Test func theTerseApiMessageIsUsedWhenThereIsNoDetailedOne() throws {
        let error = try apiError(code: "form_password_incorrect", message: "Password is incorrect.")
        #expect(error.clerkDisplayMessage == "Password is incorrect.")
    }

    @Test func anApiErrorWithNoCopyFallsBackToSomethingReadable() throws {
        let error = try apiError(code: "form_param_unknown")
        let message = error.clerkDisplayMessage
        // Not the raw "The operation couldn't be completed. (ClerkKit.ClerkAPIError
        // error 1.)" that `localizedDescription` gives an error with no message.
        #expect(!message.contains("ClerkAPIError"))
        #expect(message == "Something went wrong. Try again.")
    }

    @Test func anIncompleteSignUpNamesWhatIsMissing() {
        #expect(
            RealClerkAuth.incompleteSignUpMessage(missing: [.password])
                == "Your account needs more to finish signing up: a password."
        )
        #expect(
            RealClerkAuth.incompleteSignUpMessage(missing: [.password, .username])
                == "Your account needs more to finish signing up: a password, a username."
        )
    }

    @Test func anIncompleteSignUpWithNothingNamedStaysGeneric() {
        let message = RealClerkAuth.incompleteSignUpMessage(missing: [])
        #expect(message == "Your account needs more details to finish signing up.")
    }

    @Test func aFieldTheSdkGainedLaterStillReadsAsWords() {
        let message = RealClerkAuth.incompleteSignUpMessage(missing: [.unknown("web3_wallet")])
        #expect(message.contains("web3 wallet"))
    }

    @Test func transportFailuresKeepTheirOwnLocalizedText() {
        let offline = URLError(.notConnectedToInternet)
        #expect(offline.clerkDisplayMessage == offline.localizedDescription)
        #expect(!offline.clerkDisplayMessage.isEmpty)
    }

    // MARK: - Password reset transitions

    @Test func aSentResetCodeAdvancesAndCountsItself() {
        let start = EmailAuthState(step: .password, email: "user@example.test", submitting: true)
        let next = start.afterSendResetCode(.success)
        #expect(next.step == .resetCode)
        #expect(!next.submitting)
        #expect(next.error == nil)
        // So a resend clears the digits of the code it replaced.
        #expect(next.resetCodeGeneration == start.resetCodeGeneration + 1)
    }

    @Test func aFailedResetSendLeavesTheDigitsAlone() {
        let onStep = EmailAuthState(step: .resetCode, email: "user@example.test", submitting: true, resetCodeGeneration: 3)
        let next = onStep.afterSendResetCode(.failure("Too many requests. Start the reset again."))
        // A resend that never left the device must not clear the live code's digits.
        #expect(next.resetCodeGeneration == 3)
        #expect(next.step == .resetCode)
        #expect(next.error == "Too many requests. Start the reset again.")
        #expect(!next.submitting)
    }

    @Test func aResetTheUserAskedForDoesNotReadAsForced() {
        let answering = EmailAuthState(step: .resetCode, email: "user@example.test", submitting: true, resetWasRequested: true)
        let next = answering.afterAuthentication(.needsNewPassword)
        #expect(next.step == .resetPassword)
        #expect(!next.resetWasForced)
        #expect(!next.submitting)
    }

    @Test func aResetClerkDemandedIsMarkedAsForced() {
        // The password typed here was correct; without the flag the step reads as a
        // rejection of it.
        let signingIn = EmailAuthState(step: .password, email: "user@example.test", submitting: true)
        let next = signingIn.afterAuthentication(.needsNewPassword)
        #expect(next.step == .resetPassword)
        #expect(next.resetWasForced)
    }

    @Test func reEnteringTheResetStepKeepsWhatOnlyItCanKnow() {
        let requested = EmailAuthState(step: .resetPassword, email: "user@example.test", submitting: true, resetWasRequested: true)
        let next = requested.afterAuthentication(.needsNewPassword)
        #expect(next.step == .resetPassword)
        #expect(next.resetWasRequested)
        #expect(!next.resetWasForced)
        #expect(!next.submitting)
    }

    /// A requested reset (Clerk answers the reset code with `needs_second_factor`) and a
    /// demanded one (`needs_client_trust` fires on password sign-ins) both reach the
    /// password from the challenge, so a flag derived from the step left gets one wrong.
    @Test func whoAskedSurvivesTheChallengeInBothDirections() {
        let asked = EmailAuthState(step: .resetCode, email: "user@example.test", submitting: true, resetWasRequested: true)
        let askedThenChallenged = asked.afterAuthentication(.needsSecondFactor([.totp], .mfa))
        #expect(askedThenChallenged.step == .secondFactor)
        let askedFinish = askedThenChallenged.afterAuthentication(.needsNewPassword)
        #expect(askedFinish.step == .resetPassword)
        #expect(!askedFinish.resetWasForced)

        // Nothing recorded a request, so this is Clerk's demand, and it stays one
        // through the challenge.
        let demanded = EmailAuthState(step: .password, email: "user@example.test", submitting: true)
        let demandedThenChallenged = demanded.afterAuthentication(.needsSecondFactor([.totp], .mfa))
        let demandedFinish = demandedThenChallenged.afterAuthentication(.needsNewPassword)
        #expect(demandedFinish.step == .resetPassword)
        #expect(demandedFinish.resetWasForced)
    }

    @Test func theResetPromptSaysWhoAskedForIt() {
        let requested = EmailAuthModel.resetPasswordPrompt(email: "user@example.test", wasForced: false)
        let forced = EmailAuthModel.resetPasswordPrompt(email: "user@example.test", wasForced: true)
        #expect(requested.contains("user@example.test"))
        #expect(requested.hasPrefix("Choose a password"))
        #expect(forced.hasPrefix("A password has to be set on this account"))
        // Only one of the ways in fits "new"; the others contradict it.
        #expect(!requested.contains("new password"))
        #expect(!forced.contains("new password"))
    }

    @Test func aPromptWithNoAddressToNameOmitsIt() {
        // A forced reset is reachable through OAuth, which never writes `state.email`.
        #expect(EmailAuthModel.forEmail("") == "")
        #expect(EmailAuthModel.forEmail("   ") == "")
        #expect(EmailAuthModel.forEmail("user@example.test") == " for user@example.test")
        let anonymous = EmailAuthModel.resetPasswordPrompt(email: "", wasForced: true)
        #expect(!anonymous.contains("  "))
        #expect(anonymous.contains("Choose a password,"))
    }

    /// The state a failed reset send leaves on purpose: nothing parked, with a reset step
    /// still on screen. Needs no Clerk `Auth`, since each call returns before using it.
    @Test func aResetCallWithNothingParkedSaysSoRatherThanFailingOpaquely() async {
        let auth = RealClerkAuth()
        #expect(await auth.verifyPasswordResetCode("123456") == .failure(EmailAuthModel.noPasswordResetInProgress))
        #expect(await auth.resetPassword("correct horse battery staple") == .failure(EmailAuthModel.noPasswordResetInProgress))
    }

    /// Does not reach the branch for a code typed while a reset is parked: parking one
    /// needs a real `Auth`.
    @Test func aFirstFactorVerifyWithNothingParkedNamesTheState() async {
        #expect(await RealClerkAuth().verifyCode("123456") == .failure("No verification in progress. Request a new code."))
    }

    @Test func aDeadResetSendSaysWhyAndWhatToDo() {
        let fragment = EmailAuthModel.resetSendFailed("Too many requests")
        #expect(fragment == "Too many requests. Start the reset again.")
        // Punctuated only when Clerk's text is not already a sentence.
        let sentence = EmailAuthModel.resetSendFailed("You are trying too fast!")
        #expect(sentence == "You are trying too fast! Start the reset again.")
        #expect(EmailAuthModel.resetSendFailed("   ") == "Start the reset again.")
    }

    /// `ClerkAPIError` has no public initializer, so decode one the way the SDK does.
    private func apiError(
        code: String,
        message: String? = nil,
        longMessage: String? = nil
    ) throws -> ClerkAPIError {
        var payload: [String: String] = ["code": code]
        payload["message"] = message
        payload["long_message"] = longMessage
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ClerkAPIError.self, from: JSONSerialization.data(withJSONObject: payload))
    }
}
