import ClerkKit
import Foundation
import Testing
@testable import Pipette

/// The pure transitions behind the custom sign-in screen — the same ground
/// Android's `EmailAuthFlowTest` covers for `reduceSendCode` / `reduceVerifyCode`
/// — plus the review-account routing that sends one address down Clerk's
/// password first factor instead of the emailed code.
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
        // Back to defaults: the signed-in user flips the gate, so no code-step UI
        // should be left standing behind the app.
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

    /// The sign-up call sends `legal_accepted`, which is only defensible while
    /// the user was shown these two links first. They also have to stay in step
    /// with Android's `AuthGateScreen.kt`, which pins the same pair.
    @Test func theClickwrapPointsAtThePublishedDocuments() {
        #expect(EmailAuthState.termsURL == "https://pipette.liquid.ai/terms")
        #expect(EmailAuthState.privacyPolicyURL == "https://www.liquid.ai/privacy-policy")
        // Both have to survive URL parsing, or the notice renders as plain text
        // with nothing to tap.
        #expect(URL(string: EmailAuthState.termsURL) != nil)
        #expect(URL(string: EmailAuthState.privacyPolicyURL) != nil)
    }

    /// A verified address isn't an account: the instance requires a password at
    /// sign-up, so this transition is the whole difference between a new user
    /// getting in and dead-ending on "needs a password".
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
        // One rule for both of the email step's actions: register, or take the
        // password path. Neither needs anything but an address.
        #expect(EmailAuthState.canSubmitEmail("user@example.test", submitting: false))
    }

    @Test func aBlankAddressCannotBeSubmitted() {
        #expect(!EmailAuthState.canSubmitEmail("", submitting: false))
        // Whitespace is not an address.
        #expect(!EmailAuthState.canSubmitEmail("   ", submitting: false))
    }

    @Test func aRequestInFlightBlocksBothActions() {
        #expect(!EmailAuthState.canSubmitEmail("user@example.test", submitting: true))
    }
}

/// The second-factor challenge: which factors this client can answer, and how
/// the step it lands on is set up. Mirrors Android's `SecondFactor` coverage.
@MainActor
struct SecondFactorTests {
    @Test func strategiesMapToClerksWireValues() {
        // These strings are Clerk's, not ours — a typo here silently drops a
        // factor the account actually offers.
        #expect(SecondFactor.emailCode.rawValue == "email_code")
        #expect(SecondFactor.phoneCode.rawValue == "phone_code")
        #expect(SecondFactor.totp.rawValue == "totp")
        #expect(SecondFactor.backupCode.rawValue == "backup_code")
    }

    @Test func onlyEmailAndPhoneCodesHaveToBeSent() {
        #expect(SecondFactor.emailCode.needsSending)
        #expect(SecondFactor.phoneCode.needsSending)
        // Read off something the user already holds.
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

    /// The Clerk resource id has to survive into the challenge, because the SDK's
    /// no-argument fallback cannot recover it: `sendMfaPhoneCode()` resolves the id
    /// by matching `factor.safeIdentifier == identifier`, and the identifier here is
    /// always an email — so a phone factor never matches and the send goes out with
    /// no number attached. Clerk's own UI passes `factor.phoneNumberId` explicitly.
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
        // The masked identifiers deliberately differ from the sign-in identifier —
        // that mismatch is exactly what defeats the SDK fallback.
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

    /// Built off the wire, the way the SDK builds it — `SignIn` has no public
    /// initializer, and hand-rolling one would not exercise the same decoding.
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
        // Success drops the options and the choice with everything else — the
        // signed-in user flips the gate and this screen goes away.
        #expect(challenged.afterAuthentication(.success) == EmailAuthState())
    }

    /// Clerk can answer an attempt with another `needs_second_factor` instead of
    /// a failure. Treating that as a fresh challenge would throw away the choice
    /// and the message, so the step would look like it reset itself.
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
        // Keep what was picked when it is still on offer...
        #expect(EmailAuthState.factorToAnswer(keeping: .totp, from: [.emailCode, .totp]) == .totp)
        // ...otherwise fall back to the same rule as a fresh challenge.
        #expect(EmailAuthState.factorToAnswer(keeping: .phoneCode, from: [.emailCode, .totp]) == nil)
        #expect(EmailAuthState.factorToAnswer(keeping: .phoneCode, from: [.totp]) == .totp)
        #expect(EmailAuthState.factorToAnswer(keeping: nil, from: [.emailCode, .totp]) == nil)
    }

    @Test func aDeadEndStatusNamesItself() {
        // One shared sentence made every status with no step behind it
        // indistinguishable without a debugger. Two have left this list since:
        // needs_client_trust, which is now answered as a device check, and
        // needs_new_password, which is answered by the reset step. This one has
        // no step and is not expected to grow one, since nothing in this UI can
        // supply a first factor the account did not offer.
        let message = EmailAuthModel.incompleteSignInMessage(status: "needs_first_factor")
        #expect(message.contains("needs_first_factor"))
    }

    /// The whole path, from a `needs_client_trust` sign-in decoded off the wire to
    /// the result the screen acts on. This is the one that fails if the fix is
    /// reverted: the mapping test below would still pass with `completeSignIn`
    /// back on its old `guard signIn.status == .needsSecondFactor`, because
    /// nothing would then call the mapping.
    @Test func aClientTrustSignInIsAdmittedAsADeviceCheck() throws {
        let challenge = try signIn(
            identifier: "user@example.test",
            factors: [["strategy": "email_code", "email_address_id": "idn_email_1"]],
            status: "needs_client_trust"
        )
        #expect(RealClerkAuth().completeSignIn(challenge) == .needsSecondFactor([.emailCode], .deviceVerification))
    }

    /// The same sign-in shape under the status that always worked, so the test
    /// above is pinning the new admission rather than the decoding.
    @Test func anMfaSignInStillArrivesAsMfa() throws {
        let challenge = try signIn(
            identifier: "user@example.test",
            factors: [["strategy": "totp"]],
            status: "needs_second_factor"
        )
        #expect(RealClerkAuth().completeSignIn(challenge) == .needsSecondFactor([.totp], .mfa))
    }

    /// Was `aNewPasswordSignInIsStillADeadEnd`, which pinned the behavior this
    /// branch removes: `needs_new_password` is now answered by the reset step
    /// instead of named as a dead end. `needs_client_trust` made the same move
    /// earlier, and that is the precedent this follows.
    ///
    /// What the test still guards is unchanged, which is why it kept the second
    /// factor in its payload: a sign-in carrying *both* the status and an
    /// answerable factor has to route on the status. Reaching the second-factor
    /// step here would ask for a code against an attempt Clerk is not waiting for
    /// one on.
    @Test func aNewPasswordSignInRoutesOnItsStatusNotItsFactors() throws {
        let forced = try signIn(
            identifier: "user@example.test",
            factors: [["strategy": "email_code", "email_address_id": "idn_email_1"]],
            status: "needs_new_password"
        )
        #expect(RealClerkAuth().completeSignIn(forced) == .needsNewPassword)
    }

    /// The wording is the substance of the Client Trust fix, so it is pinned here
    /// rather than left to the view. The device sentence has to lead: on its own,
    /// the factor's own sentence makes an unannounced challenge look like the
    /// password was wrong.
    @Test func onlyTheDeviceCheckExplainsWhyItIsAsking() {
        // The real factor prompt, so a rewording of it is exercised rather than
        // shadowed by a literal.
        let source = SecondFactor.emailCode.prompt
        #expect(SecondFactorReason.mfa.prompt(source) == source)

        let device = SecondFactorReason.deviceVerification.prompt(source)
        #expect(device.hasPrefix("This is the first sign-in on this device"))
        #expect(device.hasSuffix(source))
    }

    /// The heading makes the same claim more loudly, so it gets the same guard:
    /// "Two-step verification" is a statement about the account, and Client Trust
    /// fires precisely on accounts that never turned it on.
    @Test func theDeviceCheckIsNotHeadedTwoStepVerification() {
        #expect(SecondFactorReason.deviceVerification.title == "Confirm this device")
        #expect(SecondFactorReason.mfa.title == "Two-step verification")
    }

    /// No factor's own sentence may name two-step verification, because every one
    /// of them composes under a device check that has nothing to do with it.
    @Test func noFactorPromptBlamesTwoStepVerification() {
        for factor in SecondFactor.allCases {
            // Case-insensitive: a prompt opening with "Two-step verification ..."
            // is the same mistake, and an exact match would wave it through.
            #expect(!factor.prompt.lowercased().contains("two-step verification"), "\(factor) prompt names MFA")
        }
    }

    @Test func clientTrustIsAnAnswerableChallengeAndADeadEndIsNot() {
        #expect(RealClerkAuth.secondFactorReason(for: .needsClientTrust) == .deviceVerification)
        #expect(RealClerkAuth.secondFactorReason(for: .needsSecondFactor) == .mfa)
        // Decoded the way the SDK decodes it, so a rename on the wire is caught.
        #expect(RealClerkAuth.secondFactorReason(for: SignIn.Status(rawValue: "needs_client_trust")) == .deviceVerification)
        // Everything else still has no step behind it.
        #expect(RealClerkAuth.secondFactorReason(for: .needsNewPassword) == nil)
        #expect(RealClerkAuth.secondFactorReason(for: .needsFirstFactor) == nil)
        #expect(RealClerkAuth.secondFactorReason(for: .complete) == nil)
        #expect(RealClerkAuth.secondFactorReason(for: .unknown("needs_something_new")) == nil)
    }

    /// Client Trust reaches the same step, carrying the reason that makes it read
    /// as a device check rather than as two-step verification the user never set
    /// up. Mirrors Android's `needsClientTrustAdvancesToTheSameStepAsDeviceVerification`.
    @Test func clientTrustReachesTheSameStepAsADeviceCheck() {
        let start = EmailAuthState(step: .password, email: "user@example.test", submitting: true)
        let next = start.afterAuthentication(.needsSecondFactor([.emailCode], .deviceVerification))
        #expect(next.step == .secondFactor)
        #expect(next.secondFactorReason == .deviceVerification)
        #expect(next.chosenSecondFactor == .emailCode)
        #expect(next.error == nil)
    }

    /// A client-trust challenge followed by a real MFA one re-describes itself,
    /// unlike the chosen factor, which survives a re-entry. Both statuses are
    /// answered by the same call, so Clerk can return the second after the first
    /// is cleared, and a step still saying "Confirm this device" would be
    /// describing the wrong challenge.
    @Test func aLaterMfaChallengeStopsClaimingToBeADeviceCheck() {
        let onDeviceStep = EmailAuthState(
            step: .secondFactor,
            email: "user@example.test",
            secondFactorOptions: [.emailCode],
            chosenSecondFactor: .emailCode,
            secondFactorReason: .deviceVerification
        )
        // Two options, so the lone-option rule cannot supply the answer and the
        // assertion below is actually about the drop.
        let next = onDeviceStep.afterAuthentication(.needsSecondFactor([.totp, .backupCode], .mfa))
        #expect(next.secondFactorReason == .mfa)
        // The previously chosen factor is not on offer any more, so it is dropped
        // rather than carried into a challenge that cannot answer it.
        #expect(next.chosenSecondFactor == nil)
    }

    /// Every arrival at the step is a distinct challenge, including one that
    /// changes nothing else about the state. `SecondFactorAnswer` keys its `.id`
    /// on this count, and a re-challenge sends a fresh code, so without the bump
    /// the field would keep the digits already spent on the previous one and
    /// submit them against a code that is no longer valid.
    @Test func eachChallengeIsCountedEvenWhenItLooksIdentical() {
        let challenged = EmailAuthState(
            step: .secondFactor,
            email: "user@example.test",
            secondFactorOptions: [.emailCode],
            chosenSecondFactor: .emailCode,
            secondFactorReason: .mfa
        )
        // Same reason, same lone factor: the count is the only thing left that can
        // tell the two challenges apart.
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

/// What a failed sign-in actually puts on screen. Clerk's own copy is the whole
/// point — "Incorrect code" or "Password is incorrect" beats anything generic —
/// so these pin that it is preferred and that nothing falls through to a raw
/// Swift error description.
@MainActor
struct ClerkErrorMessageTests {
    /// A challenge whose delivery this screen can't drive (Client Trust can be
    /// configured to use an email *link*, which is not a code anyone can type)
    /// names the device rather than the account, so the user is not sent off to
    /// re-read their password.
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
        // Whatever it is, it must not be the raw "The operation couldn't be
        // completed. (ClerkKit.ClerkAPIError error 1.)" that `localizedDescription`
        // produces for an error carrying no message.
        #expect(!message.contains("ClerkAPIError"))
        #expect(message == "Something went wrong. Try again.")
    }

    /// A sign-up that stops short of an account should say what it is waiting
    /// for. Today that is reliably a password — the instance requires one at
    /// sign-up and the flow does not collect it yet — so the message is the only
    /// thing telling a user (or us) why registration went nowhere.
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
        // Forward compatibility: an unmapped field must not drop out of the list
        // silently, leaving "needs more:" with nothing after it.
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
        // Bumped so a resend arriving on this step clears the digits belonging to
        // the code it replaced.
        #expect(next.resetCodeGeneration == start.resetCodeGeneration + 1)
    }

    @Test func aFailedResetSendLeavesTheDigitsAlone() {
        let onStep = EmailAuthState(step: .resetCode, email: "user@example.test", submitting: true, resetCodeGeneration: 3)
        let next = onStep.afterSendResetCode(.failure("Too many requests. Start the reset again."))
        // The whole point of moving the counter only on a send Clerk accepted: a
        // resend that never left the device must not clear a field whose digits
        // are still the live code's.
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
        // The password typed here was *correct*. Without the flag the step reads
        // as a rejection of it, and the user goes back to check a password that
        // was never the problem.
        let signingIn = EmailAuthState(step: .password, email: "user@example.test", submitting: true)
        let next = signingIn.afterAuthentication(.needsNewPassword)
        #expect(next.step == .resetPassword)
        #expect(next.resetWasForced)
    }

    @Test func reEnteringTheResetStepKeepsWhatOnlyItCanKnow() {
        // The recorded request has to survive a re-entry, or a reset the user
        // asked for starts wording itself as demanded on the second pass.
        let requested = EmailAuthState(step: .resetPassword, email: "user@example.test", submitting: true, resetWasRequested: true)
        let next = requested.afterAuthentication(.needsNewPassword)
        #expect(next.step == .resetPassword)
        #expect(next.resetWasRequested)
        #expect(!next.resetWasForced)
        #expect(!next.submitting)
    }

    /// Both directions through the second-factor step, which is where a
    /// step-derived flag cannot be right for both. A 2FA account's *requested*
    /// reset arrives at the password from the challenge (Clerk answers the reset
    /// code with `needs_second_factor`), and so does a *demanded* one, since
    /// `needs_client_trust` fires on exactly the password sign-ins the forced
    /// case is about. Deriving from the step being left gets one wrong whichever
    /// way it is written; both of these pass only because the request is recorded.
    @Test func whoAskedSurvivesTheChallengeInBothDirections() {
        let asked = EmailAuthState(step: .resetCode, email: "user@example.test", submitting: true, resetWasRequested: true)
        let askedThenChallenged = asked.afterAuthentication(.needsSecondFactor([.totp], .mfa))
        #expect(askedThenChallenged.step == .secondFactor)
        let askedFinish = askedThenChallenged.afterAuthentication(.needsNewPassword)
        #expect(askedFinish.step == .resetPassword)
        #expect(!askedFinish.resetWasForced)

        // Nothing recorded a request, so this is Clerk's demand, and it stays one
        // through the challenge. This is the direction the previous attempt broke:
        // it preserved the untouched `false` default and called it "asked for".
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
        // The requested route needs no cause: the user tapped the link and is
        // expecting this screen.
        #expect(requested.hasPrefix("Choose a password"))
        // The forced one needs one badly, since the password just typed was
        // correct.
        #expect(forced.hasPrefix("A password has to be set on this account"))
        // Neither claims the account has no password, and neither says "new":
        // only one of the ways in fits that, and the others contradict it.
        #expect(!requested.contains("new password"))
        #expect(!forced.contains("new password"))
    }

    @Test func aPromptWithNoAddressToNameOmitsIt() {
        // A forced reset is reachable through OAuth, which never writes
        // `state.email`, so this step can be reached with nothing to interpolate.
        // Unguarded it opened with a stray space, or named an empty address.
        #expect(EmailAuthModel.forEmail("") == "")
        #expect(EmailAuthModel.forEmail("   ") == "")
        #expect(EmailAuthModel.forEmail("user@example.test") == " for user@example.test")
        let anonymous = EmailAuthModel.resetPasswordPrompt(email: "", wasForced: true)
        #expect(!anonymous.contains("  "))
        #expect(anonymous.contains("Choose a password,"))
    }

    /// The states a dead reset leaves behind, which are exactly what the
    /// send-failure path creates on purpose: nothing parked, with a step that
    /// belongs to a reset still on screen. Reachable without a Clerk `Auth`,
    /// since each of these returns before it would make a call.
    @Test func aResetCallWithNothingParkedSaysSoRatherThanFailingOpaquely() async {
        let auth = RealClerkAuth()
        #expect(await auth.verifyPasswordResetCode("123456") == .failure(EmailAuthModel.noPasswordResetInProgress))
        #expect(await auth.resetPassword("correct horse battery staple") == .failure(EmailAuthModel.noPasswordResetInProgress))
    }

    /// With nothing parked, the first-factor verify names that rather than
    /// failing opaquely.
    ///
    /// It has a second answer this does not reach, for a code typed here while a
    /// *reset* is parked, which names the step that code belongs to. Driving it
    /// needs a parked reset, and parking one needs a real `Auth`, so that branch
    /// is unpinned. Said plainly rather than implied by a test name that sounds
    /// like it covers both.
    @Test func aFirstFactorVerifyWithNothingParkedNamesTheState() async {
        #expect(await RealClerkAuth().verifyCode("123456") == .failure("No verification in progress. Request a new code."))
    }

    @Test func aDeadResetSendSaysWhyAndWhatToDo() {
        // Clerk's half explains why; ours is the half only this app knows, and
        // without it the step sits there inviting a code that cannot be answered.
        let fragment = EmailAuthModel.resetSendFailed("Too many requests")
        #expect(fragment == "Too many requests. Start the reset again.")
        // Clerk's text is not reliably a sentence, so it is punctuated only when
        // it needs it, never twice.
        let sentence = EmailAuthModel.resetSendFailed("You are trying too fast!")
        #expect(sentence == "You are trying too fast! Start the reset again.")
        // Nothing worth leading with still has to say what to do.
        #expect(EmailAuthModel.resetSendFailed("   ") == "Start the reset again.")
    }

    /// `ClerkAPIError` has no public initializer, so build one the way the SDK
    /// does — off the wire.
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
