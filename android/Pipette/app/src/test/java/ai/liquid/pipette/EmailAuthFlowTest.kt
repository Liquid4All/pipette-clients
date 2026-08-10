package ai.liquid.pipette

import ai.liquid.pipette.compose.shell.EmailAuthUiState
import ai.liquid.pipette.compose.shell.applyAuthOutcome
import ai.liquid.pipette.compose.shell.beginAuthAction
import ai.liquid.pipette.compose.shell.reduceSecondFactorPrepared
import ai.liquid.pipette.compose.shell.reduceSendCode
import ai.liquid.pipette.compose.shell.reduceSendResetCode
import ai.liquid.pipette.compose.shell.reduceSignInCompletion
import ai.liquid.pipette.compose.shell.secondFactorToDeliver
import ai.liquid.pipette.fakes.FakeClerkAuth
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Covers the pure sign-in transitions ([reduceSendCode] / [reduceSignInCompletion]), the pair that wraps every request ([beginAuthAction] /
 * [applyAuthOutcome]), and a pass over [FakeClerkAuth] for each flow.
 */
class EmailAuthFlowTest {
  @Test
  fun sendSuccessAdvancesToCodeStepKeepingEmail() {
    val start = EmailAuthUiState(email = "user@example.test", submitting = true)
    val next = reduceSendCode(start, AuthActionResult.Success)
    assertEquals(EmailAuthUiState.Step.Code, next.step)
    assertEquals("user@example.test", next.email)
    assertFalse(next.submitting)
    assertNull(next.error)
  }

  @Test
  fun sendErrorStaysOnEmailAndSurfacesMessage() {
    val start = EmailAuthUiState(email = "user@example.test", submitting = true)
    val next = reduceSendCode(start, AuthActionResult.Error("That email looks invalid"))
    assertEquals(EmailAuthUiState.Step.Email, next.step)
    assertFalse(next.submitting)
    assertEquals("That email looks invalid", next.error)
  }

  @Test
  fun verifySuccessResetsToInitialState() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Code, email = "user@example.test", submitting = true)
    // Reset to defaults: the gate flips to Ready off the SignedIn state flow, so no lingering code-step UI.
    assertEquals(EmailAuthUiState(), reduceSignInCompletion(start, AuthActionResult.Success))
  }

  @Test
  fun verifyErrorKeepsCodeStepAndSurfacesMessage() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Code, email = "user@example.test", submitting = true)
    val next = reduceSignInCompletion(start, AuthActionResult.Error("Incorrect code"))
    assertEquals(EmailAuthUiState.Step.Code, next.step)
    assertFalse(next.submitting)
    assertEquals("Incorrect code", next.error)
  }

  @Test
  fun fullFlowOverFakeSendsThenVerifies() = runTest {
    val auth = FakeClerkAuth()
    var state = EmailAuthUiState(email = "user@example.test", submitting = true)

    state = reduceSendCode(state, auth.sendEmailCode("user@example.test"))
    assertEquals(EmailAuthUiState.Step.Code, state.step)

    state = reduceSignInCompletion(state, auth.verifyEmailCode("123456"))
    assertEquals(EmailAuthUiState(), state)

    assertEquals(listOf("user@example.test"), auth.sentEmails)
    assertEquals(listOf("123456"), auth.verifiedCodes)
  }

  @Test
  fun sendErrorSurfacesFromFake() = runTest {
    val auth = FakeClerkAuth(sendResult = AuthActionResult.Error("rate limited"))
    val next = reduceSendCode(EmailAuthUiState(submitting = true), auth.sendEmailCode("user@example.test"))
    assertEquals(EmailAuthUiState.Step.Email, next.step)
    assertEquals("rate limited", next.error)
  }

  @Test
  fun oauthErrorStaysOnEmailStepAndSurfacesMessage() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Email, submitting = true)
    val next = reduceSignInCompletion(start, AuthActionResult.Error("Sign-in was cancelled"))
    assertEquals(EmailAuthUiState.Step.Email, next.step)
    assertFalse(next.submitting)
    assertEquals("Sign-in was cancelled", next.error)
  }

  @Test
  fun oauthFlowOverFakeRecordsStrategy() = runTest {
    val auth = FakeClerkAuth()
    val next = reduceSignInCompletion(EmailAuthUiState(submitting = true), auth.signInWithOAuth("oauth_google"))
    assertEquals(EmailAuthUiState(), next)
    assertEquals(listOf("oauth_google"), auth.oauthStrategies)
  }

  @Test
  fun passwordFlowOverFakePassesBothCredentials() = runTest {
    val auth = FakeClerkAuth()
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = "user@example.test", submitting = true)
    val next = reduceSignInCompletion(start, auth.signInWithPassword("user@example.test", "s3cret"))
    assertEquals(EmailAuthUiState(), next)
    assertEquals(listOf("user@example.test" to "s3cret"), auth.passwordAttempts)
  }

  @Test
  fun passwordErrorSurfacesFromFake() = runTest {
    val auth = FakeClerkAuth(passwordResult = AuthActionResult.Error("Password sign-in isn't enabled"))
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = "user@example.test", submitting = true)
    val next = reduceSignInCompletion(start, auth.signInWithPassword("user@example.test", "nope"))
    assertEquals(EmailAuthUiState.Step.Password, next.step)
    // Spinner cleared, so the retype-and-resubmit path is actually reachable.
    assertFalse(next.submitting)
    assertEquals("Password sign-in isn't enabled", next.error)
  }

  // --- runAuthAction's two halves: beginAuthAction raises the flag, applyAuthOutcome decides
  // whether the outcome that eventually lands is still wanted. ---

  @Test
  fun beginRaisesSubmittingAndClearsStaleError() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Code, email = TEST_EMAIL, error = "Incorrect code")
    val next = beginAuthAction(start)
    assertTrue(next.submitting)
    // The previous attempt's message goes as the new one starts, so it can't be read as this attempt's result.
    assertNull(next.error)
    // Only the flag and the error move: the step and the address the attempt is for are left alone.
    assertEquals(EmailAuthUiState.Step.Code, next.step)
    assertEquals(TEST_EMAIL, next.email)
  }

  @Test
  fun successIsFoldedInWhileTheAttemptIsStillLive() {
    val live = beginAuthAction(EmailAuthUiState(step = EmailAuthUiState.Step.Code, email = TEST_EMAIL))
    assertEquals(EmailAuthUiState(), applyAuthOutcome(live, AuthActionResult.Success, ::reduceSignInCompletion))
  }

  @Test
  fun errorIsFoldedInWhileTheAttemptIsStillLive() {
    val live = beginAuthAction(EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL))
    val next = applyAuthOutcome(live, AuthActionResult.Error("Incorrect password"), ::reduceSignInCompletion)
    assertEquals(EmailAuthUiState.Step.Password, next.step)
    assertFalse(next.submitting)
    assertEquals("Incorrect password", next.error)
  }

  @Test
  fun lateErrorIsDiscardedAfterTheFlowWasReset() {
    // setGateBypass mid-request resets the flow, which lowers `submitting`. The error that lands afterwards
    // must not stamp itself onto the fresh email step the user is now looking at.
    val afterReset = EmailAuthUiState()
    val next = applyAuthOutcome(afterReset, AuthActionResult.Error("Incorrect code"), ::reduceSignInCompletion)
    assertEquals(EmailAuthUiState(), next)
    assertNull(next.error)
  }

  @Test
  fun lateSuccessIsDiscardedAfterTheFlowWasReset() {
    // Success is dropped for the same reason. Nothing is lost either way: the SignedIn state flow drives the
    // gate to Ready, not this reducer's return. Note this assertion cannot fail while a completing success
    // reduces to the initial state, since that is the reset state too; it pins the pair together, so a future
    // reduceSignInCompletion that returns something else has to come back through here.
    val afterReset = EmailAuthUiState()
    assertEquals(EmailAuthUiState(), applyAuthOutcome(afterReset, AuthActionResult.Success, ::reduceSignInCompletion))
  }

  @Test
  fun lateSendSuccessCannotAdvanceToACodeStepWithNoAddress() {
    // The send path's own hazard, and the reason the guard covers Success too: reduceSendCode would advance
    // to the code step, which renders the address it sent to, but the reset already cleared it.
    val afterReset = EmailAuthUiState()
    val next = applyAuthOutcome(afterReset, AuthActionResult.Success, ::reduceSendCode)
    assertEquals(EmailAuthUiState.Step.Email, next.step)
    assertEquals("", next.email)
  }

  @Test
  fun beginThenResetThenLateOutcomeLeavesTheResetState() {
    // The whole sequence runAuthAction guards against, in order.
    var state = EmailAuthUiState(email = TEST_EMAIL)
    state = beginAuthAction(state)
    state = EmailAuthUiState() // setGateBypass resets while the request is outstanding
    state = applyAuthOutcome(state, AuthActionResult.Error("rate limited"), ::reduceSendCode)
    assertEquals(EmailAuthUiState(), state)
  }

  @Test
  fun outcomeFoldsIntoALaterAttemptWhenOneStartedInBetween() {
    // Characterizes an accepted tradeoff rather than desired behavior. The flag is flow-wide, not
    // per-attempt, so a first attempt's late outcome lands on a second attempt's state when the bypass was
    // toggled and a new attempt started in between. It costs a stale message on a live step; see
    // runAuthAction's KDoc for why that beats carrying a generation counter.
    val secondAttempt = beginAuthAction(EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL))
    val next = applyAuthOutcome(secondAttempt, AuthActionResult.Error("stale message from the first attempt"), ::reduceSignInCompletion)
    assertEquals("stale message from the first attempt", next.error)
  }

  @Test
  fun needsSecondFactorAdvancesToTheMfaStepAndPreselectsALoneFactor() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    val next = reduceSignInCompletion(start, AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.EmailCode)))
    assertEquals(EmailAuthUiState.Step.SecondFactor, next.step)
    // One option means no chooser is worth rendering, so the step opens ready to accept a code.
    assertEquals(SecondFactor.EmailCode, next.selectedSecondFactor)
    assertEquals(listOf(SecondFactor.EmailCode), next.secondFactorOptions)
    assertFalse(next.submitting)
    // A challenge is progress, not a failure: nothing to apologize for on screen.
    assertNull(next.error)
  }

  @Test
  fun needsSecondFactorWithSeveralOptionsLeavesTheChoiceToTheUser() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    val options = listOf(SecondFactor.Totp, SecondFactor.BackupCode)
    val next = reduceSignInCompletion(start, AuthActionResult.NeedsSecondFactor(options))
    assertEquals(options, next.secondFactorOptions)
    assertNull(next.selectedSecondFactor)
  }

  /**
   * Client Trust reaches the same step, carrying the reason that makes it read as a device check rather than as two-step verification the user never
   * set up. The default is [SecondFactorReason.Mfa], so a reason that failed to be threaded through would show the wrong copy rather than fail to
   * compile, which is what this pins.
   */
  @Test
  fun needsClientTrustAdvancesToTheSameStepAsDeviceVerification() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    val next =
      reduceSignInCompletion(start, AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.EmailCode), SecondFactorReason.DeviceVerification))
    assertEquals(EmailAuthUiState.Step.SecondFactor, next.step)
    assertEquals(SecondFactorReason.DeviceVerification, next.secondFactorReason)
    assertEquals(SecondFactor.EmailCode, next.selectedSecondFactor)
    assertNull(next.error)
  }

  /**
   * A client-trust challenge followed by a real MFA one re-describes itself, unlike the chosen factor, which survives a re-entry. Both statuses are
   * answered by the same call, so Clerk can return the second after the first is cleared, and a step still saying "Confirm this device" would be
   * describing the wrong challenge.
   */
  @Test
  fun aLaterMfaChallengeStopsClaimingToBeADeviceCheck() {
    val onDeviceStep =
      EmailAuthUiState(
        step = EmailAuthUiState.Step.SecondFactor,
        email = TEST_EMAIL,
        secondFactorOptions = listOf(SecondFactor.EmailCode),
        selectedSecondFactor = SecondFactor.EmailCode,
        secondFactorReason = SecondFactorReason.DeviceVerification,
      )
    val next = reduceSignInCompletion(onDeviceStep, AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.Totp), SecondFactorReason.Mfa))
    assertEquals(SecondFactorReason.Mfa, next.secondFactorReason)
    // The previously chosen factor is not on offer any more, so it is dropped rather than carried into a challenge that cannot answer it.
    assertEquals(SecondFactor.Totp, next.selectedSecondFactor)
  }

  @Test
  fun emailCodeFirstFactorCanAlsoLandOnTheMfaStep() {
    // The passwordless route hits MFA too: an account with a second factor is challenged whichever first factor cleared.
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Code, email = TEST_EMAIL, submitting = true)
    val next = reduceSignInCompletion(start, AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.EmailCode)))
    assertEquals(EmailAuthUiState.Step.SecondFactor, next.step)
  }

  @Test
  fun preparedSecondFactorStaysOnTheStep() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.SecondFactor, email = TEST_EMAIL, submitting = true)
    val next = reduceSecondFactorPrepared(start, AuthActionResult.Success)
    assertEquals(EmailAuthUiState.Step.SecondFactor, next.step)
    assertFalse(next.submitting)
  }

  @Test
  fun failedSendKeepsTheUserOnTheMfaStepRatherThanBouncingThemBack() {
    val start =
      EmailAuthUiState(
        step = EmailAuthUiState.Step.SecondFactor,
        email = TEST_EMAIL,
        submitting = true,
        secondFactorOptions = listOf(SecondFactor.EmailCode),
        selectedSecondFactor = SecondFactor.EmailCode,
      )
    val next = reduceSecondFactorPrepared(start, AuthActionResult.Error("rate limited"))
    // The first factor is already spent; sending them back would make them re-enter a password Clerk accepted.
    assertEquals(EmailAuthUiState.Step.SecondFactor, next.step)
    assertEquals("rate limited", next.error)
    assertEquals(SecondFactor.EmailCode, next.selectedSecondFactor)
  }

  @Test
  fun mfaFlowOverFakePreparesThenVerifies() = runTest {
    val auth = FakeClerkAuth(passwordResult = AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.EmailCode)))
    var state = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)

    state = reduceSignInCompletion(state, auth.signInWithPassword(TEST_EMAIL, "pw"))
    assertEquals(EmailAuthUiState.Step.SecondFactor, state.step)

    state = reduceSecondFactorPrepared(beginAuthAction(state), auth.prepareSecondFactor(SecondFactor.EmailCode))
    assertEquals(EmailAuthUiState.Step.SecondFactor, state.step)

    state = reduceSignInCompletion(beginAuthAction(state), auth.verifySecondFactor(SecondFactor.EmailCode, "424242"))
    // Success resets the whole flow, MFA fields included, so nothing lingers behind the Ready gate.
    assertEquals(EmailAuthUiState(), state)

    assertEquals(listOf(SecondFactor.EmailCode), auth.preparedSecondFactors)
    assertEquals(listOf(SecondFactor.EmailCode to "424242"), auth.secondFactorAttempts)
  }

  @Test
  fun wrongMfaCodeKeepsTheStepAndItsSelection() = runTest {
    val auth = FakeClerkAuth(secondFactorResult = AuthActionResult.Error("Incorrect code"))
    val start =
      EmailAuthUiState(
        step = EmailAuthUiState.Step.SecondFactor,
        email = TEST_EMAIL,
        submitting = true,
        secondFactorOptions = listOf(SecondFactor.Totp),
        selectedSecondFactor = SecondFactor.Totp,
      )
    val next = reduceSignInCompletion(start, auth.verifySecondFactor(SecondFactor.Totp, "000000"))
    assertEquals(EmailAuthUiState.Step.SecondFactor, next.step)
    assertEquals("Incorrect code", next.error)
    assertEquals(SecondFactor.Totp, next.selectedSecondFactor)
  }

  @Test
  fun landingOnTheMfaStepDeliversALonePreselectedCode() {
    // The regression this guards: with one email_code factor there is no chooser to tap, so nothing would ask Clerk to send, and the step would
    // claim a code had been mailed while none had.
    val before = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    val after = reduceSignInCompletion(before, AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.EmailCode)))
    assertEquals(SecondFactor.EmailCode, secondFactorToDeliver(before, after))
  }

  @Test
  fun nothingIsDeliveredForFactorsTheUserAlreadyHolds() {
    val before = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    val after = reduceSignInCompletion(before, AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.Totp)))
    assertNull(secondFactorToDeliver(before, after))
  }

  @Test
  fun nothingIsDeliveredWhenTheUserStillHasToChoose() {
    val before = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    val options = listOf(SecondFactor.EmailCode, SecondFactor.Totp)
    val after = reduceSignInCompletion(before, AuthActionResult.NeedsSecondFactor(options))
    // The chooser owns the send once there is a choice to make; picking one is what asks for it.
    assertNull(secondFactorToDeliver(before, after))
  }

  /**
   * A verify answered with a *further* challenge asks for that challenge's code, even though the step it lands on is the step it was already on. The
   * regression: a cleared device check followed by the account's own two-step verification would otherwise sit there saying a code had been sent
   * while nothing had, with the code Clerk just consumed still in the field.
   */
  @Test
  fun aChainedChallengeDeliversItsOwnCode() {
    val onDeviceStep =
      EmailAuthUiState(
        step = EmailAuthUiState.Step.SecondFactor,
        email = TEST_EMAIL,
        submitting = true,
        secondFactorOptions = listOf(SecondFactor.EmailCode),
        selectedSecondFactor = SecondFactor.EmailCode,
        secondFactorReason = SecondFactorReason.DeviceVerification,
      )
    val after = reduceSignInCompletion(onDeviceStep, AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.EmailCode), SecondFactorReason.Mfa))
    assertEquals(SecondFactor.EmailCode, secondFactorToDeliver(onDeviceStep, after))
  }

  @Test
  fun alreadyOnTheMfaStepDoesNotSendAgain() {
    // Guards the recursion: the delivery itself reduces through the step. It leaves the reason and the factor where they were, which is what tells it
    // apart from the chained challenge above.
    val before =
      EmailAuthUiState(
        step = EmailAuthUiState.Step.SecondFactor,
        email = TEST_EMAIL,
        submitting = true,
        secondFactorOptions = listOf(SecondFactor.EmailCode),
        selectedSecondFactor = SecondFactor.EmailCode,
      )
    val after = reduceSecondFactorPrepared(before, AuthActionResult.Success)
    assertNull(secondFactorToDeliver(before, after))
  }

  @Test
  fun aRepeatChallengeKeepsTheFactorTheUserPicked() {
    val start =
      EmailAuthUiState(
        step = EmailAuthUiState.Step.SecondFactor,
        email = TEST_EMAIL,
        submitting = true,
        secondFactorOptions = listOf(SecondFactor.EmailCode, SecondFactor.Totp),
        selectedSecondFactor = SecondFactor.Totp,
      )
    val next = reduceSignInCompletion(start, AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.EmailCode, SecondFactor.Totp)))
    // Resetting to null here would empty the chooser and the code field keyed to it, so the step would look like it reset itself.
    assertEquals(SecondFactor.Totp, next.selectedSecondFactor)
  }

  @Test
  fun aRepeatChallengeDropsASelectionTheAccountNoLongerOffers() {
    val start =
      EmailAuthUiState(
        step = EmailAuthUiState.Step.SecondFactor,
        email = TEST_EMAIL,
        submitting = true,
        secondFactorOptions = listOf(SecondFactor.EmailCode, SecondFactor.Totp),
        selectedSecondFactor = SecondFactor.Totp,
      )
    val next = reduceSignInCompletion(start, AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.EmailCode)))
    // Down to one option, so it is pre-selected rather than left pointing at a factor that is gone.
    assertEquals(SecondFactor.EmailCode, next.selectedSecondFactor)
  }

  @Test
  fun verifyingANewAddressAdvancesToPasswordCreation() {
    // The instance requires a password at sign-up, so a verified email leaves the registration one field short of an account.
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Code, email = TEST_EMAIL, submitting = true)
    val next = reduceSignInCompletion(start, AuthActionResult.NeedsPassword)
    assertEquals(EmailAuthUiState.Step.CreatePassword, next.step)
    assertEquals(TEST_EMAIL, next.email)
    assertFalse(next.submitting)
    // Being asked for a password is the next step of registering, not a failed sign-in.
    assertNull(next.error)
  }

  @Test
  fun passwordCreationClearsAnyMfaState() {
    // A registration has no second factor yet, so nothing from a prior sign-in attempt should follow it onto the step.
    val start =
      EmailAuthUiState(
        step = EmailAuthUiState.Step.Code,
        email = TEST_EMAIL,
        submitting = true,
        secondFactorOptions = listOf(SecondFactor.EmailCode),
        selectedSecondFactor = SecondFactor.EmailCode,
      )
    val next = reduceSignInCompletion(start, AuthActionResult.NeedsPassword)
    assertEquals(emptyList<SecondFactor>(), next.secondFactorOptions)
    assertNull(next.selectedSecondFactor)
  }

  @Test
  fun creatingAPasswordCompletesTheSignUpOverTheFake() = runTest {
    val auth = FakeClerkAuth(verifyResult = AuthActionResult.NeedsPassword)
    var state = EmailAuthUiState(step = EmailAuthUiState.Step.Code, email = TEST_EMAIL, submitting = true)

    state = reduceSignInCompletion(state, auth.verifyEmailCode("300707"))
    assertEquals(EmailAuthUiState.Step.CreatePassword, state.step)

    state = reduceSignInCompletion(beginAuthAction(state), auth.createPassword("correct horse battery"))
    // Success resets the flow, exactly as a completed sign-in does: the SignedIn flip drives the gate.
    assertEquals(EmailAuthUiState(), state)
    assertEquals(listOf("correct horse battery"), auth.createdPasswords)
  }

  @Test
  fun aRejectedPasswordKeepsTheStepAndSurfacesClerksReason() = runTest {
    // Length, strength, and the breach corpus are all enforced server-side, so the message has to be Clerk's, not ours.
    val auth = FakeClerkAuth(createPasswordResult = AuthActionResult.Error("Password has been found in an online data breach."))
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.CreatePassword, email = TEST_EMAIL, submitting = true)
    val next = reduceSignInCompletion(start, auth.createPassword("password1234"))
    assertEquals(EmailAuthUiState.Step.CreatePassword, next.step)
    assertEquals("Password has been found in an online data breach.", next.error)
  }

  // --- Password reset: the route that gives an account with no password one, and the route back from a forgotten
  // one. An account in that state offers no `password` strategy at all, so the password step can only fail on it. ---

  @Test
  fun sendingAResetCodeAdvancesToItsOwnCodeStep() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    val next = reduceSendResetCode(start, AuthActionResult.Success)
    // Not Step.Code: that code answers a sign-in, this one authorizes a reset.
    assertEquals(EmailAuthUiState.Step.ResetCode, next.step)
    assertEquals(TEST_EMAIL, next.email)
    assertFalse(next.submitting)
    assertNull(next.error)
  }

  @Test
  fun aFailedResetSendStaysOnThePasswordStep() = runTest {
    // Driven through the fake like sendErrorSurfacesFromFake, so the scripted result and the reducer are exercised together. Whichever step asked for
    // the code is the step that keeps it: the one the reset was started from still has the address and the way out.
    val auth = FakeClerkAuth().apply { sendResetCodeResult = AuthActionResult.Error("Couldn't find your account.") }
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    val next = reduceSendResetCode(start, auth.sendPasswordResetCode(TEST_EMAIL))
    assertEquals(EmailAuthUiState.Step.Password, next.step)
    assertFalse(next.submitting)
    assertEquals("Couldn't find your account.", next.error)
    assertEquals(listOf(TEST_EMAIL), auth.resetCodeEmails)
  }

  @Test
  fun aResendFromTheCodeStepStaysOnTheCodeStep() = runTest {
    // The reducer's other entry point: the resend link lives on the code step, so both outcomes have to leave the user there. A success that advanced
    // "forward" would re-enter the step and wipe the field it just asked to be filled; a failure that fell back to the password step would strand the
    // code the auth layer keeps answerable on a screen with nowhere to type it.
    val auth = FakeClerkAuth()
    val onCodeStep = EmailAuthUiState(step = EmailAuthUiState.Step.ResetCode, email = TEST_EMAIL, submitting = true)

    val resent = reduceSendResetCode(onCodeStep, auth.sendPasswordResetCode(TEST_EMAIL))
    assertEquals(EmailAuthUiState.Step.ResetCode, resent.step)
    assertFalse(resent.submitting)
    assertNull(resent.error)

    auth.sendResetCodeResult = AuthActionResult.Error("Too many requests")
    val failed = reduceSendResetCode(beginAuthAction(resent), auth.sendPasswordResetCode(TEST_EMAIL))
    assertEquals(EmailAuthUiState.Step.ResetCode, failed.step)
    assertEquals("Too many requests", failed.error)
  }

  @Test
  fun aClearedResetCodeAdvancesToTheResetPasswordStep() {
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.ResetCode, email = TEST_EMAIL, submitting = true)
    val next = reduceSignInCompletion(start, AuthActionResult.NeedsNewPassword)
    assertEquals(EmailAuthUiState.Step.ResetPassword, next.step)
    assertEquals(TEST_EMAIL, next.email)
    assertFalse(next.submitting)
    // Being asked for the new password is what success looks like here, not a failure.
    assertNull(next.error)
  }

  @Test
  fun theResetPasswordStepClearsAnyMfaState() {
    // An account with two-step verification is challenged *after* the reset, so factors read off the attempt so far belong to a challenge that has
    // not been issued yet.
    val start =
      EmailAuthUiState(
        step = EmailAuthUiState.Step.ResetCode,
        email = TEST_EMAIL,
        submitting = true,
        secondFactorOptions = listOf(SecondFactor.EmailCode),
        selectedSecondFactor = SecondFactor.EmailCode,
      )
    val next = reduceSignInCompletion(start, AuthActionResult.NeedsNewPassword)
    assertEquals(emptyList<SecondFactor>(), next.secondFactorOptions)
    assertNull(next.selectedSecondFactor)
  }

  @Test
  fun theWholeResetOverTheFakeEndsSignedIn() = runTest {
    val auth = FakeClerkAuth()
    var state = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)

    state = reduceSendResetCode(state, auth.sendPasswordResetCode(TEST_EMAIL))
    assertEquals(EmailAuthUiState.Step.ResetCode, state.step)

    state = reduceSignInCompletion(beginAuthAction(state), auth.verifyPasswordResetCode("515151"))
    assertEquals(EmailAuthUiState.Step.ResetPassword, state.step)

    state = reduceSignInCompletion(beginAuthAction(state), auth.resetPassword("correct horse battery"))
    // Success resets the flow like any completing call: the SignedIn flip drives the gate to Ready.
    assertEquals(EmailAuthUiState(), state)

    assertEquals(listOf(TEST_EMAIL), auth.resetCodeEmails)
    assertEquals(listOf("515151"), auth.verifiedResetCodes)
    assertEquals(listOf("correct horse battery"), auth.resetPasswords)
  }

  @Test
  fun aWrongResetCodeKeepsItsStep() = runTest {
    val auth = FakeClerkAuth().apply { verifyResetCodeResult = AuthActionResult.Error("Incorrect code") }
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.ResetCode, email = TEST_EMAIL, submitting = true)
    val next = reduceSignInCompletion(start, auth.verifyPasswordResetCode("000000"))
    assertEquals(EmailAuthUiState.Step.ResetCode, next.step)
    assertFalse(next.submitting)
    assertEquals("Incorrect code", next.error)
  }

  @Test
  fun aRejectedResetPasswordKeepsTheStepAndSurfacesClerksReason() = runTest {
    // Same division of labour as the sign-up password: Clerk owns length, strength, and the breach corpus.
    val auth = FakeClerkAuth().apply { resetPasswordResult = AuthActionResult.Error("Password has been found in an online data breach.") }
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.ResetPassword, email = TEST_EMAIL, submitting = true)
    val next = reduceSignInCompletion(start, auth.resetPassword("password1234"))
    assertEquals(EmailAuthUiState.Step.ResetPassword, next.step)
    assertEquals("Password has been found in an online data breach.", next.error)
  }

  @Test
  fun anMfaAccountIsStillChallengedAfterItsReset() = runTest {
    // The reset clears a first factor, so Clerk can answer it with the account's second one rather than a session. An emailed factor rather than
    // TOTP, so the delivery assertion below has something to check: Clerk sends no second-factor code unless something asks, and a step claiming a
    // code was mailed while none was is the failure mode of every path into it.
    val auth = FakeClerkAuth().apply { resetPasswordResult = AuthActionResult.NeedsSecondFactor(listOf(SecondFactor.EmailCode)) }
    val before = EmailAuthUiState(step = EmailAuthUiState.Step.ResetPassword, email = TEST_EMAIL, submitting = true)
    val after = reduceSignInCompletion(before, auth.resetPassword("correct horse battery"))
    assertEquals(EmailAuthUiState.Step.SecondFactor, after.step)
    assertEquals(SecondFactor.EmailCode, after.selectedSecondFactor)
    assertEquals(SecondFactor.EmailCode, secondFactorToDeliver(before, after))
  }

  @Test
  fun theResetStepRemembersWhetherItWasAskedForOrDemanded() {
    // What the step's copy keys on. Answering a reset code is a step the user asked for and needs no explanation; a correct password answered with
    // needs_new_password is not, and the screen has to say why or it reads as a rejection. Derived from the step being left, so nothing extra crosses
    // the auth seam to carry it.
    val fromCode = EmailAuthUiState(step = EmailAuthUiState.Step.ResetCode, email = TEST_EMAIL, submitting = true)
    assertFalse(reduceSignInCompletion(fromCode, AuthActionResult.NeedsNewPassword).resetWasForced)

    val fromPassword = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    assertTrue(reduceSignInCompletion(fromPassword, AuthActionResult.NeedsNewPassword).resetWasForced)

    // The half that "remembers" is about: a rejected password re-renders the step, and a re-entry must not re-derive the answer from the step it is
    // already on, which would read as "not the code step" and reword a reset the user asked for.
    val requested = reduceSignInCompletion(fromCode, AuthActionResult.NeedsNewPassword)
    assertFalse(reduceSignInCompletion(beginAuthAction(requested), AuthActionResult.Error("Too weak")).resetWasForced)
    assertFalse(reduceSignInCompletion(beginAuthAction(requested), AuthActionResult.NeedsNewPassword).resetWasForced)
  }

  @Test
  fun aForcedResetOnAPasswordSignInLandsOnTheSameStep() = runTest {
    // Setting a password through the Clerk dashboard forces a reset on the next sign-in: the password typed was right, and Clerk still wants a
    // replacement. Same step answers it, since the same call finishes it.
    val auth = FakeClerkAuth(passwordResult = AuthActionResult.NeedsNewPassword)
    val start = EmailAuthUiState(step = EmailAuthUiState.Step.Password, email = TEST_EMAIL, submitting = true)
    val next = reduceSignInCompletion(start, auth.signInWithPassword(TEST_EMAIL, "the old one"))
    assertEquals(EmailAuthUiState.Step.ResetPassword, next.step)
    assertNull(next.error)
  }

  @Test
  fun strategyStringsRoundTripToTheClerkNames() {
    // These strings are Clerk's wire spellings, matched against `supportedSecondFactors`. A typo here silently
    // drops a factor the account offers, so pin them rather than trusting the enum's own name.
    assertEquals(SecondFactor.EmailCode, SecondFactor.fromStrategy("email_code"))
    assertEquals(SecondFactor.PhoneCode, SecondFactor.fromStrategy("phone_code"))
    assertEquals(SecondFactor.Totp, SecondFactor.fromStrategy("totp"))
    assertEquals(SecondFactor.BackupCode, SecondFactor.fromStrategy("backup_code"))
    assertNull(SecondFactor.fromStrategy("passkey"))
    assertNull(SecondFactor.fromStrategy(null))
  }

  @Test
  fun onlySentFactorsNeedDelivering() {
    assertTrue(SecondFactor.EmailCode.needsSending)
    assertTrue(SecondFactor.PhoneCode.needsSending)
    // Already in the user's hands, so the UI offers no "resend" for these.
    assertFalse(SecondFactor.Totp.needsSending)
    assertFalse(SecondFactor.BackupCode.needsSending)
  }

  /**
   * The [ClerkAuth.signOut] contract, and [FakeClerkAuth]'s fidelity to it: a reported failure leaves the session [ClerkState.SignedIn]. That pairing
   * is what the callers read, since dropping the device's account link while still signed in as the outgoing account would gate that account straight
   * back into the app and re-link the device to it, a worse trap than the mismatch screen the user is leaving.
   *
   * The branch itself is not covered here, and cannot be: it lives in `ShellViewModel.signOut`, an AndroidViewModel method that needs the SDK. This
   * pins the state it reads, the same way [AuthGateTest.unlinkedRegistrationAdmitsADifferentAccount] pins the gate half.
   */
  @Test
  fun aFailedSignOutReportsAndKeepsTheSessionSignedIn() = runTest {
    val auth = FakeClerkAuth()
    auth.stateFlow.value = ClerkState.SignedIn(userId = "user_A", email = TEST_EMAIL, sessionId = "sess_1")
    auth.signOutResult = AuthActionResult.Error("Network unavailable")

    val result = auth.signOut()

    assertEquals(AuthActionResult.Error("Network unavailable"), result)
    assertFalse(auth.signedOut)
    // Still signed in, so the caller must leave the device's account link alone.
    assertTrue(auth.stateFlow.value is ClerkState.SignedIn)
  }

  @Test
  fun aSuccessfulSignOutEndsTheSession() = runTest {
    val auth = FakeClerkAuth()

    assertEquals(AuthActionResult.Success, auth.signOut())
    assertTrue(auth.signedOut)
  }

  private companion object {
    const val TEST_EMAIL = "user@example.test"
  }
}
