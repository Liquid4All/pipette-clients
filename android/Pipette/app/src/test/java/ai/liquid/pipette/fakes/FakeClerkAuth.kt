package ai.liquid.pipette.fakes

import ai.liquid.pipette.AuthActionResult
import ai.liquid.pipette.ClerkAuth
import ai.liquid.pipette.ClerkState
import ai.liquid.pipette.OAuthProviderInfo
import ai.liquid.pipette.SecondFactor
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow

/**
 * Off-device [ClerkAuth] double: [sendEmailCode] / [verifyEmailCode] / [signInWithOAuth] / [signInWithPassword] / [prepareSecondFactor] /
 * [verifySecondFactor] / [createPassword] / [sendPasswordResetCode] / [verifyPasswordResetCode] / [resetPassword] return the scripted results and
 * record their arguments, so the auth flows can be exercised without the Clerk SDK. [state] is a mutable flow the test can drive; [oauthProviders] is
 * a scripted list.
 */
class FakeClerkAuth(
  private val sendResult: AuthActionResult = AuthActionResult.Success,
  private val verifyResult: AuthActionResult = AuthActionResult.Success,
  private val oauthResult: AuthActionResult = AuthActionResult.Success,
  private val passwordResult: AuthActionResult = AuthActionResult.Success,
  private val prepareSecondFactorResult: AuthActionResult = AuthActionResult.Success,
  private val secondFactorResult: AuthActionResult = AuthActionResult.Success,
  private val createPasswordResult: AuthActionResult = AuthActionResult.Success,
  override val oauthProviders: List<OAuthProviderInfo> = emptyList(),
) : ClerkAuth {
  val stateFlow = MutableStateFlow<ClerkState>(ClerkState.SignedOut)
  override val state: Flow<ClerkState> = stateFlow

  val sentEmails = mutableListOf<String>()
  val verifiedCodes = mutableListOf<String>()
  val oauthStrategies = mutableListOf<String>()

  /** Every (email, password) pair [signInWithPassword] was called with, in order. */
  val passwordAttempts = mutableListOf<Pair<String, String>>()

  /** Factors [prepareSecondFactor] was asked to deliver, in order. A repeat entry is a resend. */
  val preparedSecondFactors = mutableListOf<SecondFactor>()

  /** Every (factor, code) pair [verifySecondFactor] was called with, in order. */
  val secondFactorAttempts = mutableListOf<Pair<SecondFactor, String>>()
  /** Passwords [createPassword] was called with, in order. */
  val createdPasswords = mutableListOf<String>()

  /** Addresses [sendPasswordResetCode] was called with, in order. */
  val resetCodeEmails = mutableListOf<String>()

  /** Codes [verifyPasswordResetCode] was called with, in order. Kept apart from [verifiedCodes], since the two answer different Clerk strategies. */
  val verifiedResetCodes = mutableListOf<String>()

  /** Passwords [resetPassword] was called with, in order. */
  val resetPasswords = mutableListOf<String>()
  var resetCount = 0
    private set

  var signedOut = false
    private set

  override suspend fun sendEmailCode(email: String): AuthActionResult {
    sentEmails += email
    return sendResult
  }

  override suspend fun verifyEmailCode(code: String): AuthActionResult {
    verifiedCodes += code
    return verifyResult
  }

  override suspend fun signInWithOAuth(strategy: String): AuthActionResult {
    oauthStrategies += strategy
    return oauthResult
  }

  override suspend fun signInWithPassword(email: String, password: String): AuthActionResult {
    passwordAttempts += email to password
    return passwordResult
  }

  override suspend fun prepareSecondFactor(factor: SecondFactor): AuthActionResult {
    preparedSecondFactors += factor
    return prepareSecondFactorResult
  }

  override suspend fun verifySecondFactor(factor: SecondFactor, code: String): AuthActionResult {
    secondFactorAttempts += factor to code
    return secondFactorResult
  }

  override suspend fun createPassword(password: String): AuthActionResult {
    createdPasswords += password
    return createPasswordResult
  }

  // The reset trio is scripted through properties rather than three more constructor parameters, following signOutResult below. Adding them to the
  // constructor put it past detekt's parameter limit, and the next flow would have had to fight the same limit again.

  /** Result [sendPasswordResetCode] returns. */
  var sendResetCodeResult: AuthActionResult = AuthActionResult.Success

  override suspend fun sendPasswordResetCode(email: String): AuthActionResult {
    resetCodeEmails += email
    return sendResetCodeResult
  }

  /**
   * Result [verifyPasswordResetCode] returns. [AuthActionResult.NeedsNewPassword] by default rather than Success, because that is what the real call
   * answers a *correct* reset code with: the code buys the right to set a password, never a session.
   */
  var verifyResetCodeResult: AuthActionResult = AuthActionResult.NeedsNewPassword

  override suspend fun verifyPasswordResetCode(code: String): AuthActionResult {
    verifiedResetCodes += code
    return verifyResetCodeResult
  }

  /** Result [resetPassword] returns. */
  var resetPasswordResult: AuthActionResult = AuthActionResult.Success

  override suspend fun resetPassword(password: String): AuthActionResult {
    resetPasswords += password
    return resetPasswordResult
  }

  override fun resetEmailCode() {
    resetCount++
  }

  /**
   * Set to make [signOut] report a failure, as an offline or refused sign-out does. The session then stays [ClerkState.SignedIn], which is the state
   * the callers guard against: they must not unlink the device while it is still signed in as the outgoing account.
   */
  var signOutResult: AuthActionResult = AuthActionResult.Success

  override suspend fun signOut(): AuthActionResult {
    if (signOutResult is AuthActionResult.Error) return signOutResult
    signedOut = true
    stateFlow.value = ClerkState.SignedOut
    return signOutResult
  }
}
