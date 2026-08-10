package ai.liquid.pipette

import android.content.Context
import android.util.Log
import com.clerk.api.Clerk
import com.clerk.api.ClerkConfigurationOptions
import com.clerk.api.auth.types.MfaType
import com.clerk.api.auth.types.VerificationType
import com.clerk.api.network.model.error.ClerkErrorResponse
import com.clerk.api.network.serialization.ClerkResult
import com.clerk.api.network.serialization.errorMessage
import com.clerk.api.signin.SignIn
import com.clerk.api.signin.resetPassword
import com.clerk.api.signin.sendMfaEmailCode
import com.clerk.api.signin.sendMfaPhoneCode
import com.clerk.api.signin.sendResetPasswordCode
import com.clerk.api.signin.verifyCode
import com.clerk.api.signin.verifyMfaCode
import com.clerk.api.signup.SignUp
import com.clerk.api.signup.sendCode
import com.clerk.api.signup.update
import com.clerk.api.signup.verifyCode as verifySignUpCode
import com.clerk.api.sso.OAuthProvider
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine

/**
 * The single entry point that touches the Clerk SDK at startup. Isolating it here (rather than referencing `Clerk`/[RealClerkAuth] from [PipetteApp])
 * means this class — and transitively `com.clerk.*` — is only class-loaded when [create] is actually invoked, which happens in the main process only.
 * The `:benchmark` process therefore never loads the Clerk classes, not just never initializes them.
 */
internal object ClerkBootstrap {
  fun create(context: Context): ClerkAuth {
    initialize(context.applicationContext)
    return RealClerkAuth()
  }

  /**
   * Configure the SDK, once at startup.
   *
   * enableDebugMode turns on Clerk's verbose `ClerkLog` output (SDK operations + Frontend-API calls), so auth failures surface their error codes in
   * logcat. It also makes the SDK install an OkHttp HttpLoggingInterceptor at Level.BODY, which logs credentials in the clear: the email-code OTP,
   * the session JWT, and the password from the gate's password step. So it's an explicit per-developer opt-in (clerk.debugLogging) rather than every
   * debug build, and it's hard-false for release. See app/build.gradle.kts.
   */
  fun initialize(context: Context) {
    Clerk.initialize(
      context,
      ClerkConfiguration.publishableKey,
      options = ClerkConfigurationOptions(enableDebugMode = BuildConfig.CLERK_DEBUG_LOGGING),
    )
  }
}

/**
 * The auth-gate's view of Clerk state. The sign-in calls themselves are driven by the flows below (email code, OAuth, password); this layer
 * *observes* the resulting state (and offers sign-out). Keeping the SDK behind this seam lets the gate reducer and all three flows be unit-tested
 * off-device with [FakeClerkAuth].
 */
sealed interface ClerkState {
  /** SDK not yet initialized (and no init error). */
  data object Loading : ClerkState

  /** Initialization failed (e.g. bad key, offline first run) — surface, don't hang. */
  data class InitError(val message: String) : ClerkState

  /** Initialized, no signed-in user. */
  data object SignedOut : ClerkState

  /** A Clerk user is signed in. */
  data class SignedIn(val userId: String, val email: String?, val sessionId: String?) : ClerkState
}

/**
 * A second factor the account can be challenged with, mapped from Clerk's `supportedSecondFactors` strategy strings. [EmailCode] and [PhoneCode] have
 * to be *sent* before they can be verified ([ClerkAuth.prepareSecondFactor]); [Totp] and [BackupCode] are read off something the user already holds,
 * so they go straight to [ClerkAuth.verifySecondFactor].
 */
enum class SecondFactor(val strategy: String) {
  EmailCode("email_code"),
  PhoneCode("phone_code"),
  Totp("totp"),
  BackupCode("backup_code");

  /** True when Clerk must deliver the code before the user can enter it. */
  val needsSending: Boolean
    get() = this == EmailCode || this == PhoneCode

  companion object {
    fun fromStrategy(strategy: String?): SecondFactor? = entries.firstOrNull { it.strategy == strategy }
  }
}

/**
 * Why Clerk is asking for a second code. Both arrive as a non-terminal [SignIn] carrying `supportedSecondFactors`, and both are answered with
 * `prepareSecondFactor` then the strategy's verify, so the difference is entirely in what the user is told.
 */
enum class SecondFactorReason {
  /** `needs_second_factor`: the account has two-step verification switched on, and this is that second step. */
  Mfa,

  /**
   * `needs_client_trust`: Clerk's Client Trust does not recognize this device, so it wants the sign-in confirmed out of band before issuing a
   * session. Nothing to do with the account's own settings, and the user may never have set up two-step verification, which is why calling it that
   * would be wrong. It fires only on password sign-ins, and not at all once an account has MFA on, because then [Mfa] covers the same ground.
   */
  DeviceVerification,
}

/**
 * Outcome of a one-shot auth action. [Success] from any of the completing calls ([ClerkAuth.verifyEmailCode], [ClerkAuth.signInWithOAuth],
 * [ClerkAuth.signInWithPassword], [ClerkAuth.verifySecondFactor], [ClerkAuth.createPassword], [ClerkAuth.resetPassword]) means the Clerk session is
 * now active.
 */
sealed interface AuthActionResult {
  data object Success : AuthActionResult

  data class Error(val message: String) : AuthActionResult

  /**
   * The first factor was accepted but Clerk wants a code before it will issue a session, so no session exists yet. [options] is the sign-in's
   * `supportedSecondFactors`, already filtered to the ones this app can drive; it is never empty, because a challenge we can't offer any answer to is
   * surfaced as an [Error] instead of stranding the user on a step with no inputs.
   *
   * [reason] says which of the two situations this is, because they read completely differently to the user even though Clerk resolves both through
   * the same pair of calls. It defaults to [SecondFactorReason.Mfa] as the older and more common case.
   */
  data class NeedsSecondFactor(val options: List<SecondFactor>, val reason: SecondFactorReason = SecondFactorReason.Mfa) : AuthActionResult

  /**
   * A registration verified its email but Clerk still wants a password before the account exists. Raised only when a password is the *whole* of what
   * is missing, so the step it sends the user to can actually finish the sign-up; anything else is an [Error] naming the fields.
   *
   * The instance requires a password at sign-up (`password.required` in its environment config), which the passwordless flow never collects, so every
   * new account stops here. Answered by [ClerkAuth.createPassword].
   */
  data object NeedsPassword : AuthActionResult

  /**
   * Clerk is waiting for a new password on an *existing* account, and no session exists yet. Answered by [ClerkAuth.resetPassword].
   *
   * Two calls raise it. [ClerkAuth.verifyPasswordResetCode] does so on the ordinary route: the emailed reset code was accepted, which is what earns
   * the right to set a new value. [ClerkAuth.signInWithPassword] does so when the instance has forced a reset on the account, which is what setting a
   * password through the Clerk dashboard's "Set password" action does to it: the password typed was correct, and Clerk still wants a replacement
   * before it will issue a session.
   *
   * Distinct from [NeedsPassword], which belongs to a registration that has no account behind it yet.
   */
  data object NeedsNewPassword : AuthActionResult
}

/**
 * A social sign-in provider enabled on the Clerk instance, surfaced to the auth-gate UI. Sourced from the Clerk *environment config* (dashboard), not
 * hard-coded — so which buttons appear is driven entirely by what the backend has turned on. [strategy] is Clerk's stable strategy identifier (e.g.
 * `oauth_google`) and is the token passed back to [ClerkAuth.signInWithOAuth]; the SDK type never crosses this seam.
 */
data class OAuthProviderInfo(val strategy: String, val name: String, val logoUrl: String?)

// One function per call the gate can make, so this grows with the routes Clerk offers rather than with any logic of its own. Same reasoning as the
// suppression on RealClerkAuth, which mirrors this surface.
@Suppress("TooManyFunctions")
interface ClerkAuth {
  /** Cold-collectable stream of the current [ClerkState]. */
  val state: Flow<ClerkState>

  /**
   * The social sign-in providers the Clerk instance has enabled (from its environment config), for the sign-in UI to render one button each. Empty
   * until the backend enables OAuth providers and until the environment has loaded — read it when the [ClerkState.SignedOut] gate renders.
   */
  val oauthProviders: List<OAuthProviderInfo>

  /**
   * Begin the passwordless email-code flow for [email]: signs in an existing account, or registers a new one when the address is unknown, then emails
   * a 6-digit verification code. Call [verifyEmailCode] next with the code the user types.
   */
  suspend fun sendEmailCode(email: String): AuthActionResult

  /** Verify the emailed [code]. On [AuthActionResult.Success] the Clerk session is active and [state] flips to [ClerkState.SignedIn]. */
  suspend fun verifyEmailCode(code: String): AuthActionResult

  /**
   * Run the redirect-based OAuth flow for [strategy] (a value from [oauthProviders], e.g. `oauth_google`): opens the provider's page in a Custom Tab
   * and suspends until the user returns. On [AuthActionResult.Success] the Clerk session is active and [state] flips to [ClerkState.SignedIn] (same
   * mechanism as [verifyEmailCode]); no separate verify step.
   */
  suspend fun signInWithOAuth(strategy: String): AuthActionResult

  /**
   * Sign an existing account in with [email] + [password], no emailed code. Single-shot like [signInWithOAuth]: on [AuthActionResult.Success] the
   * session is active and [state] flips to [ClerkState.SignedIn].
   *
   * This is the credential path the Play Console's App access form asks for. The passwordless flow can't be handed to a reviewer, because the code
   * lands in a mailbox we don't control. It never registers: an unknown address is an error here, not a sign-up (that's [sendEmailCode]'s job).
   */
  suspend fun signInWithPassword(email: String, password: String): AuthActionResult

  /**
   * Start a password reset for [email]: creates a sign-in and emails a 6-digit `reset_password_email_code`. Call [verifyPasswordResetCode] next with
   * the code the user types, then [resetPassword] with the new value.
   *
   * This is how an account with no password at all gets one, as well as the route for a forgotten one. Such an account does not offer the `password`
   * strategy, so [signInWithPassword] can only fail on it (`strategy_for_user_invalid`) no matter what is typed. Registrations through this app do
   * set a password ([createPassword]), so the accounts in that state came from somewhere else: a social sign-in, a dashboard invite, or a sign-up
   * from before the instance required one. It never registers: an address Clerk does not know is an error here, since there is no password to reset.
   */
  suspend fun sendPasswordResetCode(email: String): AuthActionResult

  /**
   * Verify the emailed reset [code]. [AuthActionResult.NeedsNewPassword] is the success case: the code cleared, and Clerk is now waiting for the new
   * password ([resetPassword]). No session exists yet.
   */
  suspend fun verifyPasswordResetCode(code: String): AuthActionResult

  /**
   * Set [password] on the account whose reset was authorized by [verifyPasswordResetCode] (or which Clerk asked to replace its password on sign-in).
   * On [AuthActionResult.Success] the session is active and [state] flips to [ClerkState.SignedIn]; an account with two-step verification answers
   * with [AuthActionResult.NeedsSecondFactor] first.
   *
   * Clerk owns the password policy here exactly as it does in [createPassword], so a rejection arrives carrying Clerk's own wording.
   */
  suspend fun resetPassword(password: String): AuthActionResult

  /**
   * Ask Clerk to deliver the second-factor code for [factor], after a sign-in came back [AuthActionResult.NeedsSecondFactor]. Only meaningful for
   * [SecondFactor.needsSending] factors; [SecondFactor.Totp] and [SecondFactor.BackupCode] need no delivery and return [AuthActionResult.Success]
   * without a round-trip, so the UI can call this unconditionally when the user picks a factor.
   */
  suspend fun prepareSecondFactor(factor: SecondFactor): AuthActionResult

  /**
   * Answer the second-factor challenge with [code]. On [AuthActionResult.Success] the session is active and [state] flips to [ClerkState.SignedIn],
   * exactly as with [verifyEmailCode]. Requires the sign-in captured by the call that returned [AuthActionResult.NeedsSecondFactor].
   */
  suspend fun verifySecondFactor(factor: SecondFactor, code: String): AuthActionResult

  /**
   * Set the password on a registration that verified its email but came back [AuthActionResult.NeedsPassword]. On [AuthActionResult.Success] the
   * account exists, the session is active, and [state] flips to [ClerkState.SignedIn].
   *
   * Clerk owns the policy, so a too-short, too-weak, or breached password comes back as an [AuthActionResult.Error] carrying Clerk's own wording
   * rather than something guessed at here.
   */
  suspend fun createPassword(password: String): AuthActionResult

  /** Abandon any in-progress email-code flow (e.g. the user goes back to edit their email). */
  fun resetEmailCode()

  /**
   * Sign the active session out of Clerk.
   *
   * The result is returned rather than swallowed because callers act on it destructively: the gate drops the device's account link on the strength of
   * the session having ended. An [AuthActionResult.Error] means the session may have outlived the request, and unlinking under a live session would
   * let that account straight back through the gate and re-link to it, so the gate path leaves the link alone and reports instead.
   *
   * There is no local-only fallback. `Clerk.reset()` would forget the session but also un-initializes the SDK, and its re-initialize needs the
   * network that just failed, so offline it trades a live session for a gate stuck on "Sign-in unavailable". iOS has the same gap from the other
   * side: `clearAllKeychainItems()` is documented as leaving in-memory state, so the session survives in `clerk.user` regardless.
   */
  suspend fun signOut(): AuthActionResult
}

/** What the UI gate should show. Derived from [ClerkState] + the locally-linked device registration by [reduceAuthGate]. */
sealed interface AuthGate {
  data object Loading : AuthGate

  data class InitError(val message: String) : AuthGate

  data object SignedOut : AuthGate

  data class Mismatch(val linkedEmail: String?, val currentEmail: String?) : AuthGate

  data object Ready : AuthGate
}

/**
 * Pure gate reducer (no Android / SDK deps, so it's unit-testable directly).
 * - [bypass] short-circuits to [AuthGate.Ready] (debug-only dev escape hatch).
 * - A signed-in user whose id differs from the registration's previously-linked `clerkUserId` is a [AuthGate.Mismatch]; otherwise (no registration
 *   yet, or the ids match) the gate is [AuthGate.Ready].
 */
internal fun reduceAuthGate(clerk: ClerkState, registration: RegistrationData?, bypass: Boolean): AuthGate {
  if (bypass) return AuthGate.Ready
  return when (clerk) {
    is ClerkState.Loading -> AuthGate.Loading
    is ClerkState.InitError -> AuthGate.InitError(clerk.message)
    is ClerkState.SignedOut -> AuthGate.SignedOut
    is ClerkState.SignedIn -> {
      val linkedUserId = registration?.clerkUserId
      if (linkedUserId != null && linkedUserId != clerk.userId) {
        AuthGate.Mismatch(linkedEmail = registration.clerkPrimaryEmail, currentEmail = clerk.email)
      } else {
        AuthGate.Ready
      }
    }
  }
}

/**
 * Real implementation backed by the Clerk SDK singleton. Lives only in the main process (see [PipetteApp]); the `:benchmark` process never constructs
 * it.
 */
@Suppress("TooManyFunctions") // mirrors the ClerkAuth interface surface, plus one completion tail per flow
class RealClerkAuth : ClerkAuth {
  override val state: Flow<ClerkState> =
    combine(Clerk.isInitialized, Clerk.initializationError, Clerk.userFlow, Clerk.sessionFlow) { initialized, error, user, session ->
      when {
        error != null -> ClerkState.InitError(error.message ?: "Sign-in initialization failed")
        !initialized -> ClerkState.Loading
        user == null -> ClerkState.SignedOut
        else -> ClerkState.SignedIn(userId = user.id, email = user.primaryEmailAddress?.emailAddress, sessionId = session?.id)
      }
    }

  override val oauthProviders: List<OAuthProviderInfo>
    // Clerk.socialProviders is the environment's social config. Keep only providers that are enabled, usable for authentication, and meant to be
    // shown to the user (notSelectable hides e.g. link-only providers). Non-OAuth strategies (fromStrategy == UNKNOWN) are dropped so every entry is
    // one signInWithOAuth can actually run.
    get() =
      Clerk.socialProviders.values
        .filter { it.enabled && it.authenticatable && !it.notSelectable && OAuthProvider.fromStrategy(it.strategy) != OAuthProvider.UNKNOWN }
        .map { OAuthProviderInfo(strategy = it.strategy, name = it.name, logoUrl = it.logoUrl) }

  // The in-progress attempt between sendEmailCode and verifyEmailCode. A SignIn (returning user) or a
  // SignUp (new email). Written/read only from the single-threaded viewModelScope caller, so a plain
  // @Volatile ref suffices for cross-dispatch visibility.
  @Volatile private var pending: Pending? = null

  private sealed interface Pending {
    data class ReturningUser(val signIn: SignIn) : Pending

    data class NewUser(val signUp: SignUp) : Pending

    /**
     * A sign-in that cleared its first factor and is parked on a code challenge: `needs_second_factor` when the account has two-step verification
     * switched on, or `needs_client_trust` when Clerk wants a device it has not seen confirmed. [reason] records which, and it rides along here
     * rather than being re-derived, because a delivery refreshes the sign-in and the copy is worded from the reason. Kept distinct from
     * [ReturningUser] so [verifyEmailCode] can't mistake it for a first-factor attempt and submit the MFA code to the wrong endpoint.
     */
    data class SecondFactorRequired(val signIn: SignIn, val reason: SecondFactorReason) : Pending

    /**
     * A sign-in being used to set a password rather than to present one: parked between the emailed reset code and the new value, and again between a
     * forced-reset sign-in and its replacement password. Kept distinct from [ReturningUser] for the same reason [SecondFactorRequired] is, so
     * [verifyEmailCode] cannot submit a reset code as a first-factor attempt.
     */
    data class PasswordReset(val signIn: SignIn) : Pending
  }

  override suspend fun sendEmailCode(email: String): AuthActionResult {
    // Drop any prior attempt up front: a failed send must not leave a stale `pending` that a later
    // verifyEmailCode could accidentally complete.
    pending = null
    // signInWithOtp both creates the sign-in and emails the code. An unknown address fails with
    // form_identifier_not_found — that's the signal to register instead of surfacing an error.
    return when (val result = Clerk.auth.signInWithOtp { this.email = email }) {
      is ClerkResult.Success -> {
        pending = Pending.ReturningUser(result.value)
        AuthActionResult.Success
      }
      is ClerkResult.Failure ->
        if (result.isIdentifierNotFound()) signUpEmailCode(email) else AuthActionResult.Error(result.loggedErrorMessage("email-code sign-in"))
    }
  }

  /** New-address path: create the sign-up, then send the email code as a separate step (sign-up doesn't auto-send). */
  private suspend fun signUpEmailCode(email: String): AuthActionResult {
    val created =
      when (
        val result =
          Clerk.auth.signUp {
            this.email = email
            // The instance has legal consent enabled, which makes `legal_accepted` a required sign-up field: without it Clerk returns a sign-up stuck
            // at MISSING_REQUIREMENTS and no account is ever created. It's set on the strength of the clickwrap notice the email step renders (see
            // LegalNotice in AuthGateScreen), so the two belong together: if that notice goes, this flag has nothing behind it.
            this.legalAccepted = true
          }
      ) {
        is ClerkResult.Success -> result.value
        is ClerkResult.Failure -> return AuthActionResult.Error(result.loggedErrorMessage("sign-up create"))
      }
    return when (val sent = created.sendCode { this.email = email }) {
      is ClerkResult.Success -> {
        pending = Pending.NewUser(sent.value)
        AuthActionResult.Success
      }
      is ClerkResult.Failure -> AuthActionResult.Error(sent.loggedErrorMessage("sign-up code send"))
    }
  }

  override suspend fun verifyEmailCode(code: String): AuthActionResult =
    when (val p = pending) {
      // A ClerkResult.Success from verifyCode only means the HTTP call succeeded — the returned object can
      // still be non-terminal (SignIn NEEDS_SECOND_FACTOR, SignUp MISSING_REQUIREMENTS), in which case no
      // session is created and Clerk.userFlow never flips. Only status == COMPLETE means we're actually in.
      // The emailed first-factor code can itself land on an MFA challenge, so this path routes through the same handler as the password path.
      is Pending.ReturningUser -> p.signIn.verifyCode(code).foldSignIn(::completeSignIn)
      is Pending.NewUser ->
        when (val verified = p.signUp.verifySignUpCode(code, VerificationType.EMAIL)) {
          is ClerkResult.Failure -> AuthActionResult.Error(verified.loggedErrorMessage("sign-up code verify"))
          is ClerkResult.Success -> completeSignUp(verified.value)
        }
      is Pending.SecondFactorRequired ->
        AuthActionResult.Error(
          when (p.reason) {
            SecondFactorReason.Mfa -> "Enter your two-step verification code."
            SecondFactorReason.DeviceVerification -> "Enter the code that confirms this device."
          }
        )
      // Only reachable if the UI's step and the parked attempt disagree, since the reset code has its own step and its own call.
      is Pending.PasswordReset -> AuthActionResult.Error("Enter the code that lets you set a password.")
      null -> AuthActionResult.Error("No verification in progress. Request a new code.")
    }

  /**
   * Shared tail for every call that returns a [SignIn]: complete it, park a code challenge for [verifySecondFactor], or surface the status.
   *
   * Every status other than COMPLETE used to collapse into one opaque sentence, which made a `needs_second_factor` account indistinguishable from
   * `needs_new_password` or `needs_client_trust` without attaching a debugger. Three of those statuses have a step behind them; the rest name their
   * status instead.
   *
   * `needs_client_trust` is one of those, and it took a device to find: a password sign-in from a device Clerk does not recognize lands here with the
   * password already `VERIFIED`, and it used to fall through to "needs another sign-in step that isn't available here" even though the account
   * offered `email_code` and the app can drive exactly that. Clerk resolves it through the same `prepareSecondFactor`/verify pair as MFA, so the only
   * thing it needs from this function is to be let through. What differs is the wording, which [SecondFactorReason] carries.
   *
   * `needs_new_password` is another, and handling it here rather than only on the reset route is deliberate: it is the status the reset code's verify
   * lands on, *and* what a correct password gets back when the instance has forced a reset on the account. One branch serves both, because Clerk
   * finishes both with the same `resetPassword` call.
   */
  private fun completeSignIn(signIn: SignIn): AuthActionResult {
    val reason =
      when (signIn.status) {
        SignIn.Status.NEEDS_SECOND_FACTOR -> SecondFactorReason.Mfa
        SignIn.Status.NEEDS_CLIENT_TRUST -> SecondFactorReason.DeviceVerification
        else -> null
      }
    // Read off the sign-in for both statuses. Clerk populates `supportedSecondFactors` for a client-trust challenge too, with the delivery strategies
    // the instance has enabled for it, so there is no separate list to consult.
    val options =
      if (reason == null) emptyList() else signIn.supportedSecondFactors.orEmpty().mapNotNull { SecondFactor.fromStrategy(it.strategy) }.distinct()
    return when {
      signIn.status == SignIn.Status.COMPLETE -> {
        // Session is active; the SignedIn flip drives the gate to Ready.
        pending = null
        AuthActionResult.Success
      }
      // Park the sign-in: resetPassword needs this exact attempt, and it is the only thing that carries the authorization the code (or the accepted
      // password) just earned.
      signIn.status == SignIn.Status.NEEDS_NEW_PASSWORD -> {
        pending = Pending.PasswordReset(signIn)
        AuthActionResult.NeedsNewPassword
      }
      reason != null && options.isNotEmpty() -> {
        pending = Pending.SecondFactorRequired(signIn, reason)
        AuthActionResult.NeedsSecondFactor(options, reason)
      }
      else -> {
        pending = null
        AuthActionResult.Error(incompleteSignInMessage(signIn))
      }
    }
  }

  override suspend fun signInWithOAuth(strategy: String): AuthActionResult {
    // Any in-progress email-code attempt is abandoned when the user chooses a social provider instead.
    pending = null
    val provider = OAuthProvider.fromStrategy(strategy)
    if (provider == OAuthProvider.UNKNOWN) return AuthActionResult.Error("Unsupported sign-in provider.")
    // signInWithOAuth drives the full Custom Tab redirect flow and returns the resulting sign-in/sign-up. As with verifyEmailCode, a ClerkResult
    // success only means the HTTP round-trip worked — a session exists (and userFlow flips) only when the sign-in/sign-up reached COMPLETE. Clerk
    // handles the sign-in↔sign-up transfer internally, so exactly one of the two is set.
    return when (val result = Clerk.auth.signInWithOAuth(provider)) {
      is ClerkResult.Failure -> AuthActionResult.Error(result.loggedErrorMessage("OAuth sign-in ($strategy)"))
      is ClerkResult.Success -> {
        val signIn = result.value.signIn
        when {
          result.value.signUp?.status == SignUp.Status.COMPLETE -> AuthActionResult.Success
          // Clerk applies second-factor requirements to social sign-ins too, so this shares the tail with the password path rather than treating
          // anything short of COMPLETE as a dead end: an MFA-enabled account gets the second-factor step instead of an error.
          signIn != null -> completeSignIn(signIn)
          else -> AuthActionResult.Error(OAUTH_INCOMPLETE)
        }
      }
    }
  }

  override suspend fun signInWithPassword(email: String, password: String): AuthActionResult {
    // Drop any email-code attempt still pending from an earlier step, so it can't be completed by a later verifyEmailCode.
    pending = null
    // signInWithPassword completes the sign-in in one call, so there's no `pending` to record. As with verifyEmailCode, a transport-level success can
    // still be non-terminal, and only COMPLETE means a session exists, hence the same `finish` guard rather than trusting ClerkResult.Success. Two
    // statuses land here: NEEDS_SECOND_FACTOR when the account has TOTP/backup codes on, which this instance allows, and NEEDS_CLIENT_TRUST, which is
    // the one an ordinary account hits, since it fires on a password sign-in from a device Clerk does not recognize.
    val result =
      Clerk.auth.signInWithPassword {
        this.identifier = email
        this.password = password
      }
    return when (result) {
      is ClerkResult.Success -> completeSignIn(result.value)
      is ClerkResult.Failure -> {
        // Logged first either way, so the codes reach logcat whichever message the user ends up seeing.
        val message = result.loggedErrorMessage("password sign-in")
        // `strategy_for_user_invalid` means the account has no password at all, which Clerk words as "The verification strategy is not valid for this
        // account". That reads as an account defect, and it sent a real user looking for one; what it actually means is that nothing they could have
        // typed would work, and that the reset step is the way in. Reached by any account that never had a password set on it (see
        // sendPasswordResetCode), and reported from the field, so it is worth its own sentence rather than Clerk's.
        if (result.hasErrorCode("strategy_for_user_invalid")) AuthActionResult.Error(NO_PASSWORD_ON_ACCOUNT) else AuthActionResult.Error(message)
      }
    }
  }

  override suspend fun sendPasswordResetCode(email: String): AuthActionResult {
    // Held for the one failure that is worth undoing. The resend link makes this call reachable from the code step, where a reset is already parked
    // and its code is sitting in the user's inbox, and a resend that never left the device must not be what kills that code.
    val liveReset = pending as? Pending.PasswordReset
    // Dropped up front all the same, so a failure that leaves nothing usable cannot leave a stale attempt a later verify could complete.
    pending = null
    // Unlike signInWithOtp, this takes two calls, and the first only creates the attempt: no strategy, no code, nothing on it but an identifier. The
    // builder is where `email` comes from: it takes an email, a phone, or a username and collapses whichever is set into the one identifier Clerk
    // accepts, so setting `email` is how an address gets there. The send below is what picks reset_password_email_code and mails the code.
    val signIn =
      when (val created = Clerk.auth.signIn { this.email = email }) {
        is ClerkResult.Success -> created.value
        // Nothing was created, so the old attempt is still the client's live sign-in and its code is still answerable. This is the offline resend,
        // and the only failure here that can be undone.
        is ClerkResult.Failure -> {
          pending = liveReset
          return AuthActionResult.Error(created.loggedErrorMessage("password-reset sign-in create"))
        }
      }
    // No runCatching here, unlike prepareSecondFactor: this helper reads the address id off the attempt's own `email_code` factor and falls back to
    // an empty one rather than throwing, so a factor it cannot address comes back as a Clerk failure instead of taking the process down.
    return when (val sent = signIn.sendResetPasswordCode { this.email = email }) {
      is ClerkResult.Success -> {
        pending = Pending.PasswordReset(sent.value)
        AuthActionResult.Success
      }
      // Deliberately *not* restored here, unlike above. The create succeeded, and a Clerk client holds one sign-in, so the attempt this replaced is
      // no longer the one the client is on: keeping it would answer the next verify with Clerk's own not-found wording instead of this file's, and
      // the code it belongs to is dead either way. That leaves nothing parked while the user is still looking at a code step, so the message has to
      // say so. Clerk's reason comes first, since it is the half that explains why (a rate limit, or the connection dropping between the two calls);
      // the sentence after it is the half only this layer knows, and without it the step would sit there inviting a code that cannot be answered.
      is ClerkResult.Failure -> {
        val reason = sent.loggedErrorMessage("password-reset code send").asSentence()
        AuthActionResult.Error(if (reason.isEmpty()) RESET_MUST_RESTART else "$reason $RESET_MUST_RESTART")
      }
    }
  }

  override suspend fun verifyPasswordResetCode(code: String): AuthActionResult {
    val signIn = (pending as? Pending.PasswordReset)?.signIn ?: return AuthActionResult.Error(NO_PASSWORD_RESET_IN_PROGRESS)
    // The same helper the email-code path uses: it dispatches on the attempt's prepared strategy, which the send above left as
    // reset_password_email_code, so the code goes out as a reset attempt rather than a sign-in one. A cleared code lands on NEEDS_NEW_PASSWORD, which
    // completeSignIn turns into the step that collects the new value.
    return signIn.verifyCode(code).foldSignIn(::completeSignIn)
  }

  override suspend fun resetPassword(password: String): AuthActionResult {
    val signIn = (pending as? Pending.PasswordReset)?.signIn ?: return AuthActionResult.Error(NO_PASSWORD_RESET_IN_PROGRESS)
    // signOutOfOtherSessions stays false. The flow's ordinary cause is an account that never had a password, not a compromised one, and the user
    // reaching it is usually signed in on the web; ending those sessions is not something this screen was asked to do.
    return signIn.resetPassword(newPassword = password, signOutOfOtherSessions = false).foldSignIn(::completeSignIn)
  }

  override suspend fun prepareSecondFactor(factor: SecondFactor): AuthActionResult {
    val challenge = pending as? Pending.SecondFactorRequired
    val signIn = challenge?.signIn
    return when {
      signIn == null -> AuthActionResult.Error(NO_SECOND_FACTOR_IN_PROGRESS)
      // TOTP and backup codes are already in the user's hands, so there is nothing to request.
      !factor.needsSending -> AuthActionResult.Success
      // Not every failure here is a ClerkResult. Both send helpers read the address to deliver to out of the offered
      // factor and throw IllegalStateException when it carries none, before any request goes out. The strategy is all
      // fromStrategy filters on, so a factor with no `emailAddressId`/`phoneNumberId` is still offered and still
      // auto-delivered by the gate. Uncaught, that reaches the launch behind runAuthAction and takes the process with
      // it, on the fresh-install password sign-in that Client Trust now routes through here.
      else ->
        runCatching { if (factor == SecondFactor.EmailCode) signIn.sendMfaEmailCode() else signIn.sendMfaPhoneCode() }
          .fold(
            onSuccess = { sent ->
              when (sent) {
                // Clerk returns the updated sign-in; keep it, since it carries the verification the attempt is now
                // waiting on. The reason is carried over unchanged: a delivery does not change why the code was asked
                // for, and re-deriving it from the refreshed status would be fragile.
                is ClerkResult.Success -> {
                  pending = Pending.SecondFactorRequired(sent.value, challenge.reason)
                  AuthActionResult.Success
                }
                is ClerkResult.Failure -> AuthActionResult.Error(sent.loggedErrorMessage("second-factor code send ($factor)"))
              }
            },
            onFailure = { failure ->
              if (failure is CancellationException) throw failure
              Log.w(TAG, "second-factor code send ($factor) could not be attempted", failure)
              AuthActionResult.Error(UNDELIVERABLE_SECOND_FACTOR)
            },
          )
    }
  }

  override suspend fun verifySecondFactor(factor: SecondFactor, code: String): AuthActionResult {
    val signIn = (pending as? Pending.SecondFactorRequired)?.signIn ?: return AuthActionResult.Error(NO_SECOND_FACTOR_IN_PROGRESS)
    val mfaType =
      when (factor) {
        SecondFactor.EmailCode -> MfaType.EMAIL_CODE
        SecondFactor.PhoneCode -> MfaType.PHONE_CODE
        SecondFactor.Totp -> MfaType.TOTP
        SecondFactor.BackupCode -> MfaType.BACKUP_CODE
      }
    // A second-factor verify can only complete or fail: there is no third factor, so a non-COMPLETE success here is a genuine error rather than
    // another challenge to park.
    return signIn.verifyMfaCode(code, mfaType).foldSignIn(::completeSignIn)
  }

  /**
   * Shared tail for every call that returns a [SignUp], mirroring [completeSignIn]: complete it, park it for [createPassword], or name what is
   * missing. The password step is offered only when a password is the entire shortfall, since it is the only field the UI can supply.
   */
  private fun completeSignUp(signUp: SignUp): AuthActionResult =
    when {
      signUp.status == SignUp.Status.COMPLETE -> {
        pending = null
        AuthActionResult.Success
      }
      signUp.missingFields.orEmpty() == listOf(PASSWORD_FIELD) -> {
        pending = Pending.NewUser(signUp)
        AuthActionResult.NeedsPassword
      }
      else -> {
        pending = null
        AuthActionResult.Error(incompleteSignUpMessage(signUp))
      }
    }

  override suspend fun createPassword(password: String): AuthActionResult {
    val signUp = (pending as? Pending.NewUser)?.signUp ?: return AuthActionResult.Error(NO_SIGN_UP_IN_PROGRESS)
    // Clerk validates length, zxcvbn strength, and the breach corpus server-side, so a rejection arrives as a Failure with its own wording.
    return when (val updated = signUp.update { this.password = password }) {
      is ClerkResult.Failure -> AuthActionResult.Error(updated.loggedErrorMessage("sign-up password"))
      is ClerkResult.Success -> completeSignUp(updated.value)
    }
  }

  override fun resetEmailCode() {
    pending = null
  }

  override suspend fun signOut(): AuthActionResult =
    when (val result = Clerk.auth.signOut()) {
      is ClerkResult.Success -> AuthActionResult.Success
      is ClerkResult.Failure -> AuthActionResult.Error(result.loggedErrorMessage("sign-out"))
    }
}

/**
 * Unwrap a call that returns a [SignIn], surfacing a transport failure as an error and handing a success to [onSignIn].
 *
 * Top-level rather than a member so the class keeps one entry point for the sign-in tail: the OAuth path already holds a [SignIn] and needs to reach
 * that tail without a second overload to unwrap for it.
 */
private inline fun ClerkResult<SignIn, ClerkErrorResponse>.foldSignIn(onSignIn: (SignIn) -> AuthActionResult): AuthActionResult =
  when (this) {
    is ClerkResult.Failure -> AuthActionResult.Error(loggedErrorMessage("sign-in"))
    is ClerkResult.Success -> onSignIn(value)
  }

private const val TAG = "ClerkAuth"
private const val SIGN_IN_INCOMPLETE = "This account needs another sign-in step that isn't available here."
/**
 * Deliberately does not say "two-step verification": the same step answers a Client Trust device check, and by the time this fires the parked
 * challenge is gone, so there is no [SecondFactorReason] left to word it from. Naming the step rather than the reason is right for both.
 */
private const val NO_SECOND_FACTOR_IN_PROGRESS = "That sign-in step is no longer in progress. Sign in again."

/** The offered factor named a delivery the SDK could not address (see [RealClerkAuth.prepareSecondFactor]); another one, or another sign-in, may. */
private const val UNDELIVERABLE_SECOND_FACTOR = "That verification code could not be sent. Try another method, or sign in again."
private const val NO_SIGN_UP_IN_PROGRESS = "No registration in progress. Start again."
/**
 * Says to go back rather than naming a control, because it surfaces on two steps that offer different ones: the code step, which has a resend, and
 * the new-password step, whose only exit is back to the address. Going back reaches the reset from either.
 */
private const val NO_PASSWORD_RESET_IN_PROGRESS = "That password reset is no longer in progress. Go back and start it again."

/**
 * Appended when a send leaves the reset dead: the attempt it would have belonged to is gone, so whatever step the user is on, the way forward is to
 * ask for the reset again rather than to wait for a code. Names no control, for the same reason [NO_PASSWORD_RESET_IN_PROGRESS] does not.
 */
private const val RESET_MUST_RESTART = "Start the reset again."

/**
 * Ends [this] with a full stop unless it already carries one, so a sentence of ours can be appended to prose of Clerk's.
 *
 * Clerk's text is not reliably a sentence: [errorMessage] prefers the long form but falls back to the short `message` field, which is a fragment
 * ("Too many requests"). Concatenating onto that would read as one run-on line, and this is the only place in this file where third-party prose is
 * joined to ours rather than passed through whole.
 */
private fun String.asSentence(): String {
  val trimmed = trimEnd()
  return if (trimmed.isEmpty() || trimmed.last() in ".!?") trimmed else "$trimmed."
}

/**
 * Names the reset route wherever the user meets it: the link on the password step, the title of the step that finishes it, and the message
 * [NO_PASSWORD_ON_ACCOUNT] answers a passwordless account with. One string for all three, because a message that quotes a label is wrong the moment
 * the label changes, and nothing else would catch that.
 *
 * Not "Set a new password". The flow's primary case is an account that has never had one, and the message says exactly that a line above the link, so
 * "new" would contradict the sentence pointing at it. Under-describing the forgotten-password case is the cheaper of the two faults.
 */
internal const val SET_PASSWORD_ACTION = "Set a password"

/**
 * Answers the one failure the password step has that is not about the password: the account has none. Names the situation and the way out of it,
 * since Clerk's own wording for `strategy_for_user_invalid` describes a strategy the user never chose and cannot see.
 *
 * [SET_PASSWORD_ACTION] is the only control named: the step's other exit is labelled "Use a different email", which is not what someone who typed the
 * right address wants to be told to press.
 */
private const val NO_PASSWORD_ON_ACCOUNT = "This account has no password yet. Tap \"$SET_PASSWORD_ACTION\" to choose one."

/** Clerk's wire name for the password field in `missing_fields`. */
private const val PASSWORD_FIELD = "password"

/**
 * Names the status on a non-terminal sign-in the UI has no step for, so the message identifies the state instead of reporting a bare dead end. The
 * status is a short enum the user can quote back; the offered strategies are raw wire identifiers, so they go to the log rather than into the copy.
 */
private fun incompleteSignInMessage(signIn: SignIn): String {
  // Logged for every non-terminal status, not just the unsupported-MFA one. The status alone says which step Clerk wants, but not why the submitted
  // factor fell short, and the offered strategies are what say whether the account would accept something the app can actually drive. The identifier
  // is left out on purpose: it is the user's email address.
  Log.w(
    TAG,
    "Sign-in stuck at ${signIn.status}: firstFactor=${signIn.firstFactorVerification?.status}" +
      " code=${signIn.firstFactorVerification?.error?.code}" +
      " supportedFirst=${signIn.supportedFirstFactors.orEmpty().mapNotNull { it.strategy }}" +
      " supportedSecond=${signIn.supportedSecondFactors.orEmpty().mapNotNull { it.strategy }}",
  )
  return when (signIn.status) {
    // Clerk wants a factor none of the four the UI can drive, so there is no step to send the user to.
    SignIn.Status.NEEDS_SECOND_FACTOR -> "This account's two-step verification isn't supported here."
    // Same shape, different cause: the device needs confirming, but by a delivery method the app can't drive (an email *link*, say, which is a
    // Client Trust option and is not a code anyone can type). Naming the device is what stops this reading as an account problem.
    SignIn.Status.NEEDS_CLIENT_TRUST -> "This device needs to be confirmed, and the way this account does that isn't supported here."
    else -> "$SIGN_IN_INCOMPLETE (${signIn.status})"
  }
}

private const val SIGN_UP_INCOMPLETE = "Your account needs more details to finish signing up."

/**
 * Same idea as [incompleteSignInMessage] for the registration path: a verified email that still isn't COMPLETE means the instance requires a field
 * the app never collects, and `missing_fields` is the only thing that says which. Naming it turns "more details" into something actionable, since the
 * fix is usually a Clerk dashboard setting rather than anything the user can do.
 */
private fun incompleteSignUpMessage(signUp: SignUp): String {
  val missing = signUp.missingFields.orEmpty()
  Log.w(TAG, "Sign-up stuck at ${signUp.status}: missing=$missing unverified=${signUp.unverifiedFields.orEmpty()}")
  return "$SIGN_UP_INCOMPLETE (${signUp.status}${if (missing.isEmpty()) "" else ": ${missing.joinToString()}"})"
}

private const val OAUTH_INCOMPLETE = "That sign-in needs another step that isn't available here."

/**
 * True when Clerk's parsed error body names [code]. The codes are the machine-readable half of a failure and the only part worth branching on; the
 * prose alongside them is written for the user and is free to change.
 */
private fun ClerkResult.Failure<ClerkErrorResponse>.hasErrorCode(code: String): Boolean = error?.errors?.any { it.code == code } == true

/** True when a sign-in failed only because the email isn't a known account (so we should register it instead). */
private fun ClerkResult.Failure<ClerkErrorResponse>.isIdentifierNotFound(): Boolean = hasErrorCode("form_identifier_not_found")

/**
 * [errorMessage], with Clerk's machine-readable codes copied to the log on the way past. The message itself is prose written for the person reading
 * the gate, which makes it the wrong thing to search for later; the codes (`form_password_incorrect`, `strategy_for_user_invalid`, and the like) are
 * what name the failure, and nothing else records them. [operation] says which call failed, since several of them surface the same text.
 *
 * Codes are not credentials, so this stays on in release, where Clerk's own verbose logging cannot: that one installs an OkHttp body logger and would
 * print the emailed code, the session JWT, and the password (see [ClerkBootstrap.initialize]).
 *
 * A [Failure] Clerk sent no parsed error body with gets a sentence of our own, because [errorMessage] falls back to its literal "Error occurred with
 * unknown message." there, which names nothing the user can act on. The two shapes that produce it are worth telling apart: no HTTP status at all
 * means the request never left, which is what being offline looks like from here, while a status with a body we could not read means Clerk answered
 * and the fault is its end, not the connection.
 */
private fun ClerkResult.Failure<ClerkErrorResponse>.loggedErrorMessage(operation: String): String {
  Log.w(
    TAG,
    "$operation failed: type=$errorType status=$code codes=${error?.errors.orEmpty().map { it.code }} cause=${throwable?.javaClass?.simpleName}",
  )
  return when {
    error != null -> errorMessage
    code == null -> NO_RESPONSE
    else -> "Clerk returned an error ($code). Try again in a moment."
  }
}

private const val NO_RESPONSE = "Could not reach Clerk. Check your connection and try again."
