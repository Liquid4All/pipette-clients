// Cross-cutting hub: intentionally many small functions (TooManyFunctions) + time/percent literals (MagicNumber, ReturnCount).
@file:Suppress("MagicNumber", "TooManyFunctions", "ReturnCount")

package ai.liquid.pipette.compose.shell

import ai.liquid.pipette.AuthActionResult
import ai.liquid.pipette.AuthGate
import ai.liquid.pipette.BenchmarkCatalog
import ai.liquid.pipette.BenchmarkSync
import ai.liquid.pipette.BuildConfig
import ai.liquid.pipette.ClerkState
import ai.liquid.pipette.DateFormats
import ai.liquid.pipette.OAuthProviderInfo
import ai.liquid.pipette.PipetteApp
import ai.liquid.pipette.RegistrationData
import ai.liquid.pipette.RunnerState
import ai.liquid.pipette.SecondFactor
import ai.liquid.pipette.SecondFactorReason
import ai.liquid.pipette.SetupSettings
import ai.liquid.pipette.Tab
import ai.liquid.pipette.compose.PocketUi
import ai.liquid.pipette.compose.RunProgress
import ai.liquid.pipette.plural
import ai.liquid.pipette.reduceAuthGate
import ai.liquid.pipette.thermalAccentKind
import ai.liquid.pipette.thermalDescription
import ai.liquid.pipette.thermalHeadroomLabel
import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Cross-cutting app chrome the Compose Scaffold renders: gate, bottom nav, pocket overlay. */
data class ShellUiState(
  val authGate: AuthGate = AuthGate.Loading,
  val runner: RunnerState = RunnerState(),
  val selectedTab: Tab = Tab.JOBS,
  val pocket: PocketUi? = null,
  val isDebug: Boolean = BuildConfig.DEBUG,
  val isClerkAvailable: Boolean = false,
  val clerkGateBypass: Boolean = false,
  /** Social sign-in providers the Clerk instance has enabled — the sign-in gate renders one button each. Empty until the backend enables OAuth. */
  val oauthProviders: List<OAuthProviderInfo> = emptyList(),
  /** True when the device isn't registered yet — iOS shows the full-screen Setup gate before the tabbed app. */
  val needsRegistration: Boolean = false,
)

/** State of the custom sign-in flow (owned by [ShellViewModel], rendered by AuthGateScreen). */
data class EmailAuthUiState(
  val step: Step = Step.Email,
  /** The address being signed in: the one the code was sent to on the code step, the one the password belongs to on the password step. */
  val email: String = "",
  /**
   * A request is in flight: disables the submit button, shows a spinner, and guards double-submit. [ShellViewModel.runAuthAction] also reads it as
   * the marker for "this attempt still owns the state", so a reset that lowers it deliberately abandons an outstanding request's outcome.
   */
  val submitting: Boolean = false,
  /** Last failure surfaced to the user (null when clear). */
  val error: String? = null,
  /**
   * The second factors the account offers, from [AuthActionResult.NeedsSecondFactor]. Empty off the [Step.SecondFactor] step. More than one entry
   * means the step renders a chooser before the code field.
   */
  val secondFactorOptions: List<SecondFactor> = emptyList(),
  /** The factor being answered. Null until one is picked, which is immediate when [secondFactorOptions] holds exactly one. */
  val selectedSecondFactor: SecondFactor? = null,
  /**
   * Why the [Step.SecondFactor] step is being shown, which is the whole of what the step's wording keys on. Meaningless off that step, and left at
   * [SecondFactorReason.Mfa] there rather than made nullable, since the step's copy needs an answer either way.
   */
  val secondFactorReason: SecondFactorReason = SecondFactorReason.Mfa,
  /**
   * True when [Step.ResetPassword] was reached because Clerk demanded a replacement password, rather than because the user asked to set one. Same job
   * as [secondFactorReason] on its step, and for the same reason: the step is shared by routes that read completely differently.
   *
   * A user who has just answered a reset code is expecting to choose a password and needs no explanation. Anyone else arrived because Clerk answered
   * `needs_new_password`, and is not expecting this screen at all: after a correct password, after an emailed code, or after a social sign-in. Those
   * three need to be told why, which is all this distinguishes. Meaningless off that step; false there is the commoner case.
   */
  val resetWasForced: Boolean = false,
) {
  enum class Step {
    Email,
    Code,

    /** Credential sign-in for an existing account: [email] plus a password, no emailed code. Opt-in from [Email]. */
    Password,

    /**
     * A code challenge after a first factor was accepted, from either `needs_second_factor` (the account has two-step verification on) or
     * `needs_client_trust` (Clerk does not recognize this device). One step for both, because they are answered identically;
     * [EmailAuthUiState.secondFactorReason] is what makes it say which.
     */
    SecondFactor,

    /**
     * Password creation for a *new* account whose email is already verified. The instance requires a password at sign-up, so this is the last step of
     * every registration, not an optional extra. Distinct from [Password], which signs an existing account in.
     */
    CreatePassword,

    /**
     * The emailed code that authorizes a password reset, from the link on [Password]. Separate from [Code] because the code answers a different
     * strategy (`reset_password_email_code`) and leads to [ResetPassword] rather than to a session.
     */
    ResetCode,

    /**
     * Set a new password on an existing account: reached once the reset code has cleared, and also when a correct password meets a reset the instance
     * has forced on the account. Distinct from [CreatePassword] again: there is an account behind this one, and Clerk finishes it with a reset rather
     * than by completing a sign-up. Named for the call that answers it ([ShellViewModel.submitResetPassword]), which is what keeps it apart from
     * [CreatePassword]'s [ShellViewModel.submitNewPassword].
     */
    ResetPassword,
  }
}

/**
 * Pure transition after [ClerkAuth.sendEmailCode] (no Android/SDK deps, unit-testable like [reduceAuthGate]): advance to the Code step on success,
 * else clear the spinner and surface the error while staying on the Email step.
 */
internal fun reduceSendCode(current: EmailAuthUiState, result: AuthActionResult): EmailAuthUiState =
  reduceSend(current, result, EmailAuthUiState.Step.Code)

/**
 * Pure transition after [ClerkAuth.sendPasswordResetCode], the reset counterpart to [reduceSendCode]: advance to the code step that authorizes a
 * reset, or stay put and surface the failure. Staying put is what matters on this one, because it has two entry points: the password step, where the
 * reset is started, and the code step itself, where the resend link lives. Either way the user keeps the address and the way back that step already
 * offers.
 */
internal fun reduceSendResetCode(current: EmailAuthUiState, result: AuthActionResult): EmailAuthUiState =
  reduceSend(current, result, EmailAuthUiState.Step.ResetCode)

/**
 * The shape both sends share: a success advances to [codeStep], and a failure stays put with its message. They differ in nothing else, so they share
 * this rather than each carrying its own copy of every outcome, which is what would let one drift when a new [AuthActionResult] arrives.
 *
 * The three non-terminal outcomes cannot actually arrive here, since a send clears no factor and neither [ClerkAuth.sendEmailCode] nor
 * [ClerkAuth.sendPasswordResetCode] returns anything but a success or a failure. They are routed to the step that owns each one rather than dropped,
 * so that the branches say what the outcome means instead of swallowing it. Note the routing is a best effort: those steps are driven by an attempt
 * the auth layer parks, and a send parks nothing they could use, so a backend that really did answer a send this way would land on a step whose
 * submit reports that no such flow is in progress.
 */
private fun reduceSend(current: EmailAuthUiState, result: AuthActionResult, codeStep: EmailAuthUiState.Step): EmailAuthUiState =
  when (result) {
    is AuthActionResult.Success -> current.copy(step = codeStep, submitting = false, error = null)
    is AuthActionResult.Error -> current.copy(submitting = false, error = result.message)
    is AuthActionResult.NeedsSecondFactor -> enterSecondFactor(current, result)
    is AuthActionResult.NeedsPassword -> enterCreatePassword(current)
    is AuthActionResult.NeedsNewPassword -> enterResetPassword(current)
  }

/**
 * Pure transition after [ClerkAuth.prepareSecondFactor]: stay on the step either way, since the code field is already the right place to be. A failed
 * send surfaces its message there rather than bouncing the user back to a first factor they have already cleared.
 */
internal fun reduceSecondFactorPrepared(current: EmailAuthUiState, result: AuthActionResult): EmailAuthUiState =
  when (result) {
    is AuthActionResult.Success -> current.copy(submitting = false, error = null)
    is AuthActionResult.Error -> current.copy(submitting = false, error = result.message)
    is AuthActionResult.NeedsSecondFactor -> enterSecondFactor(current, result)
    is AuthActionResult.NeedsPassword -> enterCreatePassword(current)
    is AuthActionResult.NeedsNewPassword -> enterResetPassword(current)
  }

/**
 * Shared transition into [EmailAuthUiState.Step.SecondFactor]: a lone factor is pre-selected, since a chooser with one entry is just a speed bump.
 *
 * Re-entering the step keeps whatever the user already picked, provided the account still offers it. Clerk can answer an attempt with another
 * NEEDS_SECOND_FACTOR rather than a failure, and resetting the choice there would clear the chooser, the code field keyed to it, and any message,
 * leaving the step to look like it reset itself for no reason.
 */
private fun enterSecondFactor(current: EmailAuthUiState, result: AuthActionResult.NeedsSecondFactor): EmailAuthUiState {
  val reentry = current.step == EmailAuthUiState.Step.SecondFactor
  return current.copy(
    step = EmailAuthUiState.Step.SecondFactor,
    submitting = false,
    error = if (reentry) current.error else null,
    secondFactorOptions = result.options,
    selectedSecondFactor = current.selectedSecondFactor?.takeIf { reentry && it in result.options } ?: result.options.singleOrNull(),
    // Taken from the result rather than kept across a re-entry, unlike the chosen factor: the two challenges are answered the same way, so Clerk can
    // follow one with the other, and the step would then be describing the wrong one.
    secondFactorReason = result.reason,
  )
}

/**
 * The factor to ask Clerk to deliver, given the state either side of an auth action, or null when there is nothing to send.
 *
 * Clerk does not send a second-factor code on its own: something has to call `prepare`. The chooser covers that when the account offers a choice, but
 * a lone `email_code` or `phone_code` factor is pre-selected without anyone tapping it, so without this the step would sit there claiming a code had
 * been sent while nothing had been.
 *
 * A verify can be answered with a *further* challenge rather than a session, which is how a cleared client-trust check is followed by the account's
 * own two-step verification (see [enterSecondFactor]). That re-enters the step it is already on, so entering alone does not catch it, and it needs a
 * send as much as the first challenge did. Recognized by the challenge having changed, because that is also what has to hold off the send's own
 * result: a `prepare` answers with the challenge it just delivered a code for, and reading that as an arrival would ask for another code for every
 * code it sent.
 *
 * What this misses is a repeat challenge identical in both fields, which Clerk can also answer with. That one keeps the digits spent on the previous
 * code, and the user has to tap Resend to get a live one. Left to the button rather than counted here, since counting arrivals is what cannot tell a
 * repeat challenge from the delivery of the one before it. iOS reaches it from the other side: its send guard is per-factor rather than
 * per-transition, so a repeat sends on arrival there, and its Resend button covers the same failed-send case this one does.
 */
internal fun secondFactorToDeliver(before: EmailAuthUiState, after: EmailAuthUiState): SecondFactor? {
  val entered = before.step != EmailAuthUiState.Step.SecondFactor && after.step == EmailAuthUiState.Step.SecondFactor
  val chained =
    before.step == EmailAuthUiState.Step.SecondFactor &&
      after.step == EmailAuthUiState.Step.SecondFactor &&
      (before.secondFactorReason != after.secondFactorReason || before.selectedSecondFactor != after.selectedSecondFactor)
  return if (entered || chained) after.selectedSecondFactor?.takeIf { it.needsSending } else null
}

/**
 * Pure transition shared by every action that *completes* a sign-in (code verify, OAuth, password, second factor, reset code, and the reset itself),
 * as opposed to [reduceSendCode], which only advances a step: on success reset to the initial state, since the [ClerkState.SignedIn] flip is what
 * drives the gate to Ready and no sign-in UI should linger behind it; on failure clear the spinner and surface the message on whichever step invoked
 * the action, so the Code step keeps its code field, the Password step its password field, and OAuth errors land back on the Email step where the
 * provider buttons live.
 *
 * Two of those callers do not complete anything on their own: the reset code buys the right to set a password, and it is the reset that finishes.
 * They share this reducer because what they need from it is the same, and because the reset can still answer with a session or a second-factor
 * challenge.
 */
internal fun reduceSignInCompletion(current: EmailAuthUiState, result: AuthActionResult): EmailAuthUiState =
  when (result) {
    is AuthActionResult.Success -> EmailAuthUiState()
    is AuthActionResult.Error -> current.copy(submitting = false, error = result.message)
    // The first factor was accepted but the account has MFA on, so this is a step forward, not a failure.
    is AuthActionResult.NeedsSecondFactor -> enterSecondFactor(current, result)
    // Verifying a new address leaves the sign-up one field short of an account. Also a step forward.
    is AuthActionResult.NeedsPassword -> enterCreatePassword(current)
    // The reset code cleared, or a correct password met a forced reset. Both are a step forward, and both are answered by the same step.
    is AuthActionResult.NeedsNewPassword -> enterResetPassword(current)
  }

/**
 * Shared transition into whichever step asks the user to *choose* a password: [EmailAuthUiState.Step.CreatePassword] to finish a registration, or
 * [EmailAuthUiState.Step.ResetPassword] to replace an existing account's.
 *
 * Both clear any MFA state on the way, for reasons that land in the same place. A registration cannot have a second factor yet. A reset can be
 * followed by one, but only *after* the new password is accepted, so factors read off the attempt so far belong to a challenge that has not been
 * issued. Either way, leaving them set would let the step render against a challenge that does not exist.
 */
private fun enterPasswordChoice(current: EmailAuthUiState, step: EmailAuthUiState.Step): EmailAuthUiState =
  current.copy(step = step, submitting = false, error = null, secondFactorOptions = emptyList(), selectedSecondFactor = null)

/**
 * Which step each of the two password outcomes leads to, stated once rather than at all six branches that route them.
 *
 * The pairing is the part worth protecting: [AuthActionResult.NeedsPassword] and [AuthActionResult.NeedsNewPassword] read almost alike, their steps
 * read almost alike, and a swapped pair would compile and would only show up as the wrong screen at the end of a flow that is awkward to reach.
 */
private fun enterCreatePassword(current: EmailAuthUiState): EmailAuthUiState = enterPasswordChoice(current, EmailAuthUiState.Step.CreatePassword)

/**
 * Also records whether the reset was asked for or demanded, which is what the step's copy keys on (see [EmailAuthUiState.resetWasForced]).
 *
 * Read off the step being left rather than plumbed down from the auth layer, because that is where the difference actually shows: arriving from the
 * code step means the user answered a code they requested, and arriving from anywhere else means Clerk asked for a password nobody offered to change.
 * No extra signal has to cross the seam for that.
 */
private fun enterResetPassword(current: EmailAuthUiState): EmailAuthUiState {
  // Kept as it was on a re-entry, the way [enterSecondFactor] keeps the factor the user picked, and for the same reason: the step is already showing,
  // so re-deriving from a step that is now this one would answer "not the code step" and silently reword a reset the user did ask for.
  val forced = if (current.step == EmailAuthUiState.Step.ResetPassword) current.resetWasForced else current.step != EmailAuthUiState.Step.ResetCode
  return enterPasswordChoice(current, EmailAuthUiState.Step.ResetPassword).copy(resetWasForced = forced)
}

/**
 * Pure transition that opens every auth action: raise the `submitting` flag (the spinner, and what each caller's double-submit check reads) and drop
 * any error still on screen from a previous attempt. See [ShellViewModel.runAuthAction], which is the only caller.
 */
internal fun beginAuthAction(current: EmailAuthUiState): EmailAuthUiState = current.copy(submitting = true, error = null)

/**
 * Pure completion half of [ShellViewModel.runAuthAction]: fold [result] in with [reduce], or discard it when the flow was reset while the request was
 * in flight. A down `submitting` flag is what identifies that case, since [beginAuthAction] raises it before the request starts and only a reset can
 * lower it while the request is outstanding. Keeping the decision here rather than inline in the coroutine is what makes it unit-testable.
 */
internal fun applyAuthOutcome(
  current: EmailAuthUiState,
  result: AuthActionResult,
  reduce: (EmailAuthUiState, AuthActionResult) -> EmailAuthUiState,
): EmailAuthUiState = if (current.submitting) reduce(current, result) else current

/** Navigation hooks a screen ViewModel uses to talk to the shell without depending on other screens. */
interface ShellActions {
  fun navigateTo(tab: Tab)

  fun openPocketMode(jobId: String)
}

/**
 * Owns everything that isn't one screen's concern: the Clerk auth gate, the runner-state passthrough, the selected tab, pocket mode, and the
 * persisted shared prefs (setup settings + default auto-submit). Per-screen ViewModels receive this as [ShellActions] (navigation) and read its
 * [registration] / [clerkUser] / [setupSettings] / [defaultContributeResults] flows, so they never depend on each other.
 */
class ShellViewModel(app: Application) : AndroidViewModel(app), ShellActions {
  private val container = (app as PipetteApp).container
  private val storage = container.storage
  private val clerkAuth = container.clerkAuth
  private val runner = container.jobController
  private val thermalProvider = container.thermalStatusProvider

  // Local chrome state. @Volatile: written from the UI thread and from screen VMs' confine
  // threads, read when renderState() snapshots them — volatile gives the cross-thread visibility.
  @Volatile private var selectedTab: Tab = Tab.JOBS

  // Pocket-mode target job. Written and read only on [confine] (openPocketMode / exitPocketMode /
  // renderState's reset all hop onto it), so the reset's check-then-act is race-free.
  private var pocketModeJobId: String? = null

  // Cached pocket manifest so buildPocket() doesn't re-read + parse it from disk on every
  // high-frequency within-cell runner tick. Keyed on (job, run start, run-relative completed count);
  // see buildPocket for why the run start is needed and why we also reload on any full-fraction tick.
  // Confine-only.
  private var pocketManifestKey: Triple<String, Long?, Int>? = null
  private var pocketManifest: ai.liquid.pipette.JobManifest? = null

  // publish()'s registration/pocket reads hit disk; run them here, off Main.
  private val confine = kotlinx.coroutines.Dispatchers.Default.limitedParallelism(1)

  // Cross-screen data-invalidation hub: any VM that mutates on-disk state (reset, model
  // add/delete, job create/delete) calls notifyDataChanged(); every screen VM observes
  // [dataChanges] and re-derives, so no tab shows data another tab just deleted/added.
  private val _dataChanges = kotlinx.coroutines.flow.MutableSharedFlow<Unit>(extraBufferCapacity = 1)
  val dataChanges: kotlinx.coroutines.flow.SharedFlow<Unit> = _dataChanges

  fun notifyDataChanged() {
    _dataChanges.tryEmit(Unit)
  }

  private val _registration = MutableStateFlow<RegistrationData?>(null)
  val registration: StateFlow<RegistrationData?> = _registration.asStateFlow()

  // Serializes every read-modify-write of the registration record against every delete of it. The mismatch gate offers "Sign out" beside "Delete
  // device identity", each launching its own coroutine, so tapping one then the other lets the unlink's load straddle the delete and write the record
  // back with no signing key to go with it: a device that reads as registered and fails every submission. Either order is safe under the lock, since
  // an unlink that runs after the delete finds no record and does nothing.
  private val registrationMutex = Mutex()

  private val _gateBypass = MutableStateFlow(false)

  private val _setupSettings = MutableStateFlow(SetupSettings())
  val setupSettings: StateFlow<SetupSettings> = _setupSettings.asStateFlow()

  private val _defaultContributeResults = MutableStateFlow(false)
  val defaultContributeResults: StateFlow<Boolean> = _defaultContributeResults.asStateFlow()

  private val clerkStateFlow: StateFlow<ClerkState>? = clerkAuth?.state?.stateIn(viewModelScope, SharingStarted.Eagerly, ClerkState.Loading)

  val clerkUser: StateFlow<ClerkState.SignedIn?> =
    (clerkStateFlow ?: MutableStateFlow(ClerkState.Loading)).map { it as? ClerkState.SignedIn }.stateIn(viewModelScope, SharingStarted.Eagerly, null)

  /**
   * The gate, or [AuthGate.Ready] when the build has no Clerk key.
   *
   * No key means sign-in was never configured, and the honest reading of that is "this build has no auth", not "this build is broken". It used to be
   * [AuthGate.InitError] outside debug, which made a keyless release APK unusable rather than open, and so made the baked-in production key
   * load-bearing: removing it bricked the app. That is exactly what stopped a fork from building this without Liquid's Clerk instance.
   *
   * Shipping *our* release with auth off is still a mistake, but a build-time one, caught by verifyReleaseClerkKey and by CI, rather than something
   * to discover on a user's device.
   */
  private val authGate: StateFlow<AuthGate> =
    clerkStateFlow?.let { state ->
      combine(state, _registration, _gateBypass) { clerk, registration, bypass -> reduceAuthGate(clerk, registration, bypass) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, AuthGate.Loading)
    } ?: MutableStateFlow<AuthGate>(AuthGate.Ready).asStateFlow()

  val isClerkAvailable: Boolean
    get() = clerkAuth != null

  private val _state = MutableStateFlow(ShellUiState(isClerkAvailable = clerkAuth != null))
  val state: StateFlow<ShellUiState> = _state.asStateFlow()

  init {
    viewModelScope.launch {
      _setupSettings.value = container.settingsStore.readSetupSettings()
      _defaultContributeResults.value = container.settingsStore.readDefaultContributeResults()
    }
    // Load registration on confine BEFORE the collectors below can render, so needsRegistration
    // (derived from _registration) is correct on the first frame. Otherwise a build whose authGate
    // is already Ready (DEBUG, no Clerk) would flash the Setup gate for an already-registered device
    // until the async load returned.
    viewModelScope.launch(confine) {
      _registration.value = storage.loadRegistration()
      renderState()
    }
    if (BuildConfig.DEBUG && clerkAuth != null) viewModelScope.launch { _gateBypass.value = container.settingsStore.readClerkGateBypass() }
    // Hydrate the container's cached thermal-gate waiver (PIP-434). The container owns the value,
    // since both readiness gates read it and one of them lives in another process, but it has no
    // coroutine of its own to load it with. This VM is built at launch, well before a job can reach
    // a gate.
    if (BuildConfig.DEBUG) viewModelScope.launch { container.applySkipThermalGate(container.settingsStore.readSkipThermalGate()) }
    clerkStateFlow?.let { state ->
      viewModelScope.launch {
        state.collect { clerk ->
          if (clerk is ClerkState.SignedIn) linkRegistrationIfNeeded(clerk)
          // A refused sign-out leaves its message on the mismatch screen (see [signOut]). Once the session does end that screen is gone, and the
          // message would open the fresh sign-in on an error the user did not just cause, so it goes with the state it was describing. Safe as a
          // transition: ClerkState only re-emits on a change, and a rejected password leaves it SignedOut throughout.
          else if (clerk is ClerkState.SignedOut) clearAuthError()
        }
      }
    }
    // Render inline on the confine thread (not a fresh launch per emission): a slow disk-backed
    // render suspends the collector, so StateFlow conflates the high-frequency runner ticks instead
    // of piling unbounded coroutines onto the single-thread dispatcher.
    viewModelScope.launch(confine) { runner.state.collect { renderState() } }
    viewModelScope.launch(confine) { authGate.collect { renderState() } }
    publish()
    // Pull the latest benchmark catalog from the management server once per launch (mirrors iOS's registration-time sync). Screen VMs collect
    // [BenchmarkCatalog.changes] to re-render when it lands.
    syncBenchmarkCatalog()
  }

  // --- ShellActions (called by screen VMs) ---
  override fun navigateTo(tab: Tab) {
    if (selectedTab != tab) selectedTab = tab
    publish()
  }

  override fun openPocketMode(jobId: String) {
    // For the RUNNING job the pocket screen is BenchmarkActivity in the :benchmark
    // process — it must be the focused top-app activity for the CPU-affinity boost.
    // Opening the main-process Compose pocket instead would bring MAIN to top-app
    // and demote :benchmark to /foreground, silently undoing the boost mid-run.
    // BenchmarkActivity is launchMode=singleTask, so a repeat/relaunch reuses the
    // one instance rather than stacking. Today the only caller (the "Open in Pocket
    // Mode" button) is shown only for the running job, so the else branch below is a
    // defensive fallback: a non-running job has no :benchmark to demote, and the
    // richer Compose pocket is retained for a possible future device-gated path.
    if (runner.state.value.runningJobId == jobId) {
      container.launchBenchmarkPocket()
      return
    }
    // Mutate pocketModeJobId on the confine thread (the same thread renderState()'s reset runs on),
    // so a concurrent runner-tick render can't clobber this write via its check-then-act reset.
    viewModelScope.launch(confine) {
      pocketModeJobId = jobId
      renderState()
    }
  }

  // --- Chrome intents from the Scaffold ---
  fun selectTab(tab: Tab) = navigateTo(tab)

  fun exitPocketMode() {
    viewModelScope.launch(confine) {
      pocketModeJobId = null
      renderState()
    }
  }

  // --- Auth (Settings delegates these here; they change the gate) ---
  private val _emailAuth = MutableStateFlow(EmailAuthUiState())

  /** State of the custom sign-in flow rendered by [AuthGateScreen] when the gate is [AuthGate.SignedOut]. */
  val emailAuth: StateFlow<EmailAuthUiState> = _emailAuth.asStateFlow()

  /** Email-code flow, first call: send a verification code to [email]. Advances to the code step on success. */
  fun submitEmail(email: String) {
    val auth = clerkAuth ?: return
    val clean = email.trim()
    // Guard blank input and double-submit (a second tap while the request is in flight).
    if (clean.isEmpty() || _emailAuth.value.submitting) return
    // The code step reads this address back out of the state, so record it before the request starts.
    _emailAuth.value = _emailAuth.value.copy(email = clean)
    runAuthAction(::reduceSendCode) { auth.sendEmailCode(clean) }
  }

  /**
   * Email-code flow, second call: verify [code]. On success the Clerk session activates and the gate flips to [AuthGate.Ready] via the state flow.
   */
  fun submitCode(code: String) {
    val auth = clerkAuth ?: return
    val clean = code.trim()
    if (clean.isEmpty() || _emailAuth.value.submitting) return
    runAuthAction(::reduceSignInCompletion) { auth.verifyEmailCode(clean) }
  }

  /**
   * Social sign-in: run the redirect OAuth flow for [strategy] (a value from [ShellUiState.oauthProviders]). Shares the [emailAuth] submitting/error
   * state with the email flow — the browser round-trip blocks re-submits, and on success the SignedIn flip drives the gate to Ready.
   */
  fun signInWithOAuth(strategy: String) {
    val auth = clerkAuth ?: return
    if (_emailAuth.value.submitting) return
    runAuthAction(::reduceSignInCompletion) { auth.signInWithOAuth(strategy) }
  }

  /**
   * Leave the passwordless flow for the password step, carrying [email] over from the email field (which lives in Compose state until now, so the
   * step needs it passed in). Blank input is ignored, since the step has no way to edit the address, only to go back.
   */
  fun usePasswordStep(email: String) {
    // No Clerk, no auth step to move to: the same precondition the other auth intents open with, so none of them can advance the UI into a step whose
    // submit would silently do nothing.
    if (clerkAuth == null) return
    val clean = email.trim()
    if (clean.isEmpty() || _emailAuth.value.submitting) return
    _emailAuth.value = _emailAuth.value.copy(step = EmailAuthUiState.Step.Password, email = clean, error = null)
  }

  /**
   * Credential sign-in for the address already captured on the password step. On success the Clerk session activates and the gate flips to
   * [AuthGate.Ready] via the state flow, exactly as with [submitCode]. The password is never stored: it goes straight to the SDK and stays out of
   * [EmailAuthUiState], so it can't be restored into the UI across a config change or land in a state dump.
   */
  fun submitPassword(password: String) {
    val auth = clerkAuth ?: return
    val email = _emailAuth.value.email
    if (password.isEmpty() || email.isEmpty() || _emailAuth.value.submitting) return
    runAuthAction(::reduceSignInCompletion) { auth.signInWithPassword(email, password) }
  }

  /**
   * Start a password reset for the address the flow is on: Clerk emails a code, and [submitPasswordResetCode] answers it. Two things call this, and
   * both want the same request: the link on the password step, and the resend link on the code step that link leads to.
   *
   * The route serves a forgotten password and an account that has never had one alike, and for the second it is the only route there is. The address
   * comes from state rather than the caller, since neither step it is offered from has an editable email field.
   */
  fun startPasswordReset() {
    val auth = clerkAuth ?: return
    val email = _emailAuth.value.email
    if (email.isEmpty() || _emailAuth.value.submitting) return
    runAuthAction(::reduceSendResetCode) { auth.sendPasswordResetCode(email) }
  }

  /**
   * Answer the emailed reset code. Folds through [reduceSignInCompletion] like every other verify, which is what carries the
   * [AuthActionResult.NeedsNewPassword] it succeeds with onto the step that collects the new password.
   */
  fun submitPasswordResetCode(code: String) {
    val auth = clerkAuth ?: return
    val clean = code.trim()
    if (clean.isEmpty() || _emailAuth.value.submitting) return
    runAuthAction(::reduceSignInCompletion) { auth.verifyPasswordResetCode(clean) }
  }

  /**
   * Finish the reset with the new password, from [EmailAuthUiState.Step.ResetPassword]. On success the session activates and the gate flips to
   * [AuthGate.Ready]; an account with two-step verification lands on the second-factor step first.
   *
   * Not trimmed, for the same reason [submitNewPassword] isn't: spaces are legitimate password characters, and eating them here would set a password
   * the user could never type again.
   */
  fun submitResetPassword(password: String) {
    val auth = clerkAuth ?: return
    if (password.isEmpty() || _emailAuth.value.submitting) return
    runAuthAction(::reduceSignInCompletion) { auth.resetPassword(password) }
  }

  /**
   * Pick which second factor to answer, and ask Clerk to deliver the code when the factor needs sending. Also the "resend" path: choosing the
   * already-selected factor again re-sends. A no-op for a factor the account didn't offer, so the UI can't drive the SDK into an unsupported
   * strategy.
   */
  fun chooseSecondFactor(factor: SecondFactor) {
    val current = _emailAuth.value
    if (clerkAuth == null || current.submitting || factor !in current.secondFactorOptions) return
    _emailAuth.value = current.copy(selectedSecondFactor = factor, error = null)
    deliverSecondFactor(factor)
  }

  /**
   * Ask Clerk to deliver [factor]'s code. TOTP and backup codes short-circuit to Success inside the auth layer, so this stays a single call site for
   * every factor, shared by the chooser and by the automatic send on entering the step.
   */
  private fun deliverSecondFactor(factor: SecondFactor) {
    val auth = clerkAuth ?: return
    runAuthAction(::reduceSecondFactorPrepared) { auth.prepareSecondFactor(factor) }
  }

  /**
   * Finish a registration by setting its password, from [EmailAuthUiState.Step.CreatePassword]. On success the account exists, the session activates,
   * and the gate flips to [AuthGate.Ready]. Like the sign-in password, it goes straight to the SDK and never lands in [EmailAuthUiState].
   *
   * Not trimmed, unlike the codes: leading and trailing spaces are legitimate password characters, and silently eating them would set a password the
   * user could never type again.
   */
  fun submitNewPassword(password: String) {
    val auth = clerkAuth ?: return
    if (password.isEmpty() || _emailAuth.value.submitting) return
    runAuthAction(::reduceSignInCompletion) { auth.createPassword(password) }
  }

  /**
   * Answer the MFA challenge. On success the session activates and the gate flips to [AuthGate.Ready] through the state flow, exactly as with
   * [submitCode] and [submitPassword]. Like the password, the code is never held in [EmailAuthUiState].
   */
  fun submitSecondFactorCode(code: String) {
    val auth = clerkAuth ?: return
    val factor = _emailAuth.value.selectedSecondFactor ?: return
    // Authenticator apps render TOTP as "123 456" and backup codes get pasted with a trailing newline; Clerk rejects either as simply incorrect.
    val clean = code.trim()
    if (clean.isEmpty() || _emailAuth.value.submitting) return
    runAuthAction(::reduceSignInCompletion) { auth.verifySecondFactor(factor, clean) }
  }

  /**
   * Run one auth action against the SDK and fold its outcome into [_emailAuth] with [reduce]. Every request to Clerk goes through here, so the
   * `submitting` flag is raised and read in one place: raised synchronously before the request starts (which is what each caller's own double-submit
   * check reads), lowered by [reduce] when the outcome lands.
   *
   * The outcome is discarded when the flag is already down on return, because that means the flow was reset mid-request. [setGateBypass] and
   * [signOut] are the two things that can do so; the back links that also reset are disabled while a request is in flight. Without the check a late
   * outcome would write over that reset, stamping its error onto the fresh email step, or (from the send path) advancing to a code step whose address
   * has already been cleared.
   *
   * The flag is flow-wide rather than per-attempt, so an outcome can still be folded into a *later* attempt's state if the bypass was toggled and a
   * new attempt started while the first was outstanding. That costs a stale message on a live step, so it isn't worth a generation counter.
   *
   * Both transitions live outside the class as [beginAuthAction] and [applyAuthOutcome] so they can be unit-tested without an Application.
   *
   * Landing on the second-factor step is the one outcome that owes a follow-up request, since Clerk only delivers a code when asked. That is decided
   * here, from the state either side of the action, so it holds for every path into the step rather than for whichever one remembered to do it.
   */
  private fun runAuthAction(reduce: (EmailAuthUiState, AuthActionResult) -> EmailAuthUiState, action: suspend () -> AuthActionResult) {
    val before = _emailAuth.value
    _emailAuth.value = beginAuthAction(before)
    viewModelScope.launch {
      val result = action()
      val after = applyAuthOutcome(_emailAuth.value, result, reduce)
      _emailAuth.value = after
      secondFactorToDeliver(before, after)?.let(::deliverSecondFactor)
    }
  }

  /** Clear a surfaced auth error as the user edits the email/code/password field, so a stale message doesn't linger during correction. */
  fun clearAuthError() {
    if (_emailAuth.value.error != null) _emailAuth.value = _emailAuth.value.copy(error = null)
  }

  /**
   * Back out of any later step to edit the email, discarding the pending attempt if there is one. Also clears the MFA selection, so a subsequent
   * sign-in re-derives its factors from that attempt rather than inheriting stale ones.
   */
  fun changeEmail() {
    clerkAuth?.resetEmailCode()
    _emailAuth.value =
      _emailAuth.value.copy(
        step = EmailAuthUiState.Step.Email,
        submitting = false,
        error = null,
        secondFactorOptions = emptyList(),
        selectedSecondFactor = null,
      )
  }

  /**
   * Sign out and rewind the sign-in flow to its first step.
   *
   * The rewind is the point: the gate renders whatever step [emailAuth] is on, and a session usually ends *after* a flow that got somewhere, so
   * without this, signing out would return the user to a password or code step belonging to an account that is no longer signed in. Any half-finished
   * attempt inside the SDK goes with it, since it can no longer be completed.
   *
   * The device's Clerk link goes too ([unlinkRegistration]), so the next sign-in is free to be a different account.
   *
   * A failure is written to [EmailAuthUiState.error], which only the gate renders, and reported to [onProblem] for the callers that are not the gate.
   * The Setup screen is why: it sits behind a live session with "Sign out" as its only exit, so a refused sign-out there has to say something, and it
   * has no auth state on screen to say it in. Defaulted, so the gate keeps its own rendering rather than also raising a toast over it.
   */
  fun signOut(onProblem: (String) -> Unit = {}) {
    val auth = clerkAuth ?: return
    auth.resetEmailCode()
    _emailAuth.value = EmailAuthUiState()
    viewModelScope.launch {
      when (val result = auth.signOut()) {
        // The session outlived the request (offline, or Clerk refused). Leave the link alone: unlinking under a live session would gate this account
        // straight to Ready and re-link the device to it, which is a worse trap than the mismatch screen the user is trying to leave. Nothing local
        // changed, so the message is the whole outcome. The SDK offers no way to drop the session locally that survives being offline (see
        // [ClerkAuth.signOut]), so the honest answer here is to say so and let the user retry.
        is AuthActionResult.Error -> {
          Log.w("ShellViewModel", "sign-out failed, leaving the device linked: ${result.message}")
          _emailAuth.value = EmailAuthUiState(error = result.message)
          onProblem("Could not sign out: ${result.message}")
        }
        else -> unlinkRegistration()
      }
    }
  }

  fun setGateBypass(enabled: Boolean) {
    _gateBypass.value = enabled
    viewModelScope.launch { container.settingsStore.writeClerkGateBypass(enabled) }
    // Bypassing is the one way to leave the gate without completing a sign-in (every completing action already resets on success), so it's the one
    // place a later step could be left showing. Clearing the flow here means turning the bypass back off re-opens the gate on the email step, not on
    // "Enter your password" for whatever address was last typed. It also abandons any in-flight attempt's outcome (see runAuthAction).
    _emailAuth.value = EmailAuthUiState()
  }

  /**
   * Waive (or restore) the thermal readiness criterion (PIP-434). Applied to the container first so a run started before the write lands still sees
   * it; the gates read the container, never DataStore.
   */
  fun setSkipThermalGate(enabled: Boolean) {
    container.applySkipThermalGate(enabled)
    viewModelScope.launch { container.settingsStore.writeSkipThermalGate(enabled) }
  }

  val skipThermalGate: Boolean
    get() = container.skipThermalGate

  /**
   * Sign out of Settings: end the session and reset the device, leaving nothing of the previous account's session behind (PIP-459). Registration,
   * signing key, saved Hugging Face token, jobs, benchmark results, and downloaded models all go, so signing back in starts from registration. This
   * list is what callers word their confirmation from, so keep it and `settings_sign_out_confirm` in step. The synced benchmark catalog stays:
   * [LocalStorage.resetDeviceData] leaves `benchmarks/` alone here, where iOS's `Storage/LocalStorage.swift` removes it. It holds no account data,
   * and [refreshRegistration] re-syncs it either way, so the platforms differ only in whether the picker is briefly empty.
   *
   * Distinct from [signOut], which keeps the device's identity and is what the gate and the Setup screen use. That split matches iOS, where
   * `ClerkAuthGateView.signOut` ends only the session and Settings' `signOutEverywhere` clears the device. A gate that destroyed the identity would
   * make the mismatch screen's only two buttons do the same destructive thing.
   *
   * Callers own the confirmation: this deletes results that were never uploaded, and [LocalStorage.unsubmittedResultCountOnDevice] is how the
   * Settings dialog says how many. [onProblem] carries whichever half fell short, the delete or the session call, since a sign-out the user was
   * promised must not fail silently.
   */
  fun signOutAndResetDevice(onProblem: (String) -> Unit) {
    // Deliberately not gated on Clerk being wired, unlike [signOut]. The confirmation promises this device's data is deleted; a build without a Clerk
    // key (or a session that has already ended) still owes the user that, rather than a button that silently does nothing.
    clerkAuth?.resetEmailCode()
    _emailAuth.value = EmailAuthUiState()
    viewModelScope.launch(confine) {
      // Before the session call, not after: `signOut` is a network round trip, and a job left running for it keeps writing into the tree about to be
      // cleared. A request either way, since JobRunner.cancel raises a flag rather than joining, so a cell in flight can still land a write after the
      // delete. Cancellation is advisory whichever order this runs in, so that leaves a stray job, not a corrupt device.
      runner.cancel()
      // A failed session call does not skip the reset: the local wipe is what the confirmation promised, and refusing it offline would leave the user
      // unable to sign out at all. It is reported, because the consequence is visible: the SDK keeps the session (there is no local-only drop that
      // survives being offline, see [ClerkAuth.signOut]), so a wiped device can land back on Setup as the account it was wiped to leave.
      val sessionResult = clerkAuth?.signOut()
      val sessionProblem = (sessionResult as? AuthActionResult.Error)?.message
      if (sessionProblem != null) Log.w("ShellViewModel", "sign-out failed while resetting the device: $sessionProblem")
      val resetFailure =
        runCatching {
            clearIdentityMaterial()
            withContext(Dispatchers.IO) {
              // Downloads first: a transfer that outlived the reset would land a model back in the tree it just cleared, on an unregistered device.
              // Blocking work (prefs, partial-file deletes), hence in here rather than on confine.
              container.downloadCoordinator.cancelAll()
              // The outgoing user's Hugging Face credential goes with their session (PIP-459), and the confirmation says so. Not part of the device
              // identity, so [deleteDeviceIdentity] leaves it be, but a sign-out is how a shared device changes hands: gated-repo access under the
              // previous account's token must not be what the next person inherits.
              container.secrets.deleteHfToken()
              storage.resetDeviceData()
              // Nothing above throws on a failed delete (`File.delete` returns false, `SharedPreferences.apply` is fire-and-forget), and this is the
              // one outcome worth raising: the key is already gone, so a surviving record reads as registered and fails every submission.
              check(!storage.isRegistered()) { "the device registration could not be deleted" }
            }
          }
          .exceptionOrNull()
      // runCatching swallows CancellationException too, which would report a reset that was merely interrupted as one that failed, and would run the
      // tail below in a scope that can no longer honor it.
      if (resetFailure is CancellationException) throw resetFailure
      // Published even when a delete threw: the registration may already be gone, and a UI still offering a registered device would be lying about
      // the half that did succeed.
      applyDefaultContributeResults(false)
      refreshRegistration()
      notifyDataChanged()
      if (resetFailure != null) {
        Log.w("ShellViewModel", "sign-out reset did not complete", resetFailure)
        onProblem("Sign out did not finish clearing this device: ${resetFailure.message ?: resetFailure.javaClass.simpleName}")
      } else if (sessionProblem != null) {
        onProblem("This device was cleared, but the account session could not be ended: $sessionProblem")
      }
    }
  }

  /** Mismatch-gate escape hatch: clear this device's registration + key so the gate starts fresh. */
  fun deleteDeviceIdentity() {
    viewModelScope.launch(confine) {
      clearIdentityMaterial()
      applyDefaultContributeResults(false)
      refreshRegistration()
    }
  }

  /**
   * The identity half of a reset: the registration record, then the key it signed submissions with. Shared by [deleteDeviceIdentity] and
   * [signOutAndResetDevice] so the two cannot drift over what "clear the identity" means.
   *
   * On [Dispatchers.IO] rather than inheriting the caller's [confine]: both deletes block (file I/O, keystore), and confine is the single thread this
   * hub derives [renderState] on, so blocking it stalls the gate and the bottom nav. Under [registrationMutex] so a concurrent [unlinkRegistration]
   * cannot write the record back afterwards.
   */
  private suspend fun clearIdentityMaterial() {
    registrationMutex.withLock {
      withContext(Dispatchers.IO) {
        storage.deleteRegistration()
        container.secrets.deletePrivateKey()
      }
    }
  }

  val clerkGateBypass: Boolean
    get() = _gateBypass.value

  // --- Shared persisted prefs ---
  fun persistSetupSettings(settings: SetupSettings) {
    _setupSettings.value = settings
    viewModelScope.launch { container.settingsStore.writeSetupSettings(settings) }
  }

  fun applyDefaultContributeResults(enabled: Boolean): Boolean {
    val allowed = storage.isRegistered() && enabled
    _defaultContributeResults.value = allowed
    viewModelScope.launch { container.settingsStore.writeDefaultContributeResults(allowed) }
    return allowed
  }

  fun refreshRegistration() {
    viewModelScope.launch {
      _registration.value = withContext(Dispatchers.IO) { storage.loadRegistration() }
      publish()
    }
    // A registration change (register/clear) may point at a new server — re-sync the catalog. ETag-conditional, so an unchanged catalog is a cheap
    // 304.
    syncBenchmarkCatalog()
  }

  /**
   * Pull the benchmark catalog from the management server (public `GET /benchmarks`, works unregistered) and refresh [BenchmarkCatalog] from the
   * updated store. Off-main; a failure is logged, never fatal — the picker keeps the last-synced catalog. The effective server URL is the registered
   * one, else the configured Setup value (defaults to the production server).
   */
  fun syncBenchmarkCatalog() {
    viewModelScope.launch(Dispatchers.IO) {
      val serverUrl = storage.loadRegistration()?.serverUrl ?: container.settingsStore.readSetupSettings().serverUrl
      runCatching { BenchmarkSync.sync(serverUrl, container.benchmarkStore, container.managementClient) }
        // load() re-reads index.json — keep it on IO (this whole coroutine runs there) so the disk read never lands on the main thread.
        .onSuccess { BenchmarkCatalog.load(container.benchmarkStore) }
        .onFailure { Log.w("ShellViewModel", "benchmark catalog sync failed", it) }
    }
  }

  /**
   * Un-pin this device from the account it was linked to, keeping the registration itself (and so the `clientId`, the signing key, and every result
   * already submitted under it). The counterpart to [linkRegistrationIfNeeded], and the reason a second account can sign in on a device the first one
   * used: without it the link outlives the session, and the next user lands on [AuthGate.Mismatch] whose only exit deletes the device identity.
   *
   * Deliberately ordered *after* [ClerkAuth.signOut] rather than before. Clearing the link while the session is still live would briefly leave a
   * signed-in user with nothing to mismatch against, and the gate would flash [AuthGate.Ready] (the whole app) on the way out.
   *
   * A failed write is logged, not thrown: the session has already ended, so crashing the sign-out would be worse than a device that is still linked.
   * The consequence is a mismatch at the next sign-in, which "Sign out" there now resolves. (iOS surfaces this one as an alert, since its gate has a
   * place to put it.)
   */
  private suspend fun unlinkRegistration() {
    val updated =
      registrationMutex.withLock {
        withContext(Dispatchers.IO) {
          val reg = storage.loadRegistration() ?: return@withContext null
          if (reg.clerkUserId == null) return@withContext null
          runCatching { reg.withoutClerkLink().also { storage.saveRegistration(it) } }
            .onFailure { Log.w("ShellViewModel", "could not un-link this device from its account", it) }
            .getOrNull()
        }
      }
    if (updated != null) _registration.value = updated
  }

  private suspend fun linkRegistrationIfNeeded(signedIn: ClerkState.SignedIn) {
    val updated =
      registrationMutex.withLock {
        withContext(Dispatchers.IO) {
          val reg = storage.loadRegistration() ?: return@withContext null
          if (reg.clerkUserId != null && reg.clerkUserId != signedIn.userId) return@withContext null
          if (reg.clerkUserId == signedIn.userId && reg.clerkSessionId == signedIn.sessionId && reg.clerkPrimaryEmail == signedIn.email) {
            return@withContext null
          }
          reg.withClerkLink(signedIn.userId, signedIn.sessionId, signedIn.email).also { storage.saveRegistration(it) }
        }
      }
    if (updated != null) _registration.value = updated
  }

  private fun publish() {
    viewModelScope.launch(confine) { renderState() }
  }

  /** Build [_state] from the current sources. MUST run on [confine] (the collectors call it inline; other callers go through [publish]). */
  private fun renderState() {
    val runnerState = runner.state.value
    // Single snapshot: act on the same value we checked, so the reset can't null out a jobId a
    // concurrent openPocketMode just set (both now run on confine, so this stays consistent).
    val snapshot = pocketModeJobId
    val activePocketJobId = snapshot?.takeIf { runnerState.runningJobId == it }
    if (snapshot != null && activePocketJobId == null) pocketModeJobId = null
    _state.value =
      ShellUiState(
        authGate = authGate.value,
        runner = runnerState,
        selectedTab = selectedTab,
        pocket = activePocketJobId?.let { buildPocket(it, runnerState) },
        isDebug = BuildConfig.DEBUG,
        isClerkAvailable = clerkAuth != null,
        clerkGateBypass = _gateBypass.value,
        // Read live from the SDK env: empty until the environment loads, populated by the time the SignedOut gate renders (init requires env).
        oauthProviders = clerkAuth?.oauthProviders ?: emptyList(),
        // Derived from the cached registration flow (kept current by refreshRegistration), not a
        // per-render disk read — this runs on every high-frequency runner tick.
        needsRegistration = _registration.value == null,
      )
  }

  private fun buildPocket(jobId: String, runnerState: RunnerState): PocketUi {
    // Skip the disk read on the high-frequency within-cell progress ticks (fraction < 1.0, same
    // completed count) and serve the cached manifest. Reload when the key moves AND on every
    // full-fraction tick:
    //  - startedAtMillis: fresh per run() (rerun/resume too), so it discriminates runs. Without it,
    //    a completed run leaves the cache at completedInRun=0 with an all-COMPLETED manifest, and a
    //    rerun's opening ticks (completedInRun=0, fraction 0.0) would collide and serve that stale
    //    manifest for the whole first cell.
    //  - completedInRun: the cell boundary within a run.
    //  - full-fraction reload: a cell's COMPLETED status + completedCells++ are persisted
    //    (JobRunner.saveManifest) no later than the completion emit, but completedInRun only advances
    //    at the NEXT cell — so keying on it alone would show a stale count through the inter-cell
    //    cooldown. Full-fraction ticks are low-frequency (completion + cooldown status), so reloading
    //    on them is cheap.
    // manifestFraction below still uses the LIVE currentCellFraction, so progress stays smooth.
    val key = Triple(jobId, runnerState.startedAtMillis, runnerState.completedInRun)
    if (key != pocketManifestKey || runnerState.currentCellFraction >= 1.0) {
      pocketManifest = storage.loadJobManifest(jobId)
      pocketManifestKey = key
    }
    val manifest = pocketManifest
    val fraction = manifest?.let { RunProgress.manifestFraction(it, runnerState) } ?: 0.0
    // ETA from this run's progress, not the whole manifest (a resumed job starts this run at 0).
    val runFraction = RunProgress.runFraction(runnerState, fraction)
    // Compute the ETA once: estimatedTimeLeft reads System.currentTimeMillis() internally, so two
    // calls could format to different strings within one frame (compact label vs expanded line).
    val eta = RunProgress.estimatedTimeLeft(runnerState, runFraction) ?: "calculating"
    val completed = manifest?.completedCells ?: 0
    val total = manifest?.totalCells ?: 0
    val subtitle =
      manifest?.let {
        val modelCount = it.cells.map { c -> c.modelName }.toSet().size
        val benchmarkCount = it.cells.map { c -> c.benchmarkId }.toSet().size
        "$modelCount ${plural("model", modelCount)} · $benchmarkCount ${plural("benchmark", benchmarkCount)}"
      } ?: "Loading benchmark details"
    return PocketUi(
      jobId = jobId,
      title = manifest?.let { DateFormats.shortDate(it.createdAt) } ?: "Benchmark job",
      subtitle = subtitle,
      progress = fraction,
      cellsDone = "$completed/$total cells done",
      timeLeft = eta,
      thermalLabel = thermalHeadroomLabel(thermalProvider),
      thermalAccent = thermalAccentKind(thermalDescription(thermalProvider)),
      estTimeLine = "Estimated time to complete: $eta",
      currentCellLabel = runnerState.currentCellLabel,
      progressText = runnerState.currentProgressText,
      coolingSinceMillis = runnerState.coolingSinceMillis,
    )
  }

  fun cancelRunningJob() {
    runner.cancel()
  }
}
