package ai.liquid.pipette.compose.settings

import ai.liquid.pipette.AccentKind
import ai.liquid.pipette.BuildConfig
import ai.liquid.pipette.ByteFormat
import ai.liquid.pipette.DeviceInfo
import ai.liquid.pipette.FeedbackDialog
import ai.liquid.pipette.PipetteApp
import ai.liquid.pipette.RegistrationData
import ai.liquid.pipette.compose.ScreenViewModel
import ai.liquid.pipette.compose.shell.ShellViewModel
import ai.liquid.pipette.thermalAccentKind
import ai.liquid.pipette.thermalDescription
import ai.liquid.pipette.thermalLabel
import android.app.Application
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

/** Settings screen: registration summary, account, result-submission defaults, thermal, HF token, local data, debug info. */
data class SettingsUiState(
  val registration: RegistrationData? = null,
  val isRegistered: Boolean = false,
  val clerkEmail: String? = null,
  val clerkUserId: String? = null,
  val isClerkAvailable: Boolean = false,
  val runningJobPresent: Boolean = false,
  val isDebug: Boolean = BuildConfig.DEBUG,
  val clerkGateBypass: Boolean = false,
  /** Debug-only thermal readiness waiver (PIP-434); recorded on every submission the waived run produces. */
  val skipThermalGate: Boolean = false,
  val defaultContributeResults: Boolean = false,
  /**
   * Completed results still waiting to be uploaded. Signing out destroys them (PIP-459), so the confirm dialog names the number rather than letting
   * the loss be a surprise. Zero when the device has none, which drops that sentence from the copy.
   */
  val unsubmittedResultCount: Int = 0,
  /** False in a build with no PostHog project wired, which hides the opt-out row rather than offering a toggle over a sink that collects nothing. */
  val isAnalyticsAvailable: Boolean = false,
  val analyticsOptedOut: Boolean = false,
  val thermalLabel: String = "",
  val thermalDescription: String = "",
  val thermalAccent: AccentKind = AccentKind.NOMINAL,
  val savedHfToken: String = "",
  val isFeedbackAvailable: Boolean = false,
  val debugInfo: String = "",
)

sealed interface SettingsIntent {
  data object SignOut : SettingsIntent

  data object DeleteDeviceIdentity : SettingsIntent

  data class SetGateBypass(val enabled: Boolean) : SettingsIntent

  data class SetSkipThermalGate(val enabled: Boolean) : SettingsIntent

  data class SetDefaultContributeResults(val enabled: Boolean) : SettingsIntent

  data class SetAnalyticsOptOut(val optedOut: Boolean) : SettingsIntent

  data class SaveHfToken(val token: String) : SettingsIntent

  data class SubmitFeedback(val message: String, val email: String, val categoryId: String?) : SettingsIntent

  data object ResetLocalData : SettingsIntent
}

class SettingsViewModel(app: Application, shell: ShellViewModel) : ScreenViewModel(app, shell) {
  private val container = (app as PipetteApp).container
  private val storage = container.storage
  private val secrets = container.secrets
  private val thermalProvider = container.thermalStatusProvider
  private val runner = container.jobController

  private val _state = MutableStateFlow(SettingsUiState())
  val state: StateFlow<SettingsUiState> = _state.asStateFlow()

  // Single-thread confinement: derivation (disk I/O) runs here, never on Main.
  private val confine = Dispatchers.Default.limitedParallelism(1)

  init {
    viewModelScope.launch(confine) { publish() }
    viewModelScope.launch(confine) { shell.registration.collect { publish() } }
    viewModelScope.launch(confine) { shell.clerkUser.collect { publish() } }
    viewModelScope.launch(confine) { shell.defaultContributeResults.collect { publish() } }
    // unsubmittedResultCount is read off the job store, which other tabs write: submitting from the job detail clears results this dialog would
    // otherwise still be counting, and deleting a job takes its pending results with it. Both announce themselves here, so the figure the sign-out
    // warning quotes is not the one that was true when Settings was last opened.
    viewModelScope.launch(confine) { shell.dataChanges.collect { publish() } }
    // publish() re-reads the keystore and walks the job store, so the runner's progress ticks are
    // collapsed to the two values it derives from: whether a job is running, and how far through the
    // run it is. Run position is in here because unsubmittedResultCount grows as cells finish, and
    // keying on presence alone would freeze it at the job-start figure. Note `completedInRun` advances
    // when the next cell starts, not when the last payload lands, so mid-run the count can trail by
    // one result until the cooldown ends.
    viewModelScope.launch(confine) {
      runner.state.map { (it.runningJobId != null) to it.completedInRun }.distinctUntilChanged().collect { publish() }
    }
  }

  fun onIntent(intent: SettingsIntent) {
    viewModelScope.launch(confine) { handle(intent) }
  }

  private fun handle(intent: SettingsIntent) {
    when (intent) {
      // Settings sign-out resets the device (PIP-459); the gate's own sign-out stays session-only. A delete that fails surfaces here, next to the
      // dialog that promised it, the same way ResetLocalData below reports its own.
      SettingsIntent.SignOut -> shell.signOutAndResetDevice(onProblem = ::showError)
      SettingsIntent.DeleteDeviceIdentity -> {
        shell.signOut(onProblem = ::showError)
        shell.deleteDeviceIdentity()
        publish()
      }
      is SettingsIntent.SetGateBypass -> {
        shell.setGateBypass(intent.enabled)
        publish()
      }
      is SettingsIntent.SetSkipThermalGate -> {
        shell.setSkipThermalGate(intent.enabled)
        publish()
      }
      is SettingsIntent.SetDefaultContributeResults -> {
        shell.applyDefaultContributeResults(intent.enabled)
        publish()
      }
      is SettingsIntent.SetAnalyticsOptOut -> {
        // The SDK persists this itself; nothing to write to AppSettingsStore. publish() reads it
        // straight back so the row reflects what the SDK actually holds, not what we asked for.
        container.analytics.setOptedOut(intent.optedOut)
        publish()
      }
      is SettingsIntent.SaveHfToken -> {
        saveHfToken(intent.token)
        publish()
      }
      is SettingsIntent.SubmitFeedback -> FeedbackDialog.capture(intent.message, intent.email, intent.categoryId, container.analytics)
      SettingsIntent.ResetLocalData -> {
        runCatching { storage.resetDeviceData() }.onSuccess { shell.notifyDataChanged() }.onFailure { showError(it.message ?: "Reset failed") }
        publish()
      }
    }
  }

  /**
   * Store or clear the Hugging Face token. A save can fail (the Keystore was unreachable) and leaves any previously stored token intact. Surfacing
   * that matters more than the usual silent-success rule: the failure is otherwise invisible until a download rejects a token the user believes they
   * saved. Clearing cannot fail.
   */
  private fun saveHfToken(raw: String) {
    val value = raw.trim()
    if (value.isBlank()) secrets.deleteHfToken()
    else if (!secrets.saveHfToken(value)) showError("Could not save the HF token: secure storage is unavailable")
  }

  private fun publish() {
    val registration = storage.loadRegistration()
    val clerk = shell.clerkUser.value
    // Derived from the record already in hand rather than asked for again: `LocalStorage.isRegistered` is
    // `loadRegistration() != null`, so calling it here would re-read and re-parse the file this line above just read.
    val isRegistered = registration != null
    // Read once and handed to both readers, like the manifest walk below. The field wants the token and the debug panel
    // only whether there is one, and this is a keystore round trip, the costliest of the reads here.
    val hfToken = secrets.loadHfToken()
    // Listed and parsed once per publish, then handed to both readers: the sign-out warning's count and (in debug) the panel's job total are the same
    // walk, and this runs on every completed cell. Skipped entirely on an unregistered device outside debug, where neither reader is on screen: the
    // sign-out row is rendered under `isRegistered`, and jobs outlive a cleared identity, so this is not a walk over nothing.
    val manifests = if (isRegistered || BuildConfig.DEBUG) storage.loadAllJobManifests() else emptyList()
    _state.value =
      SettingsUiState(
        registration = registration,
        isRegistered = isRegistered,
        clerkEmail = clerk?.email,
        clerkUserId = clerk?.userId,
        isClerkAvailable = shell.isClerkAvailable,
        runningJobPresent = runner.state.value.runningJobId != null,
        isDebug = BuildConfig.DEBUG,
        clerkGateBypass = shell.clerkGateBypass,
        skipThermalGate = shell.skipThermalGate,
        defaultContributeResults = shell.defaultContributeResults.value,
        // Derived here rather than when the sign-out dialog opens: a count that is not ready the instant that dialog appears is a warning the user
        // never sees. See init for what this walk is collapsed to.
        unsubmittedResultCount = storage.unsubmittedResultCountOnDevice(manifests),
        isAnalyticsAvailable = container.analytics.isAvailable,
        analyticsOptedOut = container.analytics.isOptedOut(),
        thermalLabel = thermalLabel(thermalProvider),
        thermalDescription = thermalDescription(thermalProvider),
        thermalAccent = thermalAccentKind(thermalDescription(thermalProvider)),
        savedHfToken = hfToken ?: "",
        isFeedbackAvailable = FeedbackDialog.isAvailable(),
        // Debug panel is the only consumer (gated on isDebug in the UI); skip its models-dir walk
        // entirely in release builds. The manifest walk it also reads is shared with the sign-out
        // count above, so that one is hoisted rather than skipped.
        debugInfo = if (BuildConfig.DEBUG) debugInfoText(registration, manifests.size, hfToken != null) else "",
      )
  }

  private fun debugInfoText(registration: RegistrationData?, jobs: Int, hasHfToken: Boolean): String {
    val app = getApplication<Application>()
    val models = storage.availableModels().size
    val privateKeyState = if (secrets.hasPrivateKey()) "Present" else "Missing"
    val hfTokenState = if (hasHfToken) "Present" else "Missing"
    // Same record the caller already read, for the same reason it hoists the manifest walk: `isRegistered()` would parse `registration.json` again.
    val autoSubmit = if (registration != null) (if (shell.defaultContributeResults.value) "Enabled" else "Disabled") else "Unavailable"
    return listOf(
        "Client ID: ${registration?.clientId ?: "Unavailable"}",
        "Status: ${registration?.status ?: "Unavailable"}",
        "Clerk user: ${registration?.clerkUserId ?: "Unlinked"}",
        "Clerk email: ${registration?.clerkPrimaryEmail ?: "Unavailable"}",
        "Device: ${DeviceInfo.modelName()}",
        "Chip: ${DeviceInfo.chipModel()}",
        "Form factor: ${DeviceInfo.formFactor(app)}",
        "OS: Android ${DeviceInfo.osVersion()}",
        "RAM: ${ByteFormat.fileSize(DeviceInfo.ramBytes(app))}",
        "Thermal: ${thermalDescription(thermalProvider)}",
        "Auto-submit: $autoSubmit",
        "Jobs: $jobs",
        "Models: $models",
        "Private key: $privateKeyState",
        "HF token: $hfTokenState",
        "Models directory: ${storage.modelsDir.absolutePath}",
      )
      .joinToString("\n")
  }
}
