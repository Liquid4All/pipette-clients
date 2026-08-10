package ai.liquid.pipette

import android.app.Application
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

enum class Tab(val label: String) {
  JOBS("Jobs"),
  MODELS("Models"),
  SETTINGS("Settings"),
}

/**
 * Activity-scoped holder for the transient UI state that previously lived in [MainActivity] fields. Surviving here means a rotation no longer wipes
 * the user's tab, search text, selections, or run parameters. Long-lived collaborators are reached through the app-scoped [AppContainer]; background
 * work runs on [viewModelScope] (cancelled with the VM) instead of an ad-hoc thread pool.
 *
 * Rendering is driven by [uiTick]: the imperative full-rebuild render is kept (it suits programmatic XML views and is the lowest-risk move out of the
 * monolith), and [invalidate] bumps the tick to request a re-render. Async signals — the runner [state] and download callbacks — feed back through
 * the same tick.
 */
class MainViewModel(app: Application) : AndroidViewModel(app) {
  val container: AppContainer = (app as PipetteApp).container

  val runnerState: StateFlow<RunnerState>
    get() = container.jobController.state

  private val _uiTick = MutableStateFlow(0L)
  val uiTick: StateFlow<Long> = _uiTick.asStateFlow()

  private val mainHandler = Handler(Looper.getMainLooper())

  // --- Cross-screen / navigation state ---
  var selectedTab: Tab = Tab.JOBS
  @Volatile var statusText: String = ""
  var pocketModeJobId: String? = null

  /** Active step of the new-job wizard (0..2), or null when not creating a job. */
  var newJobStep: Int? = null

  // --- Job planning + list selection state (Jobs screen) ---
  val selectedModelKeys = linkedSetOf<String>()
  val selectedBenchmarkIds = linkedSetOf<String>()
  val selectedMmprojPaths = linkedSetOf<String>()
  val selectedJobQuantFilters = linkedSetOf(JobQuantFilter.ALL)
  val expandedCellIds = linkedSetOf<String>()
  val selectedRerunCellIds = linkedSetOf<String>()
  var mmprojSelectionInitialized: Boolean = false
  var jobSearchText: String = ""
  var benchmarkSearchText: String = ""
  var jobModelSearchText: String = ""
  var nGpuLayers: Int = 99
  var contextSize: Int = 4096
  var prefillBatch: Int = JobManifest.DEFAULT_PREFILL_BATCH
  var contributeResults: Boolean = false
  var pendingCsvExportText: String? = null
  var selectedJobId: String? = null

  // --- Models screen state ---
  var downloadedModelSearchText: String = ""
  var templateSearchText: String = ""
  // "Add models" sub-screen (iOS AddModelsView parity): which families are picked,
  // which quant pills are active, and whether the sub-screen is showing. Downloaded
  // models are grouped by family on the main screen; track which groups are expanded.
  var modelsShowAddScreen: Boolean = false
  val selectedAddFamilyIds = linkedSetOf<String>()
  val selectedAddQuants = ModelQuantFilter.allSelection().toCollection(linkedSetOf())
  val expandedModelGroupKeys = linkedSetOf<String>()

  // --- Persisted settings, cached in memory ---
  // Loaded once off the main thread (see init) and refreshed on write, so the
  // UI never touches DataStore on the main thread. The screens read these
  // fields; never call AppSettingsStore directly from render.
  @Volatile
  var defaultContributeResults: Boolean = false
    private set

  @Volatile
  var setupSettings: SetupSettings = SetupSettings()
    private set

  // --- Clerk auth gate ---
  // clerkAuth is null in the :benchmark process and when Clerk isn't configured
  // (see PipetteApp). Registration is read off-main and re-published after a
  // link/register/clear so the gate re-evaluates a resolved mismatch.
  private val clerkAuth: ClerkAuth? = container.clerkAuth
  private val _registration = MutableStateFlow<RegistrationData?>(null)
  // Auth is enforced by default in every build; the debug-only toggle below can
  // turn it off (e.g. to run benchmarks un-gated), persisted in AppSettingsStore.
  private val _gateBypass = MutableStateFlow(false)

  // Single shared subscription to the Clerk state stream. Both the gate reducer
  // ([authGate]) and the sign-in side effects (in [init]) observe THIS, so the
  // underlying Clerk flows are collected exactly once — no duplicate
  // subscriptions, no chance for the two paths to drift. Null when Clerk isn't
  // available (`:benchmark` process / unconfigured).
  private val clerkState: StateFlow<ClerkState>? = clerkAuth?.state?.stateIn(viewModelScope, SharingStarted.Eagerly, ClerkState.Loading)

  /** True when a Clerk SDK is available (configured + main process). */
  val isClerkAvailable: Boolean
    get() = clerkAuth != null

  /** Current debug bypass value (always effectively false in release). */
  val clerkGateBypass: Boolean
    get() = _gateBypass.value

  /** Latest signed-in Clerk identity (for pre-filling/linking registration), or null. */
  @Volatile
  var clerkUser: ClerkState.SignedIn? = null
    private set

  /**
   * What the auth gate should show. Initial value is [AuthGate.Loading] so the main UI never flashes before Clerk reports. With no Clerk available,
   * debug builds fall through ([AuthGate.Ready]) and release surfaces a config error.
   */
  val authGate: StateFlow<AuthGate> =
    clerkState?.let { state ->
      combine(state, _registration, _gateBypass) { clerk, registration, bypass -> reduceAuthGate(clerk, registration, bypass) }
        .stateIn(viewModelScope, SharingStarted.Eagerly, AuthGate.Loading)
    } ?: MutableStateFlow(if (BuildConfig.DEBUG) AuthGate.Ready else AuthGate.InitError("Clerk not configured")).asStateFlow()

  init {
    // Re-render on any download-registry change so a download resumed by WorkManager after process death (which has no live callbacks) still updates
    // the Models screen. Gated to the Models tab so a background download doesn't churn other screens; the listener fires off the main thread but
    // invalidate() is safe there.
    DownloadRegistry.onChanged = { if (selectedTab == Tab.MODELS) invalidate() }
    selectedBenchmarkIds += BenchmarkCatalog.selectable.map { it.benchmarkId.toString() }
    // AppSettingsStore's suspend API is main-safe (DataStore hops to IO
    // itself); load off the main thread on viewModelScope, then re-render.
    viewModelScope.launch {
      val store = container.settingsStore
      setupSettings = store.readSetupSettings()
      defaultContributeResults = store.readDefaultContributeResults()
      contributeResults = container.storage.isRegistered() && defaultContributeResults
      invalidate()
    }

    // Re-report the device profile + capability set the planner matches jobs against. Done on
    // every launch because its inputs drift between runs (an OS update, a security patch, an
    // APK with a different llama.cpp build); an unchanged resubmit is a server-side no-op.
    // Best-effort: a failure here is advisory and must never block the UI or the auth gate.
    viewModelScope.launch {
      withContext(Dispatchers.IO) { runCatching { container.profileRefreshService.refresh() } }
        .onFailure { Log.w("MainViewModel", "device profile refresh failed", it) }
    }

    // Auth gate: load the linked registration, honor the persisted debug bypass,
    // and link the Clerk identity into the registration once signed in.
    refreshRegistration()
    clerkState?.let { state ->
      if (BuildConfig.DEBUG) {
        viewModelScope.launch { _gateBypass.value = container.settingsStore.readClerkGateBypass() }
      }
      viewModelScope.launch {
        state.collect { clerk ->
          clerkUser = clerk as? ClerkState.SignedIn
          if (clerk is ClerkState.SignedIn) linkRegistrationIfNeeded(clerk)
        }
      }
    }
  }

  /** Re-read the device registration off-main and re-publish it to the gate. */
  fun refreshRegistration() {
    viewModelScope.launch { _registration.value = withContext(Dispatchers.IO) { container.storage.loadRegistration() } }
  }

  /**
   * Link the signed-in Clerk identity into the local registration if it isn't already current. Leaves a *different* previously-linked account
   * untouched so the gate can surface the mismatch. Local only — never sent to the mgmt server.
   */
  private suspend fun linkRegistrationIfNeeded(signedIn: ClerkState.SignedIn) {
    val updated =
      withContext(Dispatchers.IO) {
        val reg = container.storage.loadRegistration() ?: return@withContext null
        if (reg.clerkUserId != null && reg.clerkUserId != signedIn.userId) return@withContext null
        if (reg.clerkUserId == signedIn.userId && reg.clerkSessionId == signedIn.sessionId && reg.clerkPrimaryEmail == signedIn.email) {
          return@withContext null
        }
        reg.withClerkLink(signedIn.userId, signedIn.sessionId, signedIn.email).also { container.storage.saveRegistration(it) }
      }
    if (updated != null) _registration.value = updated
  }

  /** Toggle the debug-only Clerk-gate bypass (persisted). No effect in release. */
  fun setClerkGateBypass(enabled: Boolean) {
    _gateBypass.value = enabled
    viewModelScope.launch { container.settingsStore.writeClerkGateBypass(enabled) }
  }

  /**
   * Sign out of Clerk (no-op if Clerk isn't available).
   *
   * Ends only the session. Deliberately does *not* un-pin the device the way `ShellViewModel.signOut` does: that needs the [registrationMutex] this
   * class has no equivalent of, since the legacy mismatch card puts "Sign out" beside a "Delete device identity" that writes the same record from the
   * main thread. Nothing here can reach that race today ([MainActivity] is commented out of the manifest), and a second copy of the unlink is the
   * wrong way to keep it that way: reviving the view-based UI should pick up the lock-aware version rather than one that has since drifted from it.
   */
  fun signOutOfClerk() {
    val auth = clerkAuth ?: return
    viewModelScope.launch { auth.signOut() }
  }

  /** Persist the Setup form values (server/org/email) off the main thread. */
  fun persistSetupSettings(settings: SetupSettings) {
    setupSettings = settings
    viewModelScope.launch { container.settingsStore.writeSetupSettings(settings) }
  }

  /**
   * Apply (and persist) the default-auto-submit toggle. Returns the effective value (forced off when the device isn't registered). Persists off-main.
   */
  fun applyDefaultContributeResults(enabled: Boolean): Boolean {
    val allowed = container.storage.isRegistered() && enabled
    defaultContributeResults = allowed
    contributeResults = allowed
    viewModelScope.launch { container.settingsStore.writeDefaultContributeResults(allowed) }
    return allowed
  }

  override fun onCleared() {
    // Drop the global registry listener so it doesn't pin this (cleared) ViewModel.
    DownloadRegistry.onChanged = null
    super.onCleared()
  }

  /** Request a re-render of the current screen. Safe to call off the main thread. */
  fun invalidate() {
    // Atomic so concurrent off-main-thread calls can't collapse into one
    // increment (a non-atomic value=value+1 can drop a re-render).
    _uiTick.update { it + 1 }
  }

  /** Post to the main thread (for async callbacks that aren't coroutines, e.g. DownloadManager polling). */
  fun onMain(block: () -> Unit) {
    mainHandler.post(block)
  }

  /**
   * Publish a download status line and re-render the Models tab. Lives on the (retained) ViewModel so download callbacks can reference it without
   * capturing a [Screen]/Activity — the global, long-lived [DownloadRegistry] holds those callbacks for a download's lifetime, so capturing the
   * Activity there would leak it across a rotation.
   */
  fun postDownloadStatus(message: String) {
    onMain {
      statusText = message
      if (selectedTab == Tab.MODELS) invalidate()
    }
  }

  /**
   * Run [block] off the main thread, surfacing its returned message (or any failure) as [statusText]. Mirrors the old `runInBackground` contract but
   * on [viewModelScope] + [Dispatchers.IO]; resumes on the main dispatcher to publish the result.
   */
  fun runInBackground(startMessage: String, onError: (Throwable) -> Unit, block: () -> String) {
    statusText = startMessage
    invalidate()
    viewModelScope.launch {
      runCatching { withContext(Dispatchers.IO) { block() } }
        .onSuccess { message ->
          statusText = message
          invalidate()
        }
        .onFailure { error ->
          statusText = ""
          onError(error)
          invalidate()
        }
    }
  }
}
