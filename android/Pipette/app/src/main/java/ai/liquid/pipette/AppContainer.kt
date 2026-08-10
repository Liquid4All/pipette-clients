package ai.liquid.pipette

import ai.liquid.pipette.service.JobOrchestratorService
import ai.liquid.pipette.service.RemoteBenchmarkEngine
import android.content.Context
import android.content.Intent

/**
 * Manual dependency container (no Hilt — keeps the `:benchmark` process lean and the wiring explicit). Holds the process-lived singletons: the data
 * layer, the isolated benchmark engine proxy, and the [JobController] that owns the runner.
 *
 * App-scoping these is the core of Phase B: the engine proxy and runner live for the app's lifetime, so Activity recreation never tears the engine
 * down mid-run. The `:benchmark` process is still killed when idle (the proxy's own teardown), so the second-process footprint is not paid at rest.
 */
class AppContainer(
  context: Context,
  /**
   * The Clerk auth seam, or null in the `:benchmark` process / when Clerk isn't configured (see [PipetteApp]). The UI reads this to drive the auth
   * gate; a null value means the gate is effectively bypassed (no Clerk available).
   */
  val clerkAuth: ClerkAuth? = null,
  /**
   * Product-analytics sink. Defaults to [NoOpAnalytics] so tests (and any build with no PostHog project configured) construct the container without
   * initializing an SDK; [PipetteApp] passes the real one in the main process. See [Analytics].
   */
  val analytics: Analytics = NoOpAnalytics,
) {
  private val appContext = context.applicationContext

  val storage = LocalStorage(appContext)
  val settingsStore = AppSettingsStore(appContext)
  val secrets = Secrets(appContext)
  val managementClient = ManagementClient(secrets)
  // Server-synced benchmark catalog store. BenchmarkSync writes it; BenchmarkCatalog reads it (loaded from disk in init so a prior sync is available
  // before the first render). See [BenchmarkSync] / [BenchmarkCatalog].
  val benchmarkStore = FileBenchmarkStore(storage.benchmarksDir)
  val registrationService = RegistrationService(storage, secrets, managementClient, analytics)
  val profileRefreshService = ProfileRefreshService(appContext, storage, managementClient)
  // `by lazy` so merely constructing AppContainer never calls WorkManager.getInstance() before we
  // want it to. AppContainer is only ever built in the main process (PipetteApp guards it), which is
  // also where WorkManager is initialized. PipetteApp.onCreate forces this coordinator's construction
  // right after building the container, so the resume-on-launch restore() still runs at startup —
  // before a download-notification action (Pause/Cancel) can cold-start the process and race it.
  val downloadCoordinator by lazy { DownloadCoordinator(appContext, storage, analytics) }
  val submissionService = ResultSubmissionService(storage, managementClient, analytics)
  val thermalStatusProvider = AndroidThermalStatusProvider(appContext)

  /**
   * Process-wide cache of the debug thermal-gate waiver (PIP-434), the value both readiness gates consult: the between-cell one built below, and the
   * per-rep one in `:benchmark`, which gets it as a service Intent extra ([RemoteBenchmarkEngine]).
   *
   * Cached here rather than read where it's used because [AppSettingsStore] reads are `suspend` and both consumers are synchronous. The container
   * owns the value; a ViewModel supplies the coroutine that hydrates it at launch and updates it when the toggle flips ([ShellViewModel]).
   *
   * The initial `false` is not a guess at the stored value, it is the fail-closed default: until the load returns, the gate is enforced. Enforcing a
   * gate the user waived costs a wait; waiving one they didn't costs a result that silently isn't comparable.
   */
  @Volatile
  var skipThermalGate: Boolean = false
    private set

  /** Applies a loaded-or-toggled waiver. Persistence is the caller's ([ShellViewModel]); this only updates what the gates read. */
  fun applySkipThermalGate(enabled: Boolean) {
    skipThermalGate = enabled
  }

  // On every `:benchmark` (re)connect, drain crash reports and report any OOM/ANR from the process that just died. sentry-native flushes a prior
  // native crash during the respawn's init, so connect is when a new envelope becomes available; reporting here also catches the crash-then-retry
  // flow promptly, before the next launch. See [CrashEnvelopeForwarder] / [BenchmarkExitReporter].
  val benchmarkEngine: BenchmarkEngine =
    RemoteBenchmarkEngine(
      appContext,
      onBenchmarkProcessConnected = {
        CrashEnvelopeForwarder.drain(appContext)
        BenchmarkExitReporter.reportNewExits(appContext)
      },
      // Read at service-start time, not now: the proxy starts `:benchmark` when a run begins, which
      // is after the toggle has had a chance to load or change.
      skipThermalGate = { skipThermalGate },
    )
  // Launches the pocket-mode Activity in the :benchmark process so that process
  // claims top-app and inference regains the prime cores an OEM (e.g. Samsung)
  // denies a non-top-app service process. Used by JobRunner when a run's first
  // cell loads, and by the manual "Open in Pocket Mode" action (ShellViewModel)
  // for the running job — opening the main-process Compose pocket instead would
  // bring MAIN to top-app and demote :benchmark.
  val launchBenchmarkPocket: () -> Unit = {
    runCatching { appContext.startActivity(Intent(appContext, BenchmarkActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
  }

  // Holds a foreground service in THIS (main) process for the length of a run. Without it main has
  // no foreground component once the pocket screen puts :benchmark on top, so it is cached, and the
  // platform kills cached processes for sustained binder traffic, which is exactly what JobRunner's
  // progress mirror is. See JobOrchestratorService. runCatching because a background FGS start can
  // be refused outright (ForegroundServiceStartNotAllowedException); the run then proceeds without
  // the protection rather than crashing.
  private val setJobForegroundHold: (Boolean) -> Unit = { active ->
    val intent = Intent(appContext, JobOrchestratorService::class.java)
    runCatching { if (active) appContext.startForegroundService(intent) else appContext.stopService(intent) }
  }
  val jobController =
    JobController(
      storage = storage,
      engine = benchmarkEngine,
      submissionService = submissionService,
      // Use the throttled cachedHeadroom (not raw currentHeadroom): the UI now also reads thermal
      // headroom (the Pocket / running-detail "Throttling headroom" readout), and getThermalHeadroom
      // is rate-limited to ~1/sec PER PROCESS. This between-cell gate runs in the UI process, so
      // sharing the one cache keeps the UI's poll from starving the gate's read to NaN (which would
      // drop it onto the coarse status-enum fallback). The :benchmark per-rep gate is a separate
      // process with no UI reader, so it keeps the raw read.
      readiness = Readiness(thermalStatusProvider::currentStatus, thermalStatusProvider::cachedHeadroom, skipThermal = { skipThermalGate }),
      // JobRunner calls this when the run's first cell actually loads.
      host = HostHooks(launchPocketMode = launchBenchmarkPocket, setForegroundHold = setJobForegroundHold),
      analytics = analytics,
      // Gates the per-cell result upload. Bound to the app context here, not read inside the
      // runner, so the runner stays constructible in tests.
      isOnline = NetworkReachability.checker(appContext),
    )

  init {
    // BenchmarkActivity's Cancel (in :benchmark) → cancel the whole job here.
    benchmarkEngine.onJobCancelRequested = { jobController.cancel() }
    // Turn any orphaned RUNNING/PAUSED manifests (from a previous process
    // death) back into resumable state. Once per process, not per Activity.
    storage.recoverInterruptedJobs()
    // Restore the benchmark catalog from the last sync so the picker isn't empty before the first network sync of this launch completes. A small
    // JSON read, on par with the job/model reads LocalStorage already does at startup.
    BenchmarkCatalog.load(benchmarkStore)
  }
}
