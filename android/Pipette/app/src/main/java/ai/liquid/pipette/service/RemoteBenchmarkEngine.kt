package ai.liquid.pipette.service

import ai.liquid.pipette.BenchmarkCooldownCallback
import ai.liquid.pipette.BenchmarkEngine
import ai.liquid.pipette.BenchmarkProgressBus
import ai.liquid.pipette.BenchmarkProgressCallback
import ai.liquid.pipette.BenchmarkThermalCallback
import ai.liquid.pipette.CPUSET_TAG
import ai.liquid.pipette.CpuAffinityProbe
import ai.liquid.pipette.CpuAffinitySnapshot
import ai.liquid.pipette.CpusetDemotion
import ai.liquid.pipette.LlamacppLoadMode
import ai.liquid.pipette.Readiness
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.util.Log
import dalvik.system.BaseDexClassLoader
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * UI-process [BenchmarkEngine] that runs the native engine in the isolated `:benchmark` process ([PipetteBenchmarkService]). Each `*Sync` call is a
 * blocking two-way AIDL round-trip on the caller's (JobRunner executor) thread — so `JobRunner` is unchanged and binder surfaces service death as a
 * thrown exception the cell loop already handles.
 *
 * Lifecycle is driven entirely from here: bind lazily on the first op; after an [unloadSync] (the job-end signal) with no follow-up op within
 * [IDLE_GRACE_MS], tear the service down so the model's process dies and its native heap is reclaimed. The between-cell readiness gate still runs in
 * this process (it reads device-global signals and never loads a model); only the model engine is isolated.
 */
class RemoteBenchmarkEngine(
  context: Context,
  /**
   * Invoked (on the control thread) whenever the `:benchmark` process (re)connects. The main process uses this to drain native crash envelopes: a
   * prior run's crash is flushed to disk by sentry-native during *this* respawn's init, i.e. just before we connect — so connect is the right moment
   * to forward it. Defaults to a no-op (tests, in-process wiring); [AppContainer] supplies the real drain. See
   * [ai.liquid.pipette.CrashEnvelopeForwarder].
   */
  private val onBenchmarkProcessConnected: () -> Unit = {},
  /**
   * The debug thermal-gate waiver (PIP-434) to hand the `:benchmark` process, read at each service start. That process runs its own per-rep readiness
   * gate and cannot read the setting itself (DataStore is single-process), so the Intent that already starts the service carries it.
   */
  private val skipThermalGate: () -> Boolean = { false },
  /**
   * The readiness deadline (`benchmark_flags.readiness.max_wait_secs`) to hand that same process, on the same Intent and for the same reason: its
   * per-rep gate is the one this budget governs, and it cannot read the run's flags itself.
   */
  private val readinessMaxWaitMillis: () -> Long = { Readiness.COOLDOWN_MAX_MILLIS },
) : BenchmarkEngine {
  private val appContext = context.applicationContext
  private val intent = Intent(appContext, PipetteBenchmarkService::class.java)

  private val controlThread = HandlerThread("RemoteEngineControl").apply { start() }
  private val controlHandler = Handler(controlThread.looper)

  private val lock = Any()
  @Volatile private var service: IBenchmarkService? = null
  private var bound = false
  private var connectLatch: CountDownLatch? = null
  @Volatile private var cachedCommit: String? = null
  // The active CPU backend is constant once the backend registers (first model
  // load), so cache the first non-null read to survive a later :benchmark death.
  @Volatile private var cachedCpuBackend: String? = null
  // The last cpuset reading written to logcat, so the side-by-side line is emitted
  // on a change rather than once per cell. Not a cache: nothing reads it back.
  @Volatile private var lastLoggedCpuAffinity: CpuAffinitySnapshot? = null
  // The waiver the live :benchmark process was started with, or null when none is
  // running. Only the start Intent carries it, so a toggle flipped afterwards
  // cannot reach that process; this is what lets ensureBound notice.
  @Volatile private var deliveredSkipThermalGate: Boolean? = null

  @Volatile private var deliveredReadinessMaxWaitMillis: Long? = null

  private val idleTeardown = Runnable { teardownNow("idle after unload") }
  private val hardCancel = Runnable { teardownNow("hard cancel: decode ignored cooperative cancel") }

  private val connection =
    object : ServiceConnection {
      override fun onServiceConnected(name: ComponentName?, binder: IBinder) {
        val svc = IBenchmarkService.Stub.asInterface(binder)
        runCatching { binder.linkToDeath(deathRecipient, 0) }
        synchronized(lock) {
          service = svc
          connectLatch?.countDown()
        }
        // Register the reverse cancel channel so BenchmarkActivity's Cancel (in
        // the :benchmark process) can cancel the whole job in this process.
        controlHandler.post { runCatching { svc.setJobCancelCallback(jobCancelCallbackStub) } }
        // Cache the runtime commit eagerly off the main thread so writePayload
        // records the real SHA even if :benchmark dies before it asks again.
        if (cachedCommit == null) {
          controlHandler.post {
            runCatching { svc.llamaCppCommit() }.getOrNull()?.takeUnless { it.startsWith(NATIVE_PREFIX) }?.let { cachedCommit = it }
          }
        }
        // A prior run's native crash is flushed to disk during this respawn's init (which completes before the binder is returned), and :benchmark
        // shares the same filesDir, so a single drain here always sees it; anything missed is caught by the next launch/connect drain. Guarded so a
        // crash-reporting failure can't take down the connection, but logged so it isn't diagnosed blind.
        controlHandler.post { runCatching { onBenchmarkProcessConnected() }.onFailure { Log.w(TAG, "benchmark crash-report drain failed", it) } }
      }

      override fun onServiceDisconnected(name: ComponentName?) {
        // The service process vanished. Tear down fully (unbind, clear
        // `bound`, release any in-flight `connectLatch`) so the proxy doesn't
        // believe it's still bound and an ensureBound() waiter fails fast
        // instead of parking the whole timeout. Idempotent.
        teardownNow("service disconnected")
      }
    }

  private val deathRecipient =
    IBinder.DeathRecipient {
      // Process died out from under us (OOM kill, or our own hard-cancel
      // teardown). Any in-flight two-way call has already thrown; tear down so
      // `bound`/`connectLatch` stay consistent (a bound waiter fails fast) and
      // the next op cleanly re-binds. Idempotent if a teardown already ran.
      teardownNow("benchmark process died")
    }

  // Whether the native engine is packaged — resolved via the app classloader
  // WITHOUT dlopen'ing it (only `:benchmark` loads the engine). findLibrary is a
  // cheap path lookup that handles `extractNativeLibs=false` (the .so stays
  // inside the APK) and split APKs, with no APK scan — safe on the main thread.
  private val nativeLibPresent: Boolean by lazy { (appContext.classLoader as? BaseDexClassLoader)?.findLibrary(NATIVE_LIB_NAME) != null }

  override val isAvailable: Boolean
    get() = nativeLibPresent

  override fun llamaCppCommit(): String {
    cachedCommit?.let {
      return it
    }
    val svc = service ?: return PENDING_COMMIT
    return runCatching { svc.llamaCppCommit().also { cachedCommit = it } }.getOrDefault(PENDING_COMMIT)
  }

  override fun cpuBackendDescriptor(): String? {
    cachedCpuBackend?.let {
      return it
    }
    // Don't bind just to ask — it's read at payload time, after a run, when
    // the service is already up; null (the optional field omitted) otherwise.
    val svc = service ?: return null
    return runCatching { svc.cpuBackendDescriptor() }.getOrNull()?.also { cachedCpuBackend = it }
  }

  /**
   * Deliberately uncached, like [lastRunInterruption] below and unlike [llamaCppCommit] / [cpuBackendDescriptor] above.
   *
   * This used to cache the first successful read for the life of the `:benchmark` process, reasoning that the framework fixes a process's cpuset
   * group when it assigns it. That is wrong, and it is wrong in the exact way this branch is about: the group moves with process state, and a
   * demotion is that move. The cache made every cell report the first cell's reading, so a cell measured on-device in `/moderate` with both prime
   * cores gone wrote `/foreground-boost`, `0-7` and `excludes_top_tier=false` to its payload. That is worse than omitting the fields, because it
   * reads as positive evidence the cell ran clean.
   *
   * Read at payload time, which is a fair account of where the cell ran: `JobRunner` throws on [lastRunInterruption] before it reaches this call, so
   * every payload written is a cell the guard cleared. It earns its keep in the case where the guard fails open (an unbound service, see
   * [lastRunInterruption]), where this now reports the group the cell ended in rather than one from cells ago.
   *
   * Null when nothing is bound; the payload omits the fields rather than guessing.
   */
  override fun benchmarkProcessCpuAffinity(): CpuAffinitySnapshot? {
    val svc = service ?: return null
    return runCatching { svc.benchmarkProcessCpuAffinity() }
      .getOrNull()
      ?.let { CpuAffinitySnapshot.fromJson(it) }
      ?.also { snapshot ->
        // The side-by-side, this main/top-app process against the :benchmark process inference ran in:
        // `adb logcat -s pipette-cpuset`. Emitted on a change rather than per cell, so the reading moving
        // (the thing worth seeing) stands out instead of being buried in one identical line per cell.
        if (snapshot != lastLoggedCpuAffinity) {
          lastLoggedCpuAffinity = snapshot
          Log.i(CPUSET_TAG, "[main] ${CpuAffinityProbe.snapshot().summary()}")
          Log.i(CPUSET_TAG, "[:benchmark] ${snapshot.summary()}")
        }
      }
  }

  /**
   * Deliberately uncached, for a stronger reason than [benchmarkProcessCpuAffinity] above. That one is uncached because a cached reading would be
   * stale; this one would be actively harmful cached, since a retained verdict would attribute the cell that was interrupted to every cell after it.
   * The caches that do exist here ([llamaCppCommit], [cpuBackendDescriptor]) hold values that genuinely cannot change under a live process.
   *
   * An unbound service or an unreadable payload returns null, which fails open: the cell is then judged on its own result rather than failed on a
   * reading we do not have. What keeps that from meaning "one dropped reading and the job silently continues forever" is that `CellDemotionWatch`
   * carries the best state it has seen into the next cell's baseline, so a demotion that persists is still a demotion against that baseline and the
   * next cell fires on it. Without that carry this would be a hole, not a safe default.
   *
   * The retention is per `:benchmark` process, so it does not survive that process being torn down after a job ends. A job resumed later starts a
   * fresh baseline, which is sound because a resume happens with the pocket screen coming back to the foreground.
   */
  override fun lastRunInterruption(): CpusetDemotion? {
    val svc = service ?: return null
    return runCatching { svc.lastRunInterruption() }.getOrNull()?.let { CpusetDemotion.fromJson(it) }
  }

  /**
   * Never binds to deliver this. If nothing is bound the process is already gone, and with it the baseline this would have cleared, so binding purely
   * to say "forget your state" would spawn a process to tell it to forget nothing.
   */
  override fun endJob() {
    val svc = service ?: return
    // Logged, unlike the other best-effort pushes here, because a persistent failure is silent and consequential: the baseline survives the job
    // boundary and the next job's first cell fails on a state it never reached. A wrong verdict leaves no other trace to find it by.
    runCatching { svc.endJob() }.onFailure { Log.w(TAG, "endJob failed; :benchmark may carry a stale cpuset baseline", it) }
  }

  // @Volatile: set once on the main thread (AppContainer.init) but read on the
  // controlHandler thread from jobCancelCallbackStub — matches the sibling caches.
  @Volatile override var onJobCancelRequested: (() -> Unit)? = null

  // The :benchmark process invokes this (on a binder thread) when BenchmarkActivity's
  // Cancel is tapped; hop to the control thread and cancel the whole job.
  private val jobCancelCallbackStub =
    object : IJobCancelCallback.Stub() {
      override fun onCancelRequested() {
        controlHandler.post { runCatching { onJobCancelRequested?.invoke() } }
      }
    }

  override fun publishJobProgress(progress: BenchmarkProgressBus.Progress) {
    // Fire-and-forget to the pocket UI (as a JSON snapshot); skip silently if
    // :benchmark isn't bound (nothing to show), and never let a push fault the run.
    val svc = service ?: return
    runCatching { svc.updateJobProgress(progress.toJson()) }
  }

  override fun loadSync(modelPath: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int, loadMode: LlamacppLoadMode) {
    val result = ensureBound().loadModel(modelPath, nGpuLayers, contextSize, nUbatch, loadMode.cValue)
    check(result.ok) { result.errorMessage ?: "Model load failed in :benchmark" }
  }

  override fun runBenchmarkSync(
    benchmarkJson: String,
    nGpuLayers: Int,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    // Intentionally not forwarded: a UI-process cooldown callback can't be
    // invoked from the native run inside :benchmark. The mid-run readiness gate
    // runs service-side (PipetteBenchmarkService's own Readiness); only the
    // between-cell gate uses JobRunner's injected ReadinessGate. Kept to satisfy
    // the BenchmarkEngine contract the in-process EngineActor/tests rely on.
    @Suppress("UNUSED_PARAMETER") cooldown: BenchmarkCooldownCallback?,
    // Also not forwarded (same reason as cooldown): the thermal sampler reads
    // PowerManager service-side, so PipetteBenchmarkService builds its own and
    // passes it to the in-process EngineActor. Kept to satisfy the contract.
    @Suppress("UNUSED_PARAMETER") thermal: BenchmarkThermalCallback?,
  ): String {
    val svc = ensureBound()
    try {
      return resolveOrThrow(svc.runBenchmark(benchmarkJson, nGpuLayers, mmprojPath, runCallback(progress)))
    } finally {
      controlHandler.removeCallbacks(hardCancel)
    }
  }

  override fun runFreshSync(
    benchmarkJson: String,
    modelPath: String,
    nGpuLayers: Int,
    contextSize: Int,
    nUbatch: Int,
    loadMode: LlamacppLoadMode,
    mmprojPath: String?,
    progress: BenchmarkProgressCallback?,
    @Suppress("UNUSED_PARAMETER") cooldown: BenchmarkCooldownCallback?,
    @Suppress("UNUSED_PARAMETER") thermal: BenchmarkThermalCallback?,
  ): String {
    val svc = ensureBound()
    try {
      return resolveOrThrow(
        svc.runBenchmarkFresh(benchmarkJson, modelPath, nGpuLayers, contextSize, nUbatch, loadMode.cValue, mmprojPath, runCallback(progress))
      )
    } finally {
      controlHandler.removeCallbacks(hardCancel)
    }
  }

  override fun unloadSync() {
    try {
      val svc = service ?: return
      val result = svc.unloadModel()
      check(result.ok) { result.errorMessage ?: "Model unload failed in :benchmark" }
    } finally {
      // Unload is the job-end signal — always schedule teardown, even when
      // the handle is already null (a disconnect/death may have dropped it
      // while a started FGS lingers). If no new op arrives within the grace
      // window the service is torn down; a mid-job error-unload is followed
      // immediately by the next cell's load, which cancels this. teardownNow
      // is a no-op when there's genuinely nothing bound or started.
      scheduleIdleTeardown()
    }
  }

  override fun destroy() {
    teardownNow("engine destroyed")
    controlThread.quitSafely()
  }

  private fun ensureBound(): IBenchmarkService {
    controlHandler.removeCallbacks(idleTeardown)
    // A process that outlived its job (IDLE_GRACE_MS) is still configured by the
    // Intent that started it, so a waiver flipped since then never reached it.
    // Drop it and rebind rather than gate on the stale value: this runs between
    // cells, never inside one, so the cost is one cold start at a cell boundary.
    if (service != null && deliveredSkipThermalGate != skipThermalGate()) {
      teardownNow("thermal waiver changed")
    }
    // Same staleness argument as the waiver above: the Intent configured the process once, and a run whose flags carry a different deadline must not
    // be measured under the previous run's budget.
    if (service != null && deliveredReadinessMaxWaitMillis != readinessMaxWaitMillis()) {
      teardownNow("readiness deadline changed")
    }
    service?.let {
      return it
    }

    val latch = CountDownLatch(1)
    synchronized(lock) {
      service?.let {
        return it
      }
      connectLatch = latch
      // Read here rather than once at construction, and remembered, because this
      // Intent is the only channel to that process: it reads the extra in
      // onStartCommand and cannot see the setting itself (DataStore is
      // single-process). The guard above turns a stale value into a rebind.
      val waiver = skipThermalGate()
      intent.putExtra(PipetteBenchmarkService.EXTRA_SKIP_THERMAL_GATE, waiver)
      deliveredSkipThermalGate = waiver
      val deadline = readinessMaxWaitMillis()
      intent.putExtra(PipetteBenchmarkService.EXTRA_READINESS_MAX_WAIT_MILLIS, deadline)
      deliveredReadinessMaxWaitMillis = deadline
      // Best-effort FGS so the run survives backgrounding. If the app isn't in
      // an allowed state (Android 12+ background-start limits) this throws; we
      // still bind, so the run works while the app is foreground (it must be,
      // to have started the job) — just without the FGS guarantee.
      runCatching { appContext.startForegroundService(intent) }
      bound = appContext.bindService(intent, connection, Context.BIND_AUTO_CREATE)
      if (!bound) {
        // bindService refused; undo the startForegroundService side and reset.
        teardownNow("bind request rejected")
        error("Failed to bind :benchmark service")
      }
    }
    // await OUTSIDE the lock so onServiceConnected (or a racing teardown) can
    // take the lock to release us. A teardown counts the latch down too, so we
    // fail fast here instead of parking the full timeout.
    val connected = latch.await(BIND_TIMEOUT_MS, TimeUnit.MILLISECONDS)
    val svc = service
    if (svc == null) {
      // Timed out, or a teardown/death dropped the connection mid-bind. Reset
      // binding state (else the next ensureBound double-binds the same
      // connection and leaks a ref so the service is never reclaimed).
      teardownNow(if (connected) "connection dropped during bind" else "bind timed out")
      error("Could not connect to :benchmark service")
    }
    return svc
  }

  private fun runCallback(progress: BenchmarkProgressCallback?): IBenchmarkRunCallback.Stub =
    object : IBenchmarkRunCallback.Stub() {
      override fun onProgress(completed: Int, total: Int, message: String) {
        val proceed = progress?.onProgress(completed, total, message) ?: true
        if (!proceed) {
          runCatching { service?.requestCancel() }
          // Cooperative cancel may never land mid-decode; arm the hard
          // kill so cancellation is bounded even then.
          controlHandler.removeCallbacks(hardCancel)
          controlHandler.postDelayed(hardCancel, CANCEL_GRACE_MS)
        }
      }
    }

  private fun resolveOrThrow(result: BenchmarkResult): String {
    check(result.ok) { result.errorMessage ?: "Benchmark failed in :benchmark" }
    return ResultSpill.resolve(result)
  }

  private fun scheduleIdleTeardown() {
    controlHandler.removeCallbacks(idleTeardown)
    controlHandler.postDelayed(idleTeardown, IDLE_GRACE_MS)
  }

  private fun teardownNow(reason: String) {
    synchronized(lock) {
      controlHandler.removeCallbacks(idleTeardown)
      controlHandler.removeCallbacks(hardCancel)
      // Release anyone parked in ensureBound on this bind so they fail fast
      // (service is about to be null) instead of waiting out the timeout.
      connectLatch?.countDown()
      connectLatch = null
      if (!bound && service == null) return
      Log.i(TAG, "tearing down :benchmark ($reason)")
      service?.let { runCatching { it.asBinder().unlinkToDeath(deathRecipient, 0) } }
      if (bound) {
        runCatching { appContext.unbindService(connection) }
        bound = false
      }
      service = null
      // The cpuset is read fresh per cell, so there is no cache to invalidate here.
      // `lastLoggedCpuAffinity` is deliberately left alone: it only suppresses a repeated
      // log line, and a respawn into the same group has nothing new to say.

      // No process to hold a waiver any more; the next start delivers one afresh.
      deliveredSkipThermalGate = null
      deliveredReadinessMaxWaitMillis = null
      // Stop the started (foreground) component; with no clients bound the
      // service is destroyed → its onDestroy kills the process.
      runCatching { appContext.stopService(intent) }
    }
  }

  companion object {
    private const val TAG = "pipetteRemoteEngine"
    private const val NATIVE_LIB_NAME = "pipette_android"
    private const val NATIVE_PREFIX = "native-"
    private const val PENDING_COMMIT = "native-pending"
    private const val BIND_TIMEOUT_MS = 10_000L
    private const val IDLE_GRACE_MS = 5_000L
    private const val CANCEL_GRACE_MS = 12_000L
  }
}
