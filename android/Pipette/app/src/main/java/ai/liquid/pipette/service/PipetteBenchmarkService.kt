package ai.liquid.pipette.service

import ai.liquid.pipette.AndroidThermalStatusProvider
import ai.liquid.pipette.BenchmarkCooldownCallback
import ai.liquid.pipette.BenchmarkProgressBus
import ai.liquid.pipette.BenchmarkProgressCallback
import ai.liquid.pipette.BenchmarkThermalCallback
import ai.liquid.pipette.COOLING_PROGRESS_TOTAL
import ai.liquid.pipette.CancelFlag
import ai.liquid.pipette.CellDemotionWatch
import ai.liquid.pipette.CpuAffinityProbe
import ai.liquid.pipette.EngineActor
import ai.liquid.pipette.LlamacppLoadMode
import ai.liquid.pipette.Readiness
import ai.liquid.pipette.ReadinessOutcome
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.Process
import android.util.Log

/**
 * Hosts the native benchmark engine in the isolated `:benchmark` process. This is where `libpipette_android.so` is loaded and where a model's
 * multi-GB native heap lives — so when the proxy tears the service down at the end of a job, [onDestroy] kills this process and the OS reclaims every
 * byte the model held (the whole point of the isolation).
 *
 * The AIDL surface mirrors `BenchmarkEngine`: load/run/unload are blocking two-way calls served on binder threads (each parks on [EngineActor]'s
 * worker, exactly as the in-process path did), with progress streamed back over a oneway callback. Cancellation flips a per-run [CancelFlag] the
 * engine's progress/cooldown shims honor; the proxy hard-kills this process (via clean teardown) if an uninterruptible decode ignores the flag.
 */
class PipetteBenchmarkService : Service() {
  private val engineActor by lazy { EngineActor() }
  private val thermalProvider by lazy { AndroidThermalStatusProvider(this) }
  private val readiness by lazy {
    // Gate on the *cached* headroom (not a raw read) so the per-rep telemetry
    // sampler's `before` sees the same warm gate-pass value: cachedHeadroom
    // throttles to ~1 Hz and holds the last good reading, so back-to-back reads
    // (gate release → `before` sample) never rate-limit to NaN.
    Readiness(
      thermalProvider::currentStatus,
      thermalProvider::cachedHeadroom,
      skipThermal = { skipThermalGate },
      maxWaitMillis = { readinessMaxWaitMillis },
    )
  }

  /**
   * The debug thermal-gate waiver (PIP-434), delivered by the proxy on the start Intent because this process cannot read the setting itself
   * (DataStore is single-process).
   *
   * Reading it in [onStartCommand] is safe despite that callback racing [onBind]: this is only consulted from a run's cooldown callback, long after
   * both. It stays false if the foreground start is refused outright and [onStartCommand] never runs, which fails closed: the gate is enforced.
   */
  @Volatile private var skipThermalGate = false

  /**
   * The per-rep gate's wait budget, delivered on the start Intent like [skipThermalGate] and for the same reason. Defaults to the shared
   * [Readiness.COOLDOWN_MAX_MILLIS], which is what an absent `max_wait_secs` means.
   */
  @Volatile private var readinessMaxWaitMillis = Readiness.COOLDOWN_MAX_MILLIS

  @Volatile private var currentCancelFlag: CancelFlag? = null

  /**
   * The per-core `scaling_max_freq` table, read once and reused by every poll.
   *
   * Hoisted out of the watch because it is the expensive part of a sample, by a wide margin; [CpuAffinityProbe] carries the measured numbers and the
   * attribution, and they are deliberately not repeated here. Core topology does not change while the process lives, so one read is enough.
   */
  private val freqSweep by lazy { CpuAffinityProbe.freqSweep() }

  /**
   * Watches this process's cpuset for the length of a cell, load included, and holds the verdict the proxy reads back afterwards. There is no way to
   * hand that verdict through a run's return value: the run may equally have thrown, have returned a partial, or have completed in the window between
   * the demotion and the cancel taking effect, and all three are invalid measurements.
   *
   * The sampler reuses [freqSweep], which is why it is built here and not inside the watch.
   */
  private val demotionWatch by lazy { CellDemotionWatch(probe = { CpuAffinityProbe.sample(freqSweep) }) }

  @Suppress("TooManyFunctions") // mirrors the IBenchmarkService AIDL surface
  private val binder =
    object : IBenchmarkService.Stub() {
      override fun llamaCppCommit(): String = engineActor.llamaCppCommit()

      override fun cpuBackendDescriptor(): String? = engineActor.cpuBackendDescriptor()

      // Read in THIS (:benchmark) process so it reflects the scheduling group
      // inference actually runs under. Serialized as JSON for the binder hop.
      override fun benchmarkProcessCpuAffinity(): String = engineActor.benchmarkProcessCpuAffinity().toJson()

      /**
       * The demotion that invalidated the last cell, or null. A **destructive** read: the first caller takes the verdict and clears it, so one
       * demotion explains at most one cell and cannot leak into the next.
       */
      override fun lastRunInterruption(): String? = demotionWatch.consumeVerdict()?.toJson()

      override fun isAvailable(): Boolean = engineActor.isAvailable

      override fun loadModel(modelPath: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int, loadMode: Int): BenchmarkResult = guard {
        // A load opens a new cell: it starts the watch (the load is the window a screen-off most often falls in) and clears any previous verdict, so
        // a resume inside RemoteBenchmarkEngine's idle grace cannot file a load failure as the previous job's demotion. Guarded like every other part
        // of this guard: failing to arm a diagnostic must not fail the load, which would poison failedModelPaths for the whole model.
        runCatching { demotionWatch.beginCell { currentCancelFlag?.cancel() } }
        engineActor.loadSync(modelPath, nGpuLayers, contextSize, nUbatch, LlamacppLoadMode.fromCValue(loadMode))
        BenchmarkResult.loaded(HANDLE)
      }

      override fun runBenchmark(benchmarkJson: String, nGpuLayers: Int, mmprojPath: String?, callback: IBenchmarkRunCallback): BenchmarkResult =
        guard {
          val flag = beginRun()
          try {
            val json =
              engineActor.runBenchmarkSync(
                benchmarkJson,
                nGpuLayers,
                mmprojPath,
                progressShim(callback, flag),
                cooldownShim(callback, flag),
                thermalShim(),
              )
            ResultSpill.packageResult(this@PipetteBenchmarkService, json)
          } finally {
            currentCancelFlag = null
          }
        }

      override fun runBenchmarkFresh(
        benchmarkJson: String,
        modelPath: String,
        nGpuLayers: Int,
        contextSize: Int,
        nUbatch: Int,
        loadMode: Int,
        mmprojPath: String?,
        callback: IBenchmarkRunCallback,
      ): BenchmarkResult = guard {
        // Same as loadModel: this call owns the load, so the cell has to open before it rather than inside beginRun, which would leave the load
        // unguarded and take its baseline after the window it is meant to cover.
        runCatching { demotionWatch.beginCell { currentCancelFlag?.cancel() } }
        val flag = beginRun()
        try {
          val json =
            engineActor.runFreshSync(
              benchmarkJson,
              modelPath,
              nGpuLayers,
              contextSize,
              nUbatch,
              LlamacppLoadMode.fromCValue(loadMode),
              mmprojPath,
              progressShim(callback, flag),
              cooldownShim(callback, flag),
              thermalShim(),
            )
          ResultSpill.packageResult(this@PipetteBenchmarkService, json)
        } finally {
          currentCancelFlag = null
        }
      }

      override fun unloadModel(): BenchmarkResult = guard {
        // Close the cell but KEEP the baseline: an unload is not a job boundary. JobRunner unloads from its per-cell error path as well as after the
        // loop, and a mid-job unload is followed immediately by the next cell's load. Dropping the baseline here would let that next cell take its
        // own first sample as normal, so a process already demoted when the previous cell failed would look healthy and be measured on in silence.
        // [endJob] is the call that clears it.
        runCatching { demotionWatch.stop() }
        engineActor.unloadSync()
        BenchmarkResult.ok()
      }

      override fun endJob() {
        runCatching { demotionWatch.closeJob() }
      }

      override fun requestCancel() {
        currentCancelFlag?.cancel()
      }

      // Job-level progress pushed from main → forward to BenchmarkActivity's UI.
      override fun updateJobProgress(progressJson: String?) {
        val progress = progressJson?.let { BenchmarkProgressBus.Progress.fromJson(it) } ?: return
        BenchmarkProgressBus.publish(progress)
      }

      // Register the reverse cancel channel; BenchmarkActivity's Cancel invokes it
      // to cancel the whole job in the main process.
      override fun setJobCancelCallback(cb: IJobCancelCallback?) {
        BenchmarkProgressBus.setCancelAction(cb?.let { callback -> { runCatching { callback.onCancelRequested() } } })
      }
    }

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    // Default to the current value, not to false: a redelivered or extra-less
    // start must not silently re-enforce a gate the running job was waived for.
    skipThermalGate = intent?.getBooleanExtra(EXTRA_SKIP_THERMAL_GATE, skipThermalGate) ?: skipThermalGate
    // Same "default to the current value" rule as the waiver: an extra-less redelivery must not silently reset a running job's deadline.
    readinessMaxWaitMillis = intent?.getLongExtra(EXTRA_READINESS_MAX_WAIT_MILLIS, readinessMaxWaitMillis) ?: readinessMaxWaitMillis
    // The proxy starts us with startForegroundService(), so we must call
    // startForeground() promptly or the OS kills the app with "did not then
    // call Service.startForeground()". If the promotion is refused
    // (background-start limits, or the SPECIAL_USE type isn't honored), don't
    // just swallow it — drop the started-service reference with
    // stopSelf(startId) so the contract is satisfied, and keep running purely
    // as a bound service (the proxy's BIND_AUTO_CREATE binding keeps us alive
    // while the client app is foreground).
    val promoted =
      runCatching {
          // SPECIAL_USE is an API-34 foreground-service type; on API 31–33 use
          // the 2-arg startForeground rather than an unrecognized type constant.
          if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, buildNotification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
          } else {
            startForeground(NOTIFICATION_ID, buildNotification())
          }
        }
        .onFailure { Log.w(TAG, "startForeground refused; continuing as a bound service", it) }
        .isSuccess
    if (!promoted) stopSelf(startId)
    // Don't auto-restart if the OS kills us: the proxy re-binds on demand and
    // the job manifest drives recovery.
    return START_NOT_STICKY
  }

  override fun onBind(intent: Intent?): IBinder = binder

  override fun onDestroy() {
    // Reached via the proxy's clean teardown (unbind + stopService) at job end
    // or its hard-cancel path. Free the model, then kill the process so the
    // native heap is reclaimed immediately rather than lingering in a cached,
    // emptied process.
    runCatching { engineActor.destroy() }
    super.onDestroy()
    Process.killProcess(Process.myPid())
  }

  private fun beginRun(): CancelFlag {
    // Open the cell here only if nothing else already did. loadModel and runBenchmarkFresh own their loads and arm before them; re-arming now would
    // not lose the baseline (beginCell seeds the new tracker from the old one's best), but it would clear a verdict already recorded during the load
    // and not yet collected, and it would reset the confirmation streak, restarting the two-poll detection window in the middle of the cell.
    //
    // This still has to run for a run whose cell was never opened, which is what a `beginCell` that threw inside `loadModel`'s `runCatching` leaves
    // behind: arming is best-effort there so a diagnostic cannot fail a load, and this is where that miss is picked up.
    //
    // The watch takes its own baseline sample as it opens, which is also the `[:benchmark]` diagnostic line a device run reads back with
    // `adb logcat -s pipette-cpuset`, so there is no separate probe to log here.
    if (!demotionWatch.isWatching && demotionWatch.lastDemotion == null) {
      runCatching { demotionWatch.beginCell { currentCancelFlag?.cancel() } }
    }
    val flag = CancelFlag().also { currentCancelFlag = it }
    // Pre-cancel a run that is starting into an already-demoted process. The watch fires once per cell, so a demotion confirmed during the load has
    // already spent its callback on a flag this run does not own; without this the run would measure to completion under the demoted cpuset.
    if (demotionWatch.lastDemotion != null) flag.cancel()
    return flag
  }

  private fun progressShim(callback: IBenchmarkRunCallback, flag: CancelFlag): BenchmarkProgressCallback =
    BenchmarkProgressCallback { completed, total, message ->
      runCatching { callback.onProgress(completed, total, message) }
      !flag.isCancelled
    }

  private fun cooldownShim(callback: IBenchmarkRunCallback, flag: CancelFlag): BenchmarkCooldownCallback = BenchmarkCooldownCallback {
    // The readiness gate runs here, in-process with the engine, reading
    // device-global signals. Cooling status rides the progress channel tagged
    // with the COOLING sentinel total (COOLING_PROGRESS_TOTAL) so the UI shows
    // it — and drives the cooling wash/timer — without a fraction jump.
    // `completed` carries this gate invocation's elapsed millis: a cooldown has no fraction, so the
    // field is otherwise unused, and JobRunner needs the elapsed to anchor the UI's countdown to
    // THIS wait rather than to the first cooling status of the run.
    val outcome =
      readiness.waitUntilReady(flag) { text, elapsed -> runCatching { callback.onProgress(elapsed.toInt(), COOLING_PROGRESS_TOTAL, text) } }
    // The gate's verdict rather than `!flag.isCancelled`, so a device that never cooled fails the cell instead of contributing throttled numbers
    // (PIP-143). A cancel that lands during the wait outranks a timeout: cancelling is the user's instruction, and the cell should read CANCELLED
    // rather than FAILED.
    val reported = if (flag.isCancelled) ReadinessOutcome.Cancelled else outcome
    // Logged off `reported`, never off the raw `outcome`, so the log cannot announce a timeout the caller was never told about. The line lives here
    // rather than in JobRunner because this is the process the gate actually ran in.
    if (reported is ReadinessOutcome.TimedOut) Log.w(TAG, "readiness timed out before a measured rep: ${reported.observed}")
    reported.encode()
  }

  private fun thermalShim(): BenchmarkThermalCallback =
    object : BenchmarkThermalCallback {
      // Per-rep thermal telemetry, sampled in-process off the same provider the
      // readiness gate uses — so a rep's `before` sample is the gate-pass reading
      // the gate just took. cachedHeadroom holds the last good value and returns
      // NaN only until the first real reading (unsupported API, cold start); the
      // native side maps NaN / an out-of-range status to a missing sample and
      // never cancels. (Per-sensor HAL temperatures are read natively from
      // `dumpsys`.)
      override fun sampleHeadroom(): Float = thermalProvider.cachedHeadroom()

      override fun sampleStatus(): Int = thermalProvider.currentStatus()
    }

  private inline fun guard(block: () -> BenchmarkResult): BenchmarkResult =
    try {
      block()
    } catch (t: Throwable) {
      Log.w(TAG, "engine op failed", t)
      BenchmarkResult.failure(t.message ?: t.javaClass.simpleName)
    }

  private fun buildNotification(): Notification {
    val manager = getSystemService(NotificationManager::class.java)
    if (manager.getNotificationChannel(CHANNEL_ID) == null) {
      manager.createNotificationChannel(
        NotificationChannel(CHANNEL_ID, "Benchmark", NotificationManager.IMPORTANCE_LOW).apply {
          description = "Active model benchmark run"
          setShowBadge(false)
        }
      )
    }
    return Notification.Builder(this, CHANNEL_ID)
      .setContentTitle("Pipette")
      .setContentText("Running benchmark")
      .setSmallIcon(android.R.drawable.stat_sys_download)
      .setOngoing(true)
      .build()
  }

  companion object {
    /** Boolean start-Intent extra carrying the debug thermal-gate waiver into this process. See [skipThermalGate]. */
    const val EXTRA_SKIP_THERMAL_GATE = "ai.liquid.pipette.extra.SKIP_THERMAL_GATE"

    /** Long start-Intent extra carrying the run's readiness deadline in millis into this process. See [readinessMaxWaitMillis]. */
    const val EXTRA_READINESS_MAX_WAIT_MILLIS = "ai.liquid.pipette.extra.READINESS_MAX_WAIT_MILLIS"

    private const val TAG = "pipetteBenchSvc"
    private const val CHANNEL_ID = "pipette_benchmark"
    private const val NOTIFICATION_ID = 0x4243 // 'BC'
    private const val HANDLE = 1L
  }
}
