package ai.liquid.pipette.service

import ai.liquid.pipette.AndroidThermalStatusProvider
import ai.liquid.pipette.BenchmarkCooldownCallback
import ai.liquid.pipette.BenchmarkProgressBus
import ai.liquid.pipette.BenchmarkProgressCallback
import ai.liquid.pipette.BenchmarkThermalCallback
import ai.liquid.pipette.COOLING_PROGRESS_TOTAL
import ai.liquid.pipette.CancelFlag
import ai.liquid.pipette.CpuAffinityProbe
import ai.liquid.pipette.EngineActor
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
    Readiness(thermalProvider::currentStatus, thermalProvider::cachedHeadroom, skipThermal = { skipThermalGate })
  }

  /**
   * The debug thermal-gate waiver (PIP-434), delivered by the proxy on the start Intent because this process cannot read the setting itself
   * (DataStore is single-process).
   *
   * Reading it in [onStartCommand] is safe despite that callback racing [onBind]: this is only consulted from a run's cooldown callback, long after
   * both. It stays false if the foreground start is refused outright and [onStartCommand] never runs, which fails closed: the gate is enforced.
   */
  @Volatile private var skipThermalGate = false

  @Volatile private var currentCancelFlag: CancelFlag? = null

  @Suppress("TooManyFunctions") // mirrors the IBenchmarkService AIDL surface
  private val binder =
    object : IBenchmarkService.Stub() {
      override fun llamaCppCommit(): String = engineActor.llamaCppCommit()

      override fun cpuBackendDescriptor(): String? = engineActor.cpuBackendDescriptor()

      // Read in THIS (:benchmark) process so it reflects the scheduling group
      // inference actually runs under. Serialized as JSON for the binder hop.
      override fun benchmarkProcessCpuAffinity(): String = engineActor.benchmarkProcessCpuAffinity().toJson()

      override fun isAvailable(): Boolean = engineActor.isAvailable

      override fun loadModel(modelPath: String, nGpuLayers: Int, contextSize: Int, nUbatch: Int): BenchmarkResult = guard {
        engineActor.loadSync(modelPath, nGpuLayers, contextSize, nUbatch)
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
        mmprojPath: String?,
        callback: IBenchmarkRunCallback,
      ): BenchmarkResult = guard {
        val flag = beginRun()
        try {
          val json =
            engineActor.runFreshSync(
              benchmarkJson,
              modelPath,
              nGpuLayers,
              contextSize,
              nUbatch,
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
        engineActor.unloadSync()
        BenchmarkResult.ok()
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
    // Diagnostic: log the :benchmark process's cpuset/affinity at run start so a
    // device run surfaces any OEM demotion via `adb logcat -s pipette-cpuset`.
    runCatching { Log.i(CPUSET_TAG, "[:benchmark] ${CpuAffinityProbe.snapshot().summary()}") }
    return CancelFlag().also { currentCancelFlag = it }
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

    private const val TAG = "pipetteBenchSvc"
    private const val CPUSET_TAG = "pipette-cpuset"
    private const val CHANNEL_ID = "pipette_benchmark"
    private const val NOTIFICATION_ID = 0x4243 // 'BC'
    private const val HANDLE = 1L
  }
}
