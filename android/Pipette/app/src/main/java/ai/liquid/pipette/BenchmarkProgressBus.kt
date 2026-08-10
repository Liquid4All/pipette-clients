package ai.liquid.pipette

import android.os.Handler
import android.os.Looper
import org.json.JSONObject

/**
 * In-process (`:benchmark`) bridge between the main-process job orchestration and [BenchmarkActivity]'s pocket-mode UI. `PipetteBenchmarkService`
 * receives the job snapshot over AIDL (`updateJobProgress`, pushed from the main-process `JobRunner`) and forwards it here; the activity observes it
 * on the main thread and renders it to mirror the Compose `PocketModeScreen`. The activity's Cancel routes back through [requestCancel] → the
 * service's registered [IJobCancelCallback][ai.liquid.pipette.service.IJobCancelCallback] → the main-process `JobController`, cancelling the whole
 * job. No extra IPC or memory beyond the existing binding. (Thermal headroom is NOT carried here — the activity reads it locally in the `:benchmark`
 * process.)
 */
object BenchmarkProgressBus {
  /**
   * Whole-job pocket snapshot, mirroring the fields the Compose `PocketModeScreen` shows. [overallPermil] is the job fraction × 1000 (int for the
   * AIDL hop). [running] is false once the run ends (the activity finishes itself).
   */
  data class Progress(
    val title: String,
    val subtitle: String,
    val cellLabel: String,
    val statusText: String,
    val completedCells: Int,
    val totalCells: Int,
    val overallPermil: Int,
    val etaText: String,
    // Wall-clock (System.currentTimeMillis) when the readiness gate started cooling,
    // or null when not cooling. Carried (not just a bool) so the activity can tick a
    // live "Cooling m:ss / max" countdown like the Compose pocket. Both processes are
    // on the same device clock, so the elapsed math is valid across the AIDL hop.
    val coolingSinceMillis: Long?,
    val running: Boolean,
  ) {
    val cooling: Boolean
      get() = coolingSinceMillis != null

    fun toJson(): String =
      JSONObject()
        .apply {
          put("title", title)
          put("subtitle", subtitle)
          put("cell_label", cellLabel)
          put("status_text", statusText)
          put("completed_cells", completedCells)
          put("total_cells", totalCells)
          put("overall_permil", overallPermil)
          put("eta_text", etaText)
          put("cooling_since_millis", coolingSinceMillis ?: JSONObject.NULL)
          put("running", running)
        }
        .toString()

    companion object {
      fun fromJson(json: String): Progress? =
        runCatching {
            val o = JSONObject(json)
            Progress(
              title = o.optString("title"),
              subtitle = o.optString("subtitle"),
              cellLabel = o.optString("cell_label"),
              statusText = o.optString("status_text"),
              completedCells = o.optInt("completed_cells"),
              totalCells = o.optInt("total_cells"),
              overallPermil = o.optInt("overall_permil"),
              etaText = o.optString("eta_text"),
              coolingSinceMillis = if (o.isNull("cooling_since_millis")) null else o.optLong("cooling_since_millis"),
              running = o.optBoolean("running"),
            )
          }
          .getOrNull()
    }
  }

  private val mainHandler = Handler(Looper.getMainLooper())

  /** The pre-run "Starting…" snapshot re-delivered to a freshly-observing activity. */
  private fun startingSnapshot(): Progress =
    Progress(
      title = "Benchmark job",
      subtitle = "",
      cellLabel = "",
      statusText = "Starting…",
      completedCells = 0,
      totalCells = 0,
      overallPermil = 0,
      etaText = "calculating",
      coolingSinceMillis = null,
      running = true,
    )

  @Volatile private var latest = startingSnapshot()

  /**
   * Invalidate a stale *terminal* snapshot before a new run's activity observes. Called from [BenchmarkActivity]'s onCreate/onNewIntent (both run in
   * the `:benchmark` process): the `:benchmark` process is reused across jobs (the engine's teardown is idle-delayed and cancelled by the next job's
   * load), so without this a freshly-launched pocket for run N+1 could observe run N's retained `running=false` snapshot and finish() at once.
   *
   * Only clears a TERMINAL (`running=false`) snapshot — a running snapshot is the live state of the current run, so a mid-run reopen / foreground
   * (the manual "Open in Pocket Mode", or a bring-to-front) keeps showing real progress instead of resetting to "Starting…". Only sets `latest` (no
   * notify) — a currently-observing listener, if any, is unaffected.
   */
  fun markStarting() {
    if (!latest.running) latest = startingSnapshot()
  }

  /**
   * The retained snapshot, for a caller that has to re-populate freshly-built views (a fold/rotate re-inflate) without disturbing observer
   * registration. Reading it is not a substitute for [observe]: it delivers nothing on its own.
   */
  fun latestSnapshot(): Progress = latest

  @Volatile private var listener: ((Progress) -> Unit)? = null
  @Volatile private var cancelAction: (() -> Unit)? = null

  /** Called by the service when a job snapshot arrives from main. Fans out to the UI observer on the main thread. */
  fun publish(progress: Progress) {
    // Retain the snapshot — including a terminal (running=false) one — so an
    // activity that attaches AFTER the terminal was delivered (backgrounded at
    // finish, or a fast last-cell run that ends before onResume) still observes
    // it and finish()es rather than getting stuck on "Benchmarking in progress…".
    // A NEW run in a reused :benchmark process clears this stale terminal via
    // markStarting() at activity (re)launch, so it isn't re-delivered there.
    latest = progress
    // Re-read the listener when the Runnable actually runs, not at post time: if
    // the activity calls stopObserving() (onPause) between post and execution,
    // we must not deliver to a backgrounded observer.
    mainHandler.post { listener?.invoke(progress) }
  }

  /** Register the UI observer; immediately re-delivers the latest snapshot so a freshly-opened activity isn't blank. */
  fun observe(observer: (Progress) -> Unit) {
    listener = observer
    val snapshot = latest
    mainHandler.post { if (listener === observer) observer(snapshot) }
  }

  fun stopObserving() {
    listener = null
  }

  /** The service registers the job-cancel action here (invokes the reverse AIDL callback to main). */
  fun setCancelAction(action: (() -> Unit)?) {
    cancelAction = action
  }

  /** Invoked by the activity's Cancel button — cancels the whole job via the main process. */
  fun requestCancel() {
    cancelAction?.invoke()
  }
}
