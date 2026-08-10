// UI code: dp/text-size/color literals + a multi-part view builder, same as MainActivity's pocket mode / JobsViewModel.
@file:Suppress("MagicNumber", "TooManyFunctions", "LongMethod")

package ai.liquid.pipette

import android.app.Activity
import android.content.Intent
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.Outline
import android.graphics.Typeface
import android.graphics.drawable.ClipDrawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.LayerDrawable
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.TextUtils
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewOutlineProvider
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import android.window.OnBackInvokedCallback
import android.window.OnBackInvokedDispatcher
import androidx.annotation.RequiresApi

/**
 * A "Pocket Mode" run screen hosted in the `:benchmark` process (`android:process=":benchmark"`, `launchMode=singleTask`). While it is the focused
 * top activity, the framework classifies the whole `:benchmark` process as `top-app` — the only cpuset group with the prime cores on OEM-throttled
 * devices (e.g. Samsung, where `:benchmark` otherwise lands in `/foreground [0-5]`, denied the 4.47 GHz cores). Because cpuset is per-process, the
 * engine's inference threads inherit `top-app` for free.
 *
 * Deliberately built with **classic Views, not Compose** — Compose would add ~30-60 MB to this model-hosting process. It mirrors the Compose
 * [PocketModeScreen][ai.liquid.pipette.compose.shell.PocketModeScreen]: app icon, a "Throttling headroom" chip (thermal read locally in this
 * process), and a card with job title/subtitle, an overall progress bar, cells-done + ETA, and the live current-cell/status line. The job snapshot
 * arrives from the main process via [BenchmarkProgressBus]. The one exit is "Pause benchmark", which cancels the whole job; the label says Pause
 * because a job stopped with cells still pending is left `PAUSED` and resumable from the Jobs tab, not discarded. `FLAG_KEEP_SCREEN_ON` holds
 * `top-app` for the run's duration, and Back is swallowed ([refuseBackExit]) so leaving is always deliberate. Any other exit would demote the cpuset
 * mid-run.
 *
 * Note it does NOT mirror the Compose pocket's slide-to-exit: that control leaves pocket mode with the job still running, which here would mean
 * finishing this activity and handing `:benchmark` back to a throttled cpuset for the rest of the run.
 *
 * Configuration changes are handled in-place ([onConfigurationChanged]) rather than by recreation, since on a foldable a fold or rotate would
 * otherwise destroy and rebuild the very activity pinning this process to `top-app`, several times a run.
 */
class BenchmarkActivity : Activity() {
  private var progressBar: ProgressBar? = null
  private var jobTitleLabel: TextView? = null
  private var subtitleLabel: TextView? = null
  private var cellsLabel: TextView? = null
  private var timeLabel: TextView? = null
  private var currentLabel: TextView? = null
  private var statusLabel: TextView? = null
  private var estTimeLabel: TextView? = null
  private var card: LinearLayout? = null
  private var thermalDot: View? = null
  private var thermalLabel: TextView? = null
  private var cancelButton: Button? = null

  // Latest status line inputs, so the 1s ticker can re-render a live cooldown
  // countdown ("Cooling m:ss / max") without another IPC push.
  @Volatile private var coolingSinceMillis: Long? = null
  @Volatile private var lastStatusText: String = "Running…"

  // Once the user pauses, pin the status line to "Pausing…" until the terminal
  // snapshot finishes us — otherwise the 1s ticker / next render overwrite it.
  @Volatile private var cancelling = false

  // Registered OnBackInvokedCallback (API 33+), held as Any? so the field's type doesn't drag an
  // API-33 class into verification on the API 31/32 devices minSdk still allows.
  private var backGuard: Any? = null
  // Kept so a mashed Back button replaces its hint instead of queueing a toast per press.
  private var backHintToast: Toast? = null

  private val thermalProvider by lazy { AndroidThermalStatusProvider(this) }
  private val ticker = Handler(Looper.getMainLooper())
  private val thermalTick =
    object : Runnable {
      override fun run() {
        updateThermal()
        applyStatusLine() // keep the live cooldown countdown ticking
        ticker.postDelayed(this, THERMAL_POLL_MS)
      }
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    setContentView(buildContentView())
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) registerBackGuard()
    beginNewRunUi()
  }

  override fun onDestroy() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) unregisterBackGuard()
    super.onDestroy()
  }

  /**
   * The manifest hands this activity fold/rotate/density changes rather than letting the framework destroy and recreate it: on a foldable those fire
   * constantly, and a recreation both resets run-scoped state (a pending "Pausing…") and churns the activity that is holding this process at
   * `top-app`. The cost is re-inflating by hand, which is necessary rather than optional here because [dp] resolves against the display's density at
   * build time and a foldable's two displays do not share one.
   */
  override fun onConfigurationChanged(newConfig: Configuration) {
    super.onConfigurationChanged(newConfig)
    setContentView(buildContentView())
    // Re-apply state onto the new views. Not beginNewRunUi(): this is the same run, so the pause
    // affordance must survive, and the retained snapshot must not be treated as a new run's.
    cancelButton?.apply {
      isEnabled = !cancelling
      text = if (cancelling) "Pausing…" else "Pause benchmark"
    }
    updateThermal()
    render(BenchmarkProgressBus.latestSnapshot())
  }

  // singleTask: a launch for a NEW run onto the surviving instance arrives here
  // (no onCreate). Reset the same run-scoped state so the new run isn't shown a
  // prior run's stale "Pausing…"/terminal.
  override fun onNewIntent(intent: Intent?) {
    super.onNewIntent(intent)
    beginNewRunUi()
  }

  // Clear per-run UI state at the start of a run: drop any retained terminal
  // snapshot from a prior run in this reused :benchmark process, and reset the
  // pause affordance.
  private fun beginNewRunUi() {
    BenchmarkProgressBus.markStarting()
    cancelling = false
    cancelButton?.apply {
      isEnabled = true
      text = "Pause benchmark"
    }
  }

  /**
   * Back must not leave this screen. Finishing hands the `:benchmark` process straight back to a demoted cpuset mid-measurement, the exact throttle
   * this activity exists to prevent, and the run would keep going, silently slower, with no pocket screen to show it. So Back is swallowed and the
   * user is pointed at the explicit control instead. Stopping the run is a deliberate act: "Pause benchmark" ends it and leaves the job `PAUSED`,
   * resumable from the Jobs tab.
   *
   * Two registrations because both back paths are live in this app: apps targeting SDK 35+ get predictive back by default, which routes through
   * [OnBackInvokedDispatcher] and never calls [onBackPressed], while `minSdk` 31 still has devices on the legacy path.
   */
  @RequiresApi(Build.VERSION_CODES.TIRAMISU)
  private fun registerBackGuard() {
    val callback = OnBackInvokedCallback { refuseBackExit() }
    backGuard = callback
    onBackInvokedDispatcher.registerOnBackInvokedCallback(OnBackInvokedDispatcher.PRIORITY_DEFAULT, callback)
  }

  @RequiresApi(Build.VERSION_CODES.TIRAMISU)
  private fun unregisterBackGuard() {
    (backGuard as? OnBackInvokedCallback)?.let { onBackInvokedDispatcher.unregisterOnBackInvokedCallback(it) }
    backGuard = null
  }

  // Legacy (API < 33) back path. Deliberately no super call: that would finish() and drop top-app.
  //
  // GestureBackNavigation is suppressed rather than fixed: it flags that API 36+ never calls this,
  // which is true and is exactly why registerBackGuard() exists. This override is only reachable on
  // the API 31/32 devices minSdk still admits, which have no OnBackInvokedDispatcher. The lint's
  // suggested fix (AndroidX OnBackPressedDispatcher) means a ComponentActivity base class, and this
  // activity stays on the framework Activity to keep AndroidX out of the model-hosting process.
  @Deprecated("Superseded by the OnBackInvokedCallback registered in registerBackGuard() on API 33+.")
  @Suppress("DEPRECATION", "MissingSuperCall", "GestureBackNavigation")
  override fun onBackPressed() {
    refuseBackExit()
  }

  // Swallow the gesture and say why, so Back doesn't read as broken. Silent on a pause already in
  // flight: the status line is showing "Pausing…" and another toast would just add noise.
  private fun refuseBackExit() {
    if (cancelling) return
    backHintToast?.cancel()
    backHintToast = Toast.makeText(this, "Benchmark still running. Use Pause benchmark to stop it.", Toast.LENGTH_SHORT)
    backHintToast?.show()
  }

  override fun onResume() {
    super.onResume()
    // Resumed + focused → this (:benchmark) process should now be top-app.
    Log.i(CPUSET_TAG, "[:benchmark activity onResume] ${CpuAffinityProbe.snapshot().summary()}")
    BenchmarkProgressBus.observe(::render)
    updateThermal()
    ticker.postDelayed(thermalTick, THERMAL_POLL_MS)
  }

  override fun onPause() {
    super.onPause()
    BenchmarkProgressBus.stopObserving()
    ticker.removeCallbacks(thermalTick)
  }

  private fun render(snapshot: BenchmarkProgressBus.Progress) {
    // The run ended (job done or cancelled) — leave pocket mode.
    if (!snapshot.running) {
      finish()
      return
    }
    jobTitleLabel?.text = snapshot.title.ifBlank { "Benchmark job" }
    subtitleLabel?.text = snapshot.subtitle.ifBlank { "Loading benchmark details" }
    progressBar?.let {
      it.max = 1000
      it.progress = snapshot.overallPermil.coerceIn(0, 1000)
    }
    cellsLabel?.text = if (snapshot.totalCells > 0) "${snapshot.completedCells}/${snapshot.totalCells} cells done" else "Starting…"
    timeLabel?.text = snapshot.etaText
    estTimeLabel?.text = "Estimated time to complete: ${snapshot.etaText}"
    // Bare cell label, hidden when empty (the Compose JobLiveActivity omits the line rather than
    // captioning it), so the card doesn't reserve space for a placeholder.
    currentLabel?.apply {
      text = snapshot.cellLabel
      visibility = if (snapshot.cellLabel.isBlank()) View.GONE else View.VISIBLE
    }
    // The status line shows a live cooldown countdown while cooling, else the raw
    // progress text (mirrors the Compose pocket's JobLiveActivity second line).
    lastStatusText = snapshot.statusText
    coolingSinceMillis = snapshot.coolingSinceMillis
    applyStatusLine()
    // Cool wash while the readiness gate is cooling the device (mirrors the Compose pocket card).
    card?.background = cardBackground(cooling = snapshot.cooling)
  }

  // Second card line: an accented, once-a-second "Cooling m:ss / max" countdown
  // while the readiness gate is cooling; otherwise the plain progress text.
  private fun applyStatusLine() {
    if (cancelling) {
      statusLabel?.apply {
        text = "Pausing…"
        setTextColor(GRAY)
        setTypeface(null, Typeface.NORMAL)
        visibility = View.VISIBLE
      }
      return
    }
    val since = coolingSinceMillis
    statusLabel?.apply {
      if (since != null) {
        text = coolingCaption(since, System.currentTimeMillis())
        setTextColor(COOL_ACCENT)
        setTypeface(null, Typeface.BOLD)
        visibility = View.VISIBLE
      } else {
        // Blank progress text hides the line, as on the Compose side, instead of falling back to a
        // filler "Running…" that outlives the state it described.
        text = lastStatusText
        setTextColor(GRAY)
        setTypeface(null, Typeface.NORMAL)
        visibility = if (lastStatusText.isBlank()) View.GONE else View.VISIBLE
      }
    }
  }

  // Request whole-job cancellation but DON'T finish here: if the request never
  // reaches main (callback not yet registered / IPC failure), finishing would
  // orphan a still-running job with no pocket screen. Instead show a pending
  // state and let the terminal running=false snapshot (from render) finish us
  // once cancellation is actually observed.
  private fun onCancelClicked() {
    BenchmarkProgressBus.requestCancel()
    cancelling = true
    cancelButton?.apply {
      isEnabled = false
      text = "Pausing…"
    }
    applyStatusLine()
  }

  private fun updateThermal() {
    thermalLabel?.text = thermalHeadroomLabel(thermalProvider)
    (thermalDot?.background as? GradientDrawable)?.setColor(thermalAccentColor())
  }

  // Map the thermal status to a palette color via the shared [thermalAccentKind]
  // classifier (also used by the Compose screens' accentColor), so the severity
  // keyword lists live in one place. Color resources are readable from any
  // process, so no UiKit instance is needed here.
  private fun thermalAccentColor(): Int {
    val res =
      when (thermalAccentKind(thermalDescription(thermalProvider))) {
        AccentKind.CRITICAL -> R.color.pipette_thermal_critical
        AccentKind.SERIOUS -> R.color.pipette_thermal_serious
        AccentKind.NOMINAL,
        AccentKind.MUTED -> R.color.pipette_thermal_nominal
      }
    return getColor(res)
  }

  private fun buildContentView(): View =
    ScrollView(this).apply {
      setBackgroundColor(BG)
      addView(
        LinearLayout(context).apply {
          orientation = LinearLayout.VERTICAL
          gravity = Gravity.CENTER_HORIZONTAL
          setPadding(dp(24), dp(72), dp(24), dp(32))

          addView(
            ImageView(context).apply {
              // The white P-mark vector, not the launcher icon (see PocketModeScreen, which mirrors this).
              setImageResource(R.drawable.pipette_logo)
              contentDescription = "Pipette"
              // clipToOutline alone is a no-op without a provider; the Compose pocket clips the icon
              // to a 12dp rounded rect, so supply the matching outline.
              outlineProvider =
                object : ViewOutlineProvider() {
                  override fun getOutline(view: View, outline: Outline) {
                    outline.setRoundRect(0, 0, view.width, view.height, dp(12).toFloat())
                  }
                }
              clipToOutline = true
              layoutParams = LinearLayout.LayoutParams(dp(56), dp(56)).apply { setMargins(0, 0, 0, dp(20)) }
            }
          )
          addView(serifTitle("Benchmarking in progress…").apply { setPadding(0, 0, 0, dp(28)) })
          addView(thermalRow())
          addView(buildCard().also { card = it })
          estTimeLabel =
            label("Estimated time to complete: calculating", 16f, GRAY).apply {
              gravity = Gravity.CENTER
              setPadding(0, dp(24), 0, dp(20))
            }
          addView(estTimeLabel)
          cancelButton =
            Button(context).apply {
              text = "Pause benchmark"
              setTextColor(Color.WHITE)
              // Neutral surface, not the critical red it wore as "Cancel": pausing is a routine,
              // reversible act, and red read as a warning against the screen's only way out. Shaped
              // like the Compose pocket's exit control (card-toned, 14dp) so it reads as deliberate
              // rather than an unstyled rect.
              background =
                GradientDrawable().apply {
                  cornerRadius = dp(14).toFloat()
                  setColor(getColor(R.color.pipette_surface_variant))
                }
              setPadding(dp(16), dp(16), dp(16), dp(16))
              layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
              setOnClickListener { onCancelClicked() }
            }
          addView(cancelButton)
        }
      )
    }

  private fun thermalRow(): View =
    LinearLayout(this).apply {
      orientation = LinearLayout.HORIZONTAL
      gravity = Gravity.CENTER_VERTICAL
      layoutParams =
        LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
          setMargins(0, 0, 0, dp(16))
        }
      addView(label("Throttling headroom", 16f, GRAY), LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
      addView(
        LinearLayout(context).apply {
          orientation = LinearLayout.HORIZONTAL
          gravity = Gravity.CENTER_VERTICAL
          background =
            GradientDrawable().apply {
              cornerRadius = dp(20).toFloat()
              setColor(BG)
              setStroke(dp(1), Color.argb(26, 255, 255, 255))
            }
          setPadding(dp(12), dp(6), dp(12), dp(6))
          thermalDot =
            View(context).apply {
              background =
                GradientDrawable().apply {
                  shape = GradientDrawable.OVAL
                  setColor(Color.GRAY)
                }
              layoutParams = LinearLayout.LayoutParams(dp(8), dp(8)).apply { marginEnd = dp(6) }
            }
          addView(thermalDot)
          thermalLabel = label("—", 16f, Color.WHITE).apply { setTypeface(typeface, Typeface.BOLD) }
          addView(thermalLabel)
        }
      )
    }

  private fun buildCard(): LinearLayout =
    LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      gravity = Gravity.CENTER_HORIZONTAL
      background = cardBackground(cooling = false)
      setPadding(dp(32), dp(28), dp(32), dp(28))
      layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)

      jobTitleLabel = serifTitle("Benchmark job").apply { setPadding(0, 0, 0, dp(6)) }
      addView(jobTitleLabel)
      subtitleLabel =
        label("Loading benchmark details", 16f, GRAY).apply {
          gravity = Gravity.CENTER
          setPadding(0, 0, 0, dp(28))
        }
      addView(subtitleLabel)
      progressBar =
        ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
          max = 1000
          progress = 0
          // The platform default is a thin themed bar; the Compose pocket draws an 8dp fully-rounded
          // track with a white fill, so build that explicitly rather than inheriting the theme.
          progressDrawable = progressTrackDrawable()
          layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(8))
        }
      addView(progressBar)
      addView(
        LinearLayout(context).apply {
          orientation = LinearLayout.HORIZONTAL
          layoutParams =
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(8) }
          cellsLabel = label("Starting…", 16f, GRAY)
          addView(cellsLabel, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
          timeLabel = label("calculating", 16f, GRAY).apply { gravity = Gravity.END }
          addView(timeLabel)
        }
      )
      // Live block: start-aligned and truncated like the Compose JobLiveActivity, so a long model
      // name ellipsizes instead of wrapping the card taller on every cell change.
      currentLabel =
        label("", 14f, Color.WHITE).apply {
          setTypeface(typeface, Typeface.BOLD)
          maxLines = 2
          ellipsize = TextUtils.TruncateAt.END
          setPadding(0, dp(8), 0, 0)
          layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }
      addView(currentLabel)
      statusLabel =
        label("Running…", 14f, GRAY).apply {
          maxLines = 1
          ellipsize = TextUtils.TruncateAt.END
          setPadding(0, dp(6), 0, 0)
          layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }
      addView(statusLabel)
    }

  // Fully-rounded 8dp track with a white clipped fill, matching the Compose pocket's hand-drawn bar.
  private fun progressTrackDrawable(): LayerDrawable {
    val radius = dp(4).toFloat()
    val track =
      GradientDrawable().apply {
        cornerRadius = radius
        setColor(TRACK)
      }
    val fill =
      GradientDrawable().apply {
        cornerRadius = radius
        setColor(Color.WHITE)
      }
    return LayerDrawable(arrayOf(track, ClipDrawable(fill, Gravity.START, ClipDrawable.HORIZONTAL))).apply {
      setId(0, android.R.id.background)
      setId(1, android.R.id.progress)
    }
  }

  private fun cardBackground(cooling: Boolean): GradientDrawable =
    GradientDrawable().apply {
      cornerRadius = dp(20).toFloat()
      setColor(if (cooling) COOL_WASH else CARD)
      if (cooling) setStroke(dp(1), COOL_BORDER)
    }

  private fun serifTitle(text: String): TextView =
    TextView(this).apply {
      this.text = text
      setTextColor(Color.WHITE)
      typeface = Typeface.create("serif", Typeface.BOLD)
      setTextSize(TypedValue.COMPLEX_UNIT_SP, 24f)
      gravity = Gravity.CENTER
    }

  private fun label(text: String, sizeSp: Float, color: Int): TextView =
    TextView(this).apply {
      this.text = text
      setTextColor(color)
      setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp)
    }

  private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

  companion object {
    private const val CPUSET_TAG = "pipette-cpuset"
    private const val THERMAL_POLL_MS = 1_000L
    private val BG = Color.rgb(0x0A, 0x0A, 0x0A)
    private val CARD = Color.rgb(0x17, 0x17, 0x17)
    private val COOL_WASH = Color.rgb(0x1C, 0x27, 0x35)
    private val COOL_BORDER = Color.argb(0x80, 0x68, 0xA2, 0xE6)
    // Compose PocketCoolText, not the border color: the cooldown caption is a lighter blue than the
    // card's cool border.
    private val COOL_ACCENT = Color.rgb(0x89, 0xBA, 0xF7)
    private val GRAY = Color.rgb(0xA3, 0xA3, 0xA3)
    private val TRACK = Color.rgb(0x40, 0x40, 0x40)
  }
}
