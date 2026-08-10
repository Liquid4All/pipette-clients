package ai.liquid.pipette

import android.Manifest
import android.content.pm.PackageManager
import android.database.Cursor
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Shader
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.ColorUtils
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.navigation.NavigationBarView
import kotlinx.coroutines.launch

/**
 * Thin Activity shell. State and background work live in [MainViewModel]; the per-tab UI lives in the [Screen] view-controllers. The Activity only
 * wires them up, owns the system-document launchers, draws the chrome (header + nav + pocket mode), and re-renders the active screen when the
 * ViewModel signals a change. Nothing here is destroyed on a configuration change that would disturb a running benchmark — the engine and runner are
 * app-scoped.
 */
class MainActivity : AppCompatActivity() {
  private val vm: MainViewModel by viewModels()
  private lateinit var ui: UiKit
  private lateinit var root: LinearLayout
  private lateinit var content: LinearLayout
  private lateinit var nav: View
  private lateinit var bottomNav: BottomNavigationView
  private lateinit var screens: Map<Tab, Screen>

  // The body ScrollView is rebuilt on every render(); remember the last one and
  // the tab it belonged to so a re-render triggered by a runner tick (a running
  // job re-renders Jobs repeatedly) restores the user's scroll position instead
  // of snapping back to the top. A genuine tab switch resets to the top.
  private var bodyScroll: ScrollView? = null
  private var scrollTab: Tab? = null
  // Bottom system-bar inset (gesture bar). Lives on the nav bar when it's
  // visible, on the root when it's hidden (auth gate / pocket mode).
  private var bottomInset: Int = 0

  // Runner-tick rendering: a running job emits a RunnerState on every progress
  // step and cooldown countdown. Rebuilding the whole view tree each time flashes
  // (G1). Only a *structural* change (job start/stop, or a new cell starting)
  // warrants a full render(); the frequent within-cell ticks just update the live
  // text/progress views in place. The signature is what we treat as structural.
  private var lastRunnerSignature: String? = null
  private var headerStatusView: TextView? = null
  // Held references to the pocket-mode views that change on a within-cell tick, so
  // they can be updated without rebuilding the pocket screen.
  private var pocketProgressBar: ProgressBar? = null
  private var pocketStatsLabel: TextView? = null
  private var pocketCurrentLabel: TextView? = null
  private var pocketProgressLabel: TextView? = null
  private var pocketTimeLabel: TextView? = null

  private val openModelLauncher =
    registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
      if (uri != null) {
        vm.runInBackground("Copying model...", onError = { showError(it) }) {
          val name = displayName(uri) ?: uri.lastPathSegment?.substringAfterLast('/') ?: "model.gguf"
          require(name.endsWith(".gguf", ignoreCase = true)) { "Selected file must be .gguf" }
          vm.container.storage.copyModelFromUri(uri, name)
          "Copied $name"
        }
      }
    }

  // Lets the download foreground notification (with pause/cancel) show on
  // Android 13+. Result ignored — downloads still run if the user declines; the
  // notification just won't appear.
  private val notificationPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) {}

  private val exportCsvLauncher =
    registerForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri ->
      val csv = vm.pendingCsvExportText
      vm.pendingCsvExportText = null
      if (uri != null && csv != null) {
        runCatching {
            contentResolver.openOutputStream(uri).use { output ->
              requireNotNull(output) { "Unable to open export destination" }
              output.write(csv.toByteArray(Charsets.UTF_8))
            }
          }
          .onSuccess {
            vm.statusText = "Exported results CSV"
            vm.invalidate()
          }
          .onFailure { showError(it) }
      }
    }

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    ui = UiKit(this)

    val screenContext =
      ScreenContext(
        activity = this,
        vm = vm,
        ui = ui,
        openModel = { openModelLauncher.launch(arrayOf("*/*")) },
        exportCsv = { filename, csv ->
          vm.pendingCsvExportText = csv
          exportCsvLauncher.launch(filename)
        },
      )
    screens = mapOf(Tab.JOBS to JobsScreen(screenContext), Tab.MODELS to ModelsScreen(screenContext), Tab.SETTINGS to SettingsScreen(screenContext))

    root =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        layoutParams = LinearLayout.LayoutParams(match, match)
      }
    content =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        layoutParams = LinearLayout.LayoutParams(match, 0, 1f)
      }
    root.addView(content)
    nav = navBar()
    root.addView(nav)
    setContentView(root)
    applyWindowInsets()
    maybeRequestNotificationPermission()

    // The "Add models" sub-screen lives inside the Models tab (no separate
    // Activity/Fragment), so intercept system back to pop it instead of leaving
    // the app; otherwise fall through to the default behavior.
    onBackPressedDispatcher.addCallback(this) {
      if (vm.modelsShowAddScreen) {
        // Mirror ModelsScreen.closeAddModels(): drop the family selection so it doesn't linger into the next open.
        vm.selectedAddFamilyIds.clear()
        vm.modelsShowAddScreen = false
        vm.invalidate()
      } else {
        isEnabled = false
        onBackPressedDispatcher.onBackPressed()
        isEnabled = true
      }
    }

    // uiTick replays its current value on collection, so this also drives the
    // first render. Runner-state ticks only touch the Jobs/pocket views (a
    // background run shouldn't wipe an EditText the user is typing in elsewhere),
    // and only do a full rebuild on a structural change — otherwise they update
    // the live text/progress in place so the screen doesn't flash every tick.
    lifecycleScope.launch {
      repeatOnLifecycle(Lifecycle.State.STARTED) {
        launch { vm.uiTick.collect { render() } }
        // The auth gate gates the whole UI; re-render whenever it changes.
        launch { vm.authGate.collect { render() } }
        launch {
          vm.runnerState.collect { state ->
            val signature = "${state.runningJobId}|${state.currentCellLabel}"
            val structural = signature != lastRunnerSignature
            lastRunnerSignature = signature
            val inPocket = vm.pocketModeJobId != null
            when {
              structural -> if (inPocket || vm.selectedTab == Tab.JOBS) render()
              inPocket -> updatePocketLive(state)
              vm.selectedTab == Tab.JOBS -> updateHeaderStatus(state)
            }
          }
        }
      }
    }
  }

  // targetSdk 36 forces edge-to-edge, so content draws behind the status and
  // navigation bars. Pad the root's top/left/right by the system-bar + cutout
  // insets so nothing is clipped. The bottom inset goes on the nav bar itself
  // (not the root) so the nav bar's background extends to the screen edge and
  // sits flush against the gesture bar instead of floating above it.
  private fun applyWindowInsets() {
    ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
      val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
      bottomInset = bars.bottom
      view.setPadding(bars.left, bars.top, bars.right, 0)
      applyBottomInset()
      insets
    }
  }

  // Route the bottom inset to whichever view is currently the bottom edge: the
  // nav bar when it's visible, otherwise the root (gate / pocket mode).
  private fun applyBottomInset() {
    if (nav.visibility == View.VISIBLE) {
      root.setPadding(root.paddingLeft, root.paddingTop, root.paddingRight, 0)
      nav.setPadding(nav.paddingLeft, nav.paddingTop, nav.paddingRight, bottomInset)
    } else {
      nav.setPadding(nav.paddingLeft, nav.paddingTop, nav.paddingRight, 0)
      root.setPadding(root.paddingLeft, root.paddingTop, root.paddingRight, bottomInset)
    }
  }

  private fun render() {
    // The Clerk auth gate is the outermost gate: until it's Ready, show the gate
    // UI instead of the app chrome. (When gated, no job can be running.)
    val gate = vm.authGate.value
    if (gate !is AuthGate.Ready) {
      renderGate(gate)
      return
    }

    val runningJobId = vm.runnerState.value.runningJobId
    val activePocketJobId = vm.pocketModeJobId?.takeIf { runningJobId == it }
    if (vm.pocketModeJobId != null && activePocketJobId == null) {
      vm.pocketModeJobId = null
    }
    updatePocketModeWindowFlag(activePocketJobId != null)
    nav.visibility = if (activePocketJobId == null) View.VISIBLE else View.GONE
    applyBottomInset()
    if (activePocketJobId == null) {
      // Keep the bottom nav's highlight in sync with programmatic tab changes
      // (e.g. the "Go to Models" button). Setting the same id is a no-op; a
      // changed id fires the listener, which short-circuits since the tab
      // already matches.
      val targetId = menuIdForTab(vm.selectedTab)
      if (bottomNav.selectedItemId != targetId) bottomNav.selectedItemId = targetId
    }
    // Capture the outgoing scroll position before tearing the tree down, but
    // only if it belongs to the tab we're about to re-render (don't carry one
    // tab's offset onto another).
    val previousScrollY = if (scrollTab == vm.selectedTab) bodyScroll?.scrollY ?: 0 else 0
    content.removeAllViews()
    if (activePocketJobId != null) {
      bodyScroll = null
      scrollTab = null
      content.addView(pocketModeView(activePocketJobId), LinearLayout.LayoutParams(match, match))
      return
    }
    content.addView(header())
    val scroll =
      ScrollView(this).apply {
        layoutParams = LinearLayout.LayoutParams(match, 0, 1f)
        isFillViewport = true
      }
    val body =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(18), dp(14), dp(18), dp(24))
      }
    scroll.addView(body)
    content.addView(scroll)
    bodyScroll = scroll
    scrollTab = vm.selectedTab
    screens.getValue(vm.selectedTab).renderBody(body)
    if (previousScrollY > 0) {
      scroll.post { scroll.scrollTo(0, previousScrollY) }
    }
  }

  /** Render the auth gate (any non-Ready [AuthGate]) in place of the app chrome. */
  private fun renderGate(gate: AuthGate) {
    updatePocketModeWindowFlag(false)
    nav.visibility = View.GONE
    applyBottomInset()
    bodyScroll = null
    scrollTab = null
    content.removeAllViews()
    when (gate) {
      is AuthGate.Loading -> content.addView(gateMessageView("Pipette", "Loading…", showSpinner = true), LinearLayout.LayoutParams(match, match))
      is AuthGate.InitError ->
        content.addView(gateMessageView("Sign-in unavailable", gate.message, showSpinner = false), LinearLayout.LayoutParams(match, match))
      is AuthGate.SignedOut -> {
        // Legacy (disabled) launcher: sign-in now lives in the Compose app
        // (ComposeMainActivity → AuthGateScreen). This screen only keeps the debug
        // skip control so instrumentation that starts MainActivity can bypass the gate.
        content.addView(
          gateMessageView("Sign-in moved", "Sign-in is handled by the Compose app.", showSpinner = false),
          LinearLayout.LayoutParams(match, 0, 1f),
        )
        if (BuildConfig.DEBUG) content.addView(debugSkipSignInBar())
      }
      is AuthGate.Mismatch -> content.addView(mismatchView(gate.linkedEmail, gate.currentEmail), LinearLayout.LayoutParams(match, match))
      is AuthGate.Ready -> return
    }
  }

  /** Debug-only control on the sign-in screen to skip auth (turns the bypass on). */
  private fun debugSkipSignInBar(): View =
    LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(dp(18), dp(4), dp(18), dp(16))
      addView(ui.mutedLabel("Debug build: auth enforced. Skip it for local testing (toggle in Settings → Account)."))
      addView(ui.outlineButton("Skip sign-in (debug only)") { vm.setClerkGateBypass(true) })
    }

  private fun gateMessageView(title: String, message: String, showSpinner: Boolean): View =
    LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      gravity = Gravity.CENTER
      setPadding(dp(28), dp(28), dp(28), dp(28))
      addView(ui.displayTitle(title))
      if (showSpinner) {
        addView(ProgressBar(context).apply { layoutParams = LinearLayout.LayoutParams(wrap, wrap).apply { setMargins(0, dp(16), 0, dp(16)) } })
      }
      addView(ui.mutedLabel(message))
    }

  private fun mismatchView(linkedEmail: String?, currentEmail: String?): View {
    val body =
      LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(18), dp(48), dp(18), dp(24))
        addView(ui.displayTitle("Account mismatch"))
        addView(
          ui.card {
            addView(
              ui.mutedLabel(
                "This device is linked to ${linkedEmail ?: "another account"}, " + "but you're signed in as ${currentEmail ?: "a different account"}."
              )
            )
            addView(ui.primaryButton("Sign out") { vm.signOutOfClerk() })
            addView(
              ui.outlineButton("Delete device identity") {
                ui.confirm("Delete device identity? This clears registration on this device.") {
                  vm.container.storage.deleteRegistration()
                  vm.container.secrets.deletePrivateKey()
                  vm.applyDefaultContributeResults(false)
                  vm.refreshRegistration()
                }
              }
            )
          }
        )
      }
    return ScrollView(this).apply { addView(body) }
  }

  override fun onDestroy() {
    updatePocketModeWindowFlag(false)
    // The engine and runner are app-scoped (AppContainer), so there is nothing
    // to tear down here — a config-change recreation leaves a running benchmark
    // and its :benchmark process untouched.
    super.onDestroy()
  }

  // iOS has no global app header — each screen owns its title (rendered in its
  // own body). We keep only a slim contextual status strip, shown when a job is
  // running (live-updated in place via updateHeaderStatus) or when a transient
  // status message is set; otherwise there's no header and the screen title sits
  // at the top. Returns a zero-height placeholder View when there's nothing to show.
  private fun header(): View {
    val running = vm.runnerState.value.runningJobId != null
    headerStatusView = null
    if (!running && vm.statusText.isBlank()) return View(this).apply { layoutParams = LinearLayout.LayoutParams(match, 0) }
    return LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setPadding(dp(18), dp(12), dp(18), dp(6))
      if (running) {
        headerStatusView =
          TextView(context).apply {
            text = headerStatusText(vm.runnerState.value)
            textSize = 13f
            setTextColor(ui.colorMuted())
          }
        addView(headerStatusView)
      }
      if (vm.statusText.isNotBlank()) {
        addView(
          TextView(context).apply {
            text = vm.statusText
            textSize = 13f
            setTextColor(ui.colorMuted())
          }
        )
      }
    }
  }

  private fun headerStatusText(state: RunnerState): String {
    val engine = vm.container.benchmarkEngine
    return when {
      state.runningJobId != null -> "Running ${state.currentCellLabel} - ${state.currentProgressText}"
      engine.isAvailable -> {
        val commit = engine.llamaCppCommit()
        // The commit is only known once :benchmark has been bound; show a bare
        // "ready" until then (and for the unavailable sentinel).
        if (commit.startsWith("native-")) "Native benchmark engine ready" else "Native benchmark engine ready ($commit)"
      }
      else -> "Native benchmark engine missing: jobs can be planned, but cells will fail until libpipette_android.so is packaged."
    }
  }

  /** In-place header update for a within-cell runner tick (no rebuild). */
  private fun updateHeaderStatus(state: RunnerState) {
    headerStatusView?.text = headerStatusText(state)
  }

  /** In-place pocket-mode update for a within-cell runner tick (no rebuild). */
  private fun updatePocketLive(state: RunnerState) {
    val jobId = vm.pocketModeJobId ?: return
    val manifest = vm.container.storage.loadJobManifest(jobId)
    val fraction = manifest?.let { jobProgressFraction(it, state) } ?: 0.0
    val percent = (fraction.coerceIn(0.0, 1.0) * 100).toInt()
    val completed = manifest?.completedCells ?: 0
    val total = manifest?.totalCells ?: 0
    val remaining = estimatedTimeLeft(state, fraction) ?: "calculating"
    pocketProgressBar?.progress = (fraction.coerceIn(0.0, 1.0) * 1000).toInt()
    pocketStatsLabel?.text = "$percent% - $completed of $total cells complete"
    pocketCurrentLabel?.text = "Current: ${state.currentCellLabel.ifBlank { "Starting" }}"
    pocketProgressLabel?.text = state.currentProgressText.ifBlank { "Running" }
    pocketTimeLabel?.text = "Estimated time to complete: $remaining"
  }

  private fun updatePocketModeWindowFlag(enabled: Boolean) {
    if (enabled) {
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    } else {
      window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
  }

  private fun pocketModeView(jobId: String): View {
    val runnerState = vm.runnerState.value
    val manifest = vm.container.storage.loadJobManifest(jobId)
    val progressFraction = manifest?.let { jobProgressFraction(it, runnerState) } ?: 0.0
    val percent = (progressFraction.coerceIn(0.0, 1.0) * 100).toInt()
    val summary =
      manifest?.let {
        val modelCount = it.cells.map { cell -> cell.modelName }.toSet().size
        val benchmarkCount = it.cells.map { cell -> cell.benchmarkId }.toSet().size
        "$modelCount ${plural("model", modelCount)} - $benchmarkCount ${plural("benchmark", benchmarkCount)}"
      } ?: "Loading benchmark details"
    val completed = manifest?.completedCells ?: 0
    val total = manifest?.totalCells ?: 0
    val remaining = estimatedTimeLeft(runnerState, progressFraction) ?: "calculating"

    return ScrollView(this).apply {
      setBackgroundColor(Color.rgb(8, 8, 8))
      layoutParams = LinearLayout.LayoutParams(match, match)
      addView(
        LinearLayout(context).apply {
          orientation = LinearLayout.VERTICAL
          gravity = Gravity.CENTER_HORIZONTAL
          setPadding(dp(28), dp(72), dp(28), dp(32))

          addView(pocketBrandMark())
          addView(pocketTitle("Benchmarking in progress…"))
          addView(pocketLabel(summary, 14f, Color.LTGRAY, bold = false).apply { gravity = Gravity.CENTER })

          val bar =
            ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
              max = 1000
              progress = (progressFraction.coerceIn(0.0, 1.0) * 1000).toInt()
              layoutParams = LinearLayout.LayoutParams(match, wrap).apply { setMargins(0, dp(28), 0, dp(12)) }
            }
          pocketProgressBar = bar
          addView(bar)
          pocketStatsLabel = pocketLabel("$percent% - $completed of $total cells complete", 16f, Color.WHITE, bold = true)
          addView(pocketStatsLabel)
          pocketCurrentLabel = pocketLabel("Current: ${runnerState.currentCellLabel.ifBlank { "Starting" }}", 14f, Color.LTGRAY, bold = false)
          addView(pocketCurrentLabel)
          pocketProgressLabel = pocketLabel(runnerState.currentProgressText.ifBlank { "Running" }, 14f, Color.LTGRAY, bold = false)
          addView(pocketProgressLabel)
          addView(pocketThermalChip())
          pocketTimeLabel =
            pocketLabel("Estimated time to complete: $remaining", 14f, Color.LTGRAY, bold = false).apply { setPadding(0, dp(14), 0, dp(28)) }
          addView(pocketTimeLabel)

          addView(slideToExit())
          addView(
            TextView(context).apply {
              text = "Cancel job"
              textSize = 14f
              gravity = Gravity.CENTER
              setTextColor(ui.colorThermalCritical())
              setPadding(dp(12), dp(16), dp(12), dp(8))
              setOnClickListener {
                vm.container.jobController.cancel()
                setPocketMode(null)
              }
              layoutParams = LinearLayout.LayoutParams(match, wrap)
            }
          )
        }
      )
    }
  }

  private fun pocketLabel(textValue: String, size: Float, color: Int, bold: Boolean): TextView =
    TextView(this).apply {
      text = textValue
      textSize = size
      setTextColor(color)
      if (bold) setTypeface(typeface, Typeface.BOLD)
      setPadding(0, dp(4), 0, dp(8))
    }

  /**
   * Gradient Pipette "P" mark for Pocket Mode, traced from the shipped `pipette_logo_mark` artwork (~0.75 aspect, see #521) and filled with the
   * top-to-bottom #FAFAFA → #737373 gradient. Both the shape and the gradient live in the [R.drawable.pocket_brand_mark] VectorDrawable, whose
   * comment records the other two copies of this outline; drawn at 30x40dp. An earlier revision hand-tuned the path from the iOS SwiftUI source,
   * which was too wide and had an uneven arch, and the trace fixes both.
   */
  @Suppress("MagicNumber") // 30x40dp mark size + 8dp bottom margin
  private fun pocketBrandMark(): View =
    ImageView(this).apply {
      setImageResource(R.drawable.pocket_brand_mark)
      layoutParams = LinearLayout.LayoutParams(dp(30), dp(40)).apply { setMargins(0, 0, 0, dp(8)) }
    }

  /** Serif display title with a top-to-bottom white→steel gradient (iOS parity). */
  private fun pocketTitle(textValue: String): TextView =
    TextView(this).apply {
      text = textValue
      textSize = 27f
      typeface = Typeface.create("serif", Typeface.BOLD)
      setTextColor(Color.WHITE)
      gravity = Gravity.CENTER
      setPadding(0, dp(28), 0, dp(16))
      post {
        if (height > 0) {
          paint.shader =
            LinearGradient(0f, 0f, 0f, height.toFloat(), intArrayOf(Color.WHITE, Color.rgb(0x9A, 0xB4, 0xD0)), null, Shader.TileMode.CLAMP)
          invalidate()
        }
      }
    }

  /** A rounded thermal pill tinted by severity (green/orange/red). */
  private fun pocketThermalChip(): TextView {
    val accent = pocketThermalAccent()
    return TextView(this).apply {
      text = "Device temperature: ${thermalLabel(vm.container.thermalStatusProvider)}"
      textSize = 13f
      setTextColor(accent)
      setTypeface(typeface, Typeface.BOLD)
      setPadding(dp(14), dp(6), dp(14), dp(6))
      background =
        GradientDrawable().apply {
          cornerRadius = dp(999).toFloat()
          setColor(ColorUtils.setAlphaComponent(accent, 0x33))
        }
      layoutParams = LinearLayout.LayoutParams(wrap, wrap).apply { setMargins(0, dp(8), 0, dp(8)) }
    }
  }

  private fun pocketThermalAccent(): Int =
    when (thermalAccentKind(thermalDescription(vm.container.thermalStatusProvider))) {
      AccentKind.CRITICAL -> ui.colorThermalCritical()
      AccentKind.SERIOUS -> ui.colorThermalSerious()
      AccentKind.NOMINAL,
      AccentKind.MUTED -> ui.colorThermalNominal()
    }

  /** Slide-to-exit control: drag the thumb past ~60% of the track to leave pocket mode. */
  private fun slideToExit(): View {
    val trackHeight = dp(58)
    val thumbWidth = dp(64)
    val track =
      FrameLayout(this).apply {
        background =
          GradientDrawable().apply {
            cornerRadius = dp(14).toFloat()
            setColor(Color.rgb(0x1C, 0x1C, 0x20))
            setStroke(dp(1), Color.rgb(0x33, 0x33, 0x38))
          }
        layoutParams = LinearLayout.LayoutParams(match, trackHeight).apply { setMargins(0, dp(8), 0, dp(8)) }
      }
    val hint =
      TextView(this).apply {
        text = "Slide to exit"
        textSize = 14f
        setTextColor(Color.LTGRAY)
        gravity = Gravity.CENTER
        layoutParams = FrameLayout.LayoutParams(match, match)
      }
    val thumb =
      TextView(this).apply {
        text = "›"
        textSize = 24f
        gravity = Gravity.CENTER
        setTextColor(Color.BLACK)
        setTypeface(typeface, Typeface.BOLD)
        background =
          GradientDrawable().apply {
            cornerRadius = dp(12).toFloat()
            setColor(Color.WHITE)
          }
        layoutParams = FrameLayout.LayoutParams(thumbWidth, trackHeight)
      }
    thumb.setOnTouchListener { view, event ->
      when (event.actionMasked) {
        MotionEvent.ACTION_DOWN -> {
          view.tag = event.rawX - view.translationX
          true
        }
        MotionEvent.ACTION_MOVE -> {
          val origin = (view.tag as? Float) ?: event.rawX
          val maxX = (track.width - view.width).toFloat().coerceAtLeast(0f)
          view.translationX = (event.rawX - origin).coerceIn(0f, maxX)
          hint.alpha = 1f - (view.translationX / maxX.coerceAtLeast(1f))
          true
        }
        MotionEvent.ACTION_UP,
        MotionEvent.ACTION_CANCEL -> {
          val maxX = (track.width - view.width).toFloat().coerceAtLeast(1f)
          if (view.translationX > maxX * 0.6f) {
            setPocketMode(null)
          } else {
            view.animate().translationX(0f).setDuration(150).start()
            hint.animate().alpha(1f).setDuration(150).start()
          }
          true
        }
        else -> false
      }
    }
    track.addView(hint)
    track.addView(thumb)
    return track
  }

  private fun setPocketMode(jobId: String?) {
    vm.pocketModeJobId = jobId
    updatePocketModeWindowFlag(jobId != null)
    vm.invalidate()
  }

  private fun jobProgressFraction(manifest: JobManifest, runnerState: RunnerState): Double {
    if (manifest.totalCells <= 0) return 0.0
    val within =
      if (runnerState.runningJobId == manifest.jobId) {
        runnerState.currentCellFraction.coerceIn(0.0, 1.0)
      } else {
        0.0
      }
    return ((manifest.completedCells + within) / manifest.totalCells.toDouble()).coerceIn(0.0, 1.0)
  }

  private fun estimatedTimeLeft(runnerState: RunnerState, progressFraction: Double): String? {
    val startedAt = runnerState.startedAtMillis ?: return null
    if (progressFraction <= 0.02) return null
    val elapsedMs = (System.currentTimeMillis() - startedAt).coerceAtLeast(1L).toDouble()
    val totalMs = elapsedMs / progressFraction
    val remainingMs = (totalMs - elapsedMs).coerceAtLeast(0.0)
    return if (remainingMs < 60_000) {
      "${(remainingMs / 1000).toInt()}s left"
    } else {
      "${kotlin.math.round(remainingMs / 60_000).toInt()} min left"
    }
  }

  private fun navBar(): View =
    BottomNavigationView(this).apply {
      inflateMenu(R.menu.bottom_nav)
      labelVisibilityMode = NavigationBarView.LABEL_VISIBILITY_LABELED
      selectedItemId = menuIdForTab(vm.selectedTab)
      setOnItemSelectedListener { item ->
        val tab = tabForMenuId(item.itemId)
        if (vm.selectedTab != tab) {
          vm.selectedTab = tab
          vm.statusText = ""
          // Leaving the Models tab dismisses its "Add models" sub-screen so the
          // tab reopens on the downloaded list, not a stale add view.
          vm.modelsShowAddScreen = false
          vm.invalidate()
        }
        true
      }
      bottomNav = this
    }

  private fun menuIdForTab(tab: Tab): Int =
    when (tab) {
      Tab.MODELS -> R.id.nav_models
      Tab.JOBS -> R.id.nav_jobs
      Tab.SETTINGS -> R.id.nav_settings
    }

  private fun tabForMenuId(id: Int): Tab =
    when (id) {
      R.id.nav_models -> Tab.MODELS
      R.id.nav_settings -> Tab.SETTINGS
      else -> Tab.JOBS
    }

  private fun maybeRequestNotificationPermission() {
    if (
      Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
        checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
    ) {
      notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
    }
  }

  private fun dp(value: Int): Int = ui.dp(value)

  private fun showError(error: Throwable) {
    Toast.makeText(this, error.message ?: error.javaClass.simpleName, Toast.LENGTH_LONG).show()
  }

  private fun displayName(uri: Uri): String? {
    var cursor: Cursor? = null
    return try {
      cursor = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
      if (cursor != null && cursor.moveToFirst()) {
        cursor.getString(0)
      } else {
        null
      }
    } finally {
      cursor?.close()
    }
  }

  private companion object {
    const val match = ViewGroup.LayoutParams.MATCH_PARENT
    const val wrap = ViewGroup.LayoutParams.WRAP_CONTENT
  }
}
