// Root scaffold: layout literals (MagicNumber, e.g. the reserved tab-bar height).
@file:Suppress("MagicNumber")

package ai.liquid.pipette.compose

import ai.liquid.pipette.AuthGate
import ai.liquid.pipette.R
import ai.liquid.pipette.Tab
import ai.liquid.pipette.compose.jobs.JobsScreen
import ai.liquid.pipette.compose.jobs.JobsViewModel
import ai.liquid.pipette.compose.models.ModelsScreen
import ai.liquid.pipette.compose.models.ModelsViewModel
import ai.liquid.pipette.compose.nav.Route
import ai.liquid.pipette.compose.settings.SettingsScreen
import ai.liquid.pipette.compose.settings.SettingsViewModel
import ai.liquid.pipette.compose.setup.SetupScreen
import ai.liquid.pipette.compose.setup.SetupViewModel
import ai.liquid.pipette.compose.shell.AuthGateScreen
import ai.liquid.pipette.compose.shell.PocketModeScreen
import ai.liquid.pipette.compose.shell.ShellViewModel
import ai.liquid.pipette.compose.theme.PipetteTheme
import android.app.Application
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.ContentTransform
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation3.runtime.NavKey
import androidx.navigation3.runtime.entryProvider
import androidx.navigation3.runtime.rememberNavBackStack
import androidx.navigation3.ui.NavDisplay
import kotlinx.coroutines.launch

/** Floating pill tab bar reserved height (52 button + 6+6 padding), so screens can pad their scroll content clear of it. */
val PillTabBarReservedHeight = 76.dp

private const val SCREEN_ENTER_DURATION_MS = 320
private const val SCREEN_EXIT_DURATION_MS = 220

/** How far the incoming screen rises from below as it fades in. */
val ScreenChangeRise = 24.dp

/**
 * Screen-change animation used by [NavDisplay]: the outgoing screen fades out (alpha), while the incoming screen fades in and rises gently from
 * [ScreenChangeRise] below (a light upward slide).
 *
 * @param riseOffsetPx the rise distance in pixels (convert [ScreenChangeRise] with the local density)
 */
private fun screenChangeTransform(riseOffsetPx: Int): ContentTransform =
  (fadeIn(animationSpec = tween(SCREEN_ENTER_DURATION_MS)) +
    slideInVertically(animationSpec = tween(SCREEN_ENTER_DURATION_MS)) { riseOffsetPx }) togetherWith
    fadeOut(animationSpec = tween(SCREEN_EXIT_DURATION_MS))

/** Top-level destinations in the floating toolbar, in display order. */
private val NAV_MENU_ITEMS =
  listOf(
    FloatingToolbarMenuItem(Tab.JOBS, R.drawable.ic_tab_jobs, Tab.JOBS.label),
    FloatingToolbarMenuItem(Tab.MODELS, R.drawable.ic_tab_models, Tab.MODELS.label),
    FloatingToolbarMenuItem(Tab.SETTINGS, R.drawable.ic_tab_settings, Tab.SETTINGS.label),
  )

/** Compose entry point: hosts the shell + per-screen ViewModels, routes pocket / auth gate / setup gate / tabbed chrome, wires SAF launchers. */
@Composable
fun PipetteAppRoot() {
  PipetteTheme {
    val context = LocalContext.current
    val app = context.applicationContext as Application
    val shell: ShellViewModel = viewModel()
    val factory = remember(shell) { PipetteViewModelFactory(app, shell) }

    val setupVm: SetupViewModel = viewModel(factory = factory)
    val modelsVm: ModelsViewModel = viewModel(factory = factory)
    val jobsVm: JobsViewModel = viewModel(factory = factory)
    val settingsVm: SettingsViewModel = viewModel(factory = factory)

    val shellState by shell.state.collectAsStateWithLifecycle()

    // Keep the screen awake for the whole run (iOS isIdleTimerDisabled), so a benchmark — and
    // Pocket Mode — isn't interrupted by the display dimming/locking.
    val view = androidx.compose.ui.platform.LocalView.current
    val jobRunning = shellState.runner.runningJobId != null
    DisposableEffect(jobRunning) {
      view.keepScreenOn = jobRunning
      onDispose { view.keepScreenOn = false }
    }

    val modelLauncher =
      rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> if (uri != null) modelsVm.onModelUriPicked(uri) }
    val csvLauncher =
      rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri ->
        val csv = jobsVm.consumePendingCsvExport()
        if (uri != null && csv != null) {
          runCatching {
              val output = requireNotNull(context.contentResolver.openOutputStream(uri)) { "Unable to open export destination" }
              output.use { it.write(csv.toByteArray(Charsets.UTF_8)) }
            }
            .onFailure { Toast.makeText(context, it.message ?: "Export failed", Toast.LENGTH_LONG).show() }
        }
      }

    val lifecycleOwner = LocalLifecycleOwner.current
    androidx.compose.runtime.LaunchedEffect(Unit) {
      // Only collect effects while STARTED, so a launcher/Toast never fires from a STOPPED
      // activity; the buffered Channel holds emissions across the stop and replays on resume.
      lifecycleOwner.repeatOnLifecycle(Lifecycle.State.STARTED) {
        kotlinx.coroutines.coroutineScope {
          launch { setupVm.effects.collect { handleCommonEffect(it, context) } }
          launch {
            modelsVm.effects.collect { effect ->
              when (effect) {
                Effect.PickModel -> modelLauncher.launch(arrayOf("*/*"))
                else -> handleCommonEffect(effect, context)
              }
            }
          }
          launch {
            jobsVm.effects.collect { effect ->
              when (effect) {
                is Effect.ExportCsv -> csvLauncher.launch(effect.filename)
                else -> handleCommonEffect(effect, context)
              }
            }
          }
          launch { settingsVm.effects.collect { handleCommonEffect(it, context) } }
        }
      }
    }

    // One imePadding for every screen: the keyboard shrinks the whole content area, so centered
    // layouts re-center above it and scrollable ones can reach their bottom content.
    Box(modifier = Modifier.fillMaxSize().background(PipetteTheme.colors.background).imePadding()) {
      when {
        shellState.pocket != null -> PocketModeScreen(shellState.pocket!!, onExit = { shell.exitPocketMode() })
        shellState.authGate !is AuthGate.Ready -> {
          val emailAuth by shell.emailAuth.collectAsStateWithLifecycle()
          AuthGateScreen(
            gate = shellState.authGate,
            emailAuth = emailAuth,
            oauthProviders = shellState.oauthProviders,
            isDebug = shellState.isDebug,
            onSubmitEmail = { shell.submitEmail(it) },
            onSubmitCode = { shell.submitCode(it) },
            onOAuthProvider = { shell.signInWithOAuth(it) },
            onUsePassword = { shell.usePasswordStep(it) },
            onSubmitPassword = { shell.submitPassword(it) },
            onSubmitNewPassword = { shell.submitNewPassword(it) },
            onStartPasswordReset = { shell.startPasswordReset() },
            onSubmitResetCode = { shell.submitPasswordResetCode(it) },
            onSubmitResetPassword = { shell.submitResetPassword(it) },
            onChooseSecondFactor = { shell.chooseSecondFactor(it) },
            onSubmitSecondFactor = { shell.submitSecondFactorCode(it) },
            onChangeEmail = { shell.changeEmail() },
            onEditClearError = { shell.clearAuthError() },
            onSkipDebug = { shell.setGateBypass(true) },
            onSignOut = { shell.signOut() },
            onDeleteIdentity = { shell.deleteDeviceIdentity() },
          )
        }
        shellState.needsRegistration -> {
          val s by setupVm.state.collectAsStateWithLifecycle()
          SetupScreen(s, setupVm::onIntent)
        }
        else -> Chrome(shellState, shell, modelsVm, jobsVm, settingsVm)
      }
    }
  }
}

@Composable
private fun Chrome(
  shellState: ai.liquid.pipette.compose.shell.ShellUiState,
  shell: ShellViewModel,
  modelsVm: ModelsViewModel,
  jobsVm: JobsViewModel,
  settingsVm: SettingsViewModel,
) {
  val jobsState by jobsVm.state.collectAsStateWithLifecycle()
  val modelsState by modelsVm.state.collectAsStateWithLifecycle()
  val settingsState by settingsVm.state.collectAsStateWithLifecycle()
  // Top-level tab container: NavDisplay animates between the three tabs. Each screen renders its own
  // header and handles its own full-screen covers (Jobs wizard/detail, Add Models, acknowledgements,
  // feedback) inline, so there is no shared app bar here.
  val backStack = rememberNavBackStack(shellState.selectedTab.toRoute())
  LaunchedEffect(shellState.selectedTab) {
    val root = shellState.selectedTab.toRoute()
    if (backStack.lastOrNull() != root) {
      backStack.clear()
      backStack.add(root)
    }
  }

  // Full-screen covers hide the pill bar (iOS fullScreenCover): the Jobs new-job wizard / cell detail
  // and the Add Models flow.
  val hidePillBar =
    (shellState.selectedTab == Tab.JOBS &&
      (jobsState is ai.liquid.pipette.compose.jobs.JobsUiState.Wizard || jobsState is ai.liquid.pipette.compose.jobs.JobsUiState.CellDetail)) ||
      (shellState.selectedTab == Tab.MODELS && modelsState.addModelsOpen)

  Box(modifier = Modifier.fillMaxSize()) {
    val riseOffsetPx = with(LocalDensity.current) { ScreenChangeRise.roundToPx() }
    NavDisplay(
      backStack = backStack,
      // Screen change: the outgoing screen fades out (alpha), while the incoming one fades in and
      // rises gently from the bottom. Applied to forward, pop, and predictive-pop so every tab
      // switch reads the same.
      transitionSpec = { screenChangeTransform(riseOffsetPx) },
      popTransitionSpec = { screenChangeTransform(riseOffsetPx) },
      predictivePopTransitionSpec = { screenChangeTransform(riseOffsetPx) },
      entryProvider =
        entryProvider<NavKey> {
          entry<Route.Jobs> { JobsScreen(jobsState, jobsVm::onIntent) }
          entry<Route.Models> { ModelsScreen(modelsState, modelsVm::onIntent) }
          entry<Route.Settings> { SettingsScreen(settingsState, settingsVm::onIntent) }
        },
    )
    if (!hidePillBar) {
      FloatingToolbarMenu(
        items = NAV_MENU_ITEMS,
        selectedKey = shellState.selectedTab,
        onSelect = { shell.selectTab(it) },
        modifier = Modifier.align(Alignment.BottomCenter).windowInsetsPadding(WindowInsets.navigationBars).padding(vertical = 8.dp),
      )
    }
  }
}

/** Map the shell's selected [Tab] to its top-level [Route]. */
private fun Tab.toRoute(): Route.TopLevel =
  when (this) {
    Tab.JOBS -> Route.Jobs
    Tab.MODELS -> Route.Models
    Tab.SETTINGS -> Route.Settings
  }

private fun handleCommonEffect(effect: Effect, context: android.content.Context) {
  if (effect is Effect.ShowError) Toast.makeText(context, effect.message, Toast.LENGTH_LONG).show()
}
