// Root scaffold: layout literals (MagicNumber, e.g. the reserved tab-bar height).
@file:Suppress("MagicNumber")

package ai.liquid.pipette.compose

import ai.liquid.pipette.AuthGate
import ai.liquid.pipette.R
import ai.liquid.pipette.Tab
import ai.liquid.pipette.compose.jobs.CellDetailScreen
import ai.liquid.pipette.compose.jobs.CreateJobScreen
import ai.liquid.pipette.compose.jobs.JobDetailScreen
import ai.liquid.pipette.compose.jobs.JobsScreen
import ai.liquid.pipette.compose.jobs.JobsUiState
import ai.liquid.pipette.compose.jobs.JobsViewModel
import ai.liquid.pipette.compose.models.AddModelsScreen
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
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

// Adding more Nav3 entries pushed this scaffold past detekt's default 15-branch threshold. The branchiness is inherent
// to a top-level tab container that hosts every route + the state → backstack mirror + the pill-bar visibility check;
// splitting it further doesn't buy readability. Keep the suppression scoped to just this function.
@Suppress("CyclomaticComplexMethod")
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
    // Reset the stack to the tab's root on tab switch. This also drops any pushed detail (e.g. Add models)
    // so switching tabs doesn't strand a nested screen behind the scenes.
    if (backStack.lastOrNull() != root) {
      backStack.clear()
      backStack.add(root)
    }
  }
  // Mirror the Models VM's addModelsOpen flag onto the backstack: opening the flow pushes Route.AddModels,
  // closing it (via the back button, system back, or a successful download) pops the entry. The VM stays
  // the source of truth so nothing else in the app has to know about the nav wiring. Keyed on the selected
  // tab too so that returning to Models with the flow still open (the VM keeps its state across tab
  // switches) re-pushes Route.AddModels after the tab-switch effect above clears the stack — otherwise the
  // user comes back to a blank list. Guarded to no-op while another tab is active so we don't push a Models
  // route onto Jobs/Settings.
  LaunchedEffect(modelsState.addModelsOpen, shellState.selectedTab) {
    if (shellState.selectedTab != Tab.MODELS) return@LaunchedEffect
    val hasAddModels = backStack.contains(Route.AddModels)
    if (modelsState.addModelsOpen && !hasAddModels) backStack.add(Route.AddModels)
    else if (!modelsState.addModelsOpen && hasAddModels) backStack.remove(Route.AddModels)
  }

  // Mirror the Jobs VM's sealed state variant onto the backstack: JobList stays on Route.Jobs, Wizard pushes
  // Route.CreateJob, Detail pushes Route.JobDetail, CellDetail stacks Route.CellDetail on top of Route.JobDetail
  // (since a cell detail is always opened from a job detail). Popping any of those (predictive back, system
  // back, in-body back button) reduces to the equivalent JobsIntent, which flips the state back so the mirror
  // stays consistent. The VM remains the source of truth.
  //
  // Keyed on the selected tab too so that returning to Jobs after switching tabs mid-Detail (the tab-switch
  // effect above clears the stack to just Route.Jobs) re-pushes the wizard / detail / cell entries; without
  // this the list would render blank because JobsScreen renders nothing when the state is a non-JobList
  // variant, and the next interaction that assumed a live JobDetail entry would crash. Guarded to no-op
  // while another tab is active so we don't push Jobs routes on top of Models/Settings.
  LaunchedEffect(jobsState, shellState.selectedTab) {
    if (shellState.selectedTab != Tab.JOBS) return@LaunchedEffect
    val wantWizard = jobsState is JobsUiState.Wizard
    val wantDetail = jobsState is JobsUiState.Detail
    val wantCell = jobsState is JobsUiState.CellDetail
    val hasWizard = backStack.contains(Route.CreateJob)
    val hasDetail = backStack.contains(Route.JobDetail)
    val hasCell = backStack.contains(Route.CellDetail)
    if (wantWizard && !hasWizard) backStack.add(Route.CreateJob) else if (!wantWizard && hasWizard) backStack.remove(Route.CreateJob)
    // The cell detail is opened from the job detail, so the shell keeps Route.JobDetail underneath it while
    // the cell is on top. Detail == true covers both the plain-Detail state and CellDetail.
    val needsDetail = wantDetail || wantCell
    if (needsDetail && !hasDetail) backStack.add(Route.JobDetail) else if (!needsDetail && hasDetail) backStack.remove(Route.JobDetail)
    if (wantCell && !hasCell) backStack.add(Route.CellDetail) else if (!wantCell && hasCell) backStack.remove(Route.CellDetail)
  }

  // Full-screen covers hide the pill bar (iOS fullScreenCover): the Jobs new-job wizard / cell detail
  // and the Add Models flow.
  val hidePillBar =
    (shellState.selectedTab == Tab.JOBS && (jobsState is JobsUiState.Wizard || jobsState is JobsUiState.CellDetail)) ||
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
          entry<Route.AddModels> { AddModelsScreen(modelsState, modelsVm::onIntent) }
          // The wizard/detail/cell entries need a JobsUiState variant to render; during the pop animation the
          // state has already flipped back, so latch the last non-null variant and keep rendering it while the
          // Nav3 entry animates out. Once the state matches again the latch is refreshed to the live value.
          entry<Route.CreateJob> {
            var last by remember { mutableStateOf<JobsUiState.Wizard?>(null) }
            (jobsState as? JobsUiState.Wizard)?.let { last = it }
            last?.let { CreateJobScreen(it, jobsVm::onIntent) }
          }
          entry<Route.JobDetail> {
            var last by remember { mutableStateOf<JobsUiState.Detail?>(null) }
            (jobsState as? JobsUiState.Detail)?.let { last = it }
            last?.let { JobDetailScreen(it, jobsVm::onIntent) }
          }
          entry<Route.CellDetail> {
            var last by remember { mutableStateOf<JobsUiState.CellDetail?>(null) }
            (jobsState as? JobsUiState.CellDetail)?.let { last = it }
            last?.let { CellDetailScreen(it, jobsVm::onIntent) }
          }
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
