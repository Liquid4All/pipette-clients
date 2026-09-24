// Jobs list surface. The wizard (Create a job) and per-job detail (Job progress) live in their own files —
// CreateJobScreen.kt and JobDetailScreen.kt — and are wired as their own Navigation 3 destinations from
// PipetteAppRoot (Route.CreateJob / Route.JobDetail / Route.CellDetail). The shell mirrors the JobsUiState
// variant onto the backstack so predictive back and system back animate through Nav3.
@file:Suppress("TooManyFunctions", "MagicNumber", "MaxLineLength")

package ai.liquid.pipette.compose.jobs

import ai.liquid.pipette.AccentKind
import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.JobCell
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.JobStatus
import ai.liquid.pipette.R
import ai.liquid.pipette.compose.AndroidSearchBar
import ai.liquid.pipette.compose.AndroidTopAppBar
import ai.liquid.pipette.compose.IosDivider
import ai.liquid.pipette.compose.JobCardUi
import ai.liquid.pipette.compose.OutlinedAndroidCard
import ai.liquid.pipette.compose.PillTabBarReservedHeight
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import ai.liquid.pipette.compose.theme.serif
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Root Jobs tab surface. Renders the job list; the wizard and detail live in [CreateJobScreen] and [JobDetailScreen] respectively and are pushed as
 * their own Nav3 destinations. When the state is a non-list variant this composable renders nothing (the pushed screen is what the user sees).
 */
@Composable
fun JobsScreen(state: JobsUiState, onIntent: (JobsIntent) -> Unit) {
  if (state is JobsUiState.JobList) JobListScaffold(state, onIntent)
}

/** Pinned Material toolbar + independently scrolling content pane for the job list. */
@Composable
private fun JobListScaffold(state: JobsUiState.JobList, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  // Toolbar sits flush against the status-bar inset — no extra `top` padding
  // above it, so the toolbar is edge-to-edge. Post-toolbar content gets its
  // own top gap inside the scrolling pane. The "Create job" action lives in a
  // FloatingActionButton at the bottom-right (Android convention for the
  // screen's primary create action) rather than in the toolbar's actions slot.
  Box(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    Column(modifier = Modifier.fillMaxSize()) {
      AndroidTopAppBar(title = stringResource(R.string.job_list_title))
      Column(
        modifier =
          Modifier.weight(1f)
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp)
            .padding(top = 12.dp, bottom = 18.dp + PillTabBarReservedHeight)
      ) {
        JobListContent(state, onIntent)
      }
    }
    if (state.hasModels) {
      FloatingActionButton(
        onClick = { onIntent(JobsIntent.OpenWizard) },
        containerColor = colors.label,
        contentColor = colors.background,
        modifier = Modifier.align(Alignment.BottomEnd).padding(end = 16.dp, bottom = 16.dp + PillTabBarReservedHeight),
      ) {
        Icon(painter = painterResource(R.drawable.ic_plus), contentDescription = "Create job", modifier = Modifier.size(24.dp))
      }
    }
  }
}

@Composable
private fun EngineMissingBanner() {
  val colors = PipetteTheme.colors
  Box(
    modifier =
      Modifier.fillMaxWidth()
        .padding(top = 4.dp)
        .clip(RoundedCornerShape(12.dp))
        .background(colors.destructive.copy(alpha = 0.12f))
        .padding(horizontal = 16.dp, vertical = 12.dp)
  ) {
    Text(
      "Native benchmark engine missing: jobs can be planned, but cells will fail until libpipette_android.so is packaged.",
      style = TextStyle(fontSize = 13.sp, lineHeight = 18.sp),
      color = colors.destructive,
    )
  }
}

@Composable
private fun JobListContent(state: JobsUiState.JobList, onIntent: (JobsIntent) -> Unit) {
  if (!state.engineAvailable) {
    EngineMissingBanner()
    Spacer(Modifier.height(12.dp))
  }
  AndroidSearchBar(
    hint = stringResource(R.string.job_list_search),
    value = state.searchQuery,
    onValueChange = { onIntent(JobsIntent.ApplyJobSearch(it)) },
  )
  Spacer(Modifier.height(14.dp))
  when {
    !state.hasModels ->
      JobsEmptyState(
        title = stringResource(R.string.job_list_no_models_title),
        subtitle = stringResource(R.string.job_list_no_models_subtitle),
        buttonLabel = stringResource(R.string.job_list_go_to_models),
        onButton = { onIntent(JobsIntent.GoToModels) },
      )
    !state.anyJobs ->
      JobsEmptyState(
        title = stringResource(R.string.job_list_empty_title),
        subtitle = stringResource(R.string.job_list_empty_subtitle),
        buttonLabel = stringResource(R.string.job_list_create),
        buttonIcon = R.drawable.ic_plus,
        onButton = { onIntent(JobsIntent.OpenWizard) },
      )
    !state.matched ->
      JobsEmptyState(
        title = stringResource(R.string.job_list_no_match_title),
        subtitle = stringResource(R.string.job_list_no_match_subtitle),
        buttonLabel = null,
        onButton = {},
      )
    else ->
      OutlinedAndroidCard(cornerRadius = 18) {
        state.jobs.forEachIndexed { i, card ->
          if (i > 0) IosDivider(modifier = Modifier.padding(start = 20.dp))
          JobRow(card, onIntent)
        }
      }
  }
}

/** Centered empty state: faint skeleton rows + serif title + gray subtitle + optional capsule button (iOS JobsEmptyPrompt). */
@Composable
private fun JobsEmptyState(title: String, subtitle: String, buttonLabel: String?, onButton: () -> Unit, buttonIcon: Int? = null) {
  val colors = PipetteTheme.colors
  Column(modifier = Modifier.fillMaxWidth().padding(top = 80.dp), horizontalAlignment = Alignment.CenterHorizontally) {
    Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
      repeat(3) {
        Row(
          modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(colors.gray6).padding(16.dp),
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
          Box(Modifier.size(36.dp).clip(RoundedCornerShape(8.dp)).background(colors.gray5))
          Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Box(Modifier.height(10.dp).width(180.dp).clip(RoundedCornerShape(percent = 50)).background(colors.gray5))
            Box(Modifier.height(10.dp).width(110.dp).clip(RoundedCornerShape(percent = 50)).background(colors.gray5))
          }
        }
      }
    }
    Text(title, style = serif(24), color = colors.label, modifier = Modifier.padding(top = 24.dp))
    Text(
      subtitle,
      style = TextStyle(fontSize = 16.sp, lineHeight = 22.sp),
      color = colors.gray,
      textAlign = TextAlign.Center,
      modifier = Modifier.padding(top = 8.dp),
    )
    if (buttonLabel != null) {
      Box(
        modifier =
          Modifier.padding(top = 24.dp)
            .height(45.dp)
            .clip(RoundedCornerShape(percent = 50))
            .background(colors.label)
            .clickableNoRipple(onButton)
            .padding(horizontal = 24.dp),
        contentAlignment = Alignment.Center,
      ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          if (buttonIcon != null) {
            Icon(painter = painterResource(buttonIcon), contentDescription = null, tint = colors.background, modifier = Modifier.size(16.dp))
          }
          Text(buttonLabel, style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.SemiBold), color = colors.background)
        }
      }
    }
  }
}

/** Compact tappable job row: title + optional progress + meta line; whole row navigates to detail. */
@Composable
private fun JobRow(card: JobCardUi, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  Column(
    modifier =
      Modifier.fillMaxWidth().clickable { onIntent(JobsIntent.OpenJobDetail(card.manifest.jobId)) }.padding(horizontal = 20.dp, vertical = 16.dp),
    verticalArrangement = Arrangement.spacedBy(10.dp),
  ) {
    Text(card.manifest.displayTitle, style = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.SemiBold), color = colors.label, maxLines = 2)
    if (card.runningHere) {
      Box(modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(percent = 50)).background(colors.label.copy(alpha = 0.08f))) {
        Box(
          modifier =
            Modifier.fillMaxWidth(card.runProgress.coerceIn(0.0, 1.0).toFloat())
              .height(4.dp)
              .clip(RoundedCornerShape(percent = 50))
              .background(colors.label)
        )
      }
    }
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
      Text(card.rowPrimaryMeta, style = TextStyle(fontSize = 16.sp), color = colors.gray)
      Text(card.rowSecondaryMeta, style = TextStyle(fontSize = 16.sp), color = colors.gray)
    }
  }
}

@Preview
@Composable
private fun JobsScreenPreview() {
  val cells =
    mutableListOf(
      JobCell(
        benchmarkId = "mmlu",
        benchmarkType = "accuracy",
        modelPath = "/models/lfm2-1.2b-q4_k_m.gguf",
        modelName = "LFM2 1.2B",
        runStatus = CellRunStatus.COMPLETED,
      ),
      JobCell(benchmarkId = "hellaswag", benchmarkType = "accuracy", modelPath = "/models/lfm2-1.2b-q4_k_m.gguf", modelName = "LFM2 1.2B"),
    )
  PipetteTheme {
    JobsScreen(
      state =
        JobsUiState.JobList(
          hasModels = true,
          anyJobs = true,
          jobs =
            listOf(
              JobCardUi(
                manifest =
                  JobManifest(
                    createdAt = "2026-08-05T09:00:00Z",
                    nGpuLayers = 99,
                    contextSize = 4096,
                    cells = cells,
                    status = JobStatus.RUNNING,
                    title = "Nightly sweep",
                  ),
                statusAccent = AccentKind.NOMINAL,
                runningHere = true,
                runProgress = 0.5,
                countsLine = "1 of 2 cells done",
                rowPrimaryMeta = "1 model - 2 benchmarks",
                rowSecondaryMeta = "Created 2026-08-05",
                firstFailure = null,
                canResume = false,
                completedCells = 1,
                unsubmittedCount = 1,
                isRegistered = true,
              )
            ),
        ),
      onIntent = {},
    )
  }
}

@Preview
@Composable
private fun JobsScreenEmptyPreview() {
  PipetteTheme { JobsScreen(state = JobsUiState.JobList(hasModels = false, anyJobs = false), onIntent = {}) }
}
