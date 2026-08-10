// Jobs UI: list/wizard/detail across several composables (TooManyFunctions) and a branchy detail screen (CyclomaticComplexMethod).
@file:Suppress("TooManyFunctions", "CyclomaticComplexMethod", "MagicNumber")

package ai.liquid.pipette.compose.jobs

import ai.liquid.pipette.AccentKind
import ai.liquid.pipette.CellRunStatus
import ai.liquid.pipette.JobCell
import ai.liquid.pipette.JobManifest
import ai.liquid.pipette.JobStatus
import ai.liquid.pipette.R
import ai.liquid.pipette.compose.AppTextField
import ai.liquid.pipette.compose.Chip
import ai.liquid.pipette.compose.ConfirmAction
import ai.liquid.pipette.compose.IosCard
import ai.liquid.pipette.compose.IosDivider
import ai.liquid.pipette.compose.JobActivityColors
import ai.liquid.pipette.compose.JobCardUi
import ai.liquid.pipette.compose.JobLiveActivity
import ai.liquid.pipette.compose.MutedLabel
import ai.liquid.pipette.compose.OutlineButton
import ai.liquid.pipette.compose.PageHeaderLarge
import ai.liquid.pipette.compose.PillTabBarReservedHeight
import ai.liquid.pipette.compose.PrimaryButton
import ai.liquid.pipette.compose.PropertyChipRow
import ai.liquid.pipette.compose.QuantPill
import ai.liquid.pipette.compose.ResultsGridUi
import ai.liquid.pipette.compose.RotatingChevron
import ai.liquid.pipette.compose.SearchBlock
import ai.liquid.pipette.compose.SearchField
import ai.liquid.pipette.compose.accentColor
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.activity.compose.BackHandler
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Run setup (wizard), the job list, and per-job detail. */
@Composable
fun JobsScreen(state: JobsUiState, onIntent: (JobsIntent) -> Unit) {
  // The wizard and cell detail are full-screen covers with their own top chrome; the other states scroll under the pill bar.
  if (state is JobsUiState.Wizard) {
    // Back steps through the wizard; only the first step backs out to the jobs list.
    BackHandler { if (state.step > 0) onIntent(JobsIntent.WizardGoToStep(state.step - 1)) else onIntent(JobsIntent.CancelWizard) }
    WizardContent(state, onIntent)
    return
  }
  if (state is JobsUiState.CellDetail) {
    CellDetailScreen(state, onIntent)
    return
  }
  if (state is JobsUiState.Detail) BackHandler { onIntent(JobsIntent.BackToJobs) }
  Column(
    modifier =
      Modifier.fillMaxSize()
        .verticalScroll(rememberScrollState())
        .windowInsetsPadding(WindowInsets.statusBars)
        .padding(horizontal = 20.dp)
        .padding(top = 12.dp, bottom = 18.dp + PillTabBarReservedHeight)
  ) {
    when (state) {
      is JobsUiState.JobList -> JobListContent(state, onIntent)
      is JobsUiState.Detail -> DetailContent(state, onIntent)
      is JobsUiState.CellDetail -> Unit
      is JobsUiState.Wizard -> Unit
    }
  }
}

/**
 * Full-bleed cell-detail cover: a top bar with an edge-to-edge bottom rule, an info section (max-three property rows behind a "Show more" toggle), an
 * edge-to-edge separator, then the payload table. System back closes the cover rather than the app.
 */
@Composable
private fun CellDetailScreen(state: JobsUiState.CellDetail, onIntent: (JobsIntent) -> Unit) {
  val cell = state.cell
  val colors = PipetteTheme.colors
  BackHandler { onIntent(JobsIntent.CloseCellDetail) }
  var propsExpanded by remember { mutableStateOf(false) }
  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    // Top bar: back chevron + centered title, with an edge-to-edge rule below.
    Box(modifier = Modifier.fillMaxWidth().height(52.dp).padding(horizontal = 20.dp)) {
      Icon(
        painter = painterResource(R.drawable.ic_chevron_left),
        contentDescription = null,
        tint = colors.label,
        modifier = Modifier.align(Alignment.CenterStart).size(24.dp).clickableNoRipple { onIntent(JobsIntent.CloseCellDetail) },
      )
      Text(
        stringResource(R.string.cell_detail_title),
        style = ai.liquid.pipette.compose.theme.serif(20),
        color = colors.label,
        modifier = Modifier.align(Alignment.Center),
      )
    }
    IosDivider()
    Column(modifier = Modifier.weight(1f).verticalScroll(rememberScrollState())) {
      // Info section: first three properties always visible, the rest collapse behind a "Show N more" toggle.
      Column(modifier = Modifier.padding(horizontal = 20.dp).padding(top = 8.dp, bottom = 16.dp)) {
        PropertyChipRow(stringResource(R.string.property_models), listOf(cell.modelName)) { ModelChip(it) }
        PropertyChipRow(stringResource(R.string.property_quant), listOf(cell.quant))
        PropertyChipRow(stringResource(R.string.property_benchmark), listOf(cell.benchmarkLabel))
        if (propsExpanded) PropertyChipRow(stringResource(R.string.property_status), listOf(cell.statusLabel))
        MorePropertiesToggle(expanded = propsExpanded, hiddenCount = 1) { propsExpanded = !propsExpanded }
        cell.errorLine?.let {
          Spacer(Modifier.height(12.dp))
          Row(
            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(colors.red.copy(alpha = 0.08f)).padding(14.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.Top,
          ) {
            Icon(painter = painterResource(R.drawable.ic_warning), contentDescription = null, tint = colors.red, modifier = Modifier.size(14.dp))
            Text(it, style = TextStyle(fontSize = 14.sp), color = colors.label)
          }
        }
        // Per-cell actions (iOS cell/result detail): re-run a failed/cancelled cell, submit a completed one.
        if (cell.rerunSelectable) {
          Spacer(Modifier.height(12.dp))
          OutlineButton("Re-run this cell", { onIntent(JobsIntent.RerunCell(state.jobId, cell.cell.cellId)) }, leadingIcon = R.drawable.ic_retry)
        }
        if (cell.canSubmit) {
          ConfirmAction("Submit this result?", "Submit", onConfirm = { onIntent(JobsIntent.SubmitCellResult(state.jobId, cell.cell.cellId)) }) {
            trigger ->
            PrimaryButton("Submit result", trigger, loading = cell.submitting, leadingIcon = R.drawable.ic_upload)
          }
        }
      }
      IosDivider()
      // Payload table section.
      Column(modifier = Modifier.padding(horizontal = 20.dp).padding(top = 18.dp, bottom = 18.dp + PillTabBarReservedHeight)) {
        if (cell.detailRows.isEmpty()) {
          MutedLabel(stringResource(R.string.cell_detail_no_payload))
        } else {
          IosCard(cornerRadius = 20) {
            cell.detailRows.forEachIndexed { i, (label, value) ->
              if (i > 0) IosDivider()
              Row(modifier = Modifier.fillMaxWidth().height(64.dp).padding(horizontal = 20.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(label, style = TextStyle(fontSize = 16.sp), color = colors.gray, modifier = Modifier.width(168.dp))
                Text(value, style = TextStyle(fontSize = 16.sp), color = colors.label, modifier = Modifier.weight(1f))
              }
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Job list
// ---------------------------------------------------------------------------

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
  val colors = PipetteTheme.colors
  if (!state.engineAvailable) EngineMissingBanner()
  Row(modifier = Modifier.fillMaxWidth().padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
    PageHeaderLarge(stringResource(R.string.job_list_title), modifier = Modifier.weight(1f))
    if (state.hasModels) {
      Box(
        modifier =
          Modifier.size(44.dp).clip(androidx.compose.foundation.shape.CircleShape).background(colors.label).clickableNoRipple {
            onIntent(JobsIntent.OpenWizard)
          },
        contentAlignment = Alignment.Center,
      ) {
        Icon(painter = painterResource(R.drawable.ic_plus), contentDescription = null, tint = colors.background, modifier = Modifier.size(22.dp))
      }
    }
  }
  androidx.compose.foundation.layout.Spacer(Modifier.height(14.dp))
  SearchBlock(stringResource(R.string.job_list_search), state.searchQuery, { onIntent(JobsIntent.ApplyJobSearch(it)) })
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
      IosCard(cornerRadius = 18) {
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
    // Skeleton placeholder rows (the design's "ghost" list illustration).
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
    Text(title, style = ai.liquid.pipette.compose.theme.serif(24), color = colors.label, modifier = Modifier.padding(top = 24.dp))
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

/** Compact tappable job row (iOS JobRow): title + optional progress + meta line; whole row navigates to detail. */
@Composable
private fun JobRow(card: ai.liquid.pipette.compose.JobCardUi, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  Column(
    modifier =
      Modifier.fillMaxWidth()
        .clickableNoRipple { onIntent(JobsIntent.OpenJobDetail(card.manifest.jobId)) }
        .padding(horizontal = 20.dp, vertical = 16.dp),
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

// ---------------------------------------------------------------------------
// Wizard
// ---------------------------------------------------------------------------

@Composable
private fun WizardContent(state: JobsUiState.Wizard, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    // Fixed header: centered "Create a job" + close, with the segmented step-progress underline.
    Box(modifier = Modifier.fillMaxWidth().height(52.dp).padding(horizontal = 20.dp), contentAlignment = Alignment.Center) {
      Text(stringResource(R.string.job_wizard_title), style = ai.liquid.pipette.compose.theme.serif(20), color = colors.label)
      Icon(
        painter = painterResource(R.drawable.ic_close),
        contentDescription = null,
        tint = colors.label,
        modifier = Modifier.align(Alignment.CenterEnd).size(20.dp).clickableNoRipple { onIntent(JobsIntent.CancelWizard) },
      )
    }
    Row(modifier = Modifier.fillMaxWidth().height(2.dp)) {
      repeat(state.stepTitles.size) { i ->
        Box(modifier = Modifier.weight(1f).fillMaxHeight().background(if (i <= state.step) colors.label else colors.gray5))
      }
    }
    // Scrollable step body.
    Column(modifier = Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 24.dp).padding(top = 20.dp, bottom = 16.dp)) {
      when (state.step) {
        0 -> WizardStepModels(state, onIntent)
        1 -> WizardStepBenchmarks(state, onIntent)
        else -> WizardStepReview(state, onIntent)
      }
    }
    // Fixed footer.
    Row(
      modifier = Modifier.fillMaxWidth().windowInsetsPadding(WindowInsets.navigationBars).padding(horizontal = 24.dp, vertical = 12.dp),
      horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
      if (state.step > 0) {
        Box(
          modifier =
            Modifier.height(52.dp)
              .clip(RoundedCornerShape(percent = 50))
              .border(androidx.compose.foundation.BorderStroke(1.dp, colors.gray3), RoundedCornerShape(percent = 50))
              .clickableNoRipple { onIntent(JobsIntent.WizardGoToStep(state.step - 1)) }
              .padding(horizontal = 28.dp),
          contentAlignment = Alignment.Center,
        ) {
          Text(stringResource(R.string.job_wizard_back), style = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.Medium), color = colors.label)
        }
      }
      if (state.step < state.stepTitles.lastIndex) {
        WizardPrimary(stringResource(R.string.job_wizard_next), enabled = state.canAdvance, modifier = Modifier.weight(1f)) {
          onIntent(JobsIntent.WizardGoToStep(state.step + 1))
        }
      } else {
        WizardPrimary(
          text = if (state.canRun) stringResource(R.string.job_wizard_run) else state.runLabel,
          enabled = state.canRun,
          leadingIcon = if (state.canRun) R.drawable.ic_play else null,
          modifier = Modifier.weight(1f),
        ) {
          onIntent(JobsIntent.RunJob(state.nGpuLayers, state.contextSize, state.prefillBatch))
        }
      }
    }
  }
}

@Composable
private fun WizardPrimary(text: String, enabled: Boolean, modifier: Modifier = Modifier, leadingIcon: Int? = null, onClick: () -> Unit) {
  val colors = PipetteTheme.colors
  Box(
    modifier =
      modifier
        .height(52.dp)
        .clip(RoundedCornerShape(percent = 50))
        .background(if (enabled) colors.label else colors.gray3)
        .then(if (enabled) Modifier.clickableNoRipple(onClick) else Modifier),
    contentAlignment = Alignment.Center,
  ) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      if (leadingIcon != null) {
        Icon(painter = painterResource(leadingIcon), contentDescription = null, tint = colors.background, modifier = Modifier.size(18.dp))
      }
      Text(text, style = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.Medium), color = colors.background)
    }
  }
}

@Composable
private fun WizardStepModels(state: JobsUiState.Wizard, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  Text(stringResource(R.string.job_wizard_models_title), style = ai.liquid.pipette.compose.theme.serif(26), color = colors.label)
  Text(
    stringResource(R.string.job_wizard_models_subtitle),
    style = TextStyle(fontSize = 16.sp),
    color = colors.gray,
    modifier = Modifier.padding(top = 4.dp, bottom = 18.dp),
  )
  SearchField(
    value = state.modelSearch,
    onValueChange = { onIntent(JobsIntent.ApplyJobModelSearch(it)) },
    placeholder = stringResource(R.string.job_wizard_search_models),
  )
  Spacer(Modifier.height(14.dp))
  when {
    !state.anyModelGroups ->
      JobsEmptyState(
        title = stringResource(R.string.job_wizard_no_models_title),
        subtitle = stringResource(R.string.job_wizard_no_models_subtitle),
        buttonLabel = stringResource(R.string.job_wizard_go_to_models),
        onButton = { onIntent(JobsIntent.GoToModels) },
      )
    !state.modelsMatched -> MutedLabel(stringResource(R.string.job_wizard_no_models_match, state.modelSearch))
    else ->
      IosCard(cornerRadius = 16) {
        state.modelGroups.forEachIndexed { i, group ->
          if (i > 0) IosDivider()
          ModelSelectRow(group.name, group.sizeLabel, group.checked) { onIntent(JobsIntent.ToggleModelGroup(group.key, !group.checked)) }
        }
      }
  }
  Spacer(Modifier.height(28.dp))
  Text(stringResource(R.string.job_wizard_quants_title), style = ai.liquid.pipette.compose.theme.serif(21), color = colors.label)
  Text(
    stringResource(R.string.job_wizard_quants_subtitle),
    style = TextStyle(fontSize = 16.sp),
    color = colors.gray,
    modifier = Modifier.padding(top = 4.dp, bottom = 14.dp),
  )
  QuantPillRow(state, onIntent)
}

/** A model row in the wizard list: brand placeholder + name + size + a square checkbox. */
@Composable
private fun ModelSelectRow(name: String, size: String, checked: Boolean, onToggle: () -> Unit) {
  val colors = PipetteTheme.colors
  Row(
    modifier = Modifier.fillMaxWidth().clickableNoRipple(onToggle).padding(horizontal = 18.dp, vertical = 16.dp),
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(12.dp),
  ) {
    ai.liquid.pipette.compose.BrandLogo(name, size = 26.dp)
    Column(modifier = Modifier.weight(1f)) {
      Text(
        name,
        style = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.Medium),
        color = colors.label,
        maxLines = 1,
        overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
      )
      Text(size, style = TextStyle(fontSize = 14.sp), color = colors.gray, modifier = Modifier.padding(top = 2.dp))
    }
    ai.liquid.pipette.compose.WizardCheckbox(isOn = checked, size = 22)
  }
}

/** "All quants" + per-quant pills (multi-select, black when selected) with a divider after "All quants". */
@Composable
private fun QuantPillRow(state: JobsUiState.Wizard, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    state.quantFilters.forEachIndexed { i, chip ->
      QuantPill(chip.label, chip.selected) { onIntent(JobsIntent.ToggleQuantFilter(chip.filter, !chip.selected)) }
      if (i == 0) Box(Modifier.width(1.dp).height(22.dp).background(colors.gray4))
    }
  }
}

@Composable
private fun WizardStepBenchmarks(state: JobsUiState.Wizard, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  // Saveable so expanded groups survive rotation and wizard step navigation (HashSet is Serializable).
  var expanded by rememberSaveable { mutableStateOf(HashSet<String>()) }
  Text(stringResource(R.string.job_wizard_benchmarks_title), style = ai.liquid.pipette.compose.theme.serif(26), color = colors.label)
  Text(
    stringResource(R.string.job_wizard_benchmarks_subtitle),
    style = TextStyle(fontSize = 16.sp, lineHeight = 22.sp),
    color = colors.gray,
    modifier = Modifier.padding(top = 4.dp, bottom = 18.dp),
  )
  SearchField(
    value = state.benchmarkSearch,
    onValueChange = { onIntent(JobsIntent.ApplyBenchmarkSearch(it)) },
    placeholder = stringResource(R.string.job_wizard_search_benchmarks),
  )
  Spacer(Modifier.height(14.dp))
  if (!state.benchmarksMatched) {
    MutedLabel(stringResource(R.string.job_wizard_no_benchmarks_match, state.benchmarkSearch))
  } else {
    IosCard(cornerRadius = 16) {
      state.benchmarkGroups.forEachIndexed { i, group ->
        if (i > 0) IosDivider()
        BenchmarkGroup(
          group,
          expanded.contains(group.type),
          onToggleExpand = { expanded = HashSet(expanded).apply { if (!add(group.type)) remove(group.type) } },
          onIntent = onIntent,
        )
      }
    }
  }
  if (state.showMmprojCard) {
    Spacer(Modifier.height(20.dp))
    Text(stringResource(R.string.job_wizard_mmproj_title), style = ai.liquid.pipette.compose.theme.serif(21), color = colors.label)
    Text(
      stringResource(R.string.job_wizard_mmproj_subtitle),
      style = TextStyle(fontSize = 14.sp),
      color = colors.gray,
      modifier = Modifier.padding(top = 4.dp, bottom = 12.dp),
    )
    if (state.mmprojs.isEmpty()) {
      MutedLabel(stringResource(R.string.job_wizard_mmproj_empty))
    } else {
      IosCard(cornerRadius = 16) {
        state.mmprojs.forEachIndexed { i, row ->
          if (i > 0) IosDivider()
          Row(
            modifier =
              Modifier.fillMaxWidth()
                .clickableNoRipple { onIntent(JobsIntent.ToggleMmproj(row.path, !row.checked)) }
                .padding(horizontal = 18.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
          ) {
            Text(row.label, style = TextStyle(fontSize = 16.sp), color = colors.label, modifier = Modifier.weight(1f))
            ai.liquid.pipette.compose.WizardCheckbox(isOn = row.checked, size = 22)
          }
        }
      }
    }
  }
}

/** A collapsible benchmark-type group: chevron + title + description + tri-state checkbox; expands to context-size pill rows. */
@Composable
private fun BenchmarkGroup(
  group: ai.liquid.pipette.compose.BenchmarkGroupUi,
  isExpanded: Boolean,
  onToggleExpand: () -> Unit,
  onIntent: (JobsIntent) -> Unit,
) {
  val colors = PipetteTheme.colors
  Row(modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 16.dp), verticalAlignment = Alignment.Top) {
    RotatingChevron(
      expanded = isExpanded,
      tint = colors.gray,
      modifier = Modifier.width(20.dp).clickableNoRipple { if (!group.disabled) onToggleExpand() },
    )
    Column(modifier = Modifier.weight(1f).padding(start = 4.dp, end = 12.dp).clickableNoRipple { if (!group.disabled) onToggleExpand() }) {
      Text(group.displayName, style = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.Medium), color = colors.label)
      Text(
        if (group.disabled) stringResource(R.string.job_wizard_benchmark_requires_mmproj) else group.description,
        style = TextStyle(fontSize = 14.sp, lineHeight = 19.sp),
        color = colors.gray,
        modifier = Modifier.padding(top = 2.dp),
      )
    }
    if (!group.disabled) {
      Box(modifier = Modifier.clickableNoRipple { onIntent(JobsIntent.ToggleBenchmarkGroup(group.type, !group.allSelected)) }) {
        ai.liquid.pipette.compose.WizardCheckbox(isOn = group.allSelected, indeterminate = group.someSelected, size = 22)
      }
    }
  }
  if (isExpanded && !group.disabled) {
    // Indented content with a vertical guide line on the left (iOS DisclosureGroup).
    Row(modifier = Modifier.fillMaxWidth().background(colors.secondaryBackground).height(IntrinsicSize.Min)) {
      Box(modifier = Modifier.padding(start = 28.dp).width(1.dp).fillMaxHeight().background(colors.gray4))
      Column(modifier = Modifier.weight(1f)) {
        group.items.forEach { item ->
          Row(
            modifier =
              Modifier.fillMaxWidth()
                .clickableNoRipple { if (item.enabled) onIntent(JobsIntent.ToggleBenchmark(item.id, !item.checked)) }
                .padding(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
          ) {
            Chip(item.label, fontSize = 14.sp)
            Spacer(Modifier.weight(1f))
            ai.liquid.pipette.compose.WizardCheckbox(isOn = item.checked, size = 22)
          }
        }
      }
    }
  }
}

@Composable
private fun WizardStepReview(state: JobsUiState.Wizard, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  Text(
    stringResource(R.string.job_wizard_review_title),
    style = ai.liquid.pipette.compose.theme.serif(26),
    color = colors.label,
    modifier = Modifier.padding(bottom = 16.dp),
  )
  IosCard(cornerRadius = 16) {
    // Date header + divider.
    Text(
      "${state.reviewDate} · ${state.reviewSubtitle}",
      style = ai.liquid.pipette.compose.theme.serif(17),
      color = colors.label,
      modifier = Modifier.padding(horizontal = 20.dp, vertical = 18.dp),
    )
    IosDivider()
    Column(modifier = Modifier.padding(horizontal = 20.dp, vertical = 18.dp), verticalArrangement = Arrangement.spacedBy(18.dp)) {
      ReviewChipSection(stringResource(R.string.job_wizard_section_models), state.reviewModels, withLogos = true)
      ReviewChipSection(stringResource(R.string.job_wizard_section_benchmarks), state.reviewBenchmarks, withLogos = false)
      ReviewChipSection(stringResource(R.string.job_wizard_section_quants), state.reviewQuants, withLogos = false)
    }
  }
  if (state.showSkippedWarning) {
    Spacer(Modifier.height(8.dp))
    Row(verticalAlignment = Alignment.Top, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      Icon(painter = painterResource(R.drawable.ic_warning), contentDescription = null, tint = colors.orange, modifier = Modifier.size(14.dp))
      Text(state.skippedWarning, style = TextStyle(fontSize = 13.sp), color = colors.gray)
    }
  }
  ContributeRow(state.contributeResults, state.isRegistered) { onIntent(JobsIntent.SetWizardContribute(it)) }
}

/** A labeled review section: gray label + a flow of chips (model chips carry a brand logo). */
@Composable
private fun ReviewChipSection(label: String, values: List<String>, withLogos: Boolean) {
  val colors = PipetteTheme.colors
  Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
    Text(label, style = TextStyle(fontSize = 16.sp), color = colors.gray)
    androidx.compose.foundation.layout.FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
      values.forEach { v -> if (withLogos) ModelChip(v) else Chip(v) }
    }
  }
}

/** A model chip: brand logo + (truncated) name inside a hairline capsule. */
@Composable
private fun ModelChip(name: String) {
  val colors = PipetteTheme.colors
  Row(
    modifier =
      Modifier.height(34.dp)
        .clip(RoundedCornerShape(percent = 50))
        .background(colors.background)
        .border(androidx.compose.foundation.BorderStroke(1.dp, colors.label.copy(alpha = 0.10f)), RoundedCornerShape(percent = 50))
        .padding(start = 8.dp, end = 14.dp),
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(7.dp),
  ) {
    ai.liquid.pipette.compose.BrandLogo(name, size = 20.dp)
    Text(
      name,
      style = TextStyle(fontSize = 15.sp),
      color = colors.label,
      maxLines = 1,
      overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
      modifier = Modifier.widthIn(max = 160.dp),
    )
  }
}

// ---------------------------------------------------------------------------
// Detail
// ---------------------------------------------------------------------------

@Composable
private fun DetailContent(state: JobsUiState.Detail, onIntent: (JobsIntent) -> Unit) {
  val manifest = state.manifest
  val colors = PipetteTheme.colors
  var renaming by remember { mutableStateOf(false) }
  // Local visual cue only: flips the Pause button to "Pausing…" on tap; resets once the run ends.
  var pausing by remember(state.runningHere) { mutableStateOf(false) }

  DetailNavBar(state, onIntent, onRename = { renaming = true })

  Text(state.titleDate, style = ai.liquid.pipette.compose.theme.serif(24), color = colors.label, modifier = Modifier.padding(top = 8.dp))
  Text(
    state.subtitle,
    style = ai.liquid.pipette.compose.theme.serif(18),
    color = colors.gray,
    modifier = Modifier.padding(top = 2.dp, bottom = 16.dp),
  )

  // Header property rows: first three always visible, the rest collapse behind a "Show N more" toggle.
  var propsExpanded by remember { mutableStateOf(false) }
  PropertyChipRow(stringResource(R.string.property_models), state.modelChips) { ModelChip(it) }
  PropertyChipRow(stringResource(R.string.property_benchmarks), state.benchmarkChips)
  PropertyChipRow(stringResource(R.string.property_quants), state.quantChips)
  if (propsExpanded) {
    PropertyChipRow(
      stringResource(R.string.job_detail_property_gpu),
      listOf(stringResource(if (state.gpuLayers > 0) R.string.job_detail_gpu_on else R.string.job_detail_gpu_off)),
    )
    PropertyChipRow(stringResource(R.string.job_detail_property_context), listOf(state.contextSize.toString()))
  }
  MorePropertiesToggle(expanded = propsExpanded, hiddenCount = 2) { propsExpanded = !propsExpanded }

  IosDivider(modifier = Modifier.padding(vertical = 18.dp))

  if (state.runningHere) {
    Text(
      stringResource(R.string.job_detail_in_progress),
      style = ai.liquid.pipette.compose.theme.serif(28),
      color = colors.label,
      modifier = Modifier.padding(bottom = 12.dp),
    )
    DetailProgressBar(state.runProgress)
    Row(modifier = Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.SpaceBetween) {
      Text(state.runCellsDone, style = TextStyle(fontSize = 16.sp), color = colors.gray, maxLines = 1)
      Text(state.runTimeLeft, style = TextStyle(fontSize = 16.sp), color = colors.gray, maxLines = 1)
    }
    // The same live indicators Pocket Mode shows, so watching a running job here is no less
    // informative than the full-screen cover: a device-temperature (thermal-headroom) row plus the
    // shared cell/progress block. An ambient cool wash washes the block while the gate is cooling.
    val cooling = state.coolingSinceMillis != null
    Column(
      modifier =
        Modifier.fillMaxWidth()
          .padding(top = 16.dp)
          .clip(RoundedCornerShape(12.dp))
          .background(if (cooling) androidx.compose.ui.graphics.Color(0x14589BF7) else androidx.compose.ui.graphics.Color.Transparent)
          .padding(horizontal = 12.dp, vertical = 12.dp),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text("Throttling headroom", style = TextStyle(fontSize = 16.sp), color = colors.gray, modifier = Modifier.weight(1f))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
          Box(Modifier.size(8.dp).clip(androidx.compose.foundation.shape.CircleShape).background(accentColor(state.thermalAccent)))
          Text(state.thermalLabel, style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Medium), color = colors.label)
        }
      }
      JobLiveActivity(
        currentCellLabel = state.runCellLabel,
        progressText = state.runProgressText,
        coolingSinceMillis = state.coolingSinceMillis,
        colors = JobActivityColors(primaryText = colors.label, secondaryText = colors.gray, accent = androidx.compose.ui.graphics.Color(0xFF60A5FA)),
      )
    }
    PrimaryButton(
      stringResource(R.string.job_detail_open_pocket),
      { onIntent(JobsIntent.OpenPocketMode(manifest.jobId)) },
      modifier = Modifier.padding(top = 24.dp),
      leadingIcon = R.drawable.ic_pocket,
    )
    OutlineButton(
      stringResource(if (pausing) R.string.job_detail_pausing else R.string.job_detail_pause),
      {
        if (!pausing) {
          pausing = true
          onIntent(JobsIntent.CancelRunningJob)
        }
      },
      leadingIcon = R.drawable.ic_pause,
    )
    ContributeRow(state.contributeResults, state.isRegistered) { onIntent(JobsIntent.SetJobAutoSubmit(manifest.jobId, it)) }
  } else {
    val paused = state.canResume
    if (paused) {
      Text(
        stringResource(R.string.job_detail_paused),
        style = ai.liquid.pipette.compose.theme.serif(28),
        color = colors.label,
        modifier = Modifier.padding(bottom = 12.dp),
      )
      DetailProgressBar(state.runProgress)
      Spacer(Modifier.height(8.dp))
    }
    // In-body action buttons (iOS keeps these out of the overflow menu).
    if (state.unsubmittedCount > 0 && state.isRegistered) {
      ConfirmAction(
        pluralStringResource(R.plurals.job_detail_submit_confirm, state.unsubmittedCount, state.unsubmittedCount),
        stringResource(R.string.job_detail_submit_action),
        onConfirm = { onIntent(JobsIntent.SubmitJobResults(manifest.jobId)) },
      ) { trigger ->
        PrimaryButton(
          pluralStringResource(R.plurals.job_detail_submit, state.unsubmittedCount, state.unsubmittedCount),
          trigger,
          loading = state.isSubmitting,
        )
      }
    }
    if (paused) {
      PrimaryButton(stringResource(R.string.job_detail_resume), { onIntent(JobsIntent.ResumeJob(manifest.jobId)) }, leadingIcon = R.drawable.ic_play)
    }
    if (state.failedCells > 0 && !state.isRunning) {
      OutlineButton(
        stringResource(R.string.job_detail_retry),
        { onIntent(JobsIntent.RetryFailed(manifest.jobId)) },
        leadingIcon = R.drawable.ic_retry,
      )
    }

    Row(modifier = Modifier.fillMaxWidth().padding(top = 12.dp), verticalAlignment = Alignment.Top) {
      Column(modifier = Modifier.weight(1f)) {
        Text(stringResource(R.string.job_detail_results), style = ai.liquid.pipette.compose.theme.serif(24), color = colors.label)
        Text(
          stringResource(R.string.job_detail_results_hint),
          style = TextStyle(fontSize = 16.sp),
          color = colors.gray,
          modifier = Modifier.padding(top = 4.dp),
        )
      }
      if (state.completedCells > 0) {
        Icon(
          painter = painterResource(R.drawable.ic_upload),
          contentDescription = null,
          tint = colors.gray,
          modifier = Modifier.size(22.dp).clickableNoRipple { onIntent(JobsIntent.ExportCsv(manifest.jobId)) },
        )
      }
    }
    Spacer(Modifier.height(12.dp))
    state.resultsGrid?.let { ResultsTable(it) { cellId -> onIntent(JobsIntent.OpenCellDetail(cellId)) } }
      ?: MutedLabel(stringResource(R.string.job_detail_no_results))
    if (paused) ContributeRow(state.contributeResults, state.isRegistered) { onIntent(JobsIntent.SetJobAutoSubmit(manifest.jobId, it)) }
  }

  if (renaming) {
    RenameDialog(manifest.title ?: "", onDismiss = { renaming = false }) { title -> onIntent(JobsIntent.RenameJob(manifest.jobId, title)) }
  }
}

/** "Show N more properties" / "Show less" toggle with a chevron that flips down (collapsed) ↔ up (expanded). */
@Composable
private fun MorePropertiesToggle(expanded: Boolean, hiddenCount: Int, onClick: () -> Unit) {
  val colors = PipetteTheme.colors
  val rotation by animateFloatAsState(targetValue = if (expanded) 270f else 90f, label = "morePropsChevron")
  Row(
    modifier = Modifier.clickableNoRipple(onClick).padding(vertical = 8.dp),
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(8.dp),
  ) {
    Icon(
      painter = painterResource(R.drawable.ic_chevron_right),
      contentDescription = null,
      tint = colors.gray,
      modifier = Modifier.size(18.dp).graphicsLayer { rotationZ = rotation },
    )
    Text(
      if (expanded) stringResource(R.string.job_detail_show_less) else pluralStringResource(R.plurals.job_detail_show_more, hiddenCount, hiddenCount),
      style = TextStyle(fontSize = 15.sp),
      color = colors.gray,
    )
  }
}

/** Thin progress bar used on the running/paused detail pages (iOS: height 4, systemGray4 track). */
@Composable
private fun DetailProgressBar(progress: Double) {
  val colors = PipetteTheme.colors
  Box(modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(percent = 50)).background(colors.gray4)) {
    Box(
      modifier =
        Modifier.fillMaxWidth(progress.coerceIn(0.0, 1.0).toFloat()).height(4.dp).clip(RoundedCornerShape(percent = 50)).background(colors.label)
    )
  }
}

/** Detail top bar: back chevron + a "···" overflow menu (iOS: only Rename + Delete; other actions are in-body buttons). */
@Composable
private fun DetailNavBar(state: JobsUiState.Detail, onIntent: (JobsIntent) -> Unit, onRename: () -> Unit) {
  val colors = PipetteTheme.colors
  val manifest = state.manifest
  var menuOpen by remember { mutableStateOf(false) }
  var confirmDelete by remember { mutableStateOf(false) }
  Box(modifier = Modifier.fillMaxWidth().height(44.dp)) {
    Icon(
      painter = painterResource(R.drawable.ic_chevron_left),
      contentDescription = null,
      tint = colors.label,
      modifier = Modifier.align(Alignment.CenterStart).size(24.dp).clickableNoRipple { onIntent(JobsIntent.BackToJobs) },
    )
    Box(modifier = Modifier.align(Alignment.CenterEnd)) {
      Icon(
        painter = painterResource(R.drawable.ic_more),
        contentDescription = null,
        tint = colors.label,
        modifier = Modifier.size(22.dp).clickableNoRipple { menuOpen = true },
      )
      androidx.compose.material3.DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
        DropdownItem(stringResource(R.string.job_detail_rename)) {
          menuOpen = false
          onRename()
        }
        // Confirm before destroying the job + all its results (iOS parity: JobDetailView delete
        // shows a destructive confirmationDialog), matching the app's other destructive actions.
        DropdownItem(stringResource(R.string.job_detail_delete)) {
          menuOpen = false
          confirmDelete = true
        }
      }
    }
  }
  if (confirmDelete) {
    AlertDialog(
      onDismissRequest = { confirmDelete = false },
      title = { Text(stringResource(R.string.job_detail_delete_confirm_title)) },
      text = { Text(stringResource(R.string.job_detail_delete_confirm_message)) },
      confirmButton = {
        TextButton(
          onClick = {
            confirmDelete = false
            onIntent(JobsIntent.DeleteJob(manifest.jobId))
          }
        ) {
          Text(stringResource(R.string.action_delete))
        }
      },
      dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(stringResource(R.string.action_cancel)) } },
    )
  }
}

@Composable
private fun DropdownItem(text: String, onClick: () -> Unit) {
  androidx.compose.material3.DropdownMenuItem(text = { Text(text, color = PipetteTheme.colors.label) }, onClick = onClick)
}

/** Left checkbox + multi-line contribute copy (iOS contribution row). */
@Composable
private fun ContributeRow(checked: Boolean, enabled: Boolean, onToggle: (Boolean) -> Unit) {
  Row(
    modifier = Modifier.fillMaxWidth().padding(top = 24.dp).clickableNoRipple { if (enabled) onToggle(!checked) },
    horizontalArrangement = Arrangement.spacedBy(14.dp),
    verticalAlignment = Alignment.Top,
  ) {
    ai.liquid.pipette.compose.WizardCheckbox(isOn = checked, size = 22)
    Text(
      stringResource(R.string.job_contribute_text),
      style = TextStyle(fontSize = 15.sp, lineHeight = 21.sp),
      color = PipetteTheme.colors.label.copy(alpha = 0.78f),
    )
  }
}

/** Results table (iOS flat variant): frozen Model+quant column, horizontally scrollable benchmark columns, green heatmap cells. */
@Composable
private fun ResultsTable(grid: ResultsGridUi, onCellClick: (String) -> Unit) {
  val colors = PipetteTheme.colors
  val modelW = 150.dp
  val colW = 148.dp
  val rowH = 56.dp
  val headerH = 52.dp
  IosCard(cornerRadius = 12) {
    Row {
      // Frozen Model column.
      Column {
        Box(
          modifier = Modifier.width(modelW).height(headerH).background(colors.gray6).padding(horizontal = 16.dp),
          contentAlignment = Alignment.CenterStart,
        ) {
          Text(stringResource(R.string.job_detail_table_model), style = TextStyle(fontSize = 13.sp), color = colors.gray)
        }
        grid.rows.forEach { row ->
          Row(
            modifier = Modifier.width(modelW).height(rowH).padding(start = 12.dp, end = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp),
          ) {
            ai.liquid.pipette.compose.BrandLogo(row.modelName, size = 18.dp)
            Text(
              row.modelName,
              style = TextStyle(fontSize = 13.sp),
              color = colors.label,
              maxLines = 1,
              overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
              modifier = Modifier.weight(1f, fill = false),
            )
            Chip(row.quant, fontSize = 12.sp)
          }
        }
      }
      Column(modifier = Modifier.horizontalScroll(rememberScrollState())) {
        Row {
          grid.columnLabels.forEach { col ->
            Box(
              modifier = Modifier.width(colW).height(headerH).background(colors.gray6).padding(horizontal = 14.dp),
              contentAlignment = Alignment.CenterStart,
            ) {
              Text(
                col,
                style = TextStyle(fontSize = 13.sp),
                color = colors.gray,
                maxLines = 1,
                overflow = androidx.compose.ui.text.style.TextOverflow.Ellipsis,
              )
            }
          }
        }
        grid.rows.forEach { row ->
          Row {
            row.cells.forEach { cell ->
              val bg = cell.intensity?.let { resultGreen(it) }
              val textColor =
                when (cell.accent) {
                  ai.liquid.pipette.compose.ResultCellAccent.FAILED -> colors.red
                  ai.liquid.pipette.compose.ResultCellAccent.CANCELLED -> colors.orange
                  ai.liquid.pipette.compose.ResultCellAccent.NONE -> if (cell.intensity != null) colors.label else colors.gray
                }
              val tappable = cell.cell != null && cell.hasDetail
              Box(
                modifier =
                  Modifier.width(colW)
                    .height(rowH)
                    .then(if (bg != null) Modifier.background(bg) else Modifier)
                    .then(if (tappable) Modifier.clickableNoRipple { onCellClick(cell.cell!!.cellId) } else Modifier)
                    .padding(horizontal = 14.dp),
                contentAlignment = Alignment.CenterStart,
              ) {
                Text(cell.text, style = TextStyle(fontSize = 14.sp), color = textColor)
              }
            }
          }
        }
      }
    }
  }
}

/** Heatmap green for a results cell (mint → green as the value gets better). */
private fun resultGreen(intensity: Double): androidx.compose.ui.graphics.Color =
  androidx.compose.ui.graphics.Color(0xFF34C759).copy(alpha = (0.18 + intensity.coerceIn(0.0, 1.0) * 0.5).toFloat())

@Composable
private fun RenameDialog(current: String, onDismiss: () -> Unit, onSave: (String) -> Unit) {
  var text by rememberSaveable { mutableStateOf(current) }
  AlertDialog(
    onDismissRequest = onDismiss,
    title = { Text(stringResource(R.string.job_detail_rename_title)) },
    text = {
      Column {
        MutedLabel(stringResource(R.string.job_detail_rename_hint))
        AppTextField(value = text, onValueChange = { text = it }, label = stringResource(R.string.job_detail_rename_name))
      }
    },
    confirmButton = {
      TextButton(
        onClick = {
          onSave(text)
          onDismiss()
        }
      ) {
        Text(stringResource(R.string.action_save))
      }
    },
    dismissButton = {
      TextButton(
        onClick = {
          onSave("")
          onDismiss()
        }
      ) {
        Text(stringResource(R.string.action_reset))
      }
    },
  )
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
