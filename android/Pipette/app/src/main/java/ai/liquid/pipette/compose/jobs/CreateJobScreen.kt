// Full-screen "Create a job" wizard (iOS NewJobView equivalent): three steps — Models, Benchmarks, Review — with a
// fixed Android chrome above and a Back / Next / Run footer below. Wired as its own Navigation 3 destination
// (Route.CreateJob) rather than an inline sealed-state variant of JobsScreen — the shell mirrors JobsUiState.Wizard
// onto the backstack, so predictive back and system back animate through Nav3.
@file:Suppress("MagicNumber", "MaxLineLength", "TooManyFunctions", "CyclomaticComplexMethod")

package ai.liquid.pipette.compose.jobs

import ai.liquid.pipette.R
import ai.liquid.pipette.compose.AndroidSearchBar
import ai.liquid.pipette.compose.AndroidTopAppBar
import ai.liquid.pipette.compose.BenchmarkGroupUi
import ai.liquid.pipette.compose.BrandLogo
import ai.liquid.pipette.compose.Chip
import ai.liquid.pipette.compose.IosDivider
import ai.liquid.pipette.compose.MutedLabel
import ai.liquid.pipette.compose.OutlinedAndroidCard
import ai.liquid.pipette.compose.QuantPill
import ai.liquid.pipette.compose.RotatingChevron
import ai.liquid.pipette.compose.WizardCheckbox
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import ai.liquid.pipette.compose.theme.serif
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
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
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Full-screen "Create a job" destination. System back steps through wizard pages one at a time; the first step's back closes the flow (dispatching
 * [JobsIntent.CancelWizard]), so the shell will pop the Nav3 entry.
 */
@Composable
fun CreateJobScreen(state: JobsUiState.Wizard, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  // Back on any step > 0 rewinds the wizard; step 0 back exits the flow.
  BackHandler { if (state.step > 0) onIntent(JobsIntent.WizardGoToStep(state.step - 1)) else onIntent(JobsIntent.CancelWizard) }
  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    // Material 3 top app bar with a back nav icon — matches the Add models flow. The step-progress
    // underline sits directly under the toolbar (parity with the previous inline header).
    AndroidTopAppBar(
      title = stringResource(R.string.job_wizard_title),
      navigationIcon = {
        IconButton(onClick = { if (state.step > 0) onIntent(JobsIntent.WizardGoToStep(state.step - 1)) else onIntent(JobsIntent.CancelWizard) }) {
          Icon(painter = painterResource(R.drawable.ic_arrow_back), contentDescription = "Back", modifier = Modifier.size(24.dp))
        }
      },
      actions = {
        IconButton(onClick = { onIntent(JobsIntent.CancelWizard) }) {
          Icon(painter = painterResource(R.drawable.ic_close), contentDescription = "Close", modifier = Modifier.size(20.dp))
        }
      },
    )
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
              .border(BorderStroke(1.dp, colors.gray3), RoundedCornerShape(percent = 50))
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
  Text(stringResource(R.string.job_wizard_models_title), style = serif(26), color = colors.label)
  Text(
    stringResource(R.string.job_wizard_models_subtitle),
    style = TextStyle(fontSize = 16.sp),
    color = colors.gray,
    modifier = Modifier.padding(top = 4.dp, bottom = 18.dp),
  )
  AndroidSearchBar(
    hint = stringResource(R.string.job_wizard_search_models),
    value = state.modelSearch,
    onValueChange = { onIntent(JobsIntent.ApplyJobModelSearch(it)) },
  )
  Spacer(Modifier.height(14.dp))
  when {
    !state.anyModelGroups ->
      JobWizardEmptyState(
        title = stringResource(R.string.job_wizard_no_models_title),
        subtitle = stringResource(R.string.job_wizard_no_models_subtitle),
        buttonLabel = stringResource(R.string.job_wizard_go_to_models),
        onButton = { onIntent(JobsIntent.GoToModels) },
      )
    !state.modelsMatched -> MutedLabel(stringResource(R.string.job_wizard_no_models_match, state.modelSearch))
    else ->
      OutlinedAndroidCard(cornerRadius = 16) {
        state.modelGroups.forEachIndexed { i, group ->
          if (i > 0) IosDivider()
          ModelSelectRow(group.name, group.sizeLabel, group.checked) { onIntent(JobsIntent.ToggleModelGroup(group.key, !group.checked)) }
        }
      }
  }
  Spacer(Modifier.height(28.dp))
  Text(stringResource(R.string.job_wizard_quants_title), style = serif(21), color = colors.label)
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
    BrandLogo(name, size = 26.dp)
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
    WizardCheckbox(isOn = checked, size = 22)
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
  Text(stringResource(R.string.job_wizard_benchmarks_title), style = serif(26), color = colors.label)
  Text(
    stringResource(R.string.job_wizard_benchmarks_subtitle),
    style = TextStyle(fontSize = 16.sp, lineHeight = 22.sp),
    color = colors.gray,
    modifier = Modifier.padding(top = 4.dp, bottom = 18.dp),
  )
  AndroidSearchBar(
    hint = stringResource(R.string.job_wizard_search_benchmarks),
    value = state.benchmarkSearch,
    onValueChange = { onIntent(JobsIntent.ApplyBenchmarkSearch(it)) },
  )
  Spacer(Modifier.height(14.dp))
  if (!state.benchmarksMatched) {
    MutedLabel(stringResource(R.string.job_wizard_no_benchmarks_match, state.benchmarkSearch))
  } else {
    OutlinedAndroidCard(cornerRadius = 16) {
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
    Text(stringResource(R.string.job_wizard_mmproj_title), style = serif(21), color = colors.label)
    Text(
      stringResource(R.string.job_wizard_mmproj_subtitle),
      style = TextStyle(fontSize = 14.sp),
      color = colors.gray,
      modifier = Modifier.padding(top = 4.dp, bottom = 12.dp),
    )
    if (state.mmprojs.isEmpty()) {
      MutedLabel(stringResource(R.string.job_wizard_mmproj_empty))
    } else {
      OutlinedAndroidCard(cornerRadius = 16) {
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
            WizardCheckbox(isOn = row.checked, size = 22)
          }
        }
      }
    }
  }
}

/** A collapsible benchmark-type group: chevron + title + description + tri-state checkbox; expands to context-size pill rows. */
@Composable
private fun BenchmarkGroup(group: BenchmarkGroupUi, isExpanded: Boolean, onToggleExpand: () -> Unit, onIntent: (JobsIntent) -> Unit) {
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
        WizardCheckbox(isOn = group.allSelected, indeterminate = group.someSelected, size = 22)
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
            WizardCheckbox(isOn = item.checked, size = 22)
          }
        }
      }
    }
  }
}

@Composable
private fun WizardStepReview(state: JobsUiState.Wizard, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  Text(stringResource(R.string.job_wizard_review_title), style = serif(26), color = colors.label, modifier = Modifier.padding(bottom = 16.dp))
  OutlinedAndroidCard(cornerRadius = 16) {
    // Date header + divider.
    Text(
      "${state.reviewDate} · ${state.reviewSubtitle}",
      style = serif(17),
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
  JobsContributeRow(state.contributeResults, state.isRegistered) { onIntent(JobsIntent.SetWizardContribute(it)) }
}

/** A labeled review section: gray label + a flow of chips (model chips carry a brand logo). */
@Composable
private fun ReviewChipSection(label: String, values: List<String>, withLogos: Boolean) {
  val colors = PipetteTheme.colors
  Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
    Text(label, style = TextStyle(fontSize = 16.sp), color = colors.gray)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
      values.forEach { v -> if (withLogos) JobsModelChip(v) else Chip(v) }
    }
  }
}

/** Centered empty state used by the wizard's Models step when no model families are downloaded yet. */
@Composable
private fun JobWizardEmptyState(title: String, subtitle: String, buttonLabel: String?, onButton: () -> Unit) {
  val colors = PipetteTheme.colors
  Column(modifier = Modifier.fillMaxWidth().padding(top = 48.dp), horizontalAlignment = Alignment.CenterHorizontally) {
    Text(title, style = serif(22), color = colors.label)
    Text(
      subtitle,
      style = TextStyle(fontSize = 15.sp, lineHeight = 21.sp),
      color = colors.gray,
      textAlign = androidx.compose.ui.text.style.TextAlign.Center,
      modifier = Modifier.padding(top = 8.dp, start = 16.dp, end = 16.dp),
    )
    if (buttonLabel != null) {
      Box(
        modifier =
          Modifier.padding(top = 20.dp)
            .height(44.dp)
            .clip(RoundedCornerShape(percent = 50))
            .background(colors.label)
            .clickableNoRipple(onButton)
            .padding(horizontal = 24.dp),
        contentAlignment = Alignment.Center,
      ) {
        Text(buttonLabel, style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.SemiBold), color = colors.background)
      }
    }
  }
}

@Preview
@Composable
private fun CreateJobScreenPreview() {
  PipetteTheme {
    CreateJobScreen(
      state =
        JobsUiState.Wizard(
          step = 0,
          stepTitles = listOf("Models", "Benchmarks", "Review"),
          modelSearch = "",
          modelGroups = emptyList(),
          anyModelGroups = false,
          modelsMatched = true,
          selectedBaseModelCount = 0,
          quantFilters = emptyList(),
          benchmarkSearch = "",
          benchmarkGroups = emptyList(),
          benchmarksMatched = true,
          showMmprojCard = false,
          mmprojs = emptyList(),
          allMmprojSelected = false,
          nGpuLayers = 0,
          contextSize = 4096,
          prefillBatch = 512,
          contributeResults = false,
          isRegistered = true,
          reviewSummary = "",
          reviewDate = "",
          reviewSubtitle = "",
          reviewModels = emptyList(),
          reviewBenchmarks = emptyList(),
          reviewQuants = emptyList(),
          showSkippedWarning = false,
          skippedWarning = "",
          canAdvance = false,
          canRun = false,
          runLabel = "Select models and benchmarks",
        ),
      onIntent = {},
    )
  }
}
