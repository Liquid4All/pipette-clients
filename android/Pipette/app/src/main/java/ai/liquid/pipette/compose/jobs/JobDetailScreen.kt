// Job detail (progress + results heatmap + rename/delete) and the sibling cell detail cover. Wired as their own
// Navigation 3 destinations (Route.JobDetail, Route.CellDetail) mirrored from JobsUiState.Detail / .CellDetail —
// see PipetteAppRoot for the state → backstack mirror. Chrome matches the Android tabs: Material 3 top app bar
// with a back navigation icon and a "···" overflow menu.
@file:Suppress("MagicNumber", "MaxLineLength", "TooManyFunctions", "CyclomaticComplexMethod", "LongMethod")

package ai.liquid.pipette.compose.jobs

import ai.liquid.pipette.R
import ai.liquid.pipette.compose.AndroidTopAppBar
import ai.liquid.pipette.compose.AppTextField
import ai.liquid.pipette.compose.BrandLogo
import ai.liquid.pipette.compose.Chip
import ai.liquid.pipette.compose.ConfirmAction
import ai.liquid.pipette.compose.IosDivider
import ai.liquid.pipette.compose.JobActivityColors
import ai.liquid.pipette.compose.JobLiveActivity
import ai.liquid.pipette.compose.MutedLabel
import ai.liquid.pipette.compose.OutlineButton
import ai.liquid.pipette.compose.OutlinedAndroidCard
import ai.liquid.pipette.compose.PillTabBarReservedHeight
import ai.liquid.pipette.compose.PrimaryButton
import ai.liquid.pipette.compose.PropertyChipRow
import ai.liquid.pipette.compose.ResultCellAccent
import ai.liquid.pipette.compose.ResultsGridUi
import ai.liquid.pipette.compose.accentColor
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import ai.liquid.pipette.compose.theme.serif
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// ---------------------------------------------------------------------------
// Job detail
// ---------------------------------------------------------------------------

/**
 * Per-job detail — the "Job progress" surface when a run is active, or the results heatmap + resume/retry buttons when it's not. Chrome is a Material
 * 3 top app bar with a back nav icon and a "···" overflow menu (Rename, Delete). System back returns to the jobs list; the shell pops Route.JobDetail
 * in reaction to that state change.
 */
@Composable
fun JobDetailScreen(state: JobsUiState.Detail, onIntent: (JobsIntent) -> Unit) {
  BackHandler { onIntent(JobsIntent.BackToJobs) }
  var renaming by remember { mutableStateOf(false) }
  var pausing by remember(state.runningHere) { mutableStateOf(false) }
  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    JobDetailAppBar(
      onBack = { onIntent(JobsIntent.BackToJobs) },
      onRename = { renaming = true },
      onDelete = { onIntent(JobsIntent.DeleteJob(state.manifest.jobId)) },
    )
    Column(
      modifier =
        Modifier.weight(1f)
          .fillMaxWidth()
          .verticalScroll(rememberScrollState())
          .padding(horizontal = 20.dp)
          .padding(top = 4.dp, bottom = 18.dp + PillTabBarReservedHeight)
    ) {
      DetailBody(state, onIntent, pausing = pausing, onPauseTap = { pausing = true })
    }
  }
  if (renaming) {
    RenameDialog(state.manifest.title ?: "", onDismiss = { renaming = false }) { title ->
      onIntent(JobsIntent.RenameJob(state.manifest.jobId, title))
    }
  }
}

@Composable
private fun JobDetailAppBar(onBack: () -> Unit, onRename: () -> Unit, onDelete: () -> Unit) {
  var menuOpen by remember { mutableStateOf(false) }
  var confirmDelete by remember { mutableStateOf(false) }
  AndroidTopAppBar(
    title = stringResource(R.string.job_detail_title),
    navigationIcon = {
      IconButton(onClick = onBack) {
        Icon(painter = painterResource(R.drawable.ic_arrow_back), contentDescription = "Back", modifier = Modifier.size(24.dp))
      }
    },
    actions = {
      Box {
        IconButton(onClick = { menuOpen = true }) {
          Icon(painter = painterResource(R.drawable.ic_more), contentDescription = "More actions", modifier = Modifier.size(22.dp))
        }
        DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
          DropdownMenuItem(
            text = { Text(stringResource(R.string.job_detail_rename), color = PipetteTheme.colors.label) },
            onClick = {
              menuOpen = false
              onRename()
            },
          )
          // Confirm before destroying the job + all its results (iOS parity: JobDetailView delete shows a destructive confirmationDialog).
          DropdownMenuItem(
            text = { Text(stringResource(R.string.job_detail_delete), color = PipetteTheme.colors.label) },
            onClick = {
              menuOpen = false
              confirmDelete = true
            },
          )
        }
      }
    },
  )
  if (confirmDelete) {
    AlertDialog(
      onDismissRequest = { confirmDelete = false },
      title = { Text(stringResource(R.string.job_detail_delete_confirm_title)) },
      text = { Text(stringResource(R.string.job_detail_delete_confirm_message)) },
      confirmButton = {
        TextButton(
          onClick = {
            confirmDelete = false
            onDelete()
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
private fun DetailBody(state: JobsUiState.Detail, onIntent: (JobsIntent) -> Unit, pausing: Boolean, onPauseTap: () -> Unit) {
  val manifest = state.manifest
  val colors = PipetteTheme.colors
  Text(state.titleDate, style = serif(24), color = colors.label, modifier = Modifier.padding(top = 8.dp))
  Text(state.subtitle, style = serif(18), color = colors.gray, modifier = Modifier.padding(top = 2.dp, bottom = 16.dp))

  var propsExpanded by remember { mutableStateOf(false) }
  PropertyChipRow(stringResource(R.string.property_models), state.modelChips) { JobsModelChip(it) }
  PropertyChipRow(stringResource(R.string.property_benchmarks), state.benchmarkChips)
  PropertyChipRow(stringResource(R.string.property_quants), state.quantChips)
  if (propsExpanded) {
    PropertyChipRow(
      stringResource(R.string.job_detail_property_gpu),
      listOf(stringResource(if (state.gpuLayers > 0) R.string.job_detail_gpu_on else R.string.job_detail_gpu_off)),
    )
    PropertyChipRow(stringResource(R.string.job_detail_property_context), listOf(state.contextSize.toString()))
  }
  JobsMorePropertiesToggle(expanded = propsExpanded, hiddenCount = 2) { propsExpanded = !propsExpanded }

  IosDivider(modifier = Modifier.padding(vertical = 18.dp))

  if (state.runningHere) {
    RunningBlock(state, onIntent, pausing = pausing, onPauseTap = onPauseTap)
  } else {
    NotRunningBlock(state, onIntent)
  }
}

@Composable
private fun RunningBlock(state: JobsUiState.Detail, onIntent: (JobsIntent) -> Unit, pausing: Boolean, onPauseTap: () -> Unit) {
  val colors = PipetteTheme.colors
  val manifest = state.manifest
  Text(stringResource(R.string.job_detail_in_progress), style = serif(28), color = colors.label, modifier = Modifier.padding(bottom = 12.dp))
  DetailProgressBar(state.runProgress)
  Row(modifier = Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.SpaceBetween) {
    Text(state.runCellsDone, style = TextStyle(fontSize = 16.sp), color = colors.gray, maxLines = 1)
    Text(state.runTimeLeft, style = TextStyle(fontSize = 16.sp), color = colors.gray, maxLines = 1)
  }
  // Live indicators mirrored from Pocket Mode — throttling headroom row + shared cell/progress block.
  // The ambient cool wash washes the block while the gate is cooling.
  val cooling = state.coolingSinceMillis != null
  OutlinedAndroidCard(
    cornerRadius = 12,
    modifier = Modifier.padding(top = 16.dp).then(if (cooling) Modifier.background(Color(0x14589BF7), RoundedCornerShape(12.dp)) else Modifier),
  ) {
    Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
      Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text("Throttling headroom", style = TextStyle(fontSize = 16.sp), color = colors.gray, modifier = Modifier.weight(1f))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
          Box(Modifier.size(8.dp).clip(CircleShape).background(accentColor(state.thermalAccent)))
          Text(state.thermalLabel, style = TextStyle(fontSize = 16.sp), color = colors.label)
        }
      }
      JobLiveActivity(
        currentCellLabel = state.runCellLabel,
        progressText = state.runProgressText,
        coolingSinceMillis = state.coolingSinceMillis,
        colors = JobActivityColors(primaryText = colors.label, secondaryText = colors.gray, accent = Color(0xFF60A5FA)),
      )
    }
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
        onPauseTap()
        onIntent(JobsIntent.CancelRunningJob)
      }
    },
    leadingIcon = R.drawable.ic_pause,
  )
  JobsContributeRow(state.contributeResults, state.isRegistered) { onIntent(JobsIntent.SetJobAutoSubmit(manifest.jobId, it)) }
}

@Composable
private fun NotRunningBlock(state: JobsUiState.Detail, onIntent: (JobsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  val manifest = state.manifest
  val paused = state.canResume
  if (paused) {
    Text(stringResource(R.string.job_detail_paused), style = serif(28), color = colors.label, modifier = Modifier.padding(bottom = 12.dp))
    DetailProgressBar(state.runProgress)
    Spacer(Modifier.height(8.dp))
  }
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
    OutlineButton(stringResource(R.string.job_detail_retry), { onIntent(JobsIntent.RetryFailed(manifest.jobId)) }, leadingIcon = R.drawable.ic_retry)
  }

  Row(modifier = Modifier.fillMaxWidth().padding(top = 12.dp), verticalAlignment = Alignment.Top) {
    Column(modifier = Modifier.weight(1f)) {
      Text(stringResource(R.string.job_detail_results), style = serif(24), color = colors.label)
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
  if (paused) JobsContributeRow(state.contributeResults, state.isRegistered) { onIntent(JobsIntent.SetJobAutoSubmit(manifest.jobId, it)) }
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

/** Results table (iOS flat variant): frozen Model+quant column, horizontally scrollable benchmark columns, green heatmap cells. */
@Composable
private fun ResultsTable(grid: ResultsGridUi, onCellClick: (String) -> Unit) {
  val colors = PipetteTheme.colors
  val modelW = 150.dp
  val colW = 148.dp
  val rowH = 56.dp
  val headerH = 52.dp
  OutlinedAndroidCard(cornerRadius = 12) {
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
            BrandLogo(row.modelName, size = 18.dp)
            Text(
              row.modelName,
              style = TextStyle(fontSize = 13.sp),
              color = colors.label,
              maxLines = 1,
              overflow = TextOverflow.Ellipsis,
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
              Text(col, style = TextStyle(fontSize = 13.sp), color = colors.gray, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
          }
        }
        grid.rows.forEach { row ->
          Row {
            row.cells.forEach { cell ->
              val bg = cell.intensity?.let { resultGreen(it) }
              val textColor =
                when (cell.accent) {
                  ResultCellAccent.FAILED -> colors.red
                  ResultCellAccent.CANCELLED -> colors.orange
                  ResultCellAccent.NONE -> if (cell.intensity != null) colors.label else colors.gray
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
private fun resultGreen(intensity: Double): Color = Color(0xFF34C759).copy(alpha = (0.18 + intensity.coerceIn(0.0, 1.0) * 0.5).toFloat())

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

// ---------------------------------------------------------------------------
// Cell detail
// ---------------------------------------------------------------------------

/**
 * Full-bleed cell-detail cover — Material 3 top app bar (back nav icon), an info section (max-three property rows behind a "Show more" toggle), an
 * edge-to-edge separator, then the payload table. System back closes the cover rather than the app.
 */
@Composable
fun CellDetailScreen(state: JobsUiState.CellDetail, onIntent: (JobsIntent) -> Unit) {
  val cell = state.cell
  val colors = PipetteTheme.colors
  BackHandler { onIntent(JobsIntent.CloseCellDetail) }
  var propsExpanded by remember { mutableStateOf(false) }
  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    AndroidTopAppBar(
      title = stringResource(R.string.cell_detail_title),
      navigationIcon = {
        IconButton(onClick = { onIntent(JobsIntent.CloseCellDetail) }) {
          Icon(painter = painterResource(R.drawable.ic_arrow_back), contentDescription = "Back", modifier = Modifier.size(24.dp))
        }
      },
    )
    Column(modifier = Modifier.weight(1f).verticalScroll(rememberScrollState())) {
      // Info section: first three properties always visible, the rest collapse behind a "Show N more" toggle.
      Column(modifier = Modifier.padding(horizontal = 20.dp).padding(top = 8.dp, bottom = 16.dp)) {
        PropertyChipRow(stringResource(R.string.property_models), listOf(cell.modelName)) { JobsModelChip(it) }
        PropertyChipRow(stringResource(R.string.property_quant), listOf(cell.quant))
        PropertyChipRow(stringResource(R.string.property_benchmark), listOf(cell.benchmarkLabel))
        if (propsExpanded) PropertyChipRow(stringResource(R.string.property_status), listOf(cell.statusLabel))
        JobsMorePropertiesToggle(expanded = propsExpanded, hiddenCount = 1) { propsExpanded = !propsExpanded }
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
          OutlinedAndroidCard(cornerRadius = 20) {
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
