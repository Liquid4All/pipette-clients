// iOS-styled Models tab; branches over empty/list/add-models states.
@file:Suppress("CyclomaticComplexMethod", "TooManyFunctions", "MagicNumber", "MaxLineLength")

package ai.liquid.pipette.compose.models

import ai.liquid.pipette.ByteFormat
import ai.liquid.pipette.ModelFile
import ai.liquid.pipette.compose.AddModelGroupUi
import ai.liquid.pipette.compose.BrandLogo
import ai.liquid.pipette.compose.CapsuleOutlineButton
import ai.liquid.pipette.compose.Chip
import ai.liquid.pipette.compose.ConfirmAction
import ai.liquid.pipette.compose.DownloadedGroupUi
import ai.liquid.pipette.compose.IosCard
import ai.liquid.pipette.compose.IosDivider
import ai.liquid.pipette.compose.PageHeaderLarge
import ai.liquid.pipette.compose.PillTabBarReservedHeight
import ai.liquid.pipette.compose.QuantPill
import ai.liquid.pipette.compose.SearchField
import ai.liquid.pipette.compose.SectionTitle
import ai.liquid.pipette.compose.StatusBadge
import ai.liquid.pipette.compose.WizardCheckbox
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun ModelsScreen(state: ModelsUiState, onIntent: (ModelsIntent) -> Unit) {
  if (state.addModelsOpen) {
    AddModelsCover(state, onIntent)
    return
  }
  val colors = PipetteTheme.colors
  Column(
    modifier =
      Modifier.fillMaxSize()
        .verticalScroll(rememberScrollState())
        .windowInsetsPadding(WindowInsets.statusBars)
        .padding(horizontal = 20.dp)
        .padding(top = 12.dp, bottom = 18.dp + PillTabBarReservedHeight),
    verticalArrangement = Arrangement.spacedBy(14.dp),
  ) {
    Row(modifier = Modifier.fillMaxWidth().padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
      PageHeaderLarge("Your models", modifier = Modifier.weight(1f))
      CapsuleOutlineButton(
        "Add models",
        onClick = { onIntent(ModelsIntent.OpenAddModels) },
        height = 38,
        fontSize = 14,
        leadingIcon = ai.liquid.pipette.R.drawable.ic_search,
      )
    }
    SearchField(
      value = state.searchQuery,
      onValueChange = { onIntent(ModelsIntent.ApplyDownloadedSearch(it)) },
      placeholder = "Search your downloaded models",
    )

    when {
      // A download in flight is about to populate the list — don't flash the empty-state card.
      !state.hasAnyModel && state.activeDownloads.isNotEmpty() -> Unit
      !state.hasAnyModel ->
        IosCard(cornerRadius = 18) {
          Text(
            "No models downloaded. Use Add models to download for benchmarking.",
            style = TextStyle(fontSize = 16.sp, lineHeight = 22.sp),
            color = colors.gray,
            modifier = Modifier.padding(24.dp),
          )
        }
      !state.matched ->
        IosCard(cornerRadius = 18) {
          Text("No matching models.", style = TextStyle(fontSize = 15.sp), color = colors.gray, modifier = Modifier.padding(24.dp))
        }
      else ->
        IosCard(cornerRadius = 18) {
          state.downloadedGroups.forEachIndexed { i, group ->
            if (i > 0) IosDivider()
            DownloadedGroup(group, onIntent)
          }
          if (state.mmprojs.isNotEmpty()) {
            IosDivider()
            Text(
              "MMProjectors",
              style = TextStyle(fontSize = 13.sp),
              color = colors.gray,
              modifier = Modifier.padding(horizontal = 18.dp, vertical = 8.dp),
            )
            state.mmprojs.forEach { row ->
              Row(modifier = Modifier.fillMaxWidth().padding(horizontal = 18.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                  Text(row.title, style = TextStyle(fontSize = 16.sp), color = colors.label)
                  Text(row.subtitle, style = TextStyle(fontSize = 13.sp), color = colors.gray)
                }
              }
            }
          }
        }
    }

    if (state.activeDownloads.isNotEmpty()) {
      SectionTitle("Active downloads")
      IosCard(cornerRadius = 16) {
        state.activeDownloads.forEachIndexed { i, d ->
          if (i > 0) IosDivider()
          Column(modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
              Text(d.filename, style = TextStyle(fontSize = 16.sp), color = colors.label, modifier = Modifier.weight(1f))
              // A waiting row shows its state rather than a percentage: the percentage isn't moving, and saying "63%" over a stalled bar reads as a
              // hung download rather than one waiting for a network.
              val badge = if (d.totalBytes > 0 && !d.isWaitingForNetwork) "${d.bytesRead * 100 / d.totalBytes}%" else d.displayLabel
              StatusBadge(badge, if (d.isFailed) colors.red else colors.gray)
            }
            if (d.totalBytes > 0 && !d.isFailed) {
              ai.liquid.pipette.compose.AppLinearProgress(
                fraction = d.bytesRead.toDouble() / d.totalBytes.toDouble(),
                modifier = Modifier.padding(top = 8.dp),
              )
              Text(
                "${ByteFormat.fileSize(d.bytesRead)} / ${ByteFormat.fileSize(d.totalBytes)}",
                style = TextStyle(fontSize = 13.sp),
                color = colors.gray,
                modifier = Modifier.padding(top = 4.dp),
              )
            } else {
              Text(d.message, style = TextStyle(fontSize = 13.sp), color = colors.gray, modifier = Modifier.padding(top = 4.dp))
            }
            Row(modifier = Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(20.dp)) {
              val (label, action) =
                when {
                  d.isFailed -> "Resume" to ModelsIntent.ResumeDownload(d.key)
                  d.isPaused -> "Resume" to ModelsIntent.ResumeDownload(d.key)
                  else -> "Pause" to ModelsIntent.PauseDownload(d.key)
                }
              Text(label, style = TextStyle(fontSize = 14.sp), color = colors.label, modifier = Modifier.clickableNoRipple { onIntent(action) })
              Text(
                "Cancel",
                style = TextStyle(fontSize = 14.sp),
                color = colors.red,
                modifier = Modifier.clickableNoRipple { onIntent(ModelsIntent.CancelDownload(d.key)) },
              )
            }
          }
        }
      }
    }
  }
}

/** A downloaded model family: chevron + brand + name + size + "N quants" badge; expands to quant chips. Long-press to delete. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun DownloadedGroup(group: DownloadedGroupUi, onIntent: (ModelsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  var expanded by remember { mutableStateOf(false) }
  ConfirmAction(
    "Delete ${group.name} (${group.quantCount} file${if (group.quantCount == 1) "" else "s"})?",
    "Delete",
    onConfirm = { onIntent(ModelsIntent.DeleteModelGroup(group.files)) },
  ) { trigger ->
    Row(
      modifier =
        Modifier.fillMaxWidth()
          .combinedClickable(onClick = { expanded = !expanded }, onLongClick = trigger)
          .padding(horizontal = 16.dp, vertical = 15.dp),
      verticalAlignment = Alignment.CenterVertically,
      horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
      ai.liquid.pipette.compose.RotatingChevron(expanded = expanded, tint = colors.gray, modifier = Modifier.width(16.dp))
      BrandLogo(group.name, size = 22.dp)
      Column(Modifier.weight(1f)) {
        Text(
          group.name,
          style = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.Medium),
          color = colors.label,
          maxLines = 1,
          overflow = TextOverflow.Ellipsis,
        )
        Text(group.sizeLabel, style = TextStyle(fontSize = 14.sp), color = colors.gray)
      }
      Box(
        modifier = Modifier.clip(RoundedCornerShape(percent = 50)).background(colors.secondaryBackground).padding(horizontal = 11.dp, vertical = 5.dp)
      ) {
        Text("${group.quantCount} quant${if (group.quantCount == 1) "" else "s"}", style = TextStyle(fontSize = 14.sp), color = colors.label)
      }
    }
  }
  if (expanded) {
    FlowRow(
      modifier = Modifier.fillMaxWidth().padding(start = 42.dp, end = 16.dp, bottom = 12.dp),
      horizontalArrangement = Arrangement.spacedBy(8.dp),
      verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
      group.quants.forEach { Chip(it) }
    }
  }
}

/** Full-screen "Add models" cover (iOS AddModelsView): model list + Select all + quant pills + download footer. */
@Composable
private fun AddModelsCover(state: ModelsUiState, onIntent: (ModelsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  // System back closes the cover (returns to the Models list) instead of exiting the app.
  BackHandler { onIntent(ModelsIntent.CloseAddModels) }
  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    Box(modifier = Modifier.fillMaxWidth().height(52.dp).padding(horizontal = 16.dp), contentAlignment = Alignment.Center) {
      androidx.compose.material3.Icon(
        painter = androidx.compose.ui.res.painterResource(ai.liquid.pipette.R.drawable.ic_chevron_left),
        contentDescription = null,
        tint = colors.label,
        modifier = Modifier.align(Alignment.CenterStart).size(24.dp).clickableNoRipple { onIntent(ModelsIntent.CloseAddModels) },
      )
      Text("Add models", style = ai.liquid.pipette.compose.theme.serif(20), color = colors.label)
    }
    IosDivider()
    Column(modifier = Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 24.dp).padding(top = 18.dp)) {
      Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text("Download models", style = ai.liquid.pipette.compose.theme.serif(21), color = colors.label, modifier = Modifier.weight(1f))
        Row(
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(6.dp),
          modifier = Modifier.clickableNoRipple { onIntent(ModelsIntent.ToggleAddSelectAll) },
        ) {
          if (state.addAllSelected) {
            androidx.compose.material3.Icon(
              painter = androidx.compose.ui.res.painterResource(ai.liquid.pipette.R.drawable.ic_check),
              contentDescription = null,
              tint = colors.label,
              modifier = Modifier.size(16.dp),
            )
          }
          Text(
            if (state.addAllSelected) "Selected all" else "Select all",
            style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.SemiBold),
            color = colors.label,
          )
        }
      }
      Text(
        "Select the models to download for benchmarking.",
        style = TextStyle(fontSize = 15.sp),
        color = colors.gray,
        modifier = Modifier.padding(top = 4.dp, bottom = 14.dp),
      )
      SearchField(value = state.addSearch, onValueChange = { onIntent(ModelsIntent.ApplyAddSearch(it)) }, placeholder = "Search models")
      Box(Modifier.height(14.dp))
      IosCard(cornerRadius = 16) {
        state.addGroups.forEachIndexed { i, g ->
          if (i > 0) IosDivider()
          AddModelRow(g, onIntent)
        }
      }
      Box(Modifier.height(24.dp))
      Text("Quantizations", style = ai.liquid.pipette.compose.theme.serif(21), color = colors.label)
      Text(
        "Specify level of quantization to download",
        style = TextStyle(fontSize = 15.sp),
        color = colors.gray,
        modifier = Modifier.padding(top = 4.dp, bottom = 14.dp),
      )
      Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        state.addQuantPills.forEachIndexed { i, chip ->
          QuantPill(chip.label, chip.selected) { onIntent(ModelsIntent.ToggleAddQuant(chip.filter, !chip.selected)) }
          if (i == 0) Box(Modifier.width(1.dp).height(22.dp).background(colors.gray4))
        }
      }
      Box(Modifier.height(16.dp))
    }
    // Fixed download footer.
    Box(modifier = Modifier.fillMaxWidth().windowInsetsPadding(WindowInsets.navigationBars).padding(horizontal = 24.dp, vertical = 12.dp)) {
      val enabled = state.addDownloadCount > 0
      val size = if (state.addDownloadBytes > 0) " (${ByteFormat.fileSize(state.addDownloadBytes)})" else ""
      val label = "Download ${state.addDownloadCount} model${if (state.addDownloadCount == 1) "" else "s"}$size"
      val isLarge = enabled && state.addDownloadBytes > state.largeDownloadWarningBytes
      if (isLarge) {
        ConfirmAction(
          "Download ${ByteFormat.fileSize(state.addDownloadBytes)} of models? This may use significant data and storage.",
          "Download",
          onConfirm = { onIntent(ModelsIntent.DownloadAddModels) },
        ) { trigger ->
          DownloadFooterButton(label, enabled, trigger)
        }
      } else {
        DownloadFooterButton(label, enabled) { onIntent(ModelsIntent.DownloadAddModels) }
      }
    }
  }
}

@Composable
private fun AddModelRow(group: AddModelGroupUi, onIntent: (ModelsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  Row(
    modifier =
      Modifier.fillMaxWidth()
        .clickableNoRipple { onIntent(ModelsIntent.ToggleAddGroup(group.id, !group.checked)) }
        .padding(horizontal = 16.dp, vertical = 14.dp),
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(10.dp),
  ) {
    BrandLogo(group.name, size = 24.dp)
    Column(Modifier.weight(1f)) {
      Text(
        group.name,
        style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Medium),
        color = colors.label,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
      )
      Text(group.sizeLabel, style = TextStyle(fontSize = 13.sp), color = colors.gray)
    }
    WizardCheckbox(isOn = group.checked, size = 22)
  }
}

@Composable
private fun DownloadFooterButton(label: String, enabled: Boolean, onClick: () -> Unit) {
  val colors = PipetteTheme.colors
  Box(
    modifier =
      Modifier.fillMaxWidth()
        .height(52.dp)
        .clip(RoundedCornerShape(percent = 50))
        .background(if (enabled) colors.label else colors.gray3)
        .then(if (enabled) Modifier.clickableNoRipple(onClick) else Modifier),
    contentAlignment = Alignment.Center,
  ) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      androidx.compose.material3.Icon(
        painter = androidx.compose.ui.res.painterResource(ai.liquid.pipette.R.drawable.ic_download),
        contentDescription = null,
        tint = colors.background,
        modifier = Modifier.size(18.dp),
      )
      Text(label, style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Medium), color = colors.background)
    }
  }
}

@Preview
@Composable
private fun ModelsScreenPreview() {
  PipetteTheme {
    ModelsScreen(
      state =
        ModelsUiState(
          hasAnyModel = true,
          downloadedGroups =
            listOf(
              DownloadedGroupUi(
                key = "lfm2-1.2b",
                name = "LFM2 1.2B",
                sizeLabel = "731 MB",
                quantCount = 2,
                quants = listOf("q4_k_m", "q8_0"),
                files =
                  listOf(
                    ModelFile(
                      name = "lfm2-1.2b-q4_k_m.gguf",
                      path = "/models/lfm2-1.2b-q4_k_m.gguf",
                      sizeBytes = 731L * 1024 * 1024,
                      hfRepo = "LiquidAI/LFM2-1.2B-GGUF",
                    )
                  ),
              )
            ),
        ),
      onIntent = {},
    )
  }
}

@Preview
@Composable
private fun ModelsScreenEmptyPreview() {
  PipetteTheme { ModelsScreen(state = ModelsUiState(hasAnyModel = false), onIntent = {}) }
}
