// Full-screen "Add models" flow (iOS AddModelsView equivalent): model list + Select all + quant pills + download footer.
// Wired as its own Navigation 3 destination (Route.AddModels) rather than an in-place cover — keeps the backstack honest
// so predictive back and system back animate through Nav3.
@file:Suppress("MagicNumber", "MaxLineLength")

package ai.liquid.pipette.compose.models

import ai.liquid.pipette.ByteFormat
import ai.liquid.pipette.JobQuantFilter
import ai.liquid.pipette.R
import ai.liquid.pipette.compose.AddModelGroupUi
import ai.liquid.pipette.compose.AndroidSearchBar
import ai.liquid.pipette.compose.AndroidTopAppBar
import ai.liquid.pipette.compose.BrandLogo
import ai.liquid.pipette.compose.ConfirmAction
import ai.liquid.pipette.compose.IosDivider
import ai.liquid.pipette.compose.OutlinedAndroidCard
import ai.liquid.pipette.compose.QuantChipUi
import ai.liquid.pipette.compose.QuantPill
import ai.liquid.pipette.compose.WizardCheckbox
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import ai.liquid.pipette.compose.theme.serif
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Full-screen "Add models" destination. Composed as its own Nav3 entry; the Models list opens it by dispatching [ModelsIntent.OpenAddModels] (the
 * shell mirrors [ModelsUiState.addModelsOpen] onto the backstack). System back both pops the Nav3 entry and dispatches [ModelsIntent.CloseAddModels]
 * so the VM's flag stays in sync.
 */
@Composable
fun AddModelsScreen(state: ModelsUiState, onIntent: (ModelsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  // System back closes the flow (returns to the Models list) instead of exiting the app. The shell reacts
  // to the resulting state change and pops the Nav3 entry, so the backstack stays consistent either way.
  BackHandler { onIntent(ModelsIntent.CloseAddModels) }
  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    // Material 3 top app bar with a back navigation icon — same chrome the Jobs/Models list tabs use.
    AndroidTopAppBar(
      title = "Add models",
      navigationIcon = {
        IconButton(onClick = { onIntent(ModelsIntent.CloseAddModels) }) {
          Icon(painter = painterResource(R.drawable.ic_arrow_back), contentDescription = "Back", modifier = Modifier.size(24.dp))
        }
      },
    )
    Column(modifier = Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 24.dp).padding(top = 6.dp)) {
      Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text("Download models", style = serif(21), color = colors.label, modifier = Modifier.weight(1f))
        Row(
          verticalAlignment = Alignment.CenterVertically,
          horizontalArrangement = Arrangement.spacedBy(6.dp),
          modifier = Modifier.clickableNoRipple { onIntent(ModelsIntent.ToggleAddSelectAll) },
        ) {
          if (state.addAllSelected) {
            Icon(painter = painterResource(R.drawable.ic_check), contentDescription = null, tint = colors.label, modifier = Modifier.size(16.dp))
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
      AndroidSearchBar(hint = "Search models", value = state.addSearch, onValueChange = { onIntent(ModelsIntent.ApplyAddSearch(it)) })
      Box(Modifier.height(14.dp))
      OutlinedAndroidCard(cornerRadius = 16) {
        state.addGroups.forEachIndexed { i, g ->
          if (i > 0) IosDivider()
          AddModelRow(g, onIntent)
        }
      }
      Box(Modifier.height(24.dp))
      Text("Quantizations", style = serif(21), color = colors.label)
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
      Icon(painter = painterResource(R.drawable.ic_download), contentDescription = null, tint = colors.background, modifier = Modifier.size(18.dp))
      Text(label, style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Medium), color = colors.background)
    }
  }
}

@Preview
@Composable
private fun AddModelsScreenPreview() {
  PipetteTheme {
    AddModelsScreen(
      state =
        ModelsUiState(
          addModelsOpen = true,
          addSearch = "",
          addGroups =
            listOf(
              AddModelGroupUi(id = "LFM2 1.2B", name = "LFM2 1.2B", sizeLabel = "731 MB", checked = true),
              AddModelGroupUi(id = "LFM2 350M", name = "LFM2 350M", sizeLabel = "260 MB", checked = false),
              AddModelGroupUi(id = "Qwen2.5 1.5B", name = "Qwen2.5 1.5B", sizeLabel = "986 MB", checked = false),
            ),
          addAllSelected = false,
          addQuantPills = JobQuantFilter.entries.map { QuantChipUi(it, it.label, selected = it == JobQuantFilter.ALL) },
          addDownloadCount = 1,
          addDownloadBytes = 731L * 1024 * 1024,
        ),
      onIntent = {},
    )
  }
}

@Preview
@Composable
private fun AddModelsScreenEmptyPreview() {
  PipetteTheme {
    AddModelsScreen(
      state =
        ModelsUiState(
          addModelsOpen = true,
          addGroups = emptyList(),
          addQuantPills = JobQuantFilter.entries.map { QuantChipUi(it, it.label, selected = it == JobQuantFilter.ALL) },
        ),
      onIntent = {},
    )
  }
}
