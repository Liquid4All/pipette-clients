// Full-screen "Add models" flow (iOS AddModelsView equivalent): model list + Select all + quant pills + download footer.
// The flow is its own Navigation 3 destination, Route.AddModels. The backstack therefore holds an entry for it, and
// predictive back and system back animate through Nav3.
@file:Suppress("MagicNumber", "MaxLineLength")

package ai.liquid.pipette.compose.models

import ai.liquid.pipette.ByteFormat
import ai.liquid.pipette.R
import ai.liquid.pipette.compose.AddModelUi
import ai.liquid.pipette.compose.AddQuantChipUi
import ai.liquid.pipette.compose.AddQuantSectionUi
import ai.liquid.pipette.compose.AndroidSearchBar
import ai.liquid.pipette.compose.AndroidTopAppBar
import ai.liquid.pipette.compose.BrandLogo
import ai.liquid.pipette.compose.ConfirmAction
import ai.liquid.pipette.compose.FamilyChevronColumn
import ai.liquid.pipette.compose.ModelFamilyList
import ai.liquid.pipette.compose.ModelFamilyUi
import ai.liquid.pipette.compose.OutlinedAndroidCard
import ai.liquid.pipette.compose.RotatingChevron
import ai.liquid.pipette.compose.WizardCheckbox
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
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
  // System back closes the flow and returns to the Models list, instead of exiting the app. The shell reacts
  // to the state change and pops the Nav3 entry, so the backstack matches the VM flag.
  BackHandler { onIntent(ModelsIntent.CloseAddModels) }
  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    // Material 3 top app bar with a back navigation icon — same chrome the Jobs/Models list tabs use.
    AndroidTopAppBar(
      title = "Download models",
      navigationIcon = {
        IconButton(onClick = { onIntent(ModelsIntent.CloseAddModels) }) {
          Icon(painter = painterResource(R.drawable.ic_arrow_back), contentDescription = "Back", modifier = Modifier.size(24.dp))
        }
      },
    )
    Column(modifier = Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 24.dp).padding(top = 6.dp)) {
      AndroidSearchBar(hint = "Search models", value = state.addSearch, onValueChange = { onIntent(ModelsIntent.ApplyAddSearch(it)) })
      Box(Modifier.height(14.dp))
      AddModelsList(state.addFamilies, onIntent)
      Box(Modifier.height(16.dp))
    }
    // Fixed download footer.
    Box(modifier = Modifier.fillMaxWidth().windowInsetsPadding(WindowInsets.navigationBars).padding(horizontal = 24.dp, vertical = 12.dp)) {
      val enabled = state.addDownloadCount > 0
      val size = if (state.addDownloadBytes > 0) " (${ByteFormat.fileSize(state.addDownloadBytes)})" else ""
      val label = "Download ${state.addDownloadCount} model${if (state.addDownloadCount == 1) "" else "s"}$size"
      val isLarge = enabled && state.addDownloadBytes > state.largeDownloadWarningBytes
      val warning =
        when {
          !enabled -> null
          state.addHasUnknownSize -> "The size of a selected model is unknown. Download the models? This may use significant data and storage."
          isLarge -> "Download ${ByteFormat.fileSize(state.addDownloadBytes)} of models? This may use significant data and storage."
          else -> null
        }
      if (warning != null) {
        ConfirmAction(warning, "Download", onConfirm = { onIntent(ModelsIntent.DownloadAddModels) }) { trigger ->
          DownloadFooterButton(label, enabled, trigger)
        }
      } else {
        DownloadFooterButton(label, enabled) { onIntent(ModelsIntent.DownloadAddModels) }
      }
    }
  }
}

/** The catalog rows under their family headers. */
@Composable
private fun AddModelsList(families: List<ModelFamilyUi<AddModelUi>>, onIntent: (ModelsIntent) -> Unit) {
  OutlinedAndroidCard(cornerRadius = 16) {
    ModelFamilyList(
      families,
      onToggleOpen = { onIntent(ModelsIntent.ToggleAddFamilyOpen(it)) },
      onToggleChecked = { onIntent(ModelsIntent.ToggleAddFamily(it)) },
      memberKey = { it.name },
      memberIndent = 0.dp,
    ) { model, _, logoDigest ->
      AddModelRow(model, logoDigest, onIntent)
    }
  }
}

@Composable
private fun AddModelRow(model: AddModelUi, logoDigest: String?, onIntent: (ModelsIntent) -> Unit) {
  val colors = PipetteTheme.colors
  val start = 16.dp + FamilyChevronColumn
  Row(
    modifier = Modifier.fillMaxWidth().padding(start = start, end = 16.dp, top = 14.dp, bottom = 14.dp),
    verticalAlignment = Alignment.CenterVertically,
  ) {
    Row(
      modifier = Modifier.weight(1f).clickableNoRipple { onIntent(ModelsIntent.ToggleAddModelExpanded(model.name)) },
      verticalAlignment = Alignment.CenterVertically,
      horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
      BrandLogo(model.name, size = 24.dp, logoDigest = logoDigest)
      Column(Modifier.weight(1f)) {
        Text(
          model.name,
          style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Medium),
          color = colors.label,
          maxLines = 2,
          overflow = TextOverflow.Ellipsis,
        )
        Text(model.subtitle, style = TextStyle(fontSize = 13.sp), color = colors.gray)
      }
      RotatingChevron(expanded = model.expanded, tint = colors.gray, size = 16.dp)
    }
    WizardCheckbox(
      isOn = model.checked,
      indeterminate = model.mixed,
      size = 22,
      modifier = Modifier.padding(start = 12.dp).clickableNoRipple { onIntent(ModelsIntent.ToggleAddModel(model.name)) },
    )
  }
  if (model.expanded) {
    Column(
      modifier = Modifier.fillMaxWidth().padding(start = start + 34.dp, end = 16.dp, bottom = 14.dp),
      verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
      model.sections.forEach { section ->
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
          Text(
            section.label,
            style = TextStyle(fontSize = 12.sp, fontWeight = FontWeight.SemiBold, fontFamily = FontFamily.Monospace),
            color = colors.gray,
            maxLines = 1,
            overflow = TextOverflow.MiddleEllipsis,
          )
          FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            section.chips.forEach { chip -> AddQuantChip(chip) { onIntent(ModelsIntent.ToggleAddQuant(chip.key)) } }
          }
        }
      }
    }
  }
}

@Composable
private fun AddQuantChip(chip: AddQuantChipUi, onClick: () -> Unit) {
  val colors = PipetteTheme.colors
  val shape = RoundedCornerShape(percent = 50)
  val fill =
    when {
      chip.onDevice -> colors.secondaryBackground
      chip.selected -> colors.label
      else -> colors.background
    }
  val text =
    when {
      chip.onDevice -> colors.gray
      chip.selected -> colors.background
      else -> colors.label
    }
  Row(
    modifier =
      Modifier.height(32.dp)
        .clip(shape)
        .background(fill)
        .then(if (chip.onDevice || chip.selected) Modifier else Modifier.border(BorderStroke(1.dp, colors.label.copy(alpha = 0.18f)), shape))
        .then(if (chip.onDevice) Modifier else Modifier.clickableNoRipple(onClick))
        .padding(horizontal = 12.dp),
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(5.dp),
  ) {
    if (chip.onDevice)
      Icon(
        painter = painterResource(R.drawable.ic_check),
        contentDescription = stringResource(R.string.model_quant_unavailable),
        tint = text,
        modifier = Modifier.size(10.dp),
      )
    Text(chip.label, style = TextStyle(fontSize = 13.sp, fontWeight = FontWeight.Medium), color = text)
    if (chip.sizeLabel.isNotEmpty()) Text(chip.sizeLabel, style = TextStyle(fontSize = 12.sp), color = text.copy(alpha = 0.7f))
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
  val chips =
    listOf(
      AddQuantChipUi(key = "a", label = "Q4_0", sizeLabel = "696 MB", selected = true, onDevice = false),
      AddQuantChipUi(key = "b", label = "Q4_K_M", sizeLabel = "731 MB", selected = false, onDevice = true),
      AddQuantChipUi(key = "c", label = "Q8_0", sizeLabel = "1.25 GB", selected = false, onDevice = false),
    )
  PipetteTheme {
    AddModelsScreen(
      state =
        ModelsUiState(
          addModelsOpen = true,
          addSearch = "",
          addFamilies =
            listOf(
              ModelFamilyUi(
                name = "LFM2",
                logoName = "LFM2-1.2B",
                open = true,
                members =
                  listOf(
                    AddModelUi(
                      name = "LFM2-1.2B",
                      subtitle = "1 selected",
                      expanded = true,
                      checked = false,
                      mixed = true,
                      sections = listOf(AddQuantSectionUi("LiquidAI/LFM2-1.2B-GGUF", chips)),
                    ),
                    AddModelUi(name = "LFM2-350M", subtitle = "GGUF", expanded = false, checked = false, mixed = false, sections = emptyList()),
                  ),
                checked = false,
                mixed = true,
              ),
              ModelFamilyUi(
                name = "Qwen2.5",
                logoName = "Qwen2.5-1.5B",
                open = false,
                members =
                  listOf(
                    AddModelUi(name = "Qwen2.5-1.5B", subtitle = "GGUF", expanded = false, checked = false, mixed = false, sections = emptyList())
                  ),
                checked = false,
              ),
            ),
          addDownloadCount = 1,
          addDownloadBytes = 696L * 1024 * 1024,
        ),
      onIntent = {},
    )
  }
}

@Preview
@Composable
private fun AddModelsScreenEmptyPreview() {
  PipetteTheme { AddModelsScreen(state = ModelsUiState(addModelsOpen = true, addFamilies = emptyList()), onIntent = {}) }
}
