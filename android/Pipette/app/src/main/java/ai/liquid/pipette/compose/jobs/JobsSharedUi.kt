// Small composables shared between the Jobs screens (list, wizard, detail, cell detail): a brand-logo model chip,
// the "Show N more properties" toggle, and the contribute-to-Pipette checkbox row. Kept in one file so the
// screens can be reasoned about individually without cross-file spillover of helpers.
@file:Suppress("MagicNumber")

package ai.liquid.pipette.compose.jobs

import ai.liquid.pipette.R
import ai.liquid.pipette.compose.BrandLogo
import ai.liquid.pipette.compose.WizardCheckbox
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** A model chip: brand logo + (truncated) name inside a hairline capsule. Used by the wizard review and detail chip rows. */
@Composable
internal fun JobsModelChip(name: String) {
  val colors = PipetteTheme.colors
  Row(
    modifier =
      Modifier.height(34.dp)
        .clip(RoundedCornerShape(percent = 50))
        .background(colors.background)
        .border(BorderStroke(1.dp, colors.label.copy(alpha = 0.10f)), RoundedCornerShape(percent = 50))
        .padding(start = 8.dp, end = 14.dp),
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(7.dp),
  ) {
    BrandLogo(name, size = 20.dp)
    Text(
      name,
      style = TextStyle(fontSize = 15.sp),
      color = colors.label,
      maxLines = 1,
      overflow = TextOverflow.Ellipsis,
      modifier = Modifier.widthIn(max = 160.dp),
    )
  }
}

/** "Show N more properties" / "Show less" toggle with a chevron that flips down (collapsed) ↔ up (expanded). */
@Composable
internal fun JobsMorePropertiesToggle(expanded: Boolean, hiddenCount: Int, onClick: () -> Unit) {
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

/** Left checkbox + multi-line contribute copy (iOS contribution row). Shared by the wizard review and the detail body. */
@Composable
internal fun JobsContributeRow(checked: Boolean, enabled: Boolean, onToggle: (Boolean) -> Unit) {
  Row(
    modifier = Modifier.fillMaxWidth().padding(top = 24.dp).clickableNoRipple { if (enabled) onToggle(!checked) },
    horizontalArrangement = Arrangement.spacedBy(14.dp),
    verticalAlignment = Alignment.Top,
  ) {
    WizardCheckbox(isOn = checked, size = 22)
    Text(
      stringResource(R.string.job_contribute_text),
      style = TextStyle(fontSize = 15.sp, lineHeight = 21.sp),
      color = PipetteTheme.colors.label.copy(alpha = 0.78f),
    )
  }
}
