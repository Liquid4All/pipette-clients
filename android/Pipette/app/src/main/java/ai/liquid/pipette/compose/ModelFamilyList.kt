// The model lists group their rows under catalog families (iOS ModelFamilyHeader). One header and one list layout serve Add models, Your models and
// the job wizard's model step, so the three lists behave the same.
@file:Suppress("MagicNumber")

package ai.liquid.pipette.compose

import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.key
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * One family of a model list. [members] are the rows that the list shows. A family with one member shows that row with no header. [checked] and
 * [mixed] describe the selection of [members]. Set [checked] to null to hide the family checkbox. [logoDigest] names the catalog logo of the family.
 * The header and each row use their own bundled brand logo until the catalog logo loads.
 */
data class ModelFamilyUi<T>(
  val name: String,
  val logoName: String,
  val logoDigest: String? = null,
  val open: Boolean,
  val members: List<T>,
  val checked: Boolean? = null,
  val mixed: Boolean = false,
)

/** The look of a family header, set to match the rows of its list. */
data class ModelFamilyStyle(
  val horizontalPadding: Dp = 16.dp,
  val verticalPadding: Dp = 14.dp,
  val logoSize: Dp = 24.dp,
  val gap: Dp = 10.dp,
  val titleSize: Int = 16,
  val subtitleSize: Int = 13,
  val checkboxSize: Int = 22,
)

/** The chevron column of a family header: the chevron and the gap after it. A row without a chevron has this inset, so the logos line up. */
val FamilyChevronColumn: Dp = 26.dp

/** The extra indent of the models in an open family. The indent shows which models belong to the family. */
val FamilyMemberIndent: Dp = 16.dp

/** Draw [families] inside the caller's card. The function passes [memberIndent] to [row] only when the family has a header. */
@Composable
fun <T> ModelFamilyList(
  families: List<ModelFamilyUi<T>>,
  onToggleOpen: (String) -> Unit,
  onToggleChecked: (String) -> Unit = {},
  style: ModelFamilyStyle = ModelFamilyStyle(),
  memberKey: (T) -> String,
  memberIndent: Dp = FamilyMemberIndent,
  row: @Composable (member: T, indent: Dp, logoDigest: String?) -> Unit,
) {
  families.forEachIndexed { i, family ->
    if (i > 0) IosDivider()
    key(family.name) {
      val hasHeader = family.members.size > 1
      if (hasHeader) {
        ModelFamilyHeader(family, style, onToggleOpen = { onToggleOpen(family.name) }, onToggleChecked = { onToggleChecked(family.name) })
      }
      if (!hasHeader || family.open) {
        family.members.forEach { member ->
          key(memberKey(member)) {
            if (hasHeader) IosDivider(Modifier.padding(start = style.horizontalPadding + FamilyChevronColumn + memberIndent))
            row(member, if (hasHeader) memberIndent else 0.dp, family.logoDigest)
          }
        }
      }
    }
  }
}

@Composable
private fun ModelFamilyHeader(family: ModelFamilyUi<*>, style: ModelFamilyStyle, onToggleOpen: () -> Unit, onToggleChecked: () -> Unit) {
  val colors = PipetteTheme.colors
  val count = family.members.size
  Row(
    modifier = Modifier.fillMaxWidth().padding(horizontal = style.horizontalPadding, vertical = style.verticalPadding),
    verticalAlignment = Alignment.CenterVertically,
  ) {
    Row(modifier = Modifier.weight(1f).clickableNoRipple(onToggleOpen), verticalAlignment = Alignment.CenterVertically) {
      Box(Modifier.width(FamilyChevronColumn)) { RotatingChevron(expanded = family.open, tint = colors.gray, modifier = Modifier.width(16.dp)) }
      Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(style.gap)) {
        BrandLogo(family.logoName, size = style.logoSize, logoDigest = family.logoDigest)
        Column {
          Text(
            family.name,
            style = TextStyle(fontSize = style.titleSize.sp, fontWeight = FontWeight.SemiBold),
            color = colors.label,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
          )
          Text("$count model${if (count == 1) "" else "s"}", style = TextStyle(fontSize = style.subtitleSize.sp), color = colors.gray)
        }
      }
    }
    family.checked?.let { checked ->
      WizardCheckbox(
        isOn = checked,
        indeterminate = family.mixed,
        size = style.checkboxSize,
        modifier = Modifier.padding(start = 12.dp).clickableNoRipple(onToggleChecked),
      )
    }
  }
}
