// Segmented-list row: layout literals (MagicNumber).
@file:Suppress("MagicNumber")

package ai.liquid.pipette.compose

import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.ripple
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp

// Corner radii for a segmented group: outer (group top/bottom edges), inner (between adjacent
// items), and the enlarged radius every corner animates to while the row is pressed.
private val SegmentOuterCorner = 20.dp
private val SegmentInnerCorner = 6.dp
private val SegmentPressedCorner = 28.dp

/**
 * One row of a "segmented" list group (our take on Material 3's expressive SegmentedListItem).
 *
 * The corner radius is derived from [position] within a group of [count]: the first row rounds its top, the last rounds its bottom, middle rows stay
 * tight, and a lone row (count == 1) rounds all four. While pressed, every corner animates out to a larger radius for tactile feedback. Content slots
 * mirror Material's ListItem ([headlineContent] required; leading/trailing/supporting optional).
 *
 * Group rows in a `Column(verticalArrangement = Arrangement.spacedBy(2.dp))` and pass each row its index as [position] and the group size as [count].
 */
@Composable
fun SegmentedListItem(
  position: Int,
  count: Int,
  modifier: Modifier = Modifier,
  onClick: (() -> Unit)? = null,
  leadingContent: (@Composable () -> Unit)? = null,
  trailingContent: (@Composable () -> Unit)? = null,
  supportingContent: (@Composable () -> Unit)? = null,
  headlineContent: @Composable () -> Unit,
) {
  val interaction = remember { MutableInteractionSource() }
  val pressed by interaction.collectIsPressedAsState()

  val topTarget = if (pressed) SegmentPressedCorner else if (position == 0) SegmentOuterCorner else SegmentInnerCorner
  val bottomTarget = if (pressed) SegmentPressedCorner else if (position == count - 1) SegmentOuterCorner else SegmentInnerCorner
  val topRadius by animateDpAsState(topTarget, label = "segmentTopCorner")
  val bottomRadius by animateDpAsState(bottomTarget, label = "segmentBottomCorner")
  val shape = RoundedCornerShape(topStart = topRadius, topEnd = topRadius, bottomStart = bottomRadius, bottomEnd = bottomRadius)

  val clickModifier = if (onClick != null) Modifier.clickable(interactionSource = interaction, indication = ripple(), onClick = onClick) else Modifier

  Row(
    modifier =
      modifier
        .fillMaxWidth()
        .clip(shape)
        .background(PipetteTheme.colors.secondaryBackground)
        .then(clickModifier)
        .heightIn(min = 56.dp)
        .padding(horizontal = 20.dp, vertical = 14.dp),
    verticalAlignment = Alignment.CenterVertically,
  ) {
    if (leadingContent != null) {
      leadingContent()
      Spacer(Modifier.width(16.dp))
    }
    Column(modifier = Modifier.weight(1f)) {
      headlineContent()
      if (supportingContent != null) {
        Spacer(Modifier.height(2.dp))
        supportingContent()
      }
    }
    if (trailingContent != null) {
      Spacer(Modifier.width(16.dp))
      trailingContent()
    }
  }
}
