// Floating pill toolbar: layout literals (MagicNumber); file is named for the composable, not the item type (MatchingDeclarationName).
@file:Suppress("MagicNumber", "MatchingDeclarationName")

package ai.liquid.pipette.compose

import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.annotation.DrawableRes
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.expandHorizontally
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkHorizontally
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * A single entry in a [FloatingToolbarMenu].
 *
 * @param key stable identifier used to track selection.
 * @param iconRes drawable resource shown in every state.
 * @param title label revealed only while this item is selected.
 */
data class FloatingToolbarMenuItem<T>(val key: T, @DrawableRes val iconRes: Int, val title: String)

/**
 * A floating, pill-shaped toolbar that doubles as a single-select menu (adapted from the Uku app's Material 3 "toolbar" component). The selected item
 * expands into a filled pill showing its icon and title; the rest collapse to icon-only buttons. Styled with [PipetteTheme] tokens so it matches the
 * iOS-derived palette (filled pill = `label`, container = `background` with a hairline border + soft shadow).
 */
@Composable
fun <T> FloatingToolbarMenu(items: List<FloatingToolbarMenuItem<T>>, selectedKey: T, onSelect: (T) -> Unit, modifier: Modifier = Modifier) {
  val colors = PipetteTheme.colors
  val shape = RoundedCornerShape(percent = 50)
  Row(
    modifier =
      modifier
        .shadow(12.dp, shape, clip = false)
        .clip(shape)
        .background(colors.secondaryBackground)
        .border(BorderStroke(1.dp, colors.label.copy(alpha = 0.06f)), shape)
        .selectableGroup()
        .padding(6.dp),
    horizontalArrangement = Arrangement.spacedBy(4.dp),
    verticalAlignment = Alignment.CenterVertically,
  ) {
    items.forEach { item -> FloatingToolbarMenuItemContent(item = item, selected = item.key == selectedKey, onClick = { onSelect(item.key) }) }
  }
}

@Composable
private fun <T> FloatingToolbarMenuItemContent(item: FloatingToolbarMenuItem<T>, selected: Boolean, onClick: () -> Unit) {
  val colors = PipetteTheme.colors
  val containerColor by animateColorAsState(if (selected) colors.label else Color.Transparent, label = "containerColor")
  val contentColor by animateColorAsState(if (selected) colors.background else colors.gray, label = "contentColor")
  // Selected items get extra trailing room for the label; collapsed ones stay square-ish.
  val horizontalPadding by animateDpAsState(if (selected) 24.dp else 16.dp, label = "horizontalPadding")

  Row(
    modifier =
      Modifier.clip(RoundedCornerShape(percent = 50))
        .background(containerColor)
        .selectable(selected = selected, onClick = onClick, role = Role.Tab)
        .padding(horizontal = horizontalPadding, vertical = 14.dp),
    verticalAlignment = Alignment.CenterVertically,
  ) {
    Icon(painter = painterResource(item.iconRes), contentDescription = item.title, tint = contentColor, modifier = Modifier.height(24.dp))

    // Only the selected item reveals its title; others remain icon-only.
    AnimatedVisibility(
      visible = selected,
      enter = expandHorizontally(expandFrom = Alignment.Start) + fadeIn(),
      exit = shrinkHorizontally(shrinkTowards = Alignment.Start) + fadeOut(),
    ) {
      Row(verticalAlignment = Alignment.CenterVertically) {
        Spacer(modifier = Modifier.width(8.dp))
        Text(text = item.title, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = contentColor)
      }
    }
  }
}

@androidx.compose.ui.tooling.preview.Preview(name = "Floating tab bar", showBackground = true, backgroundColor = 0xFF000000)
@Composable
private fun FloatingToolbarMenuPreview() {
  PipetteTheme(darkTheme = true) {
    FloatingToolbarMenu(
      items =
        listOf(
          FloatingToolbarMenuItem(ai.liquid.pipette.Tab.JOBS, ai.liquid.pipette.R.drawable.ic_tab_jobs, "Jobs"),
          FloatingToolbarMenuItem(ai.liquid.pipette.Tab.MODELS, ai.liquid.pipette.R.drawable.ic_tab_models, "Models"),
          FloatingToolbarMenuItem(ai.liquid.pipette.Tab.SETTINGS, ai.liquid.pipette.R.drawable.ic_tab_settings, "Settings"),
        ),
      selectedKey = ai.liquid.pipette.Tab.JOBS,
      onSelect = {},
    )
  }
}
