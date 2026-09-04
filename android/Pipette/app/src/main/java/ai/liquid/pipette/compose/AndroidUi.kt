package ai.liquid.pipette.compose

import ai.liquid.pipette.R
import ai.liquid.pipette.compose.theme.PipetteTheme
import ai.liquid.pipette.compose.theme.serif
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedCard
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// ---------------------------------------------------------------------------
// Android-flavored primitives (Material 3 top app bar / search bar / elevated
// card). Live alongside the iOS-styled primitives; screens that want the
// Android look call these directly. Wired into the Jobs and Models screens as
// each tab's header + search + list card so the app reads as native Android.
// ---------------------------------------------------------------------------

/**
 * Material top app bar. Uses the app's surface color as the container and stamps the title in the same serif family the screens use for large
 * headers, so the toolbar reads as native Android chrome without breaking the serif-title language shared with iOS.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AndroidTopAppBar(title: String, modifier: Modifier = Modifier, actions: @Composable RowScope.() -> Unit = {}) {
  val colors = PipetteTheme.colors
  TopAppBar(
    title = { Text(title, style = serif(22), color = colors.label, maxLines = 1) },
    colors =
      TopAppBarDefaults.topAppBarColors(containerColor = colors.background, titleContentColor = colors.label, actionIconContentColor = colors.label),
    // The screens already inset for the status bar at the outer scroll, so the
    // toolbar itself shouldn't add another status-bar inset on top.
    windowInsets = androidx.compose.foundation.layout.WindowInsets(0),
    actions = actions,
    // No .shadow(...) — `shadow(2.dp)` draws on all four sides of the toolbar,
    // and the strip above it in the outer column let the top shadow show as
    // well. A flat toolbar reads cleaner; if a divider is wanted later, put a
    // 1 dp HorizontalDivider below in the scaffold.
    modifier = modifier.fillMaxWidth(),
  )
}

/**
 * Filled rounded Android search bar with a leading magnifier and trailing clear. Filters live (as-you-type) — the imperative render cost that made
 * the legacy view-based screens use an Apply button doesn't apply in Compose, where the field's state is remembered independently.
 */
@Composable
fun AndroidSearchBar(hint: String, value: String, onValueChange: (String) -> Unit, modifier: Modifier = Modifier) {
  val colors = PipetteTheme.colors
  val shape = RoundedCornerShape(percent = 50)
  // A BasicTextField only takes focus when tapped directly on its text glyphs —
  // taps on the leading icon or the empty space right of the hint would go
  // ignored. Route the whole row's tap to a FocusRequester so hitting anywhere
  // on the bar starts editing.
  val focusRequester = remember { FocusRequester() }
  // Hold TextFieldValue locally (not just a String). The parent stores search
  // state in a ViewModel and round-trips it back on the next recomposition; if
  // typing outruns that round-trip, the value-based overload of BasicTextField
  // treats the incoming String as authoritative and resets the cursor (users
  // see the caret jump back to after their second letter). Owning the
  // TextFieldValue here keeps the cursor with the field and only pushes the
  // string outward. When the parent replaces the query wholesale (e.g. the
  // Clear icon), sync back to the incoming value.
  var fieldValue by remember { mutableStateOf(TextFieldValue(value, TextRange(value.length))) }
  if (value != fieldValue.text) fieldValue = TextFieldValue(value, TextRange(value.length))
  Row(
    modifier =
      modifier
        .fillMaxWidth()
        .height(48.dp)
        .clip(shape)
        .background(colors.gray6)
        .clickableNoRipple { focusRequester.requestFocus() }
        .padding(horizontal = 16.dp),
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(12.dp),
  ) {
    Icon(painter = painterResource(R.drawable.ic_search), contentDescription = null, tint = colors.gray, modifier = Modifier.size(20.dp))
    Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.CenterStart) {
      BasicTextField(
        value = fieldValue,
        onValueChange = { next ->
          fieldValue = next
          if (next.text != value) onValueChange(next.text)
        },
        singleLine = true,
        textStyle = TextStyle(fontSize = 16.sp, color = colors.label),
        cursorBrush = SolidColor(colors.label),
        decorationBox = { inner ->
          if (fieldValue.text.isEmpty()) Text(hint, style = TextStyle(fontSize = 16.sp), color = colors.gray)
          inner()
        },
        modifier = Modifier.fillMaxWidth().focusRequester(focusRequester),
      )
    }
    if (fieldValue.text.isNotEmpty()) {
      Icon(
        painter = painterResource(R.drawable.ic_close),
        contentDescription = null,
        tint = colors.gray,
        modifier = Modifier.size(18.dp).clickableNoRipple { onValueChange("") },
      )
    }
  }
}

/**
 * An outlined Material card (page-color fill, hairline border, no shadow). Reads as a plain Material 3 outlined surface — the shape the Jobs/Models
 * tabs use for their list containers. Screens outside the Android-flavored tabs keep [IosCard].
 */
@Composable
fun OutlinedAndroidCard(modifier: Modifier = Modifier, cornerRadius: Int = 16, content: @Composable ColumnScope.() -> Unit) {
  val colors = PipetteTheme.colors
  OutlinedCard(
    modifier = modifier.fillMaxWidth(),
    shape = RoundedCornerShape(cornerRadius.dp),
    colors = CardDefaults.outlinedCardColors(containerColor = colors.background),
    border = BorderStroke(1.dp, colors.gray5),
    content = content,
  )
}
