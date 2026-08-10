// iOS-faithful design-system primitives: many small composables (TooManyFunctions) + exact dp/sp/alpha layout literals (MagicNumber).
@file:Suppress("MagicNumber", "TooManyFunctions", "MaxLineLength", "UnusedParameter")

package ai.liquid.pipette.compose

import ai.liquid.pipette.R
import ai.liquid.pipette.compose.theme.PipetteTheme
import ai.liquid.pipette.compose.theme.serif
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.draw.clip
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// ---------------------------------------------------------------------------
// Typography (iOS .pageHeaderLarge / .pageHeaderSmall / section titles)
// ---------------------------------------------------------------------------

/** Large page title: serif 32 (iOS pageHeaderLarge). */
@Composable
fun PageHeaderLarge(text: String, modifier: Modifier = Modifier) {
  Text(text, modifier = modifier, style = serif(32), color = PipetteTheme.colors.label, maxLines = 1)
}

/** Small page title: serif 17 (iOS pageHeaderSmall). */
@Composable
fun PageHeaderSmall(text: String, modifier: Modifier = Modifier) {
  Text(text, modifier = modifier, style = serif(17), color = PipetteTheme.colors.label)
}

/** Section heading: serif 21 in systemGray (iOS SettingsView.sectionTitle). */
@Composable
fun SectionTitle(text: String, modifier: Modifier = Modifier) {
  Text(text, modifier = modifier, style = serif(21), color = PipetteTheme.colors.gray)
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

/** Outlined rounded card: background fill + systemGray5 hairline border, continuous corner (iOS SettingsCard / ModelListCard). */
@Composable
fun IosCard(modifier: Modifier = Modifier, cornerRadius: Int = 23, content: @Composable ColumnScope.() -> Unit) {
  val shape = RoundedCornerShape(cornerRadius.dp)
  Column(
    modifier =
      modifier
        .fillMaxWidth()
        .background(PipetteTheme.colors.background, shape)
        .border(BorderStroke(1.dp, PipetteTheme.colors.gray5), shape)
        .clip(shape),
    content = content,
  )
}

/** Hairline divider tinted systemGray5 (iOS SettingsDivider). */
@Composable
fun IosDivider(modifier: Modifier = Modifier) {
  HorizontalDivider(modifier = modifier, thickness = 1.dp, color = PipetteTheme.colors.gray5)
}

/** Labeled detail row: fixed-width title + value, height 57 (iOS SettingsView.settingsRow). */
@Composable
fun PropertyRow(
  title: String,
  value: String,
  modifier: Modifier = Modifier,
  valueColor: Color = PipetteTheme.colors.gray,
  labelWidth: Int = 120,
  valueAlignment: Alignment.Horizontal = Alignment.End,
  rowHeight: Int = 57,
) {
  Row(
    modifier = modifier.fillMaxWidth().height(rowHeight.dp).padding(horizontal = 24.dp),
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(16.dp),
  ) {
    Text(title, style = TextStyle(fontSize = 16.5.sp), color = PipetteTheme.colors.label, maxLines = 1, modifier = Modifier.width(labelWidth.dp))
    Box(modifier = Modifier.fillMaxWidth(), contentAlignment = if (valueAlignment == Alignment.End) Alignment.CenterEnd else Alignment.CenterStart) {
      Text(value, style = TextStyle(fontSize = 16.5.sp), color = valueColor, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
  }
}

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------

/** Filled capsule action: Color.primary (label) when enabled, systemGray when disabled; text = systemBackground (iOS Register / Download). */
@Composable
fun CapsulePrimaryButton(
  text: String,
  onClick: () -> Unit,
  modifier: Modifier = Modifier,
  enabled: Boolean = true,
  loading: Boolean = false,
  height: Int = 48,
  fontSize: Int = 17,
  leadingIcon: Int? = null,
) {
  val colors = PipetteTheme.colors
  // While loading, show an in-button spinner and swallow taps (iOS shows the same in-button progress
  // on submit) so the action can't be re-triggered mid-flight.
  val clickable = enabled && !loading
  Box(
    modifier =
      modifier
        .fillMaxWidth()
        .height(height.dp)
        .clip(RoundedCornerShape(percent = 50))
        .background(if (clickable) colors.label else colors.gray)
        .then(if (clickable) Modifier.clickableNoRipple(onClick) else Modifier),
    contentAlignment = Alignment.Center,
  ) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      if (loading) {
        CircularProgressIndicator(modifier = Modifier.size(18.dp), color = colors.background, strokeWidth = 2.dp)
      } else if (leadingIcon != null) {
        Icon(painter = painterResource(leadingIcon), contentDescription = null, tint = colors.background, modifier = Modifier.size(18.dp))
      }
      Text(text, style = TextStyle(fontSize = fontSize.sp), color = colors.background)
    }
  }
}

/** Outlined capsule: background fill + primary@12% hairline, label text (iOS "Add models" / "Select all"). */
@Composable
fun CapsuleOutlineButton(
  text: String,
  onClick: () -> Unit,
  modifier: Modifier = Modifier,
  height: Int = 44,
  fontSize: Int = 15,
  fontWeight: FontWeight = FontWeight.SemiBold,
  leadingIcon: Int? = null,
) {
  val colors = PipetteTheme.colors
  Box(
    modifier =
      modifier
        .height(height.dp)
        .clip(RoundedCornerShape(percent = 50))
        .background(colors.background)
        .border(BorderStroke(1.dp, colors.label.copy(alpha = 0.12f)), RoundedCornerShape(percent = 50))
        .clickableNoRipple(onClick)
        .padding(horizontal = 16.dp),
    contentAlignment = Alignment.Center,
  ) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      if (leadingIcon != null) {
        Icon(painter = painterResource(leadingIcon), contentDescription = null, tint = colors.label, modifier = Modifier.size(18.dp))
      }
      Text(text, style = TextStyle(fontSize = fontSize.sp, fontWeight = fontWeight), color = colors.label)
    }
  }
}

/** Destructive filled capsule (iOS sign-out: literal red, white text). */
@Composable
fun DestructiveCapsuleButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, height: Int = 43, leadingIcon: Int? = null) {
  val colors = PipetteTheme.colors
  Box(
    modifier =
      modifier.fillMaxWidth().height(height.dp).clip(RoundedCornerShape(percent = 50)).background(colors.destructive).clickableNoRipple(onClick),
    contentAlignment = Alignment.Center,
  ) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
      if (leadingIcon != null) {
        Icon(painter = painterResource(leadingIcon), contentDescription = null, tint = Color.White, modifier = Modifier.size(18.dp))
      }
      Text(text, style = TextStyle(fontSize = 17.sp), color = Color.White)
    }
  }
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/** Rounded text field: corner 8, systemBackground fill, systemGray5 1.5dp stroke (iOS RegistrationTextField / ModelSearchField). */
@Composable
fun IosTextField(
  value: String,
  onValueChange: (String) -> Unit,
  placeholder: String,
  modifier: Modifier = Modifier,
  height: Int = 48,
  singleLine: Boolean = true,
  minLines: Int = 1,
  maxLines: Int = Int.MAX_VALUE,
  visualTransformation: VisualTransformation = VisualTransformation.None,
  keyboardOptions: KeyboardOptions = KeyboardOptions.Default,
  trailing: (@Composable () -> Unit)? = null,
  enabled: Boolean = true,
) {
  val colors = PipetteTheme.colors
  val shape = RoundedCornerShape(8.dp)
  // Single-line stays a fixed [height] with centered content; multiline grows from [height] as a
  // minimum and top-aligns its text.
  Box(
    modifier =
      modifier
        .fillMaxWidth()
        .then(if (singleLine) Modifier.height(height.dp) else Modifier.heightIn(min = height.dp))
        .clip(shape)
        .background(colors.background)
        .border(BorderStroke(1.5.dp, colors.gray5), shape)
        .padding(horizontal = 16.dp, vertical = if (singleLine) 0.dp else 12.dp),
    contentAlignment = if (singleLine) Alignment.CenterStart else Alignment.TopStart,
  ) {
    // The field takes the row's remaining width so [trailing] keeps its intrinsic size instead of
    // being squeezed out by a long value.
    Row(verticalAlignment = Alignment.CenterVertically) {
      BasicTextField(
        value = value,
        onValueChange = onValueChange,
        enabled = enabled,
        singleLine = singleLine,
        minLines = minLines,
        maxLines = maxLines,
        textStyle = TextStyle(fontSize = 17.sp, color = colors.label),
        cursorBrush = androidx.compose.ui.graphics.SolidColor(colors.label),
        visualTransformation = visualTransformation,
        keyboardOptions = keyboardOptions,
        modifier = Modifier.weight(1f),
        decorationBox = { inner ->
          if (value.isEmpty()) Text(placeholder, style = TextStyle(fontSize = 17.sp), color = colors.gray)
          inner()
        },
      )
      if (trailing != null) {
        Spacer(Modifier.width(8.dp))
        trailing()
      }
    }
  }
}

/**
 * Show/hide toggle for a password field, passed to [IosTextField]'s `trailing` slot. Stateless on purpose: the caller owns `visible` so it also owns
 * the [VisualTransformation], and the two can't drift apart.
 */
@Composable
fun PasswordVisibilityToggle(visible: Boolean, onToggle: () -> Unit) {
  val colors = PipetteTheme.colors
  // The glyph is 20dp but the tap target is 44dp, so it clears the accessibility minimum without the icon growing to match. Sized on the Box rather
  // than the Icon because a `size` before `clickable` would shrink the touchable area to the glyph.
  Box(modifier = Modifier.size(44.dp).clickableNoRipple(onToggle), contentAlignment = Alignment.Center) {
    Icon(
      painter = painterResource(if (visible) R.drawable.ic_eye_off else R.drawable.ic_eye),
      // Names the action rather than the glyph, which is what a screen reader should announce.
      contentDescription = if (visible) "Hide password" else "Show password",
      tint = colors.gray,
      modifier = Modifier.size(20.dp),
    )
  }
}

/**
 * Single-select segmented control (iOS CollectorEndpointPicker / runtime picker).
 *
 * A [selectedIndex] outside `options.indices` renders with nothing selected, which is the honest way to show "no choice made yet" rather than
 * highlighting the first segment as if it were one.
 */
@Composable
fun SegmentedControl(options: List<String>, selectedIndex: Int, onSelect: (Int) -> Unit, modifier: Modifier = Modifier) {
  val colors = PipetteTheme.colors
  val outer = RoundedCornerShape(8.dp)
  Row(
    modifier = modifier.fillMaxWidth().clip(outer).background(colors.background).border(BorderStroke(1.5.dp, colors.gray5), outer).padding(4.dp),
    horizontalArrangement = Arrangement.spacedBy(4.dp),
  ) {
    options.forEachIndexed { index, option ->
      val selected = index == selectedIndex
      Box(
        modifier =
          Modifier.weight(1f)
            .height(42.dp)
            .clip(RoundedCornerShape(7.dp))
            .background(if (selected) colors.label else Color.Transparent)
            .clickableNoRipple { onSelect(index) },
        contentAlignment = Alignment.Center,
      ) {
        Text(
          option,
          style = TextStyle(fontSize = 13.sp, fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal),
          color = if (selected) colors.background else colors.label,
          textAlign = TextAlign.Center,
          maxLines = 2,
        )
      }
    }
  }
}

/**
 * Segmented one-time-code field: [length] equal-width cells each showing a single digit, backed by one hidden [BasicTextField] (input filtered to
 * digits and capped at [length]). The next-to-fill cell is outlined in the label color. Auto-focuses and raises the number keypad on first show.
 * [onValueChange] always receives a digits-only string no longer than [length]; the caller owns the value (so it can react at full length).
 */
@Composable
fun OtpCodeField(value: String, onValueChange: (String) -> Unit, modifier: Modifier = Modifier, length: Int = 6, enabled: Boolean = true) {
  val colors = PipetteTheme.colors
  val focusRequester = remember { FocusRequester() }
  val shape = RoundedCornerShape(8.dp)
  BasicTextField(
    value = value,
    onValueChange = { entered -> onValueChange(entered.filter(Char::isDigit).take(length)) },
    enabled = enabled,
    singleLine = true,
    // Number (not NumberPassword) so IMEs keep the paste affordance and the "code from email" autofill chip.
    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
    // The caret is drawn per-cell below, so hide the field's own.
    cursorBrush = SolidColor(Color.Transparent),
    textStyle = TextStyle(color = Color.Transparent),
    // Cap the overall width so the weighted cells stay compact rather than stretching edge-to-edge.
    modifier = modifier.widthIn(max = 260.dp).fillMaxWidth().focusRequester(focusRequester),
    decorationBox = { innerTextField ->
      Box {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          repeat(length) { index ->
            val char = value.getOrNull(index)?.toString().orEmpty()
            val active = enabled && index == value.length && value.length < length
            Box(
              modifier =
                Modifier.weight(1f)
                  .height(48.dp)
                  .clip(shape)
                  .background(colors.background)
                  .border(BorderStroke(if (active) 2.dp else 1.5.dp, if (active) colors.label else colors.gray5), shape),
              contentAlignment = Alignment.Center,
            ) {
              Text(char, style = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.SemiBold, textAlign = TextAlign.Center), color = colors.label)
            }
          }
        }
        // BasicTextField requires its decorationBox to place innerTextField(); it holds the focus/IME/paste
        // input session. Its glyphs and caret are transparent (see textStyle/cursorBrush above), so it stays
        // invisible while the cells above render the digits.
        innerTextField()
      }
    },
  )
  // One-shot on first composition. Keying on `enabled` would re-focus (re-raise the keypad / steal focus)
  // every time the field re-enables after a failed submit.
  LaunchedEffect(Unit) { focusRequester.requestFocus() }
}

// ---------------------------------------------------------------------------
// Badges / toggle
// ---------------------------------------------------------------------------

/** iOS AppTextChip: white capsule, hairline border, ~30dp tall (used in property rows / quant labels). */
@Composable
fun Chip(text: String, modifier: Modifier = Modifier, fontSize: androidx.compose.ui.unit.TextUnit = 16.sp) {
  val colors = PipetteTheme.colors
  Box(
    modifier =
      modifier
        .height(30.dp)
        .clip(RoundedCornerShape(percent = 50))
        .background(colors.background)
        .border(BorderStroke(1.dp, colors.label.copy(alpha = 0.10f)), RoundedCornerShape(percent = 50))
        .padding(horizontal = 14.dp),
    contentAlignment = Alignment.Center,
  ) {
    Text(text, style = TextStyle(fontSize = fontSize), color = colors.label, maxLines = 1, overflow = TextOverflow.Ellipsis)
  }
}

/** Capsule quant filter pill: filled with label color when selected, hairline outline when not (iOS quant filter). */
@Composable
fun QuantPill(text: String, selected: Boolean, onClick: () -> Unit) {
  val colors = PipetteTheme.colors
  Box(
    modifier =
      Modifier.height(36.dp)
        .clip(RoundedCornerShape(percent = 50))
        .background(if (selected) colors.label else colors.background)
        .border(BorderStroke(1.dp, if (selected) colors.label else colors.label.copy(alpha = 0.12f)), RoundedCornerShape(percent = 50))
        .clickableNoRipple(onClick)
        .padding(horizontal = 16.dp),
    contentAlignment = Alignment.Center,
  ) {
    Text(text, style = TextStyle(fontSize = 14.sp, fontWeight = FontWeight.Medium), color = if (selected) colors.background else colors.label)
  }
}

/** Property row with an 88dp gray label and a flow of chips + optional "N more" (iOS PropertyRow). */
@Composable
fun PropertyChipRow(
  label: String,
  chips: List<String>,
  moreCount: Int = 0,
  modifier: Modifier = Modifier,
  leading: (@Composable (String) -> Unit)? = null,
) {
  val colors = PipetteTheme.colors
  Row(modifier = modifier.fillMaxWidth().padding(vertical = 10.dp), verticalAlignment = Alignment.Top) {
    Text(label, style = TextStyle(fontSize = 15.sp), color = colors.gray, modifier = Modifier.width(88.dp).padding(top = 4.dp))
    androidx.compose.foundation.layout.FlowRow(
      modifier = Modifier.padding(start = 12.dp).weight(1f),
      horizontalArrangement = Arrangement.spacedBy(8.dp),
      verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
      chips.forEach { if (leading != null) leading(it) else Chip(it) }
      if (moreCount > 0) Text("$moreCount more", style = TextStyle(fontSize = 15.sp), color = colors.gray, modifier = Modifier.padding(top = 4.dp))
    }
  }
}

/** Capsule status pill: color text over color@15% fill (iOS StatusBadge). */
@Composable
fun StatusBadge(text: String, color: Color, modifier: Modifier = Modifier) {
  Box(
    modifier = modifier.clip(RoundedCornerShape(percent = 50)).background(color.copy(alpha = 0.15f)).padding(horizontal = 8.dp, vertical = 2.dp),
    contentAlignment = Alignment.Center,
  ) {
    Text(text, style = TextStyle(fontSize = 12.sp, fontWeight = FontWeight.SemiBold), color = color)
  }
}

/** iOS-style toggle (Switch tinted to the label color). */
@Composable
fun IosToggle(checked: Boolean, onCheckedChange: (Boolean) -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true) {
  val colors = PipetteTheme.colors
  Switch(
    checked = checked,
    onCheckedChange = onCheckedChange,
    enabled = enabled,
    modifier = modifier,
    colors =
      SwitchDefaults.colors(
        checkedThumbColor = colors.background,
        checkedTrackColor = colors.label,
        uncheckedThumbColor = colors.background,
        uncheckedTrackColor = colors.gray3,
        uncheckedBorderColor = colors.gray3,
      ),
  )
}

/** iOS WizardCheckbox: rounded square, filled with label color when on/indeterminate, systemGray3 hairline when off; check icon or − dash. */
@Composable
fun WizardCheckbox(isOn: Boolean, modifier: Modifier = Modifier, size: Int = 22, indeterminate: Boolean = false) {
  val colors = PipetteTheme.colors
  val shape = RoundedCornerShape(6.dp)
  val filled = isOn || indeterminate
  Box(
    modifier =
      modifier
        .height(size.dp)
        .width(size.dp)
        .clip(shape)
        .background(if (filled) colors.label else Color.Transparent)
        .border(BorderStroke(1.5.dp, if (filled) Color.Transparent else colors.gray3), shape),
    contentAlignment = Alignment.Center,
  ) {
    if (indeterminate) {
      Text("–", style = TextStyle(fontSize = (size * 0.6).sp, fontWeight = FontWeight.Bold), color = colors.background)
    } else if (isOn) {
      Icon(
        painter = painterResource(R.drawable.ic_check),
        contentDescription = null,
        tint = colors.background,
        modifier = Modifier.size((size * 0.62).dp),
      )
    }
  }
}

/** A tappable row: leading label + trailing [WizardCheckbox] (iOS selectable rows). */
@Composable
fun CheckboxRow(text: String, checked: Boolean, onToggle: (Boolean) -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true) {
  Row(
    modifier = modifier.fillMaxWidth().clickableNoRipple { if (enabled) onToggle(!checked) }.padding(vertical = 10.dp),
    verticalAlignment = Alignment.CenterVertically,
  ) {
    Text(
      text,
      style = TextStyle(fontSize = 16.sp),
      color = if (enabled) PipetteTheme.colors.label else PipetteTheme.colors.gray,
      modifier = Modifier.weight(1f).padding(end = 12.dp),
    )
    WizardCheckbox(isOn = checked, size = 22)
  }
}

/** Wraps an action behind a confirmation dialog (replaces UiKit.confirm). [content] receives a `trigger` to invoke; confirming runs [onConfirm]. */
@Composable
fun ConfirmAction(message: String, positiveText: String = "Delete", onConfirm: () -> Unit, content: @Composable (trigger: () -> Unit) -> Unit) {
  var show by remember { mutableStateOf(false) }
  content { show = true }
  if (show) {
    AlertDialog(
      onDismissRequest = { show = false },
      text = { Text(message) },
      confirmButton = {
        TextButton(
          onClick = {
            show = false
            onConfirm()
          }
        ) {
          Text(positiveText)
        }
      },
      dismissButton = { TextButton(onClick = { show = false }) { Text("Cancel") } },
    )
  }
}

/** iOS search field: magnifier + text, 38dp, rounded-8 with primary@10% hairline (iOS AppSearchField). */
@Composable
fun SearchField(value: String, onValueChange: (String) -> Unit, placeholder: String, modifier: Modifier = Modifier) {
  val colors = PipetteTheme.colors
  val shape = RoundedCornerShape(8.dp)
  Row(
    modifier =
      modifier
        .fillMaxWidth()
        .height(38.dp)
        .clip(shape)
        .background(colors.background)
        .border(BorderStroke(1.dp, colors.label.copy(alpha = 0.10f)), shape)
        .padding(horizontal = 16.dp),
    verticalAlignment = Alignment.CenterVertically,
    horizontalArrangement = Arrangement.spacedBy(12.dp),
  ) {
    Icon(painter = painterResource(R.drawable.ic_search), contentDescription = null, tint = colors.gray, modifier = Modifier.size(18.dp))
    Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.CenterStart) {
      BasicTextField(
        value = value,
        onValueChange = onValueChange,
        singleLine = true,
        textStyle = TextStyle(fontSize = 15.sp, color = colors.label),
        cursorBrush = androidx.compose.ui.graphics.SolidColor(colors.label),
        decorationBox = { inner ->
          if (value.isEmpty()) Text(placeholder, style = TextStyle(fontSize = 15.sp), color = colors.gray)
          inner()
        },
      )
    }
    if (value.isNotEmpty())
      Icon(
        painter = painterResource(R.drawable.ic_close),
        contentDescription = null,
        tint = colors.gray,
        modifier = Modifier.size(16.dp).clickableNoRipple { onValueChange("") },
      )
  }
}

/** Chevron that animates between right (collapsed) and down (expanded) by rotating on a flag flip (iOS disclosure chevron). */
@Composable
fun RotatingChevron(expanded: Boolean, modifier: Modifier = Modifier, tint: Color = PipetteTheme.colors.gray, size: Dp = 20.dp) {
  val rotation by animateFloatAsState(targetValue = if (expanded) 90f else 0f, label = "chevronRotation")
  Icon(
    painter = painterResource(R.drawable.ic_chevron_right),
    contentDescription = null,
    tint = tint,
    modifier = modifier.size(size).graphicsLayer { rotationZ = rotation },
  )
}

// ---------------------------------------------------------------------------
// Compatibility layer (iOS-styled) for screens not yet given the pixel-faithful
// pass (Jobs, auth gate). These map the earlier primitive names onto the new
// iOS components so those screens render in the new theme unchanged; they'll be
// replaced as each screen gets its dedicated layout pass.
// ---------------------------------------------------------------------------

@Composable
fun DisplayTitle(text: String, modifier: Modifier = Modifier) {
  Text(text, modifier = modifier, style = serif(28), color = PipetteTheme.colors.label)
}

@Composable
fun AppLabel(text: String, modifier: Modifier = Modifier) {
  Text(text, modifier = modifier.padding(vertical = 4.dp), style = TextStyle(fontSize = 16.sp), color = PipetteTheme.colors.label)
}

@Composable
fun MutedLabel(text: String, modifier: Modifier = Modifier) {
  Text(text, modifier = modifier.padding(vertical = 2.dp), style = TextStyle(fontSize = 14.sp), color = PipetteTheme.colors.gray)
}

@Composable
fun AppCard(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
  IosCard(modifier = modifier.padding(vertical = 8.dp), cornerRadius = 18) {
    Column(modifier = Modifier.fillMaxWidth().padding(16.dp), content = content)
  }
}

@Composable
fun AppTile(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
  Column(
    modifier =
      modifier
        .fillMaxWidth()
        .padding(top = 6.dp)
        .clip(RoundedCornerShape(12.dp))
        .background(PipetteTheme.colors.secondaryBackground)
        .padding(start = 12.dp, top = 10.dp, end = 12.dp, bottom = 12.dp),
    content = content,
  )
}

@Composable
fun PrimaryButton(
  text: String,
  onClick: () -> Unit,
  modifier: Modifier = Modifier,
  enabled: Boolean = true,
  loading: Boolean = false,
  leadingIcon: Int? = null,
) {
  CapsulePrimaryButton(
    text,
    onClick,
    modifier = modifier.padding(vertical = 6.dp),
    enabled = enabled,
    loading = loading,
    height = 44,
    fontSize = 16,
    leadingIcon = leadingIcon,
  )
}

@Composable
fun OutlineButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true, leadingIcon: Int? = null) {
  CapsuleOutlineButton(
    text,
    onClick,
    modifier = modifier.fillMaxWidth().padding(vertical = 6.dp),
    height = 44,
    fontSize = 16,
    leadingIcon = leadingIcon,
  )
}

@Composable
fun AppTextButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true) {
  Text(
    text,
    style = TextStyle(fontSize = 15.sp, fontWeight = FontWeight.Medium),
    color = if (enabled) PipetteTheme.colors.label else PipetteTheme.colors.gray,
    modifier = modifier.padding(vertical = 6.dp, horizontal = 4.dp).clickableNoRipple { if (enabled) onClick() },
  )
}

@Composable
fun AppCheckbox(text: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true) {
  CheckboxRow(text, checked, onCheckedChange, modifier = modifier, enabled = enabled)
}

@Composable
fun AppFilterChip(text: String, selected: Boolean, onToggle: (Boolean) -> Unit, modifier: Modifier = Modifier, enabled: Boolean = true) {
  val colors = PipetteTheme.colors
  Box(
    modifier =
      modifier
        .clip(RoundedCornerShape(percent = 50))
        .background(if (selected) colors.label else Color.Transparent)
        .border(BorderStroke(1.dp, if (selected) Color.Transparent else colors.label.copy(alpha = 0.12f)), RoundedCornerShape(percent = 50))
        .clickableNoRipple { if (enabled) onToggle(!selected) }
        .padding(horizontal = 14.dp, vertical = 5.dp)
  ) {
    Text(text, style = TextStyle(fontSize = 13.sp, fontWeight = FontWeight.Medium), color = if (selected) colors.background else colors.label)
  }
}

@Composable
fun AppLinearProgress(fraction: Double, modifier: Modifier = Modifier) {
  val colors = PipetteTheme.colors
  Box(
    modifier =
      modifier
        .fillMaxWidth()
        .height(4.dp)
        .padding(vertical = 0.dp)
        .clip(RoundedCornerShape(percent = 50))
        .background(colors.label.copy(alpha = 0.08f))
  ) {
    Box(
      modifier =
        Modifier.fillMaxWidth(fraction.coerceIn(0.0, 1.0).toFloat()).height(4.dp).clip(RoundedCornerShape(percent = 50)).background(colors.label)
    )
  }
}

@Composable
fun AppTextField(
  value: String,
  onValueChange: (String) -> Unit,
  label: String,
  modifier: Modifier = Modifier,
  singleLine: Boolean = true,
  visualTransformation: VisualTransformation = VisualTransformation.None,
  keyboardOptions: KeyboardOptions = KeyboardOptions.Default,
) {
  IosTextField(
    value,
    onValueChange,
    placeholder = label,
    modifier = modifier,
    visualTransformation = visualTransformation,
    keyboardOptions = keyboardOptions,
  )
}

/** Live search (iOS searches as you type); [onApply] receives each change. */
@Composable
fun SearchBlock(hint: String, current: String, onApply: (String) -> Unit, modifier: Modifier = Modifier) {
  SearchField(value = current, onValueChange = onApply, placeholder = hint, modifier = modifier.padding(vertical = 4.dp))
}

@Composable
fun TwoButtonRow(content: @Composable RowScope.() -> Unit) {
  Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp), content = content)
}

/** Heatmap fill for a results cell: label color at intensity-scaled alpha (matches iOS resultCellColor direction). */
@Composable
fun heatmapColor(intensity: Double): Color {
  val a = (0.16 + intensity.coerceIn(0.0, 1.0) * 0.34).toFloat()
  return PipetteTheme.colors.label.copy(alpha = a)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Tap handling without the Material ripple (iOS buttons have no ripple). */
fun Modifier.clickableNoRipple(onClick: () -> Unit): Modifier = composed {
  val interaction = remember { androidx.compose.foundation.interaction.MutableInteractionSource() }
  clickable(interactionSource = interaction, indication = null, onClick = onClick)
}
