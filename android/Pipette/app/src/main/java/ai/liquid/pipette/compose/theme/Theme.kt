// Typography sizes are ad-hoc per the SwiftUI source; serif uses bundled Charter (a Bitstream-Charter transitional serif ~ iOS IowanOldStyle).
@file:Suppress("MagicNumber")

package ai.liquid.pipette.compose.theme

import ai.liquid.pipette.R
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/** Bundled Charter (≈ iOS IowanOldStyle): the serif used for page titles and card headings. */
val PipetteSerif = FontFamily(Font(R.font.charter_regular, FontWeight.Normal), Font(R.font.charter_bold, FontWeight.Bold))

/** A serif [TextStyle] at [size]sp, matching iOS `.serif(size)`. Line height ~1.3× so Charter's tall ascenders aren't clipped. */
fun serif(size: Int, weight: FontWeight = FontWeight.Normal): TextStyle =
  TextStyle(fontFamily = PipetteSerif, fontWeight = weight, fontSize = size.sp, lineHeight = (size * 1.3).sp)

val LocalPipetteColors = staticCompositionLocalOf { LightPipetteColors }

object PipetteTheme {
  val colors: PipetteColors
    @Composable get() = LocalPipetteColors.current
}

@Composable
fun PipetteTheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
  val colors = if (darkTheme) DarkPipetteColors else LightPipetteColors
  // Map the Material scheme too (a few Material widgets — TextField, Checkbox, AlertDialog — still read it). primary = the iOS label
  // (adaptive black/white) so filled controls match iOS's `Color.primary` fills.
  val scheme =
    if (darkTheme) {
      darkColorScheme(
        primary = colors.label,
        onPrimary = colors.background,
        background = colors.background,
        onBackground = colors.label,
        surface = colors.background,
        onSurface = colors.label,
        surfaceVariant = colors.secondaryBackground,
        onSurfaceVariant = colors.gray,
        outline = colors.gray5,
        error = colors.red,
      )
    } else {
      lightColorScheme(
        primary = colors.label,
        onPrimary = colors.background,
        background = colors.background,
        onBackground = colors.label,
        surface = colors.background,
        onSurface = colors.label,
        surfaceVariant = colors.secondaryBackground,
        onSurfaceVariant = colors.gray,
        outline = colors.gray5,
        error = colors.red,
      )
    }
  CompositionLocalProvider(LocalPipetteColors provides colors) { MaterialTheme(colorScheme = scheme, content = content) }
}
