// iOS system color values (sRGB), light + dark, mirrored from Apple's UIColor system palette so the Compose UI matches the SwiftUI client.
@file:Suppress("MagicNumber", "MatchingDeclarationName")

package ai.liquid.pipette.compose.theme

import androidx.compose.ui.graphics.Color

/**
 * The iOS-style semantic palette. iOS views use `Color.primary` (= the adaptive label color), `Color(.systemGray*)`, and `Color(.systemBackground)`
 * rather than a fixed brand palette, so we reproduce Apple's actual light/dark system values and switch on the device appearance. `label` is black in
 * light / white in dark; `background` is white / black; the grays differ per appearance.
 */
data class PipetteColors(
  val background: Color,
  val secondaryBackground: Color,
  val label: Color,
  val secondaryLabel: Color,
  val tertiaryLabel: Color,
  val gray: Color,
  val gray2: Color,
  val gray3: Color,
  val gray4: Color,
  val gray5: Color,
  val gray6: Color,
  val red: Color,
  val orange: Color,
  val green: Color,
  /** Sign-out button red literal from SettingsView (Color(red:0.91 green:0.12 blue:0.14)). */
  val destructive: Color,
  /** Thermal "nominal" green literal from SettingsView (Color(red:0.12 green:0.75 blue:0.32)). */
  val thermalNominal: Color,
  val isDark: Boolean,
)

internal val LightPipetteColors =
  PipetteColors(
    background = Color(0xFFFFFFFF),
    secondaryBackground = Color(0xFFF2F2F7),
    label = Color(0xFF000000),
    secondaryLabel = Color(0x993C3C43), // 60% opacity overlay
    tertiaryLabel = Color(0x4D3C3C43), // 30%
    gray = Color(0xFF8E8E93),
    gray2 = Color(0xFFAEAEB2),
    gray3 = Color(0xFFC7C7CC),
    gray4 = Color(0xFFD1D1D6),
    gray5 = Color(0xFFE5E5EA),
    gray6 = Color(0xFFF2F2F7),
    red = Color(0xFFFF3B30),
    orange = Color(0xFFFF9500),
    green = Color(0xFF34C759),
    destructive = Color(red = 0.91f, green = 0.12f, blue = 0.14f),
    thermalNominal = Color(red = 0.12f, green = 0.75f, blue = 0.32f),
    isDark = false,
  )

internal val DarkPipetteColors =
  PipetteColors(
    background = Color(0xFF000000),
    secondaryBackground = Color(0xFF1C1C1E),
    label = Color(0xFFFFFFFF),
    secondaryLabel = Color(0x99EBEBF5),
    tertiaryLabel = Color(0x4DEBEBF5),
    gray = Color(0xFF8E8E93),
    gray2 = Color(0xFF636366),
    gray3 = Color(0xFF48484A),
    gray4 = Color(0xFF3A3A3C),
    gray5 = Color(0xFF2C2C2E),
    gray6 = Color(0xFF1C1C1E),
    red = Color(0xFFFF453A),
    orange = Color(0xFFFF9F0A),
    green = Color(0xFF30D158),
    destructive = Color(red = 0.91f, green = 0.12f, blue = 0.14f),
    thermalNominal = Color(red = 0.16f, green = 0.80f, blue = 0.38f),
    isDark = true,
  )
