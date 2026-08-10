package ai.liquid.pipette.compose

import ai.liquid.pipette.R
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Brand logo for a model, mirroring iOS `ModelBrand.detect` + `BrandLogoView`. Logos are the project's own assets, ported from the iOS asset catalog
 * into res/drawable. Brands without a bundled asset (Meta/Llama, DeepSeek, Microsoft) fall back to a neutral initial badge.
 */
@Composable
fun BrandLogo(modelName: String, modifier: Modifier = Modifier, size: Dp = 24.dp) {
  val res = brandDrawable(modelName)
  if (res != null) {
    androidx.compose.foundation.Image(
      painter = painterResource(res),
      contentDescription = null,
      contentScale = ContentScale.Fit,
      modifier = modifier.size(size),
    )
  } else {
    Box(
      modifier = modifier.size(size).clip(RoundedCornerShape(percent = 50)).background(PipetteTheme.colors.gray5),
      contentAlignment = Alignment.Center,
    ) {
      Text(
        modelName.trim().take(1).uppercase().ifBlank { "?" },
        style = TextStyle(fontSize = (size.value * 0.5).sp, fontWeight = FontWeight.SemiBold),
        color = PipetteTheme.colors.gray,
      )
    }
  }
}

/** Maps a model name to a bundled brand drawable, or null when none exists (caller shows a fallback). Detection order matches iOS ModelBrand. */
private fun brandDrawable(name: String): Int? {
  val n = name.lowercase()
  return when {
    n.contains("lfm") || n.contains("liquid") -> R.drawable.brand_liquid
    n.contains("gemma") -> R.drawable.brand_google
    n.contains("granite") || n.contains("ibm") -> R.drawable.brand_ibm
    n.contains("qwen") -> R.drawable.brand_qwen
    n.contains("mistral") || n.contains("ministral") || n.contains("mixtral") -> R.drawable.brand_mistral
    else -> null // meta/llama, deepseek, microsoft, unknown → fallback badge
  }
}

@androidx.compose.ui.tooling.preview.Preview(name = "Brand logo", showBackground = true, backgroundColor = 0xFF000000)
@Composable
private fun BrandLogoPreview() {
  PipetteTheme(darkTheme = true) { BrandLogo("Qwen 3.5 2B", size = 40.dp) }
}
