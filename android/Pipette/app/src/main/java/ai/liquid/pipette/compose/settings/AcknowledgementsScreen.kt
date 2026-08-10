// iOS-styled Acknowledgements (Settings → Open source licenses): layout literals + license body lines.
@file:Suppress("MagicNumber", "MaxLineLength")

package ai.liquid.pipette.compose.settings

import ai.liquid.pipette.compose.IosCard
import ai.liquid.pipette.compose.IosDivider
import ai.liquid.pipette.compose.SectionTitle
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Open-source licenses, reachable from Settings: a list of bundled components → per-item license text. [onBack] returns to Settings. */
@Composable
fun AcknowledgementsScreen(onBack: () -> Unit) {
  val context = LocalContext.current
  val items = remember { Acknowledgements.all(context) }
  var selected by remember { mutableStateOf<Acknowledgement?>(null) }

  val current = selected
  if (current != null) {
    LicenseDetail(current, onBack = { selected = null })
    return
  }

  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    NavBar(title = "Open Source Licenses", onBack = onBack)
    Column(modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 36.dp)) {
      SectionTitle("Bundled Components", modifier = Modifier.padding(vertical = 12.dp))
      IosCard(cornerRadius = 18) {
        items.forEachIndexed { i, item ->
          if (i > 0) IosDivider()
          Row(
            modifier = Modifier.fillMaxWidth().clickableNoRipple { selected = item }.padding(horizontal = 18.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
          ) {
            Column(modifier = Modifier.weight(1f)) {
              Text(item.name, style = TextStyle(fontSize = 16.5.sp), color = PipetteTheme.colors.label)
              Text(item.license, style = TextStyle(fontSize = 13.sp), color = PipetteTheme.colors.gray, modifier = Modifier.padding(top = 3.dp))
            }
            androidx.compose.material3.Icon(
              painter = androidx.compose.ui.res.painterResource(ai.liquid.pipette.R.drawable.ic_chevron_right),
              contentDescription = null,
              tint = PipetteTheme.colors.gray3,
              modifier = Modifier.size(18.dp),
            )
          }
        }
      }
    }
  }
}

@Composable
private fun LicenseDetail(item: Acknowledgement, onBack: () -> Unit) {
  Column(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars)) {
    NavBar(title = item.name, onBack = onBack)
    Text(
      item.text,
      style = TextStyle(fontSize = 12.sp, fontFamily = FontFamily.Monospace),
      color = PipetteTheme.colors.label,
      modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp).padding(bottom = 36.dp),
    )
  }
}

@Composable
private fun NavBar(title: String, onBack: () -> Unit) {
  Box(modifier = Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 16.dp), contentAlignment = Alignment.Center) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.align(Alignment.CenterStart).clickableNoRipple(onBack)) {
      androidx.compose.material3.Icon(
        painter = androidx.compose.ui.res.painterResource(ai.liquid.pipette.R.drawable.ic_chevron_left),
        contentDescription = null,
        tint = PipetteTheme.colors.label,
        modifier = Modifier.size(22.dp),
      )
      Text("Back", style = TextStyle(fontSize = 17.sp), color = PipetteTheme.colors.label)
    }
    Text(title, style = TextStyle(fontSize = 17.sp), color = PipetteTheme.colors.label, maxLines = 1)
  }
}

@Preview
@Composable
private fun AcknowledgementsScreenPreview() {
  PipetteTheme { AcknowledgementsScreen(onBack = {}) }
}
