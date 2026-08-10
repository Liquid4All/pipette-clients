// iOS-styled Setup header: layout literals (MagicNumber), as on the screen this was split out of.
@file:Suppress("MagicNumber")

package ai.liquid.pipette.compose.setup.component

import ai.liquid.pipette.R
import ai.liquid.pipette.compose.theme.PipetteTheme
import ai.liquid.pipette.compose.theme.serif
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Logo, welcome copy, and — when a Clerk session got us here — the account the registration will be filed under. */
@Composable
internal fun SetupHeader(clerkEmail: String) {
  val colors = PipetteTheme.colors

  Image(painter = painterResource(R.drawable.pipette_logo_mark), contentDescription = null, modifier = Modifier.size(52.dp))

  Text(text = "Welcome to Pipette", style = serif(26), color = colors.label, textAlign = TextAlign.Center, modifier = Modifier.padding(top = 14.dp))

  Text(
    text = "Measure model performance on your device",
    style = TextStyle(fontSize = 16.sp),
    color = colors.gray,
    textAlign = TextAlign.Center,
    modifier = Modifier.padding(top = 6.dp),
  )

  if (clerkEmail.isNotBlank()) {
    Text(
      text = "Signed in as $clerkEmail",
      style = TextStyle(fontSize = 13.sp),
      color = colors.gray,
      textAlign = TextAlign.Center,
      modifier = Modifier.padding(top = 10.dp),
    )
  }
}
