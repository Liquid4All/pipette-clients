package ai.liquid.pipette.compose.setup.component

import ai.liquid.pipette.CollectorEndpoint
import ai.liquid.pipette.CollectorEndpointOption
import ai.liquid.pipette.compose.IosTextField
import ai.liquid.pipette.compose.SegmentedControl
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Collector choice: the production collector, or a URL typed in for a self-hosted deployment. */
@Composable
internal fun CollectorPicker(selectedIndex: Int, onSelect: (Int) -> Unit, customUrl: String, onCustomUrlChange: (String) -> Unit) {
  val colors = PipetteTheme.colors
  val isCustom = CollectorEndpointOption.entries[selectedIndex] == CollectorEndpointOption.CUSTOM

  Text(
    text = "Collector",
    style = TextStyle(fontSize = 12.sp),
    color = colors.gray,
    modifier = Modifier.fillMaxWidth().padding(top = 16.dp, start = 6.dp, bottom = 8.dp),
  )

  SegmentedControl(options = CollectorEndpointOption.entries.map { it.title }, selectedIndex = selectedIndex, onSelect = onSelect)

  // Expand rather than appear: the field pushes the form down, and a jump makes the segment tap read as a
  // screen change instead of one control opening.
  AnimatedVisibility(visible = isCustom, enter = fadeIn() + expandVertically(), exit = fadeOut() + shrinkVertically()) {
    Column(modifier = Modifier.fillMaxWidth()) {
      IosTextField(
        value = customUrl,
        onValueChange = onCustomUrlChange,
        placeholder = "Custom collector URL",
        // Uri type keeps the IME off autocorrect/capitalization, which would mangle a host as it is typed.
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
        modifier = Modifier.padding(top = 12.dp),
      )

      // Animated too, so the message doesn't snap in and out under the caret as a URL is typed.
      AnimatedVisibility(
        visible = customUrl.isNotBlank() && CollectorEndpoint.normalizedCustomUrl(customUrl) == null,
        enter = fadeIn() + expandVertically(),
        exit = fadeOut() + shrinkVertically(),
      ) {
        Text(
          text = "Enter a valid HTTPS collector URL.",
          style = TextStyle(fontSize = 13.sp),
          color = colors.destructive,
          modifier = Modifier.fillMaxWidth().padding(top = 6.dp, start = 6.dp),
        )
      }
    }
  }
}
