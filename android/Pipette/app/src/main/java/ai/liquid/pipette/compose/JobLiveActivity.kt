// Hard-coded dp/sp layout literals + clock math (MagicNumber), as in the sibling PocketModeScreen.
@file:Suppress("MagicNumber")

package ai.liquid.pipette.compose

import ai.liquid.pipette.coolingCaption
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

/**
 * The "what is the benchmark doing right now" block shared by the running-job detail page and Pocket Mode (iOS JobLiveActivityView): the current cell
 * on one line, and the fine-grained per-rep progress on the next. While the thermal gate is cooling, that second line shows a live "Cooling m:ss /
 * max" timer instead — ticked once a second off [coolingSinceMillis], with the ambient "we're cooling" cue left to the card background wash (which
 * costs no layout).
 */
@Composable
fun JobLiveActivity(
  currentCellLabel: String,
  progressText: String,
  coolingSinceMillis: Long?,
  colors: JobActivityColors,
  modifier: Modifier = Modifier,
) {
  // Re-read the clock once a second while cooling so only this block recomposes — no manual timer,
  // no whole-screen invalidation. Idle (returns the initial value) whenever we're not cooling.
  val now by
    produceState(initialValue = System.currentTimeMillis(), coolingSinceMillis) {
      if (coolingSinceMillis == null) return@produceState
      while (true) {
        value = System.currentTimeMillis()
        delay(1_000L)
      }
    }

  Column(modifier = modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
    if (currentCellLabel.isNotBlank()) {
      Text(
        currentCellLabel,
        style = TextStyle(fontSize = 14.sp, fontWeight = FontWeight.Medium),
        color = colors.primaryText,
        maxLines = 2,
        overflow = TextOverflow.Ellipsis,
      )
    }
    // Cooling wins the second line when active (emphasized), else the raw progress text.
    val cooling = coolingSinceMillis != null
    val secondLine = if (cooling) coolingCaption(coolingSinceMillis, now) else progressText
    if (secondLine.isNotBlank()) {
      Text(
        secondLine,
        style = TextStyle(fontSize = 14.sp, fontWeight = if (cooling) FontWeight.SemiBold else FontWeight.Normal),
        color = if (cooling) colors.accent else colors.secondaryText,
        maxLines = 1,
        overflow = TextOverflow.Ellipsis,
      )
    }
  }
}
