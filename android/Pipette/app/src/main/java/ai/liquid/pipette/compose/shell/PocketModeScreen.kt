// Pocket-mode overlay: hard-coded dp/sp/alpha layout literals, mirroring the iOS PocketModeView (always dark).
@file:Suppress("MagicNumber", "MaxLineLength")

package ai.liquid.pipette.compose.shell

import ai.liquid.pipette.AccentKind
import ai.liquid.pipette.R
import ai.liquid.pipette.compose.JobActivityColors
import ai.liquid.pipette.compose.JobLiveActivity
import ai.liquid.pipette.compose.PocketUi
import ai.liquid.pipette.compose.accentColor
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.Orientation
import androidx.compose.foundation.gestures.draggable
import androidx.compose.foundation.gestures.rememberDraggableState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

private val PocketBg = Color(0xFF0A0A0A)
private val PocketCard = Color(0xFF171717)
private val PocketGray = Color(0xFFA3A3A3)
private val PocketTrack = Color(0xFF404040)
private val PocketSerif = FontFamily.Serif
// Cool wash for the card while the readiness gate cools the device (iOS PocketPalette.cool*): the
// card color pre-blended with a cool tint (opaque, so it reads on the dark screen), a light cool
// border, and a light cool text for the cooldown caption.
private val PocketCoolWash = Color(0xFF1C2735)
private val PocketCoolBorder = Color(0x8068A2E6)
private val PocketCoolText = Color(0xFF89BAF7)

/** Full-screen pocket mode (iOS PocketModeView), always dark. Exiting does NOT cancel the job — it keeps running in the background. */
@Composable
fun PocketModeScreen(pocket: PocketUi, onExit: () -> Unit) {
  Column(
    modifier = Modifier.fillMaxSize().background(PocketBg).windowInsetsPadding(WindowInsets.statusBars).padding(horizontal = 24.dp, vertical = 24.dp),
    horizontalAlignment = Alignment.CenterHorizontally,
  ) {
    Spacer(Modifier.height(128.dp))
    // The white P-mark vector (fill baked into pipette_logo.xml), not the launcher icon (that is
    // an adaptive icon built from ic_launcher_foreground.xml).
    Image(
      painter = painterResource(R.drawable.pipette_logo),
      contentDescription = "Pipette",
      modifier = Modifier.size(42.dp).clip(RoundedCornerShape(12.dp)),
    )
    Text(
      "Benchmarking in progress...",
      style = TextStyle(fontSize = 24.sp, fontFamily = PocketSerif),
      color = Color.White,
      textAlign = TextAlign.Center,
      modifier = Modifier.padding(top = 20.dp, bottom = 28.dp),
    )

    Row(modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp), verticalAlignment = Alignment.CenterVertically) {
      Text("Throttling headroom", style = TextStyle(fontSize = 16.sp), color = PocketGray, modifier = Modifier.weight(1f))
      val accent = accentColor(pocket.thermalAccent)
      Row(
        modifier =
          Modifier.clip(RoundedCornerShape(percent = 50))
            .background(PocketBg)
            .border(androidx.compose.foundation.BorderStroke(1.dp, Color.White.copy(alpha = 0.10f)), RoundedCornerShape(percent = 50))
            .padding(horizontal = 12.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
      ) {
        Box(Modifier.size(8.dp).clip(CircleShape).background(accent))
        Text(pocket.thermalLabel, style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.Medium), color = Color.White)
      }
    }

    val cooling = pocket.coolingSinceMillis != null
    Column(
      modifier =
        Modifier.fillMaxWidth()
          .clip(RoundedCornerShape(20.dp))
          .background(if (cooling) PocketCoolWash else PocketCard)
          .then(
            if (cooling) Modifier.border(androidx.compose.foundation.BorderStroke(1.dp, PocketCoolBorder), RoundedCornerShape(20.dp)) else Modifier
          )
          .padding(horizontal = 32.dp, vertical = 28.dp),
      horizontalAlignment = Alignment.CenterHorizontally,
    ) {
      Text(pocket.title, style = TextStyle(fontSize = 24.sp, fontFamily = PocketSerif), color = Color.White)
      Text(pocket.subtitle, style = TextStyle(fontSize = 16.sp), color = PocketGray, modifier = Modifier.padding(top = 6.dp, bottom = 28.dp))
      Box(modifier = Modifier.fillMaxWidth().height(8.dp).clip(RoundedCornerShape(percent = 50)).background(PocketTrack)) {
        Box(
          modifier =
            Modifier.fillMaxWidth(pocket.progress.coerceIn(0.0, 1.0).toFloat())
              .height(8.dp)
              .clip(RoundedCornerShape(percent = 50))
              .background(Color.White)
        )
      }
      Row(modifier = Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(pocket.cellsDone, style = TextStyle(fontSize = 16.sp), color = PocketGray)
        Text(pocket.timeLeft, style = TextStyle(fontSize = 16.sp), color = PocketGray)
      }
      if (pocket.currentCellLabel.isNotBlank() || pocket.progressText.isNotBlank() || cooling) {
        JobLiveActivity(
          currentCellLabel = pocket.currentCellLabel,
          progressText = pocket.progressText,
          coolingSinceMillis = pocket.coolingSinceMillis,
          colors = JobActivityColors(primaryText = Color.White, secondaryText = PocketGray, accent = PocketCoolText),
          modifier = Modifier.padding(top = 8.dp),
        )
      }
    }

    Spacer(Modifier.weight(1f))
    SlideToExit(onExit)
    Text(pocket.estTimeLine, style = TextStyle(fontSize = 16.sp), color = PocketGray, modifier = Modifier.padding(top = 14.dp))
  }
}

/** Slide-to-exit control (iOS): drag the thumb past ~72% of the track to leave pocket mode. */
@Composable
private fun SlideToExit(onExit: () -> Unit) {
  val trackHeight = 58.dp
  val thumbWidth = 64.dp
  val density = LocalDensity.current
  var trackWidthPx by remember { mutableFloatStateOf(0f) }
  var offsetPx by remember { mutableFloatStateOf(0f) }
  val thumbWidthPx = with(density) { thumbWidth.toPx() }
  Box(
    modifier =
      Modifier.fillMaxWidth().padding(vertical = 8.dp).height(trackHeight).clip(RoundedCornerShape(14.dp)).background(PocketCard).onSizeChanged {
        trackWidthPx = it.width.toFloat()
      },
    contentAlignment = Alignment.Center,
  ) {
    Text("Slide to exit pocket mode", style = TextStyle(fontSize = 14.sp), color = PocketGray)
    Box(
      modifier =
        Modifier.offset { IntOffset(offsetPx.roundToInt(), 0) }
          .width(thumbWidth)
          .height(trackHeight)
          .align(Alignment.CenterStart)
          .clip(RoundedCornerShape(12.dp))
          .background(Color.White)
          .draggable(
            orientation = Orientation.Horizontal,
            state =
              rememberDraggableState { delta ->
                val max = (trackWidthPx - thumbWidthPx).coerceAtLeast(0f)
                offsetPx = (offsetPx + delta).coerceIn(0f, max)
              },
            onDragStopped = {
              val max = (trackWidthPx - thumbWidthPx).coerceAtLeast(1f)
              if (offsetPx > max * 0.72f) onExit() else offsetPx = 0f
            },
          ),
      contentAlignment = Alignment.Center,
    ) {
      androidx.compose.material3.Icon(
        painter = androidx.compose.ui.res.painterResource(ai.liquid.pipette.R.drawable.ic_chevron_right),
        contentDescription = null,
        tint = Color.Black,
        modifier = Modifier.size(22.dp),
      )
    }
  }
}

@Preview
@Composable
fun PocketModeScreenPreview() {
  PocketModeScreen(
    pocket =
      PocketUi(
        jobId = "preview-job",
        title = "Camera Thermal",
        subtitle = "Phase 1 of 3",
        progress = 0.45,
        cellsDone = "45 / 100",
        timeLeft = "2m 30s",
        thermalLabel = "Green",
        thermalAccent = AccentKind.NOMINAL,
        estTimeLine = "Est. 5m 45s total",
        currentCellLabel = "Cell 45: Measuring brightness response",
        progressText = "1.2x - 1.5x",
        coolingSinceMillis = null,
      ),
    onExit = {},
  )
}
