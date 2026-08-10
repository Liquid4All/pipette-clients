// iOS-styled Settings: layout literals (MagicNumber) + long copy strings (MaxLineLength).
@file:Suppress("MagicNumber", "MaxLineLength")

package ai.liquid.pipette.compose.settings

import ai.liquid.pipette.AccentKind
import ai.liquid.pipette.FeedbackDialog
import ai.liquid.pipette.RegistrationData
import ai.liquid.pipette.compose.AppFilterChip
import ai.liquid.pipette.compose.CapsuleOutlineButton
import ai.liquid.pipette.compose.ConfirmAction
import ai.liquid.pipette.compose.DestructiveCapsuleButton
import ai.liquid.pipette.compose.IosCard
import ai.liquid.pipette.compose.IosDivider
import ai.liquid.pipette.compose.IosTextField
import ai.liquid.pipette.compose.IosToggle
import ai.liquid.pipette.compose.PageHeaderLarge
import ai.liquid.pipette.compose.PillTabBarReservedHeight
import ai.liquid.pipette.compose.PropertyRow
import ai.liquid.pipette.compose.SectionTitle
import ai.liquid.pipette.compose.accentColor
import ai.liquid.pipette.compose.clickableNoRipple
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Settings tab (iOS SettingsView): Account, Device, About, Debugging sections in outlined cards. */
@Composable
fun SettingsScreen(state: SettingsUiState, onIntent: (SettingsIntent) -> Unit) {
  var showAcknowledgements by remember { mutableStateOf(false) }
  if (showAcknowledgements) {
    AcknowledgementsScreen(onBack = { showAcknowledgements = false })
    return
  }
  var showFeedback by remember { mutableStateOf(false) }
  if (showFeedback) {
    FeedbackDialogUi(
      defaultEmail = state.clerkEmail ?: state.registration?.contactEmail ?: "",
      onDismiss = { showFeedback = false },
      onSubmit = { message, email, categoryId -> onIntent(SettingsIntent.SubmitFeedback(message, email, categoryId)) },
    )
  }
  val colors = PipetteTheme.colors
  Column(
    modifier =
      Modifier.fillMaxSize()
        .verticalScroll(rememberScrollState())
        .windowInsetsPadding(WindowInsets.statusBars)
        .padding(horizontal = 24.dp)
        .padding(top = 12.dp, bottom = 36.dp + PillTabBarReservedHeight)
  ) {
    PageHeaderLarge("Settings", modifier = Modifier.padding(bottom = 28.dp))

    SectionTitle("Account")
    IosCard(modifier = Modifier.padding(top = 16.dp)) {
      PropertyRow("Email", state.clerkEmail ?: state.registration?.clerkPrimaryEmail ?: state.registration?.contactEmail ?: "—")
      IosDivider()
      PropertyRow("Organization", state.registration?.organization?.ifBlank { "—" } ?: "—")
      IosDivider()
      PropertyRow("Registered", state.registration?.registeredAt ?: "—")
    }
    if (state.isRegistered) {
      // Snapshotted on the tap rather than read live, so the warning is settled before the dialog appears and its text cannot change while the
      // user is reading a destructive prompt. iOS holds the same value in `pendingResultsAtSignOut` for the same reason.
      var pendingResultsAtSignOut by remember { mutableStateOf(0) }
      ConfirmAction(
        message = signOutConfirmMessage(pendingResultsAtSignOut),
        positiveText = "Sign Out",
        onConfirm = { onIntent(SettingsIntent.SignOut) },
      ) { trigger ->
        DestructiveCapsuleButton(
          "Sign out",
          onClick = {
            pendingResultsAtSignOut = state.unsubmittedResultCount
            trigger()
          },
          modifier = Modifier.padding(top = 17.dp),
          leadingIcon = ai.liquid.pipette.R.drawable.ic_signout,
        )
      }
    }
    if (state.isRegistered) {
      ToggleRow(
        "By default, auto-submit benchmark results to the public dataset when jobs finish. Only performance metrics are shared, never personal or device data.",
        checked = state.defaultContributeResults,
        enabled = true,
        onCheckedChange = { onIntent(SettingsIntent.SetDefaultContributeResults(it)) },
        modifier = Modifier.padding(top = 34.dp),
      )
    }
    AnalyticsToggle(state, onIntent)

    SectionTitle("Device", modifier = Modifier.padding(top = 46.dp))
    IosCard(modifier = Modifier.padding(top = 19.dp)) {
      PropertyRow(
        "Thermal state",
        state.thermalLabel,
        valueColor = accentColor(state.thermalAccent),
        labelWidth = 150,
        valueAlignment = Alignment.Start,
        rowHeight = 53,
      )
    }
    ResetButton(onIntent)

    SectionTitle("About", modifier = Modifier.padding(top = 46.dp))
    IosCard(modifier = Modifier.padding(top = 19.dp)) {
      Row(
        modifier = Modifier.fillMaxWidth().clickableNoRipple { showAcknowledgements = true }.padding(horizontal = 24.dp, vertical = 18.dp),
        verticalAlignment = Alignment.CenterVertically,
      ) {
        Text("Open source licenses", style = TextStyle(fontSize = 16.5.sp), color = colors.label, modifier = Modifier.weight(1f))
        androidx.compose.material3.Icon(
          painter = androidx.compose.ui.res.painterResource(ai.liquid.pipette.R.drawable.ic_chevron_right),
          contentDescription = null,
          tint = colors.gray3,
          modifier = Modifier.size(18.dp),
        )
      }
    }

    if (state.isFeedbackAvailable) {
      SectionTitle("Feedback", modifier = Modifier.padding(top = 46.dp))
      CapsuleOutlineButton(
        FeedbackDialog.BUTTON_LABEL,
        onClick = { showFeedback = true },
        modifier = Modifier.padding(top = 19.dp).fillMaxWidth(),
        height = 42,
      )
    }

    if (state.isDebug) {
      SectionTitle("Debugging", modifier = Modifier.padding(top = 46.dp))
      ToggleRow(
        "Bypass auth gate (debug only)",
        checked = state.clerkGateBypass,
        enabled = true,
        onCheckedChange = { onIntent(SettingsIntent.SetGateBypass(it)) },
        modifier = Modifier.padding(top = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
      )
      // Left-aligned (the row default), unlike the toggle above: this label wraps, and centering a
      // switch against two lines of text puts it beside the gap between them.
      ToggleRow(
        "Skip thermal readiness (debug only). Cells run without waiting for the device to cool, and every result records that the gate was waived.",
        checked = state.skipThermalGate,
        enabled = true,
        onCheckedChange = { onIntent(SettingsIntent.SetSkipThermalGate(it)) },
        modifier = Modifier.padding(top = 16.dp),
      )
      IosCard(modifier = Modifier.padding(top = 19.dp)) {
        Text(
          state.debugInfo,
          style = TextStyle(fontSize = 12.5.sp, fontFamily = FontFamily.Monospace),
          color = colors.gray,
          modifier = Modifier.padding(horizontal = 18.dp, vertical = 12.dp),
        )
      }
    }
  }
}

/**
 * Sign-out confirmation copy. The reset is described unconditionally; the count of results about to be lost is appended only when there are any, so a
 * device with nothing pending doesn't read a sentence about zero.
 */
@Composable
private fun signOutConfirmMessage(unsubmittedResultCount: Int): String {
  val base = stringResource(ai.liquid.pipette.R.string.settings_sign_out_confirm)
  if (unsubmittedResultCount <= 0) return base
  val warning = pluralStringResource(ai.liquid.pipette.R.plurals.settings_sign_out_unsubmitted, unsubmittedResultCount, unsubmittedResultCount)
  return "$base\n\n$warning"
}

@Composable
private fun ResetButton(onIntent: (SettingsIntent) -> Unit) {
  ConfirmAction(
    message = "This deletes local jobs, benchmark results, and downloaded models. Your device identity is kept.",
    positiveText = "Reset",
    onConfirm = { onIntent(SettingsIntent.ResetLocalData) },
  ) { trigger ->
    CapsuleOutlineButton(
      "Reset data on this device",
      onClick = trigger,
      modifier = Modifier.padding(top = 34.dp).fillMaxWidth(),
      height = 42,
      leadingIcon = ai.liquid.pipette.R.drawable.ic_retry,
    )
  }
}

/**
 * Analytics opt-out, drawn only when a real sink is wired: a toggle over `NoOpAnalytics` would be a control that does nothing.
 *
 * Phrased as opting IN so "on" is the permissive setting, like every other toggle on this screen; the stored flag is PostHog's opt-OUT, hence the
 * inversion on both sides. A composable of its own rather than an `if` inside `SettingsScreen`, which is already at detekt's complexity ceiling.
 *
 * Not gated on registration the way the auto-submit row is: analytics start at launch, before this device has registered, so the control that stops
 * them has to be reachable then too. That is also why the top padding varies: without the auto-submit row above it, this row follows a button.
 */
@Composable
private fun AnalyticsToggle(state: SettingsUiState, onIntent: (SettingsIntent) -> Unit) {
  if (!state.isAnalyticsAvailable) return
  ToggleRow(
    "Share anonymous usage analytics: which app features are used and whether benchmark runs succeed. Never your results, prompts, or account details.",
    checked = !state.analyticsOptedOut,
    enabled = true,
    onCheckedChange = { onIntent(SettingsIntent.SetAnalyticsOptOut(!it)) },
    modifier = Modifier.padding(top = if (state.isRegistered) 24.dp else 34.dp),
  )
}

@Composable
private fun ToggleRow(
  text: String,
  checked: Boolean,
  enabled: Boolean,
  onCheckedChange: (Boolean) -> Unit,
  modifier: Modifier = Modifier,
  verticalAlignment: Alignment.Vertical = Alignment.Top,
) {
  Row(modifier = modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(16.dp), verticalAlignment = verticalAlignment) {
    IosToggle(checked = checked, onCheckedChange = onCheckedChange, enabled = enabled)
    Text(text, style = TextStyle(fontSize = 15.5.sp, lineHeight = 21.sp), color = PipetteTheme.colors.label.copy(alpha = 0.75f))
  }
}

/**
 * Compose feedback dialog: optional category (single-select), required message, optional reply email. Submit is disabled until a message is entered.
 */
@Composable
private fun FeedbackDialogUi(defaultEmail: String, onDismiss: () -> Unit, onSubmit: (String, String, String?) -> Unit) {
  val colors = PipetteTheme.colors
  var message by remember { mutableStateOf("") }
  var email by remember { mutableStateOf(defaultEmail) }
  var categoryId by remember { mutableStateOf<String?>(null) }
  AlertDialog(
    onDismissRequest = onDismiss,
    title = { Text(FeedbackDialog.DIALOG_TITLE) },
    text = {
      Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(FeedbackDialog.DIALOG_DESCRIPTION, style = TextStyle(fontSize = 14.sp), color = colors.gray)
        Text("What's this about? (optional)", style = TextStyle(fontSize = 13.sp), color = colors.gray)
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
          FeedbackDialog.options.forEach { (id, label) ->
            AppFilterChip(label, selected = categoryId == id, onToggle = { on -> categoryId = if (on) id else null })
          }
        }
        IosTextField(
          value = message,
          onValueChange = { message = it },
          placeholder = "Tell us more *",
          singleLine = false,
          minLines = 4,
          maxLines = 6,
        )
        IosTextField(
          value = email,
          onValueChange = { email = it },
          placeholder = "you@example.com (optional)",
          keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
        )
      }
    },
    confirmButton = {
      TextButton(
        enabled = message.isNotBlank(),
        onClick = {
          onSubmit(message.trim(), email.trim(), categoryId)
          onDismiss()
        },
      ) {
        Text("Submit")
      }
    },
    dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
  )
}

@Preview
@Composable
private fun SettingsScreenPreview() {
  PipetteTheme {
    SettingsScreen(
      state =
        SettingsUiState(
          registration =
            RegistrationData(
              clientId = "pipette-preview-device",
              status = "approved",
              serverUrl = "https://pipette.liquid.ai",
              organization = "Liquid AI",
              contactEmail = "preview@liquid.ai",
              registeredAt = "2026-08-05T09:00:00Z",
            ),
          isRegistered = true,
          clerkEmail = "preview@liquid.ai",
          isClerkAvailable = true,
          thermalLabel = "Nominal",
          thermalDescription = "Device is cool; benchmarks can run.",
          thermalAccent = AccentKind.NOMINAL,
        ),
      onIntent = {},
    )
  }
}
