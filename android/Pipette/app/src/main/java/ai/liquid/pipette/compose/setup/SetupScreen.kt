@file:Suppress("MagicNumber", "MaxLineLength")

package ai.liquid.pipette.compose.setup

import ai.liquid.pipette.AppSettingsStore
import ai.liquid.pipette.CollectorEndpointOption
import ai.liquid.pipette.compose.AppTextButton
import ai.liquid.pipette.compose.CapsulePrimaryButton
import ai.liquid.pipette.compose.IosTextField
import ai.liquid.pipette.compose.setup.component.CollectorPicker
import ai.liquid.pipette.compose.setup.component.IdentityFields
import ai.liquid.pipette.compose.setup.component.SetupHeader
import ai.liquid.pipette.compose.theme.PipetteTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

/**
 * Registration gate: logo + welcome + organization/email form (org mandatory), shown before the tabbed app.
 *
 * [SetupIntent.SignOut] is the only way out of this screen. It sits between a live Clerk session and the tabbed app, so once signed in there is
 * otherwise nothing to press but Register: no back, and Settings is on the far side of the registration it is asking for. Without an exit, the
 * account that signed in is the account the device is stuck with.
 */
@Composable
fun SetupScreen(state: SetupUiState, onIntent: (SetupIntent) -> Unit) {
  // Not keyed on the async-loaded settings/clerk values: prefill once when they arrive and only if
  // the user hasn't typed, so a cold-start DataStore emission can't clobber in-progress input.
  var email by rememberSaveable { mutableStateOf("") }
  var organization by rememberSaveable { mutableStateOf("") }
  // The ordinal, not the enum: it is what SegmentedControl speaks and what rememberSaveable stores without a custom saver.
  var collectorIndex by rememberSaveable { mutableIntStateOf(CollectorEndpointOption.PRODUCTION.ordinal) }
  var customCollectorUrl by rememberSaveable { mutableStateOf("") }
  // `remember`, NOT `rememberSaveable`: the key is a secret, and saved-instance-state can be persisted to disk on process death.
  var preauthKey by remember { mutableStateOf("") }
  LaunchedEffect(state.settings.contactEmail, state.clerkEmail) {
    if (email.isBlank()) email = state.settings.contactEmail.ifBlank { state.clerkEmail }
  }
  LaunchedEffect(state.settings.organization) { if (organization.isBlank()) organization = state.settings.organization }
  // A device already pointing at a non-production collector reopens on Custom with that URL, so re-registering doesn't
  // silently move it to production. Prefill-once, like the fields above: only while the user hasn't touched the picker.
  LaunchedEffect(state.settings.serverUrl) {
    val stored = state.settings.serverUrl.trim()
    val untouched = customCollectorUrl.isBlank() && collectorIndex == CollectorEndpointOption.PRODUCTION.ordinal
    if (untouched && stored.isNotEmpty() && stored != AppSettingsStore.DEFAULT_SERVER_URL) {
      collectorIndex = CollectorEndpointOption.CUSTOM.ordinal
      customCollectorUrl = stored
    }
  }

  // The header is fixed and only the form scrolls: focusing a field near the bottom would otherwise scroll the screen's
  // identity away and leave an unlabeled stack of inputs. `fill = false` keeps the form at its natural height, so the
  // whole thing stays centred until the content or the IME needs the space.
  Column(
    modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.statusBars).padding(horizontal = 32.dp, vertical = 32.dp),
    horizontalAlignment = Alignment.CenterHorizontally,
    verticalArrangement = Arrangement.Center,
  ) {
    SetupHeader(state.clerkEmail)

    Column(modifier = Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState()), horizontalAlignment = Alignment.CenterHorizontally) {
      IdentityFields(
        clerkEmail = state.clerkEmail,
        email = email,
        onEmailChange = { email = it },
        organization = organization,
        onOrganizationChange = { organization = it },
      )

      val collectorOption = CollectorEndpointOption.entries[collectorIndex]

      CollectorPicker(
        selectedIndex = collectorIndex,
        onSelect = { collectorIndex = it },
        customUrl = customCollectorUrl,
        onCustomUrlChange = { customCollectorUrl = it },
      )
      // Offered for either collector: the management API takes `preauth_key` on any register call, so a key issued
      // against the production collector has to be usable here too.
      IosTextField(
        value = preauthKey,
        onValueChange = { preauthKey = it },
        placeholder = "Pre-auth key (optional)",
        // Password type so the IME doesn't capitalize, autocorrect, or offer to save the token; the text stays visible.
        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
        modifier = Modifier.padding(top = 12.dp),
      )
      // When signed in via Clerk, the email field is hidden and the Clerk identity wins — otherwise a
      // stale persisted contactEmail (prefilled above) would be submitted instead of the account shown
      // as "Signed in as …".
      val effectiveEmail = state.clerkEmail.ifBlank { email }
      // Null while a Custom entry is empty or malformed, which is what disables Register: an unparsable URL would
      // only fail at the first request, after a keypair had already been generated.
      val serverUrl = collectorOption.serverUrl(customCollectorUrl)
      CapsulePrimaryButton(
        text = "Register",
        onClick = { serverUrl?.let { onIntent(SetupIntent.Register(it, organization.trim(), effectiveEmail, preauthKey.trim())) } },
        enabled = effectiveEmail.isNotBlank() && organization.isNotBlank() && serverUrl != null,
        loading = state.isRegistering,
        modifier = Modifier.padding(top = 26.dp),
      )
      // Only when there is a session to end. A build with no Clerk key reaches this screen signed out, and offering to sign out of nothing would be
      // a dead control.
      if (state.clerkEmail.isNotBlank()) {
        AppTextButton(text = "Sign out", onClick = { onIntent(SetupIntent.SignOut) }, enabled = !state.isRegistering)
      }
    }
  }
}

@Preview
@Composable
private fun SetupScreenPreview() {
  PipetteTheme { SetupScreen(state = SetupUiState(clerkEmail = "", isRegistering = false), onIntent = {}) }
}
