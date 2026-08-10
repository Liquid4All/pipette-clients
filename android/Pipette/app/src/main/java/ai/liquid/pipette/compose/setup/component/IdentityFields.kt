package ai.liquid.pipette.compose.setup.component

import ai.liquid.pipette.compose.IosTextField
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp

/**
 * Who is registering: email (only when Clerk hasn't already told us) then organization.
 *
 * Organization is asked for whichever collector is selected — it names the registration on the collector side, and production needs it as much as a
 * self-hosted one does.
 */
@Composable
internal fun IdentityFields(
  clerkEmail: String,
  email: String,
  onEmailChange: (String) -> Unit,
  organization: String,
  onOrganizationChange: (String) -> Unit,
) {
  if (clerkEmail.isBlank()) {
    IosTextField(
      value = email,
      onValueChange = onEmailChange,
      placeholder = "Email",
      keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
      modifier = Modifier.padding(top = 44.dp),
    )
  }

  IosTextField(
    value = organization,
    onValueChange = onOrganizationChange,
    placeholder = "Organization name",
    modifier = Modifier.padding(top = if (clerkEmail.isBlank()) 12.dp else 28.dp),
  )
}
