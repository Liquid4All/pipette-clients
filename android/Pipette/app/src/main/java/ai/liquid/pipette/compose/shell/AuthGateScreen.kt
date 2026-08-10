// Auth gate: several small composables (TooManyFunctions) + iOS layout dp/sp literals (MagicNumber).
@file:Suppress("MagicNumber", "TooManyFunctions")

package ai.liquid.pipette.compose.shell

import ai.liquid.pipette.AuthGate
import ai.liquid.pipette.OAuthProviderInfo
import ai.liquid.pipette.R
import ai.liquid.pipette.SET_PASSWORD_ACTION
import ai.liquid.pipette.SecondFactor
import ai.liquid.pipette.SecondFactorReason
import ai.liquid.pipette.compose.AppCard
import ai.liquid.pipette.compose.AppLabel
import ai.liquid.pipette.compose.AppTextButton
import ai.liquid.pipette.compose.ConfirmAction
import ai.liquid.pipette.compose.DisplayTitle
import ai.liquid.pipette.compose.IosTextField
import ai.liquid.pipette.compose.MutedLabel
import ai.liquid.pipette.compose.OtpCodeField
import ai.liquid.pipette.compose.OutlineButton
import ai.liquid.pipette.compose.PasswordVisibilityToggle
import ai.liquid.pipette.compose.PrimaryButton
import ai.liquid.pipette.compose.SegmentedControl
import ai.liquid.pipette.compose.theme.PipetteTheme
import ai.liquid.pipette.compose.theme.serif
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withLink
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Renders any non-Ready [AuthGate] in place of the app chrome (mirrors MainActivity.renderGate). */
@Composable
fun AuthGateScreen(
  gate: AuthGate,
  emailAuth: EmailAuthUiState,
  oauthProviders: List<OAuthProviderInfo>,
  isDebug: Boolean,
  onSubmitEmail: (String) -> Unit,
  onSubmitCode: (String) -> Unit,
  onOAuthProvider: (String) -> Unit,
  onUsePassword: (String) -> Unit,
  onSubmitPassword: (String) -> Unit,
  onSubmitNewPassword: (String) -> Unit,
  onStartPasswordReset: () -> Unit,
  onSubmitResetCode: (String) -> Unit,
  onSubmitResetPassword: (String) -> Unit,
  onChooseSecondFactor: (SecondFactor) -> Unit,
  onSubmitSecondFactor: (String) -> Unit,
  onChangeEmail: () -> Unit,
  onEditClearError: () -> Unit,
  onSkipDebug: () -> Unit,
  onSignOut: () -> Unit,
  onDeleteIdentity: () -> Unit,
) {
  when (gate) {
    is AuthGate.Loading ->
      // First-land loading splash: centered PIPETTE wordmark.
      Column(
        modifier = Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
      ) {
        Image(
          painter = painterResource(R.drawable.pipette_logo),
          contentDescription = "Pipette",
          colorFilter = ColorFilter.tint(PipetteTheme.colors.label),
          modifier = Modifier.size(180.dp),
        )
      }
    is AuthGate.InitError -> GateMessage("Sign-in unavailable", gate.message, showSpinner = false)
    is AuthGate.SignedOut ->
      Column(modifier = Modifier.fillMaxSize()) {
        // Custom sign-in / sign-up steps (replaces Clerk's prebuilt AuthView): email code, or a password for an account that has one.
        Column(modifier = Modifier.fillMaxSize().weight(1f)) {
          SignInStepsScreen(
            state = emailAuth,
            oauthProviders = oauthProviders,
            onSubmitEmail = onSubmitEmail,
            onSubmitCode = onSubmitCode,
            onOAuthProvider = onOAuthProvider,
            onUsePassword = onUsePassword,
            onSubmitPassword = onSubmitPassword,
            onSubmitNewPassword = onSubmitNewPassword,
            onStartPasswordReset = onStartPasswordReset,
            onSubmitResetCode = onSubmitResetCode,
            onSubmitResetPassword = onSubmitResetPassword,
            onChooseSecondFactor = onChooseSecondFactor,
            onSubmitSecondFactor = onSubmitSecondFactor,
            onChangeEmail = onChangeEmail,
            onEditClearError = onEditClearError,
          )
        }
        if (isDebug) {
          Column(modifier = Modifier.padding(horizontal = 18.dp, vertical = 12.dp)) {
            MutedLabel("Debug build: auth enforced. Skip it for local testing (toggle in Settings → Account).")
            OutlineButton("Skip sign-in (debug only)", onSkipDebug)
          }
        }
      }
    is AuthGate.Mismatch ->
      // Centered like the other gate states, which also keeps the title clear of the status bar.
      Column(modifier = Modifier.fillMaxSize().padding(18.dp), verticalArrangement = Arrangement.Center) {
        DisplayTitle("Account mismatch")
        AppCard {
          val linked = gate.linkedEmail ?: "another account"
          val current = gate.currentEmail ?: "a different account"
          MutedLabel("This device is linked to $linked, but you're signed in as $current.")
          PrimaryButton("Sign out", onSignOut)
          // A sign-out that could not reach Clerk leaves this screen exactly as it was, since the session and the link both survive it (see
          // [ShellViewModel.signOut]). Without this the button would read as doing nothing at all, on the one screen whose whole purpose is
          // getting out of this state. Shares [EmailAuthUiState.error] with the sign-in steps because it is the same field the failure is
          // reported into, and no sign-in step is on screen to compete for it.
          ErrorText(emailAuth.error)
          ConfirmAction(
            message = "Delete this device's identity? This clears its registration and private key.",
            positiveText = "Delete",
            onConfirm = onDeleteIdentity,
          ) { trigger ->
            OutlineButton("Delete device identity", trigger)
          }
        }
      }
    is AuthGate.Ready -> Unit
  }
}

/**
 * The steps behind the signed-out gate. The default route is the emailed 6-digit code, which both signs an existing account in and registers a new
 * one when the address is unknown (two outcomes of the one [ClerkAuth.sendEmailCode] call, which is why the email step's CTA reads "Register"). The
 * opt-in password step signs in only. The text fields are local (Compose) state; each step delegates the actual SDK call to [ShellViewModel] via the
 * callbacks. [EmailAuthUiState.submitting] drives the in-button spinner and blocks re-submits.
 */
@Composable
private fun SignInStepsScreen(
  state: EmailAuthUiState,
  oauthProviders: List<OAuthProviderInfo>,
  onSubmitEmail: (String) -> Unit,
  onSubmitCode: (String) -> Unit,
  onOAuthProvider: (String) -> Unit,
  onUsePassword: (String) -> Unit,
  onSubmitPassword: (String) -> Unit,
  onSubmitNewPassword: (String) -> Unit,
  onStartPasswordReset: () -> Unit,
  onSubmitResetCode: (String) -> Unit,
  onSubmitResetPassword: (String) -> Unit,
  onChooseSecondFactor: (SecondFactor) -> Unit,
  onSubmitSecondFactor: (String) -> Unit,
  onChangeEmail: () -> Unit,
  onEditClearError: () -> Unit,
) {
  // Back from any later step returns to the email step rather than leaving the app, so a wrong address or a
  // mistaken "use a password" is recoverable with the gesture that already means "go back". Only registered
  // off the first step, which leaves the system default (exit) intact there.
  //
  // Mid-request it stays registered and does nothing, which is not the same as leaving it disabled: a disabled
  // BackHandler falls through to the system default, so back during a slow sign-in would close the app and
  // lose the attempt. The back *links* can be disabled because a dead button swallows the tap; back can't.
  BackHandler(enabled = state.step != EmailAuthUiState.Step.Email) { if (!state.submitting) onChangeEmail() }
  // Scrollable so the form stays reachable when the keyboard (root imePadding) shrinks the viewport. A
  // scrollable column is measured with unbounded height and wraps its content, which would make
  // Arrangement.Center a no-op — the min-height pins the content to at least the viewport so short forms
  // still center while tall ones scroll.
  BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
    Column(
      modifier =
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).heightIn(min = this@BoxWithConstraints.maxHeight).padding(horizontal = 24.dp),
      verticalArrangement = Arrangement.Center,
      horizontalAlignment = Alignment.CenterHorizontally,
    ) {
      when (state.step) {
        EmailAuthUiState.Step.Email ->
          EmailStep(
            state = state,
            oauthProviders = oauthProviders,
            onSubmit = onSubmitEmail,
            onOAuthProvider = onOAuthProvider,
            onUsePassword = onUsePassword,
            onEditClearError = onEditClearError,
          )
        EmailAuthUiState.Step.Code ->
          CodeStep(state = state, onSubmit = onSubmitCode, onChangeEmail = onChangeEmail, onEditClearError = onEditClearError)
        EmailAuthUiState.Step.Password ->
          PasswordStep(
            state = state,
            onSubmit = onSubmitPassword,
            onStartReset = onStartPasswordReset,
            onChangeEmail = onChangeEmail,
            onEditClearError = onEditClearError,
          )
        EmailAuthUiState.Step.CreatePassword ->
          SetPasswordStep(
            state = state,
            title = "Create a password",
            prompt = "Your email is verified. Choose a password to finish creating the account${forEmail(state.email)}.",
            cta = "Create account",
            onSubmit = onSubmitNewPassword,
            onChangeEmail = onChangeEmail,
            onEditClearError = onEditClearError,
          )
        EmailAuthUiState.Step.ResetCode ->
          CodeStep(
            state = state,
            then = "Enter it to set a password.",
            onSubmit = onSubmitResetCode,
            onChangeEmail = onChangeEmail,
            onEditClearError = onEditClearError,
            onResend = onStartPasswordReset,
          )
        EmailAuthUiState.Step.ResetPassword ->
          SetPasswordStep(
            state = state,
            title = SET_PASSWORD_ACTION,
            prompt = resetPasswordPrompt(state.email, state.resetWasForced),
            cta = "Save and sign in",
            onSubmit = onSubmitResetPassword,
            onChangeEmail = onChangeEmail,
            onEditClearError = onEditClearError,
          )
        EmailAuthUiState.Step.SecondFactor ->
          SecondFactorStep(
            state = state,
            onChooseFactor = onChooseSecondFactor,
            onSubmit = onSubmitSecondFactor,
            onChangeEmail = onChangeEmail,
            onEditClearError = onEditClearError,
          )
      }
    }
  }
}

@Composable
private fun EmailStep(
  state: EmailAuthUiState,
  oauthProviders: List<OAuthProviderInfo>,
  onSubmit: (String) -> Unit,
  onOAuthProvider: (String) -> Unit,
  onUsePassword: (String) -> Unit,
  onEditClearError: () -> Unit,
) {
  val colors = PipetteTheme.colors
  var email by rememberSaveable { mutableStateOf(state.email) }
  // Welcome header (logo + title + subtitle), mirroring the Setup screen — no card. The logo is a monochrome
  // vector tinted to the label color, so it's black on light and white on dark.
  Image(
    painter = painterResource(R.drawable.pipette_logo),
    contentDescription = "Pipette",
    colorFilter = ColorFilter.tint(colors.label),
    modifier = Modifier.size(52.dp),
  )
  Text("Welcome to Pipette", style = serif(26), color = colors.label, textAlign = TextAlign.Center, modifier = Modifier.padding(top = 14.dp))
  Text(
    "Measure model performance on your device",
    style = TextStyle(fontSize = 16.sp),
    color = colors.gray,
    textAlign = TextAlign.Center,
    modifier = Modifier.padding(top = 6.dp),
  )
  Spacer(Modifier.height(32.dp))
  IosTextField(
    value = email,
    onValueChange = {
      email = it
      onEditClearError()
    },
    placeholder = "Email",
    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
  )
  ErrorText(state.error)
  Spacer(Modifier.height(8.dp))
  PrimaryButton(text = "Register", onClick = { onSubmit(email) }, enabled = email.isNotBlank() && !state.submitting, loading = state.submitting)
  LegalNotice()
  // Opt-in credential path for an account that already has a password. Needs the address first, so it stays disabled until the field is filled, then
  // carries that value to the password step rather than asking for it twice.
  AppTextButton("Sign in with a password", { onUsePassword(email) }, enabled = email.isNotBlank() && !state.submitting)
  // Social sign-in: one button per provider enabled in the Clerk config. Absent entirely until the backend enables any (then the divider + buttons
  // appear without a client change). Disabled while any auth request (email or a prior OAuth tap) is in flight.
  if (oauthProviders.isNotEmpty()) {
    OrDivider()
    oauthProviders.forEach { provider ->
      OutlineButton(text = "Continue with ${provider.name}", onClick = { onOAuthProvider(provider.strategy) }, enabled = !state.submitting)
    }
  }
}

private const val TERMS_URL = "https://pipette.liquid.ai/terms"
private const val PRIVACY_URL = "https://www.liquid.ai/privacy-policy"

/**
 * Clickwrap notice under the email step's CTA. The Clerk instance has legal consent enabled, so a sign-up has to send `legal_accepted` or it never
 * completes, and `RealClerkAuth.signUpEmailCode` sends it on the strength of *this* text. That makes the pair load-bearing rather than decorative:
 * the flag is only honest while a user cannot register without having been shown this.
 *
 * The links open in the browser through Compose's own URL handling. It sits on the email step alone, since that's the only step that can register.
 */
@Composable
private fun LegalNotice() {
  val colors = PipetteTheme.colors
  val linkStyles = TextLinkStyles(style = SpanStyle(color = colors.label, textDecoration = TextDecoration.Underline))
  val notice = buildAnnotatedString {
    append("By continuing you agree to the ")
    withLink(LinkAnnotation.Url(TERMS_URL, linkStyles)) { append("Terms") }
    append(" and ")
    withLink(LinkAnnotation.Url(PRIVACY_URL, linkStyles)) { append("Privacy Policy") }
    append(".")
  }
  Text(
    notice,
    style = TextStyle(fontSize = 13.sp),
    color = colors.gray,
    textAlign = TextAlign.Center,
    modifier = Modifier.padding(top = 10.dp, start = 8.dp, end = 8.dp),
  )
}

/** A centered "or" flanked by hairlines, separating the email form from the social buttons. */
@Composable
private fun OrDivider() {
  Row(modifier = Modifier.fillMaxWidth().padding(vertical = 16.dp), verticalAlignment = Alignment.CenterVertically) {
    HorizontalDivider(modifier = Modifier.weight(1f), color = PipetteTheme.colors.gray5)
    Text("or", style = TextStyle(fontSize = 13.sp), color = PipetteTheme.colors.gray, modifier = Modifier.padding(horizontal = 12.dp))
    HorizontalDivider(modifier = Modifier.weight(1f), color = PipetteTheme.colors.gray5)
  }
}

private const val CODE_LENGTH = 6

/**
 * The emailed 6-digit code, for either code this flow can be waiting on: the sign-in first factor, or the one that authorizes a password reset. The
 * field and its auto-submit are identical, and so is the sentence saying a code was sent, so [then] carries the only part that differs: what
 * answering it leads to. A suffix rather than the whole prompt, so the shared sentence stays in one place.
 *
 * [onResend] adds a "Resend code" link when the step's code can be asked for again. Omitted on the sign-in route, whose send also registers an
 * unknown address, so re-running it is not the same request twice.
 */
@Composable
private fun CodeStep(
  state: EmailAuthUiState,
  onSubmit: (String) -> Unit,
  onChangeEmail: () -> Unit,
  onEditClearError: () -> Unit,
  then: String? = null,
  onResend: (() -> Unit)? = null,
) {
  // Keyed on the email so requesting a code for a different address starts with an empty field,
  // rather than restoring (and auto-submitting) the previous attempt's code. The two call sites are
  // separate composition groups, so neither one's code can be restored under the other.
  var code by rememberSaveable(state.email) { mutableStateOf("") }
  DisplayTitle("Check your email")
  MutedLabel(listOfNotNull("We sent a $CODE_LENGTH-digit code to ${state.email}.", then).joinToString(" "))
  Spacer(Modifier.height(16.dp))
  OtpCodeField(
    value = code,
    onValueChange = {
      code = it
      onEditClearError()
    },
    length = CODE_LENGTH,
    enabled = !state.submitting,
  )
  ErrorText(state.error)
  Spacer(Modifier.height(8.dp))
  PrimaryButton(text = "Verify", onClick = { onSubmit(code) }, enabled = code.length == CODE_LENGTH && !state.submitting, loading = state.submitting)
  // Same affordance the second-factor step has, and for the same reason: without it, a code that never arrived or arrived too late is only
  // recoverable by rewinding to the email step and walking back in.
  if (onResend != null) {
    AppTextButton(
      "Resend code",
      {
        // The digits already typed answer the code being replaced, so they go with it. Leaving them would arm the auto-submit below: a resend clears
        // the error, and a full field with no error is the one state that submits itself when this step is re-created across a config change.
        code = ""
        onResend()
      },
      enabled = !state.submitting,
    )
  }
  AppTextButton("Use a different email", onChangeEmail, enabled = !state.submitting)
  // Auto-submit once all digits are entered (standard OTP UX); keyed on the value so a failed attempt
  // (error shown, code unchanged) doesn't re-fire, but a re-edit back to full length does. The
  // `error == null` guard stops a full code that's restored across a config change (with its error still
  // set) from auto-resubmitting itself on recomposition.
  LaunchedEffect(code) { if (code.length == CODE_LENGTH && !state.submitting && state.error == null) onSubmit(code) }
}

/**
 * The password field the three password steps share.
 *
 * Visibility lives here and is deliberately NOT remembered across composition: a field that came back revealed after a config change would expose the
 * password to whoever is looking at the screen. The keyboard type stays Password while revealed, so the IME keeps suggestions and autocorrect off, on
 * the same reasoning: a password shown on screen still must not be learned by the dictionary.
 *
 * The value itself stays with the caller, and none of the three lift it into [EmailAuthUiState] (see [PasswordStep]).
 */
@Composable
private fun PasswordField(value: String, onValueChange: (String) -> Unit, enabled: Boolean) {
  var visible by remember { mutableStateOf(false) }
  IosTextField(
    value = value,
    onValueChange = onValueChange,
    placeholder = "Password",
    visualTransformation = if (visible) VisualTransformation.None else PasswordVisualTransformation(),
    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
    trailing = { PasswordVisibilityToggle(visible = visible, onToggle = { visible = !visible }) },
    enabled = enabled,
  )
}

/**
 * Credential step for an existing account: the address carried over from [EmailStep] plus its password. Unlike [CodeStep] there's no auto-submit (a
 * password has no known length) and the value is never lifted into [EmailAuthUiState]: it lives in this composable and goes straight to the SDK.
 *
 * The reset link is not only for a forgotten password, which is why it does not say "Forgot": an account can have no password at all (a social
 * sign-in, a dashboard invite, or a sign-up from before the instance required one), and this step is the only way in to giving it one. Its label is
 * [SET_PASSWORD_ACTION], shared with the sentence Clerk's `strategy_for_user_invalid` failure is rewritten into, so the error names something visible
 * on screen.
 */
@Composable
private fun PasswordStep(
  state: EmailAuthUiState,
  onSubmit: (String) -> Unit,
  onStartReset: () -> Unit,
  onChangeEmail: () -> Unit,
  onEditClearError: () -> Unit,
) {
  // Deliberately NOT rememberSaveable: a saved-instance-state bundle is persisted to disk by the
  // system, so keeping the password there would outlive the sign-in. A plain remember is dropped
  // when this step leaves composition (and on a config change), which is the whole point.
  var password by remember { mutableStateOf("") }
  DisplayTitle("Enter your password")
  MutedLabel("Signing in as ${state.email}.")
  Spacer(Modifier.height(16.dp))
  PasswordField(
    value = password,
    onValueChange = {
      password = it
      onEditClearError()
    },
    enabled = !state.submitting,
  )
  ErrorText(state.error)
  Spacer(Modifier.height(8.dp))
  PrimaryButton(text = "Sign in", onClick = { onSubmit(password) }, enabled = password.isNotEmpty() && !state.submitting, loading = state.submitting)
  // Shared with the auth layer, which quotes this label in the message it answers a passwordless account with (SET_PASSWORD_ACTION).
  AppTextButton(SET_PASSWORD_ACTION, onStartReset, enabled = !state.submitting)
  AppTextButton("Use a different email", onChangeEmail, enabled = !state.submitting)
}

/**
 * The step for *choosing* a password, shared by the two flows that end in one: finishing a registration, and resetting an existing account's
 * password. Both ask for one value under the same rules, so what differs is only what the user is told, and that arrives as [title], [prompt], and
 * [cta].
 *
 * Unlike [PasswordStep] the user is picking a value rather than recalling one, which is why neither call site says "Enter". The length hint mirrors
 * the instance's `password_settings.min_length` so the button doesn't invite a round-trip that can only fail; everything else Clerk enforces (breach
 * corpus, zxcvbn strength) comes back as a message from Clerk, which is the only party that actually knows.
 *
 * The only exit is back to the email step in both cases: a half-finished sign-up cannot be resumed, and a reset that has consumed its code cannot
 * either.
 */
@Composable
private fun SetPasswordStep(
  state: EmailAuthUiState,
  title: String,
  prompt: String,
  cta: String,
  onSubmit: (String) -> Unit,
  onChangeEmail: () -> Unit,
  onEditClearError: () -> Unit,
) {
  // Same reasoning as PasswordStep: never rememberSaveable, since that bundle is persisted to disk.
  var password by remember { mutableStateOf("") }
  DisplayTitle(title)
  MutedLabel(prompt)
  Spacer(Modifier.height(16.dp))
  PasswordField(
    value = password,
    onValueChange = {
      password = it
      onEditClearError()
    },
    enabled = !state.submitting,
  )
  // Shown up front rather than as a post-submit error, since the rule is knowable before the user commits to anything.
  MutedLabel("At least $MIN_PASSWORD_LENGTH characters.")
  ErrorText(state.error)
  Spacer(Modifier.height(8.dp))
  PrimaryButton(
    text = cta,
    onClick = { onSubmit(password) },
    enabled = password.length >= MIN_PASSWORD_LENGTH && !state.submitting,
    loading = state.submitting,
  )
  AppTextButton("Use a different email", onChangeEmail, enabled = !state.submitting)
}

/**
 * Mirrors `password_settings.min_length` on the Clerk instance (12). The SDK keeps its parsed environment internal, so this cannot be read at
 * runtime; it is a hint to keep the button honest, and Clerk stays the authority. If the instance policy changes, the worst case is a rejected submit
 * with Clerk's own message, not a user locked out.
 */
private const val MIN_PASSWORD_LENGTH = 12

/**
 * Code challenge on a sign-in that has already cleared its first factor: the account's own two-step verification, or Clerk's client-trust check on a
 * device it has not seen. [EmailAuthUiState.secondFactorReason] is which, and it owns the copy that names it.
 *
 * Reached only past the credential either way, so the copy says "verification code" and not "sign in": telling the user otherwise reads like their
 * password was rejected.
 */
@Composable
private fun SecondFactorStep(
  state: EmailAuthUiState,
  onChooseFactor: (SecondFactor) -> Unit,
  onSubmit: (String) -> Unit,
  onChangeEmail: () -> Unit,
  onEditClearError: () -> Unit,
) {
  val factor = state.selectedSecondFactor
  // Keyed on the factor and the reason together, so either moving starts an empty field. A code typed for one factor is not a valid answer to
  // another, and carrying it over invites submitting it against the wrong challenge; a chained challenge can arrive on the same factor, and the code
  // just consumed must not sit prefilled under a step that has asked for a fresh one. Saveable, unlike the password field: a one-time code is not a
  // lasting secret, and losing it to a rotation means waiting on another email.
  var code by rememberSaveable(factor, state.secondFactorReason) { mutableStateOf("") }
  DisplayTitle(
    when (state.secondFactorReason) {
      SecondFactorReason.Mfa -> "Two-step verification"
      // Not "two-step verification": Client Trust fires on accounts that have never switched that on, and telling someone to enter a code from a
      // feature they do not use is how a solvable step reads as a dead end.
      SecondFactorReason.DeviceVerification -> "Confirm this device"
    }
  )
  MutedLabel(secondFactorPrompt(factor, state.email, state.secondFactorReason))
  Spacer(Modifier.height(16.dp))
  // Only worth a chooser when the account actually offers a choice; a single factor is pre-selected upstream.
  if (state.secondFactorOptions.size > 1) {
    val options = state.secondFactorOptions
    SegmentedControl(
      options = options.map(::secondFactorLabel),
      // -1 while nothing is chosen yet, which the control draws as no segment selected. Coercing to 0 would claim the user picked the first one.
      selectedIndex = options.indexOf(factor),
      onSelect = { onChooseFactor(options[it]) },
    )
    Spacer(Modifier.height(12.dp))
  }
  IosTextField(
    value = code,
    onValueChange = {
      code = it
      onEditClearError()
    },
    placeholder = if (factor == SecondFactor.BackupCode) "Backup code" else "Verification code",
    // Backup codes are alphanumeric; every other factor is a numeric OTP.
    keyboardOptions = KeyboardOptions(keyboardType = if (factor == SecondFactor.BackupCode) KeyboardType.Text else KeyboardType.NumberPassword),
    // Frozen while a verify or a send is in flight, matching the code step's OTP field.
    enabled = !state.submitting,
  )
  ErrorText(state.error)
  Spacer(Modifier.height(8.dp))
  PrimaryButton(
    text = "Verify",
    onClick = { onSubmit(code) },
    enabled = code.isNotEmpty() && factor != null && !state.submitting,
    loading = state.submitting,
  )
  // Re-requesting only makes sense for a factor Clerk delivers; TOTP and backup codes have nothing to resend.
  if (factor != null && factor.needsSending) {
    AppTextButton("Resend code", { onChooseFactor(factor) }, enabled = !state.submitting)
  }
  AppTextButton("Use a different email", onChangeEmail, enabled = !state.submitting)
}

private fun secondFactorLabel(factor: SecondFactor): String =
  when (factor) {
    SecondFactor.EmailCode -> "Email"
    SecondFactor.PhoneCode -> "SMS"
    SecondFactor.Totp -> "Authenticator"
    SecondFactor.BackupCode -> "Backup code"
  }

/**
 * " for <address>", or nothing at all when there isn't one, for the steps whose copy names the account being changed.
 *
 * Real rather than defensive on the new-password step, and [secondFactorPrompt] handles the same case a few lines down for the same reason: the OAuth
 * route never collects an address, and `completeSignIn` can answer it with `needs_new_password`. Without this the sentence renders with a hole in it,
 * naming nobody.
 *
 * The create-password step is the defensive one. Every route to it runs through the email step, so it always has an address, and it shares this only
 * because two steps asking the same thing should not word it two ways.
 *
 * The code steps need no guard at all, hence no "to <address>" counterpart here: each is entered only by a send that required an address, which is
 * why [CodeStep] names one outright.
 */
private fun forEmail(email: String): String = if (email.isBlank()) "" else " for $email"

/**
 * What the new-password step says, which depends on whether the user asked for the reset or Clerk demanded it (see
 * [EmailAuthUiState.resetWasForced]).
 *
 * The requested route needs no cause at all: the user tapped the link, answered the code, and is expecting this screen. The demanded route needs one
 * badly, since the password just typed was *correct*, and a screen that asks for another without a word reads as a rejection. Same division of labour
 * as [secondFactorPrompt] a few lines down, which exists for the same reason.
 *
 * Neither sentence claims the account has no password. It usually does: only one of the three ways in (an account that never had one) fits that
 * description, and the other two would be flatly contradicted by it.
 */
private fun resetPasswordPrompt(email: String, wasForced: Boolean): String {
  val choose = "Choose a password${forEmail(email)}, and you'll be signed in with it."
  // Says a password has to be set, not that one was rejected or has to be replaced. The demanded route is reached with a password submitted (the
  // password step) and without one (an emailed code, or OAuth), so a cause sentence that mentions the credential the user just presented would be
  // naming nothing on two of the three.
  return if (wasForced) "A password has to be set on this account before you can sign in. $choose" else choose
}

/**
 * Where the code is coming from, and for device verification why it is being asked for at all.
 *
 * The MFA wording can stay terse because the user set two-step verification up and is expecting it. Client Trust arrives unannounced on an ordinary
 * password sign-in, so the device sentence leads: without it the screen looks like the password was wrong.
 */
private fun secondFactorPrompt(factor: SecondFactor?, email: String, reason: SecondFactorReason): String {
  val source =
    when (factor) {
      // The address is unknown on the OAuth route, which never collects one, so name the channel rather than an empty string.
      SecondFactor.EmailCode -> if (email.isBlank()) "Enter the code sent to your email." else "Enter the code sent to $email."
      SecondFactor.PhoneCode -> "Enter the code sent to your phone."
      SecondFactor.Totp -> "Enter the code from your authenticator app."
      SecondFactor.BackupCode -> "Enter one of your backup codes."
      null -> "Choose how to verify."
    }
  return when (reason) {
    SecondFactorReason.Mfa -> source
    SecondFactorReason.DeviceVerification -> "This is the first sign-in on this device, so it needs confirming. $source"
  }
}

@Composable
private fun ErrorText(error: String?) {
  if (error != null) {
    Text(error, modifier = Modifier.padding(top = 8.dp), style = TextStyle(fontSize = 14.sp), color = PipetteTheme.colors.destructive)
  }
}

@Composable
private fun GateMessage(title: String, message: String, showSpinner: Boolean) {
  Column(
    modifier = Modifier.fillMaxSize().padding(28.dp),
    verticalArrangement = Arrangement.Center,
    horizontalAlignment = Alignment.CenterHorizontally,
  ) {
    DisplayTitle(title)
    if (showSpinner) CircularProgressIndicator(modifier = Modifier.padding(vertical = 16.dp))
    AppLabel(message)
  }
}

@Preview
@Composable
private fun AuthGateScreenPreview() {
  PipetteTheme {
    AuthGateScreen(
      gate = AuthGate.SignedOut,
      emailAuth = EmailAuthUiState(),
      oauthProviders = emptyList(),
      isDebug = false,
      onSubmitEmail = {},
      onSubmitCode = {},
      onOAuthProvider = {},
      onUsePassword = {},
      onSubmitPassword = {},
      onSubmitNewPassword = {},
      onStartPasswordReset = {},
      onSubmitResetCode = {},
      onSubmitResetPassword = {},
      onChooseSecondFactor = {},
      onSubmitSecondFactor = {},
      onChangeEmail = {},
      onEditClearError = {},
      onSkipDebug = {},
      onSignOut = {},
      onDeleteIdentity = {},
    )
  }
}
