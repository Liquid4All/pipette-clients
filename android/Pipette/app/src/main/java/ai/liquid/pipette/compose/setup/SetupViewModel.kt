package ai.liquid.pipette.compose.setup

import ai.liquid.pipette.ManagementClientException
import ai.liquid.pipette.PipetteApp
import ai.liquid.pipette.RegistrationData
import ai.liquid.pipette.SetupSettings
import ai.liquid.pipette.compose.ScreenViewModel
import ai.liquid.pipette.compose.shell.ShellViewModel
import android.app.Application
import androidx.lifecycle.viewModelScope
import java.io.IOException
import java.net.ConnectException
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Device-registration screen state: either the registered summary, or the registration form. */
data class SetupUiState(
  val registration: RegistrationData? = null,
  val settings: SetupSettings = SetupSettings(),
  val clerkEmail: String = "",
  val isRegistering: Boolean = false,
)

sealed interface SetupIntent {
  /** [preauthKey] is blank when the user left the optional field empty, which registers keylessly. */
  data class Register(val serverUrl: String, val organization: String, val contactEmail: String, val preauthKey: String = "") : SetupIntent

  data object ClearRegistration : SetupIntent

  /**
   * Sign out of the account, leaving this device's registration alone. Routed through the VM rather than calling [ShellViewModel.signOut] from the
   * composable so a refused sign-out has somewhere to land: this screen's only exit is that button, and it renders no auth state of its own.
   */
  data object SignOut : SetupIntent
}

/** One VM per screen: owns only the Setup screen's state. Registration / Clerk identity / persisted settings come from the [shell] hub. */
class SetupViewModel(app: Application, shell: ShellViewModel) : ScreenViewModel(app, shell) {
  private val container = (app as PipetteApp).container
  private val storage = container.storage
  private val secrets = container.secrets
  private val registrationService = container.registrationService

  private val _state = MutableStateFlow(SetupUiState())
  val state: StateFlow<SetupUiState> = _state.asStateFlow()

  // In-flight registration flag for the Register button's spinner. Mutated only on confine.
  @Volatile private var registering = false

  // Single-thread confinement: publish()'s registration read (disk I/O) and ClearRegistration's
  // file deletes run here, never on Main — matching the other screen VMs.
  private val confine = Dispatchers.Default.limitedParallelism(1)

  init {
    viewModelScope.launch(confine) { publish() }
    viewModelScope.launch(confine) { shell.registration.collect { publish() } }
    viewModelScope.launch(confine) { shell.clerkUser.collect { publish() } }
    viewModelScope.launch(confine) { shell.setupSettings.collect { publish() } }
  }

  fun onIntent(intent: SetupIntent) {
    viewModelScope.launch(confine) { handle(intent) }
  }

  private fun handle(intent: SetupIntent) {
    when (intent) {
      is SetupIntent.Register -> register(intent)
      SetupIntent.SignOut -> shell.signOut(onProblem = ::showError)
      SetupIntent.ClearRegistration -> {
        storage.deleteRegistration()
        secrets.deletePrivateKey()
        shell.applyDefaultContributeResults(false)
        shell.refreshRegistration()
        publish()
      }
    }
  }

  private fun register(intent: SetupIntent.Register) {
    // Re-entrancy guard: the Register button only stops accepting taps once isRegistering round-trips
    // to a recomposition, so a fast double-tap can enqueue two Register intents. Both would race on the
    // single pending-signing-key slot (double registration + a promoted key that doesn't match the
    // registered public key). handle() runs on confine, so this check is serialized with the flag.
    if (registering) return
    val server = intent.serverUrl.trim()
    val email = intent.contactEmail.trim()
    // Organization is mandatory (iOS parity, enforced by the Register button's enabled state), so it
    // arrives non-blank here.
    val org = intent.organization.trim()
    // Passed straight to the request and never persisted or logged; blank means keyless registration, left for
    // manual approval on the collector side.
    val preauthKey = intent.preauthKey.trim().ifBlank { null }
    shell.persistSetupSettings(SetupSettings(server, org, email))
    val clerk = shell.clerkUser.value
    registering = true
    publish()
    // On success the registration flow refreshes and the gate advances off Setup — that transition is
    // the feedback (iOS parity); failures surface as an error toast. The Register button shows an
    // in-button spinner while this runs.
    viewModelScope.launch {
      runCatching {
          withContext(Dispatchers.IO) {
            registrationService.register(
              serverUrl = server,
              organization = org,
              contactEmail = email,
              preauthKey = preauthKey,
              clerkUserId = clerk?.userId,
              clerkSessionId = clerk?.sessionId,
              clerkPrimaryEmail = clerk?.email,
            )
            shell.refreshRegistration()
          }
        }
        .onFailure { showError(registrationErrorMessage(it)) }
      withContext(confine) {
        registering = false
        publish()
      }
    }
  }

  /**
   * Map a registration failure to a user-facing message. A 401/403 on this path is the server's pre-auth-key verdict (mgmt httpapi §2.2), so name the
   * cause instead of leaking a raw "HTTP 401" — the Setup screen offers the key field, so the user can act on it. Everything else falls back to the
   * underlying message. Mirrors iOS humanizedRegistrationError.
   */
  private fun registrationErrorMessage(error: Throwable): String {
    if (error is ManagementClientException) {
      return when (error.statusCode) {
        HttpURLConnection.HTTP_UNAUTHORIZED ->
          "Registration was rejected: the pre-auth key is invalid, expired, or already used. Check the key and try again."
        HttpURLConnection.HTTP_FORBIDDEN -> "This collector requires a pre-auth key to register. Enter a valid key and try again."
        else -> error.message ?: "Registration failed (HTTP ${error.statusCode})."
      }
    }
    // Connectivity cases get plain language instead of a raw exception string (iOS parity).
    return when (error) {
      is SocketTimeoutException -> "The request timed out. Check your connection and try again."
      is UnknownHostException,
      is ConnectException -> "Couldn't reach the registration server. Please try again in a moment."
      is IOException -> "No internet connection. Check your network and try again."
      else -> error.message ?: error.javaClass.simpleName
    }
  }

  private fun publish() {
    _state.value =
      SetupUiState(
        registration = storage.loadRegistration(),
        settings = shell.setupSettings.value,
        clerkEmail = shell.clerkUser.value?.email.orEmpty(),
        isRegistering = registering,
      )
  }
}
