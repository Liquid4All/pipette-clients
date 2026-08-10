package ai.liquid.pipette

import android.text.InputType
import android.widget.CheckBox
import android.widget.LinearLayout

/**
 * Device registration, result-submission defaults, HF token, local data, and debug info. (Registration absorbed the former Setup tab — iOS parity.)
 */
class SettingsScreen(ctx: ScreenContext) : Screen(ctx) {
  override fun renderBody(body: LinearLayout) {
    val registration = storage.loadRegistration()
    body.addView(displayTitle("Settings"))
    body.addView(registrationCard(registration))

    body.addView(
      card {
        addView(sectionTitle("Account"))
        val clerk = vm.clerkUser
        when {
          clerk != null -> {
            addView(mutedLabel("Signed in as ${clerk.email ?: clerk.userId}"))
            // Signing out swaps the app chrome for the auth gate, which would
            // hide an in-flight run's progress/cancel controls. Block it until
            // the run finishes (or is cancelled from the Jobs screen).
            if (vm.runnerState.value.runningJobId != null) {
              addView(mutedLabel("Sign out is unavailable while a benchmark is running."))
            } else {
              addView(
                outlineButton("Sign out") {
                  vm.signOutOfClerk()
                  statusText = "Signed out"
                  render()
                }
              )
            }
          }
          vm.isClerkAvailable -> addView(mutedLabel("Not signed in."))
          else -> addView(mutedLabel("Sign-in is not configured for this build."))
        }
        if (BuildConfig.DEBUG) {
          addView(
            CheckBox(activity).apply {
              text = "Bypass auth gate (debug only)"
              isChecked = vm.clerkGateBypass
              setOnCheckedChangeListener { _, checked ->
                vm.setClerkGateBypass(checked)
                statusText = if (checked) "Auth gate bypassed" else "Auth gate enforced"
                render()
              }
            }
          )
        }
      }
    )

    body.addView(
      card {
        addView(sectionTitle("Results"))
        addView(
          CheckBox(activity).apply {
            text = "Auto-submit benchmark results by default"
            isChecked = storage.isRegistered() && vm.defaultContributeResults
            isEnabled = storage.isRegistered()
            setOnCheckedChangeListener { _, checked -> setDefaultContributeResults(checked) }
          }
        )
        if (!storage.isRegistered()) {
          addView(mutedLabel("Register this device before enabling default result submission."))
        }
      }
    )

    body.addView(
      card {
        addView(sectionTitle("Thermal state"))
        addView(statusBadge(thermalStateLabel(), thermalAccent()))
        addView(mutedLabel(thermalStateDescription()))
      }
    )

    body.addView(
      card {
        addView(sectionTitle("Hugging Face"))
        val token = input("HF token", secrets.loadHfToken() ?: "")
        token.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        addView(token)
        addView(
          primaryButton("Save HF token") {
            val value = token.text.toString().trim()
            // Report what actually happened. A save can fail (the Keystore was unreachable), and
            // claiming success would leave the user with a download that later fails on a token
            // they believe they saved. Clearing cannot fail, so only the save branch is checked.
            statusText =
              if (value.isBlank()) {
                secrets.deleteHfToken()
                "HF token cleared"
              } else if (secrets.saveHfToken(value)) {
                "HF token updated"
              } else {
                "Could not save the HF token: secure storage is unavailable"
              }
            render()
          }
        )
      }
    )

    body.addView(
      card {
        addView(sectionTitle("Local data"))
        addView(
          outlineButton("Reset jobs and models") {
            confirm("Delete local jobs and downloaded models?") {
              storage.resetDeviceData()
              selectedModelKeys.clear()
              selectedMmprojPaths.clear()
              mmprojSelectionInitialized = false
              render()
            }
          }
        )
      }
    )

    // Only shown when Sentry initialized (DSN wired via the manifest) — mirrors the
    // dashboard's FEEDBACK_ENABLED gate so there's no dead button without a backend.
    if (FeedbackDialog.isAvailable()) {
      body.addView(
        card {
          addView(sectionTitle("Feedback"))
          addView(mutedLabel("Report a bug or tell us what's missing. Sent to the team via Sentry."))
          addView(
            primaryButton(FeedbackDialog.BUTTON_LABEL) {
              FeedbackDialog.show(
                activity,
                ui,
                defaultEmail = vm.clerkUser?.email,
                analytics = (activity.application as? PipetteApp)?.containerOrNull?.analytics ?: NoOpAnalytics,
              ) {
                statusText = "Feedback submitted, thank you"
                render()
              }
            }
          )
        }
      )
    }

    body.addView(
      card {
        addView(sectionTitle("Debugging"))
        addView(mutedLabel(debugInfoText(registration)))
      }
    )
  }

  /** Device registration — folded in from the former Setup tab. Shows the register form when unregistered, otherwise a summary + clear. */
  private fun registrationCard(registration: RegistrationData?): android.view.View = card {
    addView(sectionTitle("Device registration"))
    if (registration != null) {
      addView(statusBadge("Registered", ui.colorThermalNominal()))
      addView(mutedLabel("Client: ${registration.clientId}\nStatus: ${registration.status}\nServer: ${registration.serverUrl}"))
      addView(
        outlineButton("Clear registration") {
          confirm("Clear registration?") {
            storage.deleteRegistration()
            secrets.deletePrivateKey()
            vm.applyDefaultContributeResults(false) // unregistered → off, persisted off-main
            vm.refreshRegistration() // re-publish to the auth gate
            render()
          }
        }
      )
      return@card
    }
    addView(mutedLabel("Register this device to submit benchmark results."))
    val settings = vm.setupSettings
    val server = input("Server URL", settings.serverUrl)
    val org = input("Organization", settings.organization)
    // Pre-fill the contact email from the signed-in Clerk account (iOS parity).
    val clerk = vm.clerkUser
    val email =
      input(
        "Contact email",
        settings.contactEmail.ifBlank { clerk?.email.orEmpty() },
        InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS,
      )
    addView(server)
    addView(org)
    addView(email)
    addView(
      primaryButton("Register device") {
        vm.persistSetupSettings(
          SetupSettings(
            serverUrl = server.text.toString().trim(),
            organization = org.text.toString().trim(),
            contactEmail = email.text.toString().trim(),
          )
        )
        runInBackground("Registering...") {
          val result =
            registrationService.register(
              serverUrl = server.text.toString().trim(),
              organization = org.text.toString().trim(),
              contactEmail = email.text.toString().trim(),
              clerkUserId = clerk?.userId,
              clerkSessionId = clerk?.sessionId,
              clerkPrimaryEmail = clerk?.email,
            )
          onMain {
            contributeResults = vm.defaultContributeResults
            vm.refreshRegistration()
          }
          "Registered ${result.clientId} (${result.status})"
        }
      }
    )
  }

  /** Color the thermal badge by severity, via the shared [thermalAccentKind] classifier. */
  private fun thermalAccent(): Int =
    when (thermalAccentKind(thermalStateDescription())) {
      AccentKind.CRITICAL -> ui.colorThermalCritical()
      AccentKind.SERIOUS -> ui.colorThermalSerious()
      AccentKind.NOMINAL,
      AccentKind.MUTED -> ui.colorThermalNominal()
    }

  private fun setDefaultContributeResults(enabled: Boolean) {
    val allowed = vm.applyDefaultContributeResults(enabled)
    statusText =
      if (allowed) {
        "Default auto-submit enabled"
      } else {
        "Default auto-submit disabled"
      }
    render()
  }

  private fun debugInfoText(registration: RegistrationData?): String {
    val models = storage.availableModels().size
    val jobs = storage.loadAllJobManifests().size
    val privateKeyState = if (secrets.hasPrivateKey()) "Present" else "Missing"
    val hfTokenState = if (secrets.loadHfToken() == null) "Missing" else "Present"
    val autoSubmit =
      if (storage.isRegistered()) {
        if (vm.defaultContributeResults) "Enabled" else "Disabled"
      } else {
        "Unavailable"
      }
    return listOf(
        "Client ID: ${registration?.clientId ?: "Unavailable"}",
        "Status: ${registration?.status ?: "Unavailable"}",
        "Clerk user: ${registration?.clerkUserId ?: "Unlinked"}",
        "Clerk email: ${registration?.clerkPrimaryEmail ?: "Unavailable"}",
        "Device: ${DeviceInfo.modelName()}",
        "Chip: ${DeviceInfo.chipModel()}",
        "Form factor: ${DeviceInfo.formFactor(activity)}",
        "OS: Android ${DeviceInfo.osVersion()}",
        "RAM: ${ByteFormat.fileSize(DeviceInfo.ramBytes(activity))}",
        "Thermal: ${thermalStateDescription()}",
        "Auto-submit: $autoSubmit",
        "Jobs: $jobs",
        "Models: $models",
        "Private key: $privateKeyState",
        "HF token: $hfTokenState",
        "Models directory: ${storage.modelsDir.absolutePath}",
      )
      .joinToString("\n")
  }
}
