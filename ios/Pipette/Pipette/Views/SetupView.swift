import SwiftUI
import UIKit

/// Initial registration view shown once before the main app.
///
/// Collects organization and contact email, then calls pipette-ios's
/// registration service to generate a CryptoKit signing keypair and register with
/// the management server.
/// - Note: `onSignOut` is the only way off this screen. It sits between a live
///   Clerk session and the app, so once signed in there is otherwise nothing to
///   press but Register: no back, and Settings is on the far side of the
///   registration being asked for. Without an exit, the account that signed in
///   is the account the device is stuck with.
struct SetupView: View {
    @Binding var isRegistered: Bool
    let clerkUserId: String?
    let clerkSessionId: String?
    let clerkPrimaryEmail: String?
    let onSignOut: () -> Void

    @Environment(\.storage) private var storage
    @State private var collectorEndpointOption: CollectorEndpointOption = .production
    @State private var customCollectorURL = ""
    @State private var organization = ""
    @State private var contactEmail: String
    @State private var preauthKey = ""
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var registrationResult: (clientId: String, status: String)?
    @FocusState private var focusedField: RegistrationField?

    init(
        isRegistered: Binding<Bool>,
        clerkUserId: String? = nil,
        clerkSessionId: String? = nil,
        clerkPrimaryEmail: String? = nil,
        onSignOut: @escaping () -> Void = {}
    ) {
        self._isRegistered = isRegistered
        self.clerkUserId = clerkUserId
        self.clerkSessionId = clerkSessionId
        self.clerkPrimaryEmail = clerkPrimaryEmail
        self.onSignOut = onSignOut
        self._contactEmail = State(initialValue: clerkPrimaryEmail ?? "")
    }

    var body: some View {
        NavigationStack {
            GeometryReader { proxy in
                ScrollView {
                    VStack(spacing: 0) {
                        PipetteLogoMark()
                            .frame(width: 52, height: 52)
                            .padding(.bottom, 14)

                        Text("Welcome to Pipette")
                            .font(.serif(26))
                            .lineSpacing(0)
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.primary)
                            .padding(.bottom, 15)

                        Text("Measure model performance on your device")
                            .font(.system(size: 16, weight: .regular))
                            .lineLimit(1)
                            .minimumScaleFactor(0.9)
                            .multilineTextAlignment(.center)
                            .foregroundStyle(Color(.systemGray))
                            .fixedSize(horizontal: false, vertical: true)

                        if let clerkPrimaryEmail {
                            Text("Signed in as \(clerkPrimaryEmail)")
                                .font(.system(size: 13, weight: .regular))
                                .multilineTextAlignment(.center)
                                .foregroundStyle(Color(.systemGray))
                                .padding(.top, 10)
                        }

                        VStack(spacing: 12) {
                            // Only ask for an email when Clerk hasn't provided
                            // one; otherwise registration uses the signed-in
                            // account's email directly.
                            if clerkPrimaryEmail == nil {
                                PipetteTextField(
                                    placeholder: "Email",
                                    text: $contactEmail,
                                    keyboardType: .emailAddress,
                                    textContentType: .emailAddress,
                                    capitalization: .never,
                                    submitLabel: .next
                                )
                                .focused($focusedField, equals: .email)
                                .onSubmit {
                                    focusedField = .organization
                                }
                            }

                            PipetteTextField(
                                placeholder: "Organization name",
                                text: $organization,
                                keyboardType: .default,
                                textContentType: .organizationName,
                                capitalization: .words,
                                submitLabel: collectorEndpointOption == .custom ? .next : .done
                            )
                            .focused($focusedField, equals: .organization)
                            .onSubmit {
                                focusedField = collectorEndpointOption == .custom ? .customCollectorURL : nil
                            }

                            CollectorEndpointPicker(
                                options: CollectorEndpointOption.allCases,
                                selection: $collectorEndpointOption
                            )

                            if collectorEndpointOption == .custom {
                                PipetteTextField(
                                    placeholder: "Custom collector URL",
                                    text: $customCollectorURL,
                                    keyboardType: .URL,
                                    textContentType: .URL,
                                    capitalization: .never,
                                    submitLabel: .done
                                )
                                .focused($focusedField, equals: .customCollectorURL)
                                .onSubmit {
                                    focusedField = nil
                                }

                                if let customCollectorValidationMessage {
                                    Text(customCollectorValidationMessage)
                                        .font(.footnote)
                                        .foregroundStyle(.red)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .fixedSize(horizontal: false, vertical: true)
                                        .padding(.horizontal, 6)
                                }

                                // Custom only: the Liquid collector approves a device on its own
                                // side, so a key has nothing to admit there. Optional even here —
                                // blank registers exactly as before, and a self-hosted deployment
                                // that issues keys can skip the manual approval hop.
                                PipetteTextField(
                                    placeholder: "Pre-auth key (optional)",
                                    text: $preauthKey,
                                    keyboardType: .asciiCapable,
                                    textContentType: nil,
                                    capitalization: .never,
                                    submitLabel: .done
                                )
                                .focused($focusedField, equals: .preauthKey)
                                .onSubmit {
                                    focusedField = nil
                                }
                            }
                        }
                        .padding(.top, 44)

                        Button(action: register) {
                            ZStack {
                                if isLoading {
                                    ProgressView()
                                        .tint(Color(.systemBackground))
                                } else {
                                    Text("Register")
                                        .font(.system(size: 17, weight: .regular))
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 48)
                            .foregroundStyle(Color(.systemBackground))
                            .background(
                                Capsule()
                                    .fill(canRegister ? Color.primary : Color(.systemGray))
                            )
                        }
                        .buttonStyle(.plain)
                        .disabled(!canRegister || isLoading)
                        .padding(.top, 26)

                        if let error = errorMessage {
                            Text(error)
                                .font(.footnote)
                                .multilineTextAlignment(.center)
                                .foregroundStyle(.red)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.top, 18)
                        }

                        // Only when there is a session to end. A gate that reached
                        // here without Clerk has nothing to sign out of, and a dead
                        // control is worse than none.
                        if clerkPrimaryEmail != nil {
                            Button("Sign out", action: onSignOut)
                                .font(.system(size: 15, weight: .medium))
                                .foregroundStyle(isLoading ? Color(.systemGray) : .primary)
                                .buttonStyle(.plain)
                                .disabled(isLoading)
                                .padding(.top, 22)
                        }
                    }
                    .padding(.horizontal, horizontalPadding(for: proxy.size.width))
                    .padding(.vertical, 32)
                    .frame(minHeight: proxy.size.height, alignment: .center)
                    .frame(maxWidth: .infinity)
                }
                .scrollDismissesKeyboard(.interactively)
            }
            .background(Color(.systemBackground))
            .navigationBarHidden(true)
            .onChange(of: collectorEndpointOption) { _, option in
                if option == .custom {
                    focusedField = .customCollectorURL
                } else {
                    // Drop the key but keep the URL. A key typed under Custom and then
                    // abandoned would otherwise still be sent to Liquid's collector,
                    // which is the one thing hiding the field promises won't happen. The
                    // URL is inert by comparison — `serverURL(customURL:)` ignores it on
                    // `.production` — so keeping it means toggling back doesn't retype it.
                    preauthKey = ""
                    if focusedField == .customCollectorURL || focusedField == .preauthKey {
                        focusedField = nil
                    }
                }
            }
        }
    }

    private var canRegister: Bool {
        !isLoading &&
        !organization.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !contactEmail.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        selectedServerURL != nil
    }

    private var selectedServerURL: String? {
        collectorEndpointOption.serverURL(customURL: customCollectorURL)
    }

    private var customCollectorValidationMessage: String? {
        let trimmed = customCollectorURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard collectorEndpointOption == .custom, !trimmed.isEmpty else { return nil }
        return CollectorEndpoint.normalizedCustomURL(trimmed) == nil
            ? "Enter a valid HTTPS collector URL."
            : nil
    }

    private func horizontalPadding(for width: CGFloat) -> CGFloat {
        min(max(width * 0.11, 32), 48)
    }

    private func register() {
        guard canRegister, let url = selectedServerURL else { return }

        isLoading = true
        errorMessage = nil
        focusedField = nil

        // Capture @State values for the detached task
        let org = organization.trimmingCharacters(in: .whitespacesAndNewlines)
        let email = contactEmail.trimmingCharacters(in: .whitespacesAndNewlines)
        // Passed straight to the register request; never written to disk,
        // Keychain, or logs. An empty field means keyless registration.
        let trimmedPreauthKey = preauthKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let key = trimmedPreauthKey.isEmpty ? nil : trimmedPreauthKey

        // Use Task.detached so the synchronous Rust FFI call runs off the
        // main thread — a plain Task inherits @MainActor from the View.
        let storage = storage
        Task.detached {
            do {
                let result = try await RegistrationService.register(
                    serverUrl: ServerURL(url),
                    organization: org,
                    contactEmail: email,
                    preauthKey: key,
                    clerkUserId: clerkUserId,
                    clerkSessionId: clerkSessionId,
                    clerkPrimaryEmail: clerkPrimaryEmail,
                    storage: storage
                )
                await MainActor.run {
                    registrationResult = (result.clientId.value, result.status)
                    isRegistered = true
                }
            } catch {
                await MainActor.run {
                    errorMessage = humanizedRegistrationError(error, preauthContext: true)
                }
            }
            await MainActor.run {
                isLoading = false
            }
        }
    }

}

private enum RegistrationField: Hashable {
    case email
    case organization
    case customCollectorURL
    case preauthKey
}

private struct CollectorEndpointPicker: View {
    let options: [CollectorEndpointOption]
    @Binding var selection: CollectorEndpointOption

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Collector")
                .font(.system(size: 12, weight: .regular))
                .foregroundStyle(Color(.systemGray))
                .padding(.horizontal, 6)

            HStack(spacing: 4) {
                ForEach(options) { option in
                    Button {
                        selection = option
                    } label: {
                        Text(option.title)
                            .font(.system(size: 13, weight: selection == option ? .semibold : .regular))
                            .foregroundStyle(selection == option ? Color(.systemBackground) : Color.primary)
                            .lineLimit(2)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: .infinity)
                            .frame(height: 42)
                            .background(
                                RoundedRectangle(cornerRadius: 7, style: .continuous)
                                    .fill(selection == option ? Color.primary : Color.clear)
                            )
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(selection == option ? .isSelected : [])
                    .accessibilityLabel(option.title)
                }
            }
            .geometryGroup()
            .padding(4)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color(.systemBackground))
                    .shadow(color: Color.black.opacity(0.08), radius: 3, y: 2)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(Color(.systemGray5), lineWidth: 1.5)
            )
        }
        .accessibilityLabel("Collector")
        .accessibilityValue(selection.title)
    }
}

#if DEBUG
#Preview("Setup · Clerk email") {
    SetupView(
        isRegistered: .constant(false),
        clerkPrimaryEmail: "alex@example.com"
    )
}

#Preview("Setup · No email") {
    SetupView(isRegistered: .constant(false))
}
#endif
