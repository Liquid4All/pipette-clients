import SwiftUI
import UIKit

/// The app's rounded onboarding input: 48pt tall, 8pt corners, a `systemGray5`
/// hairline and a soft drop shadow. Shared by the sign-in screen and the
/// registration setup screen so the two consecutive screens match; Android's
/// `IosTextField` is the port of this same recipe.
struct PipetteTextField: View {
    let placeholder: String
    @Binding var text: String
    let keyboardType: UIKeyboardType
    /// Optional: a secret field (pre-auth key) passes nil to opt out of autofill
    /// associations so the token isn't offered to the system keychain.
    let textContentType: UITextContentType?
    var capitalization: TextInputAutocapitalization = .never
    var submitLabel: SubmitLabel = .done
    /// Masks the entry, and offers a reveal toggle to go with it.
    var isSecure: Bool = false

    /// Whether the masked text is currently shown. Local, and reset by anything
    /// that rebuilds the field — a revealed password should not outlive the
    /// screen it was typed on.
    @State private var isRevealed = false
    @FocusState private var isFocused: Bool

    var body: some View {
        field
            .font(.system(size: 17, weight: .regular))
            .foregroundStyle(.primary)
            .textContentType(textContentType)
            .keyboardType(keyboardType)
            .textInputAutocapitalization(capitalization)
            .autocorrectionDisabled()
            .submitLabel(submitLabel)
            .focused($isFocused)
            .padding(.leading, 24)
            // Room for the eye, so a long password doesn't run under it.
            .padding(.trailing, isSecure ? 52 : 24)
            .frame(maxWidth: .infinity)
            .frame(height: 48)
            .overlay(alignment: .trailing) {
                if isSecure {
                    revealToggle
                }
            }
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

    @ViewBuilder
    private var field: some View {
        if isSecure, !isRevealed {
            SecureField(placeholder, text: $text)
        } else {
            TextField(placeholder, text: $text)
        }
    }

    private var revealToggle: some View {
        Button {
            isRevealed.toggle()
            // Swapping SecureField for TextField hands first responder back to
            // nobody, which drops the keyboard mid-entry. Take it straight back.
            isFocused = true
        } label: {
            Image(systemName: isRevealed ? "eye.slash" : "eye")
                .font(.system(size: 16, weight: .regular))
                .foregroundStyle(Color(.systemGray))
                .frame(width: 52, height: 48)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(isRevealed ? "Hide password" : "Show password")
    }
}

#if DEBUG
#Preview("Text Fields") {
    VStack(spacing: 12) {
        PipetteTextField(
            placeholder: "Email",
            text: .constant(""),
            keyboardType: .emailAddress,
            textContentType: .emailAddress
        )
        PipetteTextField(
            placeholder: "Password",
            text: .constant("hunter2"),
            keyboardType: .default,
            textContentType: .password,
            isSecure: true
        )
    }
    .padding(24)
}
#endif
