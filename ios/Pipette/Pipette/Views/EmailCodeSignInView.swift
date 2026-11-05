import ClerkKit
import SwiftUI

/// The sign-in screen, one view per step of `EmailAuthState.Step`. Draws the same
/// screens as Android (`AuthGateScreen.kt`) instead of ClerkKit's prebuilt `AuthView`.
///
/// The short path is an address, then the 6-digit code. The password sign-in exists for
/// App Review, whose reviewer cannot read a code sent to a mailbox we do not control
/// (`docs/pipette-ios/app-review.md`).
struct EmailCodeSignInView: View {
    @Environment(Clerk.self) private var clerk
    @State private var model = EmailAuthModel()

    var body: some View {
        GeometryReader { proxy in
            ScrollView {
                VStack(spacing: 0) {
                    switch model.state.step {
                    case .email:
                        EmailStep(model: model, providers: clerk.oauthProviders)
                    case .password:
                        PasswordStep(model: model)
                    case .code:
                        CodeStep(model: model, onSubmit: model.submitCode)
                    case .createPassword:
                        SetPasswordStep(
                            model: model,
                            title: "Create a password",
                            prompt: "Your email is verified. Choose a password to finish creating the account"
                                + "\(EmailAuthModel.forEmail(model.state.email)).",
                            cta: "Create account",
                            onSubmit: model.submitNewPassword
                        )
                    case .secondFactor:
                        SecondFactorStep(model: model)
                    case .resetCode:
                        // Keyed on the generation so an accepted resend starts an empty
                        // field: the typed digits answer the code it replaced.
                        CodeStep(
                            model: model,
                            then: "Enter it to choose a password.",
                            onSubmit: model.submitResetCode,
                            onResend: model.resendPasswordResetCode
                        )
                        .id(model.state.resetCodeGeneration)
                    case .resetPassword:
                        SetPasswordStep(
                            model: model,
                            title: EmailAuthModel.setPasswordAction,
                            prompt: EmailAuthModel.resetPasswordPrompt(
                                email: model.state.email,
                                wasForced: model.state.resetWasForced
                            ),
                            cta: "Save password",
                            onSubmit: model.submitResetPassword
                        )
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 32)
                .frame(minHeight: proxy.size.height, alignment: .center)
                .frame(maxWidth: .infinity)
            }
            .scrollDismissesKeyboard(.interactively)
        }
        .background(Color(.systemBackground))
    }
}

// MARK: - Email step

private struct EmailStep: View {
    let model: EmailAuthModel
    let providers: [OAuthProviderInfo]

    @State private var email: String

    init(model: EmailAuthModel, providers: [OAuthProviderInfo]) {
        self.model = model
        self.providers = providers
        // Seeded so "Use a different email" restores the address to edit.
        self._email = State(initialValue: model.state.email)
    }

    private var canSubmit: Bool {
        EmailAuthState.canSubmitEmail(email, submitting: model.state.submitting)
    }

    var body: some View {
        PipetteLogoMark()
            .padding(.bottom, 14)

        Text("Welcome to Pipette")
            .font(.serif(26))
            .multilineTextAlignment(.center)
            .foregroundStyle(.primary)
            .padding(.bottom, 15)

        Text("Measure model performance on your device")
            .font(.system(size: 16, weight: .regular))
            .multilineTextAlignment(.center)
            .foregroundStyle(Color(.systemGray))
            .fixedSize(horizontal: false, vertical: true)

        PipetteTextField(
            placeholder: "Email",
            text: $email,
            keyboardType: .emailAddress,
            textContentType: .emailAddress,
            submitLabel: .go
        )
        .padding(.top, 32)
        .onChange(of: email) { _, _ in model.clearError() }
        .onSubmit(submit)

        AuthErrorText(model.state.error)

        AuthPrimaryButton(
            title: "Register",
            isEnabled: canSubmit,
            isLoading: model.state.submitting,
            action: submit
        )
        .padding(.top, 20)

        LegalNotice()

        // Needs the address first, and carries it forward rather than asking twice.
        Button("Sign in with a password") {
            model.usePassword(email)
        }
        .font(.system(size: 15, weight: .medium))
        .foregroundStyle(canSubmit ? .primary : Color(.systemGray))
        .buttonStyle(.plain)
        .disabled(!canSubmit)
        .padding(.top, 14)

        // One button per provider enabled in the Clerk dashboard, so providers appear
        // without a client change.
        if !providers.isEmpty {
            OrDivider()
                .padding(.vertical, 16)

            VStack(spacing: 12) {
                ForEach(providers) { provider in
                    AuthOutlineButton(
                        title: "Continue with \(provider.name)",
                        isEnabled: !model.state.submitting
                    ) {
                        Task { await model.signInWithOAuth(strategy: provider.strategy) }
                    }
                }
            }
        }
    }

    private func submit() {
        guard canSubmit else { return }
        Task { await model.submitEmail(email) }
    }
}

// MARK: - Password step

/// Sign an existing account in with its password. Never registers: an unknown address
/// is an error here. An account created by social sign-in or a dashboard invite has no
/// password, which is what the reset link is for.
private struct PasswordStep: View {
    let model: EmailAuthModel

    @State private var password = ""
    @FocusState private var isFocused: Bool

    private var canSubmit: Bool {
        !password.isEmpty && !model.state.submitting
    }

    var body: some View {
        Text("Enter your password")
            .font(.serif(28))
            .multilineTextAlignment(.center)
            .foregroundStyle(.primary)

        Text("Signing in as \(model.state.email).")
            .font(.system(size: 14, weight: .regular))
            .multilineTextAlignment(.center)
            .foregroundStyle(Color(.systemGray))
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, 6)

        PipetteTextField(
            placeholder: "Password",
            text: $password,
            keyboardType: .default,
            textContentType: .password,
            submitLabel: .go,
            isSecure: true
        )
        .focused($isFocused)
        .padding(.top, 16)
        .onChange(of: password) { _, _ in model.clearError() }
        .onSubmit(submit)

        AuthErrorText(model.state.error)

        AuthPrimaryButton(
            title: "Sign in",
            isEnabled: canSubmit,
            isLoading: model.state.submitting,
            action: submit
        )
        .padding(.top, 20)

        // Also the way in for an account with no password. Shares its label with
        // `noPasswordOnAccount`, so that message names a button that exists.
        Button(EmailAuthModel.setPasswordAction) {
            isFocused = false
            Task { await model.startPasswordReset(model.state.email) }
        }
        .font(.system(size: 15, weight: .medium))
        .foregroundStyle(model.state.submitting ? Color(.systemGray) : .primary)
        .buttonStyle(.plain)
        .disabled(model.state.submitting)
        .padding(.top, 14)

        SecondFactorStep.useDifferentEmail(model)
            .onAppear { isFocused = true }
    }

    private func submit() {
        guard canSubmit else { return }
        isFocused = false
        Task {
            await model.submitPassword(email: model.state.email, password: password)
        }
    }
}

// MARK: - Code step

/// A 6-digit code Clerk emailed, for the first-factor `email_code` and the reset's
/// `reset_password_email_code`. One view for both, as on Android.
private struct CodeStep: View {
    let model: EmailAuthModel
    /// Appended to "We sent a 6-digit code to <address>." to say what the code is for.
    var then: String?
    let onSubmit: (String) async -> Void
    /// Only on the reset, where it is the only way back from an expired code or a
    /// failed send.
    var onResend: (() async -> Void)?

    @State private var code = ""

    private static let codeLength = 6

    var body: some View {
        Text("Check your email")
            .font(.serif(28))
            .multilineTextAlignment(.center)
            .foregroundStyle(.primary)

        Text(
            [
                "We sent a \(Self.codeLength)-digit code to \(model.state.email).",
                then,
            ]
            .compactMap { $0 }
            .joined(separator: " ")
        )
        .font(.system(size: 14, weight: .regular))
        .multilineTextAlignment(.center)
        .foregroundStyle(Color(.systemGray))
        .fixedSize(horizontal: false, vertical: true)
        .padding(.top, 6)

        OtpCodeField(code: $code, length: Self.codeLength, isEnabled: !model.state.submitting)
            .padding(.top, 16)
            .onChange(of: code) { _, entered in
                model.clearError()
                // Fires on edits only, so a rejected code sits until the user changes it.
                // The model's in-flight guard covers the double-tap.
                if entered.count == Self.codeLength {
                    submit()
                }
            }

        AuthErrorText(model.state.error)

        AuthPrimaryButton(
            title: "Verify",
            isEnabled: code.count == Self.codeLength,
            isLoading: model.state.submitting,
            action: submit
        )
        .padding(.top, 20)

        if let onResend {
            SecondFactorStep.resendCode(model, action: onResend)
        }

        SecondFactorStep.useDifferentEmail(model)
    }

    private func submit() {
        guard code.count == Self.codeLength else { return }
        Task { await onSubmit(code) }
    }
}

// MARK: - Choosing a password

/// Choosing a password, for `createPassword` and `resetPassword`. One view for both,
/// as on Android, so the stated minimum cannot differ between them.
///
/// Clerk owns the real policy and rejects in its own words; the minimum is stated up
/// front because it is knowable before the user commits.
private struct SetPasswordStep: View {
    let model: EmailAuthModel
    let title: String
    let prompt: String
    let cta: String
    let onSubmit: (String) async -> Void

    @State private var password = ""
    @FocusState private var isFocused: Bool

    private var canSubmit: Bool {
        password.count >= EmailAuthState.minimumPasswordLength && !model.state.submitting
    }

    var body: some View {
        Text(title)
            .font(.serif(28))
            .multilineTextAlignment(.center)
            .foregroundStyle(.primary)

        Text(prompt)
            .font(.system(size: 14, weight: .regular))
            .multilineTextAlignment(.center)
            .foregroundStyle(Color(.systemGray))
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, 6)

        PipetteTextField(
            placeholder: "Password",
            text: $password,
            keyboardType: .default,
            textContentType: .newPassword,
            submitLabel: .go,
            isSecure: true
        )
        .focused($isFocused)
        .padding(.top, 16)
        .onChange(of: password) { _, _ in model.clearError() }
        .onSubmit(submit)

        Text("At least \(EmailAuthState.minimumPasswordLength) characters.")
            .font(.system(size: 13, weight: .regular))
            .foregroundStyle(Color(.systemGray))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, 8)
            .padding(.horizontal, 6)

        AuthErrorText(model.state.error)

        AuthPrimaryButton(
            title: cta,
            isEnabled: canSubmit,
            isLoading: model.state.submitting,
            action: submit
        )
        .padding(.top, 20)

        SecondFactorStep.useDifferentEmail(model)
            .onAppear { isFocused = true }
    }

    private func submit() {
        guard canSubmit else { return }
        isFocused = false
        Task { await onSubmit(password) }
    }
}

// MARK: - Second factor

/// The code challenge: pick a factor (when more than one is offered), then answer it.
/// Serves both an account's two-step verification and Clerk's Client Trust device
/// check, which differ only in wording. The model has already sent the chosen
/// factor's code.
private struct SecondFactorStep: View {
    let model: EmailAuthModel

    private var options: [SecondFactor] { model.state.secondFactorOptions }
    private var chosen: SecondFactor? { model.state.chosenSecondFactor }
    private var reason: SecondFactorReason { model.state.secondFactorReason }

    var body: some View {
        Text(reason.title)
            .font(.serif(28))
            .multilineTextAlignment(.center)
            .foregroundStyle(.primary)

        if let chosen {
            // A new factor or a re-challenge starts an empty field: the typed code
            // answers neither. The reason only changes on a new challenge, which the
            // generation already counts.
            SecondFactorAnswer(model: model, factor: chosen, reason: reason, canChangeMethod: options.count > 1)
                .id(SecondFactorAnswer.Identity(factor: chosen, generation: model.state.challengeGeneration))
        } else {
            picker
        }
    }

    @ViewBuilder
    private var picker: some View {
        Text(reason.prompt("Choose how to verify it's you."))
            .font(.system(size: 14, weight: .regular))
            .multilineTextAlignment(.center)
            .foregroundStyle(Color(.systemGray))
            .padding(.top, 6)

        VStack(spacing: 12) {
            ForEach(options) { option in
                AuthOutlineButton(title: option.title, isEnabled: !model.state.submitting) {
                    Task { await model.chooseSecondFactor(option) }
                }
            }
        }
        .padding(.top, 20)

        AuthErrorText(model.state.error)

        Self.useDifferentEmail(model)
    }

    /// Shared by the reset step and a delivered second factor. It is the only way back
    /// from an expired code or a failed send on the reset, and on a challenge offering a
    /// single factor, where the chooser is hidden.
    fileprivate static func resendCode(
        _ model: EmailAuthModel,
        action: @escaping () async -> Void
    ) -> some View {
        Button("Resend code") {
            Task { await action() }
        }
        .font(.system(size: 15, weight: .medium))
        .foregroundStyle(model.state.submitting ? Color(.systemGray) : .primary)
        .buttonStyle(.plain)
        .disabled(model.state.submitting)
        .padding(.top, 14)
    }

    /// The way back to the address field, shared by every step past the first. Without
    /// it, an account with MFA enrolled and no access to its factor is stuck.
    fileprivate static func useDifferentEmail(_ model: EmailAuthModel) -> some View {
        Button("Use a different email") {
            model.changeEmail()
        }
        .font(.system(size: 15, weight: .medium))
        .foregroundStyle(model.state.submitting ? Color(.systemGray) : .primary)
        .buttonStyle(.plain)
        .disabled(model.state.submitting)
        .padding(.top, 14)
    }
}

/// Entry for one chosen factor. Its own view so the typed code is scoped to the factor
/// by identity: the parent re-creates it on every switch, so a code typed for one
/// factor cannot be submitted against another.
private struct SecondFactorAnswer: View {
    /// What this view's `.id` is keyed on.
    struct Identity: Hashable {
        let factor: SecondFactor
        let generation: Int
    }

    let model: EmailAuthModel
    let factor: SecondFactor
    let reason: SecondFactorReason
    let canChangeMethod: Bool

    @State private var code = ""

    private var canSubmit: Bool {
        guard let digits = factor.digitCount else { return !code.isEmpty }
        return code.count == digits
    }

    var body: some View {
        Text(reason.prompt(factor.prompt))
            .font(.system(size: 14, weight: .regular))
            .multilineTextAlignment(.center)
            .foregroundStyle(Color(.systemGray))
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, 6)

        // A backup code is alphanumeric and varies in length, so it gets a plain field.
        if let digits = factor.digitCount {
            OtpCodeField(code: $code, length: digits, isEnabled: !model.state.submitting)
                .padding(.top, 16)
                .onChange(of: code) { _, entered in
                    model.clearError()
                    if entered.count == digits { submit() }
                }
        } else {
            PipetteTextField(
                placeholder: "Backup code",
                text: $code,
                keyboardType: .asciiCapable,
                textContentType: nil,
                submitLabel: .go
            )
            .padding(.top, 16)
            .onChange(of: code) { _, _ in model.clearError() }
            .onSubmit(submit)
        }

        AuthErrorText(model.state.error)

        AuthPrimaryButton(
            title: "Verify",
            isEnabled: canSubmit,
            isLoading: model.state.submitting,
            action: submit
        )
        .padding(.top, 20)

        // An authenticator app and a backup code have nothing to resend.
        if factor.needsSending {
            SecondFactorStep.resendCode(model) { await model.resendSecondFactorCode() }
        }

        if canChangeMethod {
            Button("Use a different method") {
                model.changeSecondFactor()
            }
            .font(.system(size: 15, weight: .medium))
            .foregroundStyle(model.state.submitting ? Color(.systemGray) : .primary)
            .buttonStyle(.plain)
            .disabled(model.state.submitting)
            .padding(.top, 14)
        }

        SecondFactorStep.useDifferentEmail(model)
    }

    private func submit() {
        guard canSubmit else { return }
        Task { await model.submitSecondFactorCode(code) }
    }
}

// MARK: - Components

/// Segmented one-time-code field: digit cells drawn over a single invisible text
/// field that owns focus, the keypad and paste.
private struct OtpCodeField: View {
    @Binding var code: String
    let length: Int
    let isEnabled: Bool

    @FocusState private var isFocused: Bool

    var body: some View {
        ZStack {
            // Near-zero opacity rather than hidden, so it keeps the input session.
            TextField("", text: digitsOnly)
                .keyboardType(.numberPad)
                .textContentType(.oneTimeCode)
                .focused($isFocused)
                .opacity(0.01)
                .accessibilityLabel("Verification code")

            cells
                .allowsHitTesting(false)
                .accessibilityHidden(true)
        }
        .frame(maxWidth: 260)
        .contentShape(Rectangle())
        .onTapGesture { isFocused = true }
        .onAppear { isFocused = true }
        .disabled(!isEnabled)
    }

    /// Filters on the way in, so a pasted "12 34 56" cannot read as a complete code
    /// and auto-submit before it is normalized.
    private var digitsOnly: Binding<String> {
        Binding(
            get: { code },
            set: { entered in code = String(entered.filter(\.isNumber).prefix(length)) }
        )
    }

    private var cells: some View {
        HStack(spacing: 8) {
            ForEach(0..<length, id: \.self) { index in
                let isActive = isEnabled && index == code.count && code.count < length
                Text(digit(at: index))
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundStyle(.primary)
                    .frame(maxWidth: .infinity)
                    .frame(height: 48)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color(.systemBackground))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(
                                isActive ? Color.primary : Color(.systemGray5),
                                lineWidth: isActive ? 2 : 1.5
                            )
                    )
            }
        }
    }

    private func digit(at index: Int) -> String {
        guard index < code.count else { return "" }
        return String(code[code.index(code.startIndex, offsetBy: index)])
    }
}

/// Clickwrap notice, matching Android's `LegalNotice`. Only on the email step, the only
/// one that can register: the sign-up sends `legalAccepted` on the strength of it.
private struct LegalNotice: View {
    var body: some View {
        Text(notice)
            .font(.system(size: 13, weight: .regular))
            .multilineTextAlignment(.center)
            .foregroundStyle(Color(.systemGray))
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, 10)
            .padding(.horizontal, 8)
    }

    private var notice: AttributedString {
        var text = AttributedString("By continuing you agree to the ")
        text += link("Terms", to: EmailAuthState.termsURL)
        text += AttributedString(" and ")
        text += link("Privacy Policy", to: EmailAuthState.privacyPolicyURL)
        text += AttributedString(".")
        return text
    }

    private func link(_ label: String, to url: String) -> AttributedString {
        var segment = AttributedString(label)
        segment.link = URL(string: url)
        segment.underlineStyle = .single
        segment.foregroundColor = .primary
        return segment
    }
}

private struct OrDivider: View {
    var body: some View {
        HStack(spacing: 12) {
            line
            Text("or")
                .font(.system(size: 13, weight: .regular))
                .foregroundStyle(Color(.systemGray))
            line
        }
    }

    private var line: some View {
        Rectangle()
            .fill(Color(.systemGray5))
            .frame(height: 1)
    }
}

/// The filled capsule action, styled like `SetupView`'s Register button.
struct AuthPrimaryButton: View {
    let title: String
    let isEnabled: Bool
    let isLoading: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                if isLoading {
                    ProgressView()
                        .tint(Color(.systemBackground))
                } else {
                    Text(title)
                        .font(.system(size: 17, weight: .regular))
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 48)
            .foregroundStyle(Color(.systemBackground))
            .background(Capsule().fill(isEnabled ? Color.primary : Color(.systemGray)))
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled || isLoading)
    }
}

/// The outlined capsule used for the social providers.
private struct AuthOutlineButton: View {
    let title: String
    let isEnabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 17, weight: .regular))
                .frame(maxWidth: .infinity)
                .frame(height: 48)
                .foregroundStyle(isEnabled ? .primary : Color(.systemGray))
                .background(Capsule().fill(Color(.systemBackground)))
                .capsuleBorder()
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
    }
}

/// Renders nothing when there is no error, so it costs no space.
private struct AuthErrorText: View {
    let message: String?

    init(_ message: String?) {
        self.message = message
    }

    var body: some View {
        if let message {
            Text(message)
                .font(.system(size: 14, weight: .regular))
                .multilineTextAlignment(.center)
                .foregroundStyle(.red)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 10)
        }
    }
}

#if DEBUG
#Preview("Sign In · Email") {
    EmailCodeSignInView()
        .environment(Clerk.preview { preview in
            preview.isSignedIn = false
        })
}

#Preview("Sign In · Code") {
    CodeStepPreview()
        .environment(Clerk.preview { preview in
            preview.isSignedIn = false
        })
}

/// Starts at the code step, without a network round-trip.
private struct CodeStepPreview: View {
    @State private var model = EmailAuthModel.previewAtCodeStep(email: "tester@example.com")

    var body: some View {
        VStack(spacing: 0) {
            CodeStep(model: model, onSubmit: model.submitCode)
        }
        .padding(24)
    }
}
#endif
