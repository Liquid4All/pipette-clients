import ClerkKit
import SwiftUI

/// The sign-in screen, one view per step of `EmailAuthState.Step`. Replaces
/// ClerkKit's prebuilt `AuthView` with the same screens Android draws
/// (`AuthGateScreen.kt`), so the two clients match.
///
/// The passwordless path is the short one: an address, then the 6-digit code.
/// Off it hang a password sign-in for an account that has one (which App Review
/// needs, since a reviewer cannot read a code sent to a mailbox we do not
/// control: see `docs/pipette-ios/app-review.md`), the password a new account
/// finishes registering with, a second-factor challenge, and the reset that
/// gets a password onto an account with none.
///
/// The steps that ask the same question share a view rather than repeating it:
/// `CodeStep` serves the two first-factor-shaped codes (the sign-in code and the
/// reset code) and `SetPasswordStep` both password choices, as their Android
/// counterparts do. A second-factor code is entered in `SecondFactorAnswer`
/// instead, since it varies in width and can be alphanumeric.
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
                        // Keyed on the generation so a resend Clerk accepted starts
                        // an empty field: the digits already typed answer the code
                        // it just replaced. Same device as `SecondFactorAnswer`'s
                        // identity, one part shorter, since there is no factor to
                        // switch between here.
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
        // Seeded rather than empty so coming back from a later step ("Use a
        // different email") restores the address to edit.
        self._email = State(initialValue: model.state.email)
    }

    /// Both actions on this step need an address and no request in flight.
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

        // The credential path, for an account that already has a password. Needs
        // the address first, so it stays disabled until the field is filled and
        // then carries that value forward rather than asking for it twice.
        Button("Sign in with a password") {
            model.usePassword(email)
        }
        .font(.system(size: 15, weight: .medium))
        .foregroundStyle(canSubmit ? .primary : Color(.systemGray))
        .buttonStyle(.plain)
        .disabled(!canSubmit)
        .padding(.top, 14)

        // Social sign-in: one button per provider enabled in the Clerk dashboard.
        // Absent entirely until the backend enables any — then the divider and the
        // buttons appear without a client change.
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

/// The credential path: sign an existing account in with its password, no code.
///
/// It exists because a store reviewer cannot read a code sent to a mailbox we do
/// not control, the same reason Android added it, and it is offered to everyone
/// because most accounts have a password. Not all do: a social sign-in or a
/// dashboard invite creates one without, at any time, which is what the reset
/// link below is for. It never registers: an unknown address is an error here,
/// not a sign-up.
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

        // The way in for an account that has no password at all, as much as for a
        // forgotten one. Labelled from the same constant the reset step's title
        // and `noPasswordOnAccount` use, so the message that tells someone to tap
        // this cannot end up naming something that isn't here.
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

/// Answering a 6-digit code Clerk emailed, for either strategy that sends one:
/// the first-factor `email_code`, and the reset's `reset_password_email_code`.
///
/// One view for both, as Android has one `CodeStep`. They ask the same question
/// with the same field and the same auto-submit; what differs is the sentence
/// after the address, where the answer goes, and whether there is a resend. Two
/// copies of this drifted apart on Android before it was shared, which is the
/// reason to share it here before that happens rather than after.
private struct CodeStep: View {
    let model: EmailAuthModel
    /// Appended to "We sent a 6-digit code to <address>.", when the step needs to
    /// say what the code is *for*. The first factor does not: a code that signs
    /// you in needs no explaining.
    var then: String?
    let onSubmit: (String) async -> Void
    /// Absent on the first-factor step, which has no resend. Present on the
    /// reset, where it is the only way back from a code that expired or a send
    /// that failed, since that step offers nothing to switch to.
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
                // A previous failure stops applying the moment the code is edited.
                model.clearError()
                // Auto-submit once every digit is in, the standard OTP affordance. This
                // fires on edits only, so a rejected code sits there until the user
                // changes it; editing back to full length submits the new attempt. The
                // model's own in-flight guard covers the double-tap.
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

/// Choosing a password, for both steps that ask for one: finishing a new account
/// (`createPassword`) and finishing a reset on an account that already exists
/// (`resetPassword`).
///
/// One view for both, as Android has one `SetPasswordStep`. Everything that
/// matters is identical, the secure field, the stated minimum, the focus
/// handling, and only the three strings differ. Sharing it is what keeps the
/// minimum from being stated as one number on one step and another elsewhere,
/// which `EmailAuthState.minimumPasswordLength` alone would not have caught once
/// the hint text was duplicated too.
///
/// Clerk owns the real policy, so rejections still arrive in its words; the
/// minimum here is stated up front because the rule is knowable before the user
/// commits to anything.
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

/// The code challenge: pick a factor (when more than one is offered), then answer
/// it. Serves both an account's own two-step verification and Clerk's Client
/// Trust device check, which differ only in wording. Mirrors Android's
/// second-factor step.
///
/// The code for whichever factor is chosen has already been sent by the model —
/// asking for a code this screen never requested was the dead end Android hit
/// before it stopped waiting for a tap.
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
            // Identified by the factor and by which challenge this is, so either
            // changing starts an empty field. A code typed for one factor is not a
            // valid answer to another, and one already submitted is not an answer
            // to the re-challenge that just emailed a fresh code. The reason is not
            // part of this: it only ever changes on a new challenge, which the
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

    /// Ask for another code, shared by the two steps that can. The reset step's
    /// only way back from an expired code or a failed send, and on a challenge
    /// offering one delivered factor the only way back there too, since the
    /// chooser is hidden with nothing to switch to. Android pairs the same button
    /// with the same condition.
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

    /// The way back to the address field, shared by every step past the first.
    /// It began as the way out of a challenge the user cannot answer, which is
    /// still the case it matters most for: an account with MFA enrolled and no
    /// access to its factor is otherwise stuck. It is also the only exit from
    /// both password steps and from a reset whose code never arrived. Left on
    /// this type because that is where it started; it belongs to none of them in
    /// particular now.
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

/// Entry for one chosen factor. Its own view so the typed code is scoped to the
/// factor by identity rather than by remembering to clear it: the parent
/// re-creates this on every switch, which is what makes a code typed for the
/// authenticator app unable to be submitted against an emailed one.
private struct SecondFactorAnswer: View {
    /// What this view's `.id` is keyed on: a change to either part is a different
    /// challenge and has to start an empty field.
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

        // A fixed-width numeric code gets the segmented field; a backup code is
        // alphanumeric and varies in length, so it gets a plain one.
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

        // Only for a factor Clerk delivers: an authenticator app and a backup code
        // have nothing to resend. On a challenge offering one delivered factor this
        // is the only way back from a code that expired or a send that failed,
        // since the chooser below is hidden with nothing to switch to. Android
        // pairs the same button with the same condition.
        if factor.needsSending {
            SecondFactorStep.resendCode(model) { await model.resendSecondFactorCode() }
        }

        // Only worth offering when there is something else to switch to.
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

/// Segmented one-time-code field: `length` equal-width cells, each showing one
/// digit, backed by a single invisible text field that owns focus, the keypad
/// and paste. The next cell to fill is outlined in the label colour.
private struct OtpCodeField: View {
    @Binding var code: String
    let length: Int
    let isEnabled: Bool

    @FocusState private var isFocused: Bool

    var body: some View {
        ZStack {
            // Kept in the layout (not hidden) so it can hold the input session; its
            // glyphs and caret are what the cells below draw, so it renders at a
            // hair of opacity rather than being drawn twice.
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

    /// Filters on the way in, so the binding the caller watches only ever holds
    /// digits and never overshoots `length` — a pasted "12 34 56" must not read as
    /// a complete code and auto-submit before it is normalized.
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

/// Clickwrap notice under the email step's button, matching Android's
/// `LegalNotice`.
///
/// It sits on the email step alone, because that is the only step that can
/// register, and registering is what makes it necessary: the sign-up sends
/// `legal_accepted` on the strength of this sentence. The links open in Safari.
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

    /// Underlined and in the label colour, so the two targets read as tappable
    /// against the gray sentence around them.
    private func link(_ label: String, to url: String) -> AttributedString {
        var segment = AttributedString(label)
        segment.link = URL(string: url)
        segment.underlineStyle = .single
        segment.foregroundColor = .primary
        return segment
    }
}

/// A centered "or" flanked by hairlines, separating the email form from the
/// social buttons.
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

/// The filled capsule action — the same 48pt button `SetupView` registers with.
/// Also used by the gate's unavailable state for its retry.
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

/// The failure line under whichever field it belongs to. Renders nothing when
/// there is no error, so it costs no space in the layout.
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

/// Drives the model to the code step so the preview renders it without a
/// network round-trip.
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
