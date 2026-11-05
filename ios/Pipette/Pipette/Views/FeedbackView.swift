import Sentry
import SwiftUI

/// In-app feedback, recorded as a Sentry User Feedback record. Mirrors Android's
/// `FeedbackDialog` and the pipette-dashboard web flow, with the same category taxonomy so
/// the `category` tag means the same thing on every platform.
///
/// No honeypot or min-interval spam guard: a native form has no bot surface.
struct FeedbackView: View {
    /// Pre-fills the optional reply address (the signed-in Clerk email).
    let defaultEmail: String?

    @Environment(\.dismiss) private var dismiss
    @State private var category: FeedbackCategory?
    @State private var message: String = ""
    @State private var email: String = ""
    @State private var didSubmit = false

    private var trimmedMessage: String {
        message.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var canSubmit: Bool {
        !trimmedMessage.isEmpty
    }

    var body: some View {
        NavigationStack {
            Group {
                if didSubmit {
                    successView
                } else {
                    formView
                }
            }
            .background(Color(.systemBackground))
            .navigationTitle("Submit feedback")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if !didSubmit {
                    ToolbarItem(placement: .cancellationAction) {
                        Button("Cancel") { dismiss() }
                    }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Submit") { submit() }
                            .fontWeight(.semibold)
                            .disabled(!canSubmit)
                    }
                }
            }
        }
        .onAppear {
            if email.isEmpty, let defaultEmail { email = defaultEmail }
        }
    }

    private var formView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                Text("Tell us what's missing or broken. We read every message.")
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)

                field(label: "What's this about?", kind: .optional()) {
                    Menu {
                        Button("None") { category = nil }
                        ForEach(FeedbackCategory.allCases) { option in
                            Button(option.label) { category = option }
                        }
                    } label: {
                        HStack {
                            Text(category?.label ?? "Select a category")
                                .foregroundStyle(category == nil ? .secondary : .primary)
                            Spacer()
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(Color(.systemGray3))
                        }
                        .padding(.horizontal, 14)
                        .frame(height: 48)
                        .roundedFieldBackground()
                    }
                }

                field(label: "Tell us more", kind: .required) {
                    ZStack(alignment: .topLeading) {
                        if message.isEmpty {
                            Text("What would you like to tell us?")
                                .font(.system(size: 16))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 18)
                                .padding(.vertical, 16)
                        }
                        TextEditor(text: $message)
                            .font(.system(size: 16))
                            .scrollContentBackground(.hidden)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .frame(minHeight: 120)
                    }
                    .roundedFieldBackground()
                }

                field(label: "Your email", kind: .optional(hint: "if you want a reply")) {
                    TextField("you@example.com", text: $email)
                        .keyboardType(.emailAddress)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .padding(.horizontal, 14)
                        .frame(height: 48)
                        .roundedFieldBackground()
                }
            }
            .padding(24)
        }
    }

    private var successView: some View {
        VStack(spacing: 14) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 44))
                .foregroundStyle(Color(red: 0.12, green: 0.75, blue: 0.32))
            Text("Thanks, we got your feedback.")
                .font(.serif(19))
            Text("We read every submission.")
                .font(.system(size: 14))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task {
            // Mirror the web flow's brief success state, then auto-close.
            try? await Task.sleep(for: .milliseconds(1800))
            dismiss()
        }
    }

    /// How a field's label is annotated. One value, so required and optional are exclusive.
    private enum FieldKind {
        case required
        case optional(hint: String? = nil)
    }

    private func field(
        label: String,
        kind: FieldKind,
        @ViewBuilder control: () -> some View
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 4) {
                Text(label)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.secondary)
                switch kind {
                case .required:
                    Text("*").foregroundStyle(.red)
                case let .optional(hint):
                    Text(hint.map { "(optional, \($0))" } ?? "(optional)")
                        .font(.system(size: 13))
                        .foregroundStyle(Color(.systemGray3))
                }
            }
            control()
        }
    }

    private func submit() {
        guard canSubmit else { return }
        let trimmedEmail = email.trimmingCharacters(in: .whitespacesAndNewlines)

        let feedback = SentryFeedback(
            message: trimmedMessage,
            name: nil,
            email: trimmedEmail.isEmpty ? nil : trimmedEmail,
            source: .custom
        )

        // `capture(feedback:)` applies the global scope (the public API has no cloned-scope
        // overload), so the feedback-only tags are set around the call and removed right
        // after, so they never ride along on a later crash. capture() applies the scope
        // synchronously, so the tags are baked in before the removal. `source` matches the
        // Android and web tag so cross-platform filtering includes iOS. Device, chip, os and
        // app_version are already global (`SentryConfiguration`).
        SentrySDK.configureScope { scope in
            scope.setTag(value: "ios-settings", key: "source")
            if let category {
                scope.setTag(value: category.id, key: "category")
            }
        }
        SentrySDK.capture(feedback: feedback)
        SentrySDK.configureScope { scope in
            scope.removeTag(key: "source")
            scope.removeTag(key: "category")
        }

        // The feedback content stays in Sentry: PostHog gets only the category, never the
        // message or contact email.
        Analytics.capture(AnalyticsEvents.feedbackSubmitted, [AnalyticsEvents.category: category?.id])

        didSubmit = true
    }
}

private extension View {
    func roundedFieldBackground() -> some View {
        background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Color(.systemGray4), lineWidth: 1)
            )
    }
}
