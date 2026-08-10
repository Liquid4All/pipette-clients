import Sentry
import SwiftUI

/// In-app user feedback / bug reporting, recorded as a Sentry User Feedback record via
/// `SentrySDK.capture(feedback:)`. Mirrors the Android `FeedbackDialog` and the
/// pipette-dashboard web flow: an optional category, a required message, and an optional
/// email (prefilled from the signed-in Clerk user). The same category taxonomy is reused so
/// the `category` tag means the same thing across web, Android, and iOS in Sentry.
///
/// Device / chip / OS / app-version tags are set once globally in `SentryConfiguration`, so
/// they ride along with this feedback (and crashes) automatically; the per-submission
/// `category` and the `source` ("ios-settings") tag are set on the scope around the capture
/// (see `submit()`). The `source` tag matches the Android/web taxonomy so cross-platform
/// `source` filtering includes iOS. No honeypot/min-interval spam guard — a native form has
/// no bot surface.
struct FeedbackView: View {
    /// Pre-fills the optional reply address (the signed-in Clerk email, when available).
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

    /// How a field's label is annotated. Modeled as one value so the required and optional
    /// states are mutually exclusive (no nonsensical `required && optional`), and the hint
    /// only exists where it's meaningful.
    private enum FieldKind {
        case required
        case optional(hint: String? = nil)
    }

    /// A labeled field group: a small caption (with an "*"/"(optional…)" suffix) above the control.
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

        // The static `capture(feedback:)` applies the *global* scope (the public SDK feedback
        // API has no cloned-scope overload — only the lower-level client does), so set the
        // feedback-only tags on the scope around the call and remove them immediately after,
        // so they never ride along on a later crash. capture() applies the scope synchronously
        // before returning, so the tags are baked into the feedback before the removal.
        // `source` marks the submission origin (the iOS counterpart of Android's `source` tag,
        // so cross-platform `source` filtering includes iOS); `category` is per-submission.
        // Device / chip / os / app_version are already global.
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

        // The feedback CONTENT stays in Sentry. PostHog records only that feedback was sent and
        // under which category, so the funnel can show how often people reach for it. No message,
        // no contact email.
        Analytics.capture(AnalyticsEvents.feedbackSubmitted, [AnalyticsEvents.category: category?.id])

        didSubmit = true
    }
}

/// Optional feedback category. Kept in sync with the Android `FeedbackDialog` categories and
/// pipette-dashboard's `FEEDBACK_CATEGORIES` so the `category` tag is consistent everywhere.
enum FeedbackCategory: String, CaseIterable, Identifiable {
    case reportBug = "report_bug"
    case reportIncorrectData = "report_incorrect_data"
    case requestModel = "request_model"
    case requestRuntime = "request_runtime"
    case requestHardware = "request_hardware"
    case requestEval = "request_eval"
    case other

    var id: String { rawValue }

    var label: String {
        switch self {
        case .reportBug: return "Report a bug"
        case .reportIncorrectData: return "Report incorrect data"
        case .requestModel: return "Request a model"
        case .requestRuntime: return "Request a runtime"
        case .requestHardware: return "Request hardware"
        case .requestEval: return "Request an evaluation dataset"
        case .other: return "Something else"
        }
    }
}

private extension View {
    /// The app's standard rounded, hairline-bordered field box.
    func roundedFieldBackground() -> some View {
        background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(Color(.systemGray4), lineWidth: 1)
            )
    }
}
