import SwiftUI

/// Shared page heading for wizard steps.
struct WizardStepHeader: View {
    let title: String
    let description: String?

    init(title: String, description: String? = nil) {
        self.title = title
        self.description = description
    }

    var body: some View {
        VStack(alignment: .leading, spacing: description == nil ? 0 : 8) {
            Text(title)
                .font(.serif(21))
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)

            if let description {
                Text(description)
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
                    .lineSpacing(5)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
