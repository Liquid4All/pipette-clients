import SwiftUI

/// A warning or an information note on a grey card.
struct NoticeCallout: View {
    let message: String
    var isWarning = true
    var font: Font = .system(size: 15)

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: isWarning ? "exclamationmark.triangle.fill" : "info.circle.fill")
                .foregroundStyle(isWarning ? Color.orange : Color.secondary)
            Text(message)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(font)
        .padding(14)
        .background(Color(.secondarySystemBackground),
                    in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}

#Preview {
    VStack(spacing: 16) {
        NoticeCallout(message: MemoryFreshness.pickerNote)
        if let notice = MemoryFreshness.runNotice(
            .newJob, publicBuild: true, memoryCells: 1, totalCells: 1, ranEngineWork: false
        ) {
            NoticeCallout(message: notice.message, isWarning: notice.isWarning, font: .caption)
        }
    }
    .padding()
}
