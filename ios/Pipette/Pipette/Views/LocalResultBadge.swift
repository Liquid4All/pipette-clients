import SwiftUI

/// Marks a benchmark or result that came from the generated `local/` catalog half.
///
/// The server never sanctioned these definitions, so their results are not submitted
/// (``JobCell/isSubmittable``). Without the mark a local run looks identical to a synced
/// one, and the user never learns nothing was published.
struct LocalResultBadge: View {
    /// Shown alongside the badge where there is room to explain it.
    var showsCaption = false

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "iphone")
                .font(.system(size: 10, weight: .semibold))
            Text(showsCaption ? "Local, stays on this device" : "Local")
                .font(.system(size: 11, weight: .semibold))
                .lineLimit(1)
        }
        .foregroundStyle(.secondary)
        .padding(.horizontal, 8)
        .frame(height: 20)
        .background(Color(.secondarySystemBackground), in: Capsule())
        .accessibilityLabel("Local benchmark. Results stay on this device.")
    }
}
