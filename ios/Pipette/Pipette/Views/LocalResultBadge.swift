import SwiftUI

/// Marks a benchmark or result that came from the generated `local/` catalog half.
///
/// The distinction is not cosmetic: the server never sanctioned these definitions, so
/// their results are not submitted (``JobCell/isSubmittable``). Without the mark a user
/// can pick one from the New Job screen, measure for an hour, and never learn that
/// nothing was published — the run looks identical to a synced one at every step.
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
