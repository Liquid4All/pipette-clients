import SwiftUI

/// Branded launch splash shown at cold start before the Clerk auth gate.
/// White background, the Pipette mark + wordmark centered, and the
/// "by ◆ Liquid" lockup pinned near the bottom — mirroring the design mockup.
///
/// The splash is a plain visual; timing and the fade-out to the app live in
/// `PipetteApp` (see the `showSplash` gate), so this view stays testable and
/// preview-able in isolation.
struct SplashView: View {
    var body: some View {
        ZStack {
            Color.white
                .ignoresSafeArea()

            // Centered mark + wordmark lockup (single vector asset).
            Image("pipette-logo-splash")
                .resizable()
                .scaledToFit()
                .frame(width: 120)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            // "by ◆ Liquid" lockup pinned to the bottom.
            VStack {
                Spacer()
                Image("by-liquid")
                    .resizable()
                    .scaledToFit()
                    .frame(height: 56)
                    .padding(.bottom, 20)
            }
        }
    }
}

#Preview {
    SplashView()
}
