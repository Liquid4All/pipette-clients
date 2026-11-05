import SwiftUI

/// Branded launch splash shown at cold start before the Clerk auth gate.
/// Timing and the fade-out live in `PipetteApp` (`showSplash`).
struct SplashView: View {
    var body: some View {
        ZStack {
            Color.white
                .ignoresSafeArea()

            Image("pipette-logo-splash")
                .resizable()
                .scaledToFit()
                .frame(width: 120)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

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
