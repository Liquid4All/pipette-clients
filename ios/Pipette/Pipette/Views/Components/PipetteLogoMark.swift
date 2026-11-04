import SwiftUI

/// The Pipette pocket mark that heads the onboarding screens.
///
/// The asset carries transparent padding around the glyph, so it is scaled up
/// inside a fixed frame and clipped back to it.
struct PipetteLogoMark: View {
    var size: CGFloat = 52

    var body: some View {
        Image("pipette-logo")
            .resizable()
            .scaledToFit()
            .scaleEffect(1.85)
            .frame(width: size, height: size)
            .clipped()
            .accessibilityHidden(true)
    }
}

#if DEBUG
#Preview("Logo Mark") {
    PipetteLogoMark()
}
#endif
