import SwiftUI

/// Rounded checkbox matching the job wizard mockup: filled with the foreground
/// color and a white check when on, an outlined empty square when off.
struct WizardCheckbox: View {
    let isOn: Bool
    var isMixed: Bool = false
    var size: CGFloat = 26

    var body: some View {
        RoundedRectangle(cornerRadius: 6)
            .fill((isOn || isMixed) ? Color.primary : Color.clear)
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder((isOn || isMixed) ? Color.clear : Color(.systemGray3), lineWidth: 1.5)
            )
            .overlay(
                Image(systemName: isMixed ? "minus" : "checkmark")
                    .font(.system(size: 13, weight: .bold))
                    .foregroundStyle(Color(.systemBackground))
                    .opacity((isOn || isMixed) ? 1 : 0)
            )
            .frame(width: size, height: size)
    }
}
