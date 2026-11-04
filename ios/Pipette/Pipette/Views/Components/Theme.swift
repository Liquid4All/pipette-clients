import SwiftUI
import UIKit

// MARK: - Typography

private let serifFontName = "IowanOldStyle-Roman"

extension Font {
    /// The app's serif display font, for page titles and card headings.
    static func serif(_ size: CGFloat) -> Font {
        .custom(serifFontName, size: size)
    }

    /// The serif display font, scaling with Dynamic Type relative to `textStyle`.
    static func serif(_ size: CGFloat, relativeTo textStyle: Font.TextStyle) -> Font {
        .custom(serifFontName, size: size, relativeTo: textStyle)
    }
}

// MARK: - Page headers

extension View {
    /// Large page title: 32pt serif at 120% line height.
    func pageHeaderLarge() -> some View {
        pageHeader(size: 32, lineHeightMultiple: 1.2)
    }

    /// Small page title: 17pt serif at 140% line height.
    func pageHeaderSmall() -> some View {
        pageHeader(size: 17, lineHeightMultiple: 1.4)
    }

    /// `lineSpacing` is additive on top of the font's intrinsic line height, so
    /// derive the extra spacing from the real font metrics to hit the target
    /// multiple (e.g. 120% / 140%) instead of assuming a 1.0x baseline.
    private func pageHeader(size: CGFloat, lineHeightMultiple: CGFloat) -> some View {
        let intrinsic = UIFont(name: serifFontName, size: size)?.lineHeight ?? size
        let extraSpacing = max(0, size * lineHeightMultiple - intrinsic)
        return font(.serif(size)).lineSpacing(extraSpacing)
    }
}

// MARK: - Card chrome

extension View {
    /// The hairline border around cards and grouped containers, drawn over
    /// whatever background the card sets.
    func cardBorder(cornerRadius: CGFloat) -> some View {
        overlay(
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(Color(.systemGray5), lineWidth: 1)
        )
    }

    /// A card container: `systemBackground` fill plus the `cardBorder` hairline.
    /// Clip and shadow stay caller-side, since only some cards want them.
    func appCard(cornerRadius: CGFloat) -> some View {
        background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .cardBorder(cornerRadius: cornerRadius)
    }

    /// The same hairline border as `cardBorder`, for pill-shaped containers.
    func capsuleBorder() -> some View {
        overlay(
            Capsule()
                .strokeBorder(Color(.systemGray5), lineWidth: 1)
        )
    }
}

// MARK: - Status badge

/// A capsule status pill, for job and per-cell run status.
struct StatusBadge: View {
    let label: String
    let color: Color

    var body: some View {
        Text(label)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(Capsule())
    }
}
