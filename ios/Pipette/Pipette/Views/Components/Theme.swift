import SwiftUI
import UIKit

// MARK: - Typography

private let serifFontName = "IowanOldStyle-Roman"

extension Font {
    /// The app's serif display font (used for page titles and card headings).
    ///
    /// Centralizes the font-family name so it lives in exactly one place. It
    /// was previously spelled out as `.custom("IowanOldStyle-Roman", size:)` at
    /// ~20 call sites — a single typo there would silently fall back to the
    /// system font with no compiler warning.
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
    /// The hairline border drawn around cards and grouped containers. Replaces
    /// the `RoundedRectangle(...).strokeBorder(Color(.systemGray5), lineWidth: 1)`
    /// overlay that was hand-rolled in ~10 places. Compose it over whatever
    /// background the card already sets.
    func cardBorder(cornerRadius: CGFloat) -> some View {
        overlay(
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(Color(.systemGray5), lineWidth: 1)
        )
    }

    /// A card container: the standard `systemBackground` fill in a continuous
    /// rounded rect plus the `cardBorder` hairline — the `.background(…, in:).cardBorder(…)`
    /// pair that recurred across the New Job steps and the settings cards. Clip and
    /// shadow stay caller-side, since only some cards want them.
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

/// A small capsule pill showing a status label tinted by its colour — used for
/// the overall job status and for per-cell run status, which previously
/// repeated the same six modifiers inline.
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
