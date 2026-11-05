import SwiftUI

/// Layout shared by the `PropertyRow` lists in `RunDetailHeaderView` and
/// `CellDetailView`.
enum PropertyList {
    /// Spacing between stacked `PropertyRow`s.
    static let rowGap: CGFloat = 16
    static let labelColumnWidth: CGFloat = 88
}

struct PropertyRow<Content: View>: View {
    let title: String
    let content: Content

    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            Text(title)
                .font(.system(size: 15))
                .foregroundStyle(.secondary)
                .frame(width: PropertyList.labelColumnWidth, alignment: .leading)

            content
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// A property row whose value is a flow of chips.
struct PropertyChipRow<Chip: View>: View {
    let title: String
    let values: [String]
    let chip: (String) -> Chip

    var body: some View {
        PropertyRow(title: title) {
            ChipFlowLayout(horizontalSpacing: 10, verticalSpacing: 8) {
                ForEach(values, id: \.self) { value in
                    chip(value)
                }
            }
        }
    }
}

/// The chips used inside `PropertyChipRow`, so every property list renders
/// models and values at the same size.
enum PropertyChip {
    /// `brandSource` is the raw model name to detect the brand from when `text`
    /// is a prettified one.
    static func model(_ text: String, brandSource: String? = nil) -> some View {
        AppModelChip(
            text: text,
            brand: ModelBrand.detect(name: brandSource ?? text, hfRepo: nil),
            maxWidth: 235
        )
    }

    static func text(_ text: String) -> some View {
        AppTextChip(text: text)
    }
}
