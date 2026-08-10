import SwiftUI

/// A labeled detail row: a fixed-width title column on the left, arbitrary
/// content on the right. Shared by `RunDetailHeaderView` and `CellDetailView`,
/// which both previously hand-rolled this layout at slightly different
/// densities — now a single consistent style.
enum PropertyList {
    /// Spacing between stacked `PropertyRow`s. Exposed so each list's container
    /// can match without re-hardcoding the value.
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

/// A property row whose value is a flow of chips, collapsing any chips beyond
/// `limit` into an "N more" label. The single way labeled chip lists render
/// across the run-detail header and cell-detail page, so they look identical.
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

/// The canonical chips used inside `PropertyChipRow`, so every property list
/// renders models and values at the same size.
enum PropertyChip {
    /// A model chip with brand logo. `brandSource` overrides which string the
    /// brand is detected from, for when the displayed text is a prettified name
    /// that differs from the raw model name.
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
