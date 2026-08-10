import SwiftUI

/// A model family collapsed across its quant variants.
struct ModelGroup: Identifiable {
    let key: String
    let name: String
    let paramLabel: String?
    let files: [DiscoveredModel]

    var id: String { key }

    var brand: ModelBrand {
        ModelBrand.detect(name: name, hfRepo: files.first?.hfRepo)
    }

    var sizeLabel: String {
        paramLabel ?? files.first?.sizeFormatted ?? ""
    }

    /// What of this family is already on disk, or nil when none of it is.
    ///
    /// Marks the *present* rather than the absent: the run screen offers the whole
    /// catalog, so most rows are not downloaded and a "not downloaded" badge would be
    /// noise on nearly every one. Nothing is blocked either way — a benchmark started
    /// against an absent model fetches it before the cell runs.
    var downloadedHint: String? {
        let onDisk = files.filter(\.isDownloaded).count
        guard onDisk > 0 else { return nil }
        return onDisk == files.count ? "Downloaded" : "\(onDisk) of \(files.count) downloaded"
    }

    var quantCountLabel: String {
        "\(files.count) quant\(files.count == 1 ? "" : "s")"
    }

    /// True when the group's quants come from more than one repo (e.g.
    /// Ministral: Q4_0 from unsloth, K-quants from mistralai). The signal to
    /// label each quant with its source so the mix is legible.
    var spansRepos: Bool {
        Set(files.map(\.hfRepo)).count > 1
    }
}

enum ModelCatalog {
    /// Group downloaded files by **model identity** (normalized stem), not by
    /// `hf_repo`. A single model whose quants are intentionally sourced from
    /// different repos — e.g. Ministral Q4_0 from `unsloth` and its K-quants
    /// from `mistralai` upstream — then shows as one card instead of two.
    /// Each `DiscoveredModel` keeps its own `hfRepo`, so per-cell provenance and
    /// reporting are unaffected.
    static func groups(from models: [DiscoveredModel]) -> [ModelGroup] {
        // Family identity: the spec's own `familyId` derivation, shared across a
        // model's quants *and* formats — so a stamped catalog download and a
        // sideloaded copy of the same file group together (the GGUF derivation is the
        // normalized filename stem either way), and a model's GGUF and MLX builds unify.
        func key(_ model: DiscoveredModel) -> String {
            model.familyId
        }

        var seen = Set<String>()
        var keys: [String] = []
        for model in models where seen.insert(key(model)).inserted {
            keys.append(key(model))
        }

        let byKey = Dictionary(grouping: models, by: key)

        return keys.compactMap { k in
            byKey[k].map { group in
                // Prefer the declared family name, then stored provenance,
                // then the filename stem.
                let name = CatalogEntry.familyIdToName[k]
                    ?? group.first?.displayName
                    ?? LocalStorage.modelStem(from: group.first?.name ?? k)
                let param = LocalStorage.parseParamSize(
                    name: group.first?.name ?? name,
                    hfRepo: group.first?.hfRepo
                )
                return ModelGroup(key: k, name: name, paramLabel: param, files: group)
            }
        }
    }

    static func displayName(for key: String) -> String {
        let repoName = key.contains("/") ? String(key.split(separator: "/").last!) : key
        return repoName.hasSuffix("-GGUF") ? String(repoName.dropLast(5)) : repoName
    }

    /// Authored display name for a model *spec* (identity). Prefers the catalog's
    /// repo→name map, then the family catalog keyed by the spec's `familyId`, then a
    /// derived-stem fallback (for sideloads / repos absent from the catalog). The one
    /// place a `Model` spec's user-facing name is resolved, so authored names don't
    /// live on the enums.
    static func displayName(for model: Model) -> String {
        if let repo = model.repo, let name = CatalogEntry.repoToName[repo.description] { return name }
        if let name = CatalogEntry.familyIdToName[model.familyId] { return name }
        // AFM has no repo/file to derive a name from; its authored display name is
        // the engine label ("Apple Foundation"), not the raw "apple-foundation" slug.
        if case .appleFoundationText = model { return model.engineLabel }
        // Sideload fallback: GGUF derives its stem from the weight filename; MLX has no
        // file, so it falls back to the repo slug.
        // `reference` names the weights whichever arm named them, so this holds for a
        // sideload, a store form, and a URL alike.
        let key: String = switch model {
        case let .ggufText(m): m.source.reference
        case let .ggufVision(m): m.source.modelReference
        case let .mlx(m): m.source.reference
        case .appleFoundationText: model.familyId
        }
        return displayName(for: key)
    }

    /// A clear, human-facing label for a downloaded model file: the family display
    /// name plus its quant (e.g. `LFM 2.5 350M · Q4_K_M`) — never the raw on-disk
    /// filename. Falls back to the family name alone when the quant is unknown.
    static func displayLabel(for model: DiscoveredModel) -> String {
        let name = displayName(for: model.source)
        guard let quant = model.quant else { return name }
        return "\(name) · \(quant)"
    }
}

/// Quantization filter pills. `all` mirrors a fully selected concrete quant set.
enum QuantPill: String, CaseIterable, Hashable {
    case all, q1, q2, q4, q4km, q5km

    static var specificCases: [QuantPill] {
        allCases.filter { $0 != .all }
    }

    static func allSelection(disabled disabledQuants: Set<QuantPill> = []) -> Set<QuantPill> {
        Set([.all] + specificCases.filter { !disabledQuants.contains($0) })
    }

    static func toggledSelection(
        _ selection: Set<QuantPill>,
        toggling pill: QuantPill,
        disabled disabledQuants: Set<QuantPill> = [],
        allowsEmpty: Bool = true
    ) -> Set<QuantPill> {
        guard !disabledQuants.contains(pill) else { return selection }

        if pill == .all {
            guard selection.contains(.all) else { return allSelection(disabled: disabledQuants) }
            return allowsEmpty ? [] : selection
        }

        var nextSelection = selection
        nextSelection.remove(.all)
        if nextSelection.contains(pill) {
            nextSelection.remove(pill)
        } else {
            nextSelection.insert(pill)
        }

        return normalizedSelection(
            nextSelection,
            disabled: disabledQuants,
            defaultsToAll: !allowsEmpty
        )
    }

    static func normalizedSelection(
        _ selection: Set<QuantPill>,
        disabled disabledQuants: Set<QuantPill> = [],
        defaultsToAll: Bool = false
    ) -> Set<QuantPill> {
        let enabledSpecificCases = specificCases.filter { !disabledQuants.contains($0) }
        var normalized = selection
        normalized.subtract(disabledQuants)

        if normalized.contains(.all) {
            return allSelection(disabled: disabledQuants)
        }

        normalized.remove(.all)

        if !enabledSpecificCases.isEmpty && enabledSpecificCases.allSatisfy({ normalized.contains($0) }) {
            return allSelection(disabled: disabledQuants)
        }

        if normalized.isEmpty && defaultsToAll {
            return allSelection(disabled: disabledQuants)
        }

        return normalized
    }

    var label: String {
        switch self {
        case .all: return "All quants"
        case .q1: return "q1_0"
        case .q2: return "q2_0"
        case .q4: return "q4_0"
        case .q4km: return "q4_km"
        case .q5km: return "q5_km"
        }
    }

    /// Matches a parsed quant token (e.g. `Q4_0`, `Q4_K_M`) case-insensitively.
    func matches(_ quant: String?) -> Bool {
        guard let q = quant?.uppercased() else { return false }
        switch self {
        case .all: return true
        case .q1: return q == "Q1_0"
        case .q2: return q == "Q2_0"
        case .q4: return q == "Q4_0"
        case .q4km: return q == "Q4_K_M"
        case .q5km: return q == "Q5_K_M"
        }
    }
}

struct AppSearchField: View {
    @Binding var text: String
    var placeholder: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 16, weight: .regular))
                .foregroundStyle(.secondary)
            TextField(placeholder, text: $text)
                .font(.system(size: 15))
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
        }
        .frame(height: 38)
        .padding(.horizontal, 16)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.10), lineWidth: 1)
        )
    }
}

struct AppListCard<Content: View>: View {
    var cornerRadius: CGFloat = 16
    let content: Content

    init(cornerRadius: CGFloat = 16, @ViewBuilder content: () -> Content) {
        self.cornerRadius = cornerRadius
        self.content = content()
    }

    var body: some View {
        VStack(spacing: 0) {
            content
        }
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.10), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }
}

typealias ModelSearchField = AppSearchField
typealias ModelListCard = AppListCard

struct ModelsEmptyStateIllustration: View {
    var width: CGFloat = 250

    var body: some View {
        Image("models-empty-state")
            .resizable()
            .scaledToFit()
            .frame(width: width, height: width * 113 / 275)
            .accessibilityHidden(true)
    }
}

struct EmptyModelsPrompt: View {
    let message: String
    let actionTitle: String
    var actionSystemImage: String?
    let action: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            ModelsEmptyStateIllustration()
                .padding(.bottom, 8)

            Text("No models downloaded")
                .font(.serif(25))
                .foregroundStyle(.primary)

            Text(message)
                .font(.system(size: 16))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineSpacing(5)

            Button(action: action) {
                HStack(spacing: 8) {
                    if let actionSystemImage {
                        Image(systemName: actionSystemImage)
                            .font(.system(size: 14, weight: .semibold))
                    }
                    Text(actionTitle)
                        .font(.system(size: 16, weight: .semibold))
                }
                .foregroundStyle(Color(.systemBackground))
                .padding(.horizontal, 24)
                .frame(height: 45)
                .background(Color.primary, in: Capsule())
            }
            .buttonStyle(.plain)
            .padding(.top, 8)
        }
        .frame(maxWidth: .infinity)
    }
}

extension EmptyModelsPrompt {
    /// The "download models before you can start a job" prompt shown identically
    /// by the Jobs list and New Job's model step. Only the tap action differs.
    static func needModelsForJobs(action: @escaping () -> Void) -> EmptyModelsPrompt {
        EmptyModelsPrompt(
            message: "To create jobs, you need to first download\nmodels to select for benchmarking.",
            actionTitle: "Go to Models",
            action: action)
    }
}

typealias NoDownloadedModelsState = EmptyModelsPrompt

struct ModelFamilyRow<Accessory: View>: View {
    let title: String
    let subtitle: String
    let brand: ModelBrand
    var logoSize: CGFloat = 18
    var titleFont: Font = .system(size: 16, weight: .semibold)
    var subtitleFont: Font = .system(size: 13)
    var titleLineLimit: Int = 1
    let accessory: Accessory

    init(
        title: String,
        subtitle: String,
        brand: ModelBrand,
        logoSize: CGFloat = 18,
        titleFont: Font = .system(size: 16, weight: .semibold),
        subtitleFont: Font = .system(size: 13),
        titleLineLimit: Int = 1,
        @ViewBuilder accessory: () -> Accessory
    ) {
        self.title = title
        self.subtitle = subtitle
        self.brand = brand
        self.logoSize = logoSize
        self.titleFont = titleFont
        self.subtitleFont = subtitleFont
        self.titleLineLimit = titleLineLimit
        self.accessory = accessory()
    }

    var body: some View {
        HStack(spacing: 12) {
            BrandLogoView(brand: brand, size: logoSize)
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(titleFont)
                    .foregroundStyle(.primary)
                    .lineLimit(titleLineLimit)
                    .truncationMode(.tail)
                if !subtitle.isEmpty {
                    Text(subtitle)
                        .font(subtitleFont)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            Spacer(minLength: 8)
            accessory
        }
    }
}

struct SelectableModelGroupRow: View {
    let group: ModelGroup
    let isSelected: Bool
    var checkboxSize: CGFloat = 22
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ModelFamilyRow(
                title: group.name,
                subtitle: [group.sizeLabel, group.downloadedHint]
                    .compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · "),
                brand: group.brand,
                logoSize: 18
            ) {
                WizardCheckbox(isOn: isSelected, size: checkboxSize)
            }
            .contentShape(Rectangle())
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
        }
        .buttonStyle(.plain)
    }
}

struct QuantizationSelector: View {
    @Binding var selectedQuants: Set<QuantPill>
    var title: String = "Quantizations"
    var subtitle: String = "Specify level of quantization to download"
    var disabledQuants: Set<QuantPill> = []
    var style: QuantizationSelectorStyle = .pills
    // Whether the selection may be cleared entirely (e.g. the download sheet,
    // where "nothing selected" just disables the Download button). Contexts
    // where an empty selection is meaningless (the job wizard's quant filter)
    // pass false, so deselecting falls back to "All quants" instead.
    var allowsEmptySelection: Bool = true
    @State private var clearedSelectionForDisabledQuants = false

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text(title)
                .font(.serif(19))
            Text(subtitle)
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
            controls
        }
        .onAppear { sanitizeSelection() }
        .onChange(of: disabledQuants) { _, _ in
            sanitizeSelection()
        }
    }

    @ViewBuilder
    private var controls: some View {
        switch style {
        case .pills:
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 9) {
                    QuantPillButton(pill: .all, isSelected: selectedQuants.contains(.all)) {
                        toggleQuant(.all)
                    }
                    Divider()
                        .frame(height: 22)
                    ForEach(QuantPill.allCases.filter { $0 != .all }, id: \.self) { pill in
                        QuantPillButton(
                            pill: pill,
                            isSelected: selectedQuants.contains(pill),
                            isDisabled: disabledQuants.contains(pill)
                        ) {
                            toggleQuant(pill)
                        }
                    }
                }
                .padding(.vertical, 2)
            }
        case .list:
            ModelListCard(cornerRadius: 14) {
                ForEach(Array(QuantPill.allCases.enumerated()), id: \.element) { index, pill in
                    if index > 0 {
                        Divider()
                            .padding(.leading, 16)
                    }
                    QuantPillRow(
                        pill: pill,
                        isSelected: selectedQuants.contains(pill),
                        isDisabled: disabledQuants.contains(pill)
                    ) {
                        toggleQuant(pill)
                    }
                }
            }
        }
    }

    private func toggleQuant(_ pill: QuantPill) {
        guard !disabledQuants.contains(pill) else { return }
        let nextSelection = QuantPill.toggledSelection(
            selectedQuants,
            toggling: pill,
            disabled: disabledQuants,
            allowsEmpty: allowsEmptySelection
        )

        clearedSelectionForDisabledQuants = nextSelection.isEmpty
        selectedQuants = nextSelection
    }

    private func sanitizeSelection() {
        let previousSelection = selectedQuants
        var nextSelection = selectedQuants
        nextSelection.subtract(disabledQuants)

        if nextSelection.isEmpty && !previousSelection.isEmpty {
            clearedSelectionForDisabledQuants = true
        } else if !nextSelection.isEmpty {
            clearedSelectionForDisabledQuants = false
        }

        nextSelection = QuantPill.normalizedSelection(
            nextSelection,
            disabled: disabledQuants,
            defaultsToAll: !allowsEmptySelection || !clearedSelectionForDisabledQuants
        )

        if selectedQuants != nextSelection {
            selectedQuants = nextSelection
        }
    }
}

enum QuantizationSelectorStyle {
    case pills
    case list
}

private struct QuantPillRow: View {
    let pill: QuantPill
    let isSelected: Bool
    var isDisabled = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Text(pill.label)
                    .font(.system(size: 14))
                    .foregroundStyle(isDisabled ? Color(.systemGray2) : Color.primary)
                Spacer(minLength: 12)
                WizardCheckbox(isOn: isSelected, size: 26)
                    .opacity(isDisabled ? 0.35 : 1)
            }
            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
            .padding(.horizontal, 16)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
    }
}

private struct QuantPillButton: View {
    let pill: QuantPill
    let isSelected: Bool
    var isDisabled = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(pill.label)
                .font(.system(size: 13, weight: .medium))
                .lineLimit(1)
                .fixedSize()
                .padding(.horizontal, 14)
                .frame(height: 28)
                .background(backgroundColor, in: Capsule())
                .overlay(
                    Capsule()
                        .strokeBorder(borderColor, lineWidth: 1)
                )
                .foregroundStyle(foregroundColor)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
    }

    private var foregroundColor: Color {
        if isDisabled { return Color(.systemGray2) }
        return isSelected ? Color(.systemBackground) : Color.primary
    }

    private var backgroundColor: Color {
        if isDisabled { return Color(.secondarySystemBackground) }
        return isSelected ? Color.primary : Color.clear
    }

    private var borderColor: Color {
        if isDisabled { return Color.primary.opacity(0.06) }
        return isSelected ? Color.clear : Color.primary.opacity(0.12)
    }
}

struct ModelQuantBadge: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 14, weight: .regular))
            .foregroundStyle(.primary)
            .lineLimit(1)
            .padding(.horizontal, 11)
            .frame(height: 25)
            .background(Color(.systemBackground), in: Capsule())
            .overlay(
                Capsule()
                    .strokeBorder(Color.primary.opacity(0.10), lineWidth: 1)
            )
    }
}

struct AppTextChip: View {
    let text: String
    var font: Font = .system(size: 16)
    var height: CGFloat = 30
    var horizontalPadding: CGFloat = 14
    var width: CGFloat?
    var maxWidth: CGFloat?

    var body: some View {
        Group {
            if let width {
                content
                    .frame(width: width, height: height, alignment: .leading)
            } else {
                content
                    .frame(height: height)
                    .frame(maxWidth: maxWidth, alignment: .leading)
            }
        }
        .background(Color(.systemBackground), in: Capsule())
        .capsuleBorder()
    }

    private var content: some View {
        Text(text)
            .font(font)
            .foregroundStyle(.primary)
            .lineLimit(1)
            .truncationMode(.tail)
            .padding(.horizontal, horizontalPadding)
    }
}

struct AppModelChip: View {
    let text: String
    let brand: ModelBrand
    var logoSize: CGFloat = 20
    var font: Font = .system(size: 16)
    var height: CGFloat = 30
    var leadingPadding: CGFloat = 10
    var trailingPadding: CGFloat = 14
    var width: CGFloat?
    var maxWidth: CGFloat?

    var body: some View {
        Group {
            if let width {
                content
                    .frame(width: width, height: height, alignment: .leading)
            } else {
                content
                    .frame(height: height)
                    .frame(maxWidth: maxWidth, alignment: .leading)
            }
        }
        .background(Color(.systemBackground), in: Capsule())
        .capsuleBorder()
    }

    private var content: some View {
        HStack(spacing: 7) {
            BrandLogoView(brand: brand, size: logoSize)
            Text(text)
                .font(font)
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .padding(.leading, leadingPadding)
        .padding(.trailing, trailingPadding)
    }
}
