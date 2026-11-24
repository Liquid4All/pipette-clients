import SwiftUI

/// A model collapsed across its quant variants.
struct ModelGroup: FamilyMember {
    let key: String
    let name: String
    let paramLabel: String?
    let files: [DiscoveredModel]
    /// The catalog family, or ``CatalogEntry/defaultFamily(_:)`` of the name for a model
    /// absent from the catalog.
    let family: String
    var id: String { key }

    var brand: ModelBrand {
        ModelBrand.detect(name: name, hfRepo: files.first?.hfRepo)
    }

    var sizeLabel: String {
        paramLabel ?? files.first?.sizeFormatted ?? ""
    }

    /// What of this model is already on disk, or nil when none of it is. Marks the
    /// present, not the absent: most catalog rows are not downloaded, and a run fetches
    /// an absent model anyway. A partly downloaded model names its downloaded quants.
    var downloadedHint: String? {
        let onDisk = files.filter(\.isDownloaded)
        guard !onDisk.isEmpty else { return nil }
        if onDisk.count == files.count { return "Downloaded" }
        let quants = onDisk.compactMap(\.quant)
        guard quants.count == onDisk.count else { return "\(onDisk.count) of \(files.count) downloaded" }
        return "\(Set(quants).sorted().joined(separator: ", ")) downloaded"
    }

    var quantCountLabel: String {
        "\(files.count) quant\(files.count == 1 ? "" : "s")"
    }

    /// The group's quants come from more than one repo (e.g. Ministral Q4_0 from
    /// unsloth, K-quants from mistralai), so each quant is labelled with its source.
    var spansRepos: Bool {
        Set(files.map(\.hfRepo)).count > 1
    }
}

enum ModelCatalog {
    /// Group downloaded files by model identity, not `hf_repo`, so a model whose quants
    /// come from different repos shows as one card. Each `DiscoveredModel` keeps its own
    /// `hfRepo`, so per-cell provenance is unaffected.
    ///
    /// The groups are sorted by name, ignoring case.
    static func groups(
        from models: [DiscoveredModel],
        entries: [String: CatalogEntry] = CatalogEntry.firstEntryByModelId
    ) -> [ModelGroup] {
        // `modelId` is shared across quants and formats, so a catalog download and a
        // sideloaded copy of the same file group together, as do GGUF and MLX builds.
        return Dictionary(grouping: models, by: \.modelId).map { k, group in
            let name = entries[k]?.name
                ?? group.first?.displayName
                ?? LocalStorage.modelStem(from: group.first?.name ?? k)
            let param = LocalStorage.parseParamSize(
                name: group.first?.name ?? name,
                hfRepo: group.first?.hfRepo
            )
            return ModelGroup(key: k, name: name, paramLabel: param, files: group,
                              family: CatalogEntry.family(of: entries[k], name: name))
        }
        .sorted { $0.name == $1.name ? $0.key < $1.key : namesAscending($0.name, $1.name) }
    }

    static func displayName(for key: String) -> String {
        // `last` rather than `last!`: a key of "/" contains a slash but splits to nothing.
        let repoName = key.contains("/")
            ? key.split(separator: "/").last.map(String.init) ?? key
            : key
        return LocalStorage.repoStem(repoName)
    }

    /// Display name for a model spec: the catalog's repo name, then its model name,
    /// then a derived stem for sideloads and uncatalogued repos. The one place a `Model`
    /// spec's user-facing name is resolved.
    static func displayName(for model: Model) -> String {
        if let repo = model.repo, let name = CatalogEntry.repoToName[repo.description] { return name }
        if let name = CatalogEntry.modelIdToName[model.modelId] { return name }
        // AFM has no repo or file to derive a name from.
        if case .appleFoundationText = model { return model.engineLabel }
        // The GGUF arms prefer their repo because their references name a file
        // (`{repo}:{path}`, `{a}+{b}`), which would label as the raw coordinate. The
        // directory-shaped arms keep their reference, whose `prefix` tells same-repo
        // siblings apart. A source with no repo derives from the weights.
        let key: String = switch model {
        case let .ggufText(m): model.repo?.description ?? m.source.reference
        case let .ggufVision(m): model.repo?.description ?? m.source.weightsReference
        case let .ggufVisionSpeculative(m):
            model.repo?.description ?? m.source.vision.weightsReference
        case let .mlx(m): m.source.reference
        case let .onnxGenai(m): m.source.reference
        case .appleFoundationText: model.modelId
        }
        return displayName(for: key)
    }

    /// Model name plus quant (e.g. `LFM 2.5 350M · Q4_K_M`), never the on-disk filename.
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
    /// Shown by the Jobs list and New Job's model step; only the tap action differs.
    static func needModelsForJobs(action: @escaping () -> Void) -> EmptyModelsPrompt {
        EmptyModelsPrompt(
            message: "To create jobs, you need to first download\nmodels to select for benchmarking.",
            actionTitle: "Go to Models",
            action: action)
    }
}

typealias NoDownloadedModelsState = EmptyModelsPrompt

enum ModelRowLayout {
    static let logoSize: CGFloat = 28
    static let spacing: CGFloat = 12
}

struct ModelRow<Accessory: View>: View {
    let title: String
    let subtitle: String
    let brand: ModelBrand
    var logoDigest: String?
    var logoSize: CGFloat
    var titleFont: Font = .system(size: 16, weight: .semibold)
    var subtitleFont: Font = .system(size: 13)
    var titleLineLimit: Int = 1
    let accessory: Accessory

    init(
        title: String,
        subtitle: String,
        brand: ModelBrand,
        logoDigest: String? = nil,
        logoSize: CGFloat = ModelRowLayout.logoSize,
        titleFont: Font = .system(size: 16, weight: .semibold),
        subtitleFont: Font = .system(size: 13),
        titleLineLimit: Int = 1,
        @ViewBuilder accessory: () -> Accessory
    ) {
        self.title = title
        self.subtitle = subtitle
        self.brand = brand
        self.logoDigest = logoDigest
        self.logoSize = logoSize
        self.titleFont = titleFont
        self.subtitleFont = subtitleFont
        self.titleLineLimit = titleLineLimit
        self.accessory = accessory()
    }

    var body: some View {
        HStack(spacing: ModelRowLayout.spacing) {
            BrandLogoView(brand: brand, logoDigest: logoDigest, size: logoSize)
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
    let logoDigest: String?
    var checkboxSize: CGFloat = 22
    /// Extra leading space, so the logo lines up under a family header's logo.
    var leadingInset: CGFloat = 0
    /// Set to false when every listed model is on disk. The hint would say nothing.
    var showsDownloadedHint = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ModelRow(
                title: group.name,
                subtitle: [group.sizeLabel, showsDownloadedHint ? group.downloadedHint : nil]
                    .compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · "),
                brand: group.brand,
                logoDigest: logoDigest
            ) {
                WizardCheckbox(isOn: isSelected, size: checkboxSize)
            }
            .contentShape(Rectangle())
            .padding(.leading, 16 + leadingInset)
            .padding(.trailing, 16)
            .padding(.vertical, 14)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Model families

/// A model that a list groups under its catalog family.
protocol FamilyMember: Identifiable {
    var name: String { get }
    var family: String { get }
    var brand: ModelBrand { get }
}

/// The models of one catalog family, such as `LFM2.5`.
struct ModelFamily<Member: FamilyMember>: Identifiable {
    let name: String
    let members: [Member]
    let logoDigest: String?

    var id: String { name }

    var brand: ModelBrand { members.first?.brand ?? .unknown }

    /// Families sorted by name, ignoring case, each with its members sorted the same way.
    static func grouping(_ members: [Member], logoDigests: [String: String] = [:]) -> [ModelFamily] {
        Dictionary(grouping: members, by: \.family)
            .map { name, members in
                ModelFamily(name: name, members: members.sorted { namesAscending($0.name, $1.name) },
                            logoDigest: logoDigests[name.lowercased()])
            }
            .sorted { namesAscending($0.name, $1.name) }
    }
}

/// Case-insensitive order, with a case-sensitive tie-break so the order is stable.
private func namesAscending(_ lhs: String, _ rhs: String) -> Bool {
    let order = lhs.caseInsensitiveCompare(rhs)
    return order == .orderedSame ? lhs < rhs : order == .orderedAscending
}

/// The header of a family with more than one model. It shows, from left to right:
///
/// - the chevron that opens and closes the family
/// - the logo
/// - the name and the model count
/// - the checkbox for the whole family
///
/// The checkbox appears only when the list selects models.
struct ModelFamilyHeader: View {
    /// The width of the chevron column in the pickers. A model row has this inset.
    static let chevronWidth: CGFloat = 22
    /// The extra indent of the models in an open family. The indent shows which models
    /// belong to the family.
    static let memberIndent: CGFloat = 16

    let name: String
    let count: Int
    let brand: ModelBrand
    var logoDigest: String?
    let isOpen: Bool
    var isSelected = false
    var isMixed = false
    var chevronColumnWidth: CGFloat = ModelFamilyHeader.chevronWidth
    var logoSize: CGFloat = 28
    var titleFont: Font = .system(size: 16, weight: .semibold)
    var subtitleFont: Font = .system(size: 13)
    var checkboxSize: CGFloat = 22
    var horizontalPadding: CGFloat = 16
    var verticalPadding: CGFloat = 14
    let onToggleOpen: () -> Void
    /// Nil for a list without selection, which shows no checkbox.
    var onToggleSelection: (() -> Void)?

    var body: some View {
        HStack(spacing: 0) {
            Button(action: onToggleOpen) {
                HStack(spacing: 0) {
                    Image(systemName: "chevron.down")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .rotationEffect(.degrees(isOpen ? 0 : -90))
                        .frame(width: chevronColumnWidth, alignment: .leading)
                    ModelRow(
                        title: name,
                        subtitle: "\(count) \("model".pluralized(count))",
                        brand: brand,
                        logoDigest: logoDigest,
                        logoSize: logoSize,
                        titleFont: titleFont,
                        subtitleFont: subtitleFont
                    ) { EmptyView() }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if let onToggleSelection {
                Button(action: onToggleSelection) {
                    WizardCheckbox(isOn: isSelected, isMixed: isMixed, size: checkboxSize)
                }
                .buttonStyle(.plain)
                .padding(.leading, 12)
            }
        }
        .padding(.horizontal, horizontalPadding)
        .padding(.vertical, verticalPadding)
    }
}

struct QuantizationSelector: View {
    @Binding var selectedQuants: Set<QuantPill>
    var title: String = "Quantizations"
    var subtitle: String = "Specify level of quantization to download"
    var disabledQuants: Set<QuantPill> = []
    var style: QuantizationSelectorStyle = .pills
    // False where an empty selection is meaningless (the job wizard's quant filter),
    // so deselecting falls back to "All quants".
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
