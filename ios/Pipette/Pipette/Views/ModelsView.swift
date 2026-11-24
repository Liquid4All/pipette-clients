import Combine
import SwiftUI

/// Tab view for managing downloaded models.
struct ModelsView: View {
    @Environment(\.pillTabBarReservedHeight) private var pillTabBarReservedHeight
    @Environment(\.ws) private var ws
    @Environment(ModelStore.self) private var modelStore
    @Environment(DownloadCoordinator.self) private var downloadCoordinator

    @State private var searchText: String = ""
    @State private var expandedDownloadedKeys: Set<String> = []
    /// The families closed by hand. Families start open. A new download reopens its
    /// family. A search opens every family with a match.
    @State private var closedFamilies: Set<String> = []
    @State private var showAddModels = false
    @State private var isManagingModels = false
    @State private var selectedModelIDs: Set<String> = []
    @State private var pendingDeleteModels: [DiscoveredModel] = []
    @State private var downloadError: String?
    @State private var lastActiveDownloadModelKeys: Set<String> = []
    @State private var catalogEntries: [String: CatalogEntry] = [:]
    @State private var catalogLogoOwner: ModelShortList.Owner?
    @State private var familyLogoDigests: [String: String] = [:]

    private var models: [DiscoveredModel] { modelStore.models }

    private var downloadedKeys: Set<ModelStorageKey> {
        Set(models.compactMap { ModelStorageKey.of($0.source) })
    }

    /// Includes the built-in AFM model. The mmproj filter guards against a sideloaded
    /// bare projector; a vision model's projector is part of its own entry.
    private var displayableModelFiles: [DiscoveredModel] {
        models.filter { $0.name.range(of: "mmproj", options: .caseInsensitive) == nil }
    }

    /// The delete/manage pool: file-backed models only, so not the built-in AFM.
    private var downloadedModelFiles: [DiscoveredModel] {
        displayableModelFiles.filter { if case .appleFoundationText = $0.source { false } else { true } }
    }

    private var downloadedGroups: [ModelGroup] {
        ModelCatalog.groups(from: displayableModelFiles, entries: catalogEntries)
    }

    private var filteredDownloadedGroups: [ModelGroup] {
        let q = searchQuery
        guard !q.isEmpty else { return downloadedGroups }
        return downloadedGroups.filter { group in
            group.name.lowercased().contains(q)
                || group.key.lowercased().contains(q)
                || group.files.contains { $0.name.lowercased().contains(q) }
        }
    }

    private var filteredActiveDownloads: [DownloadCoordinator.Download] {
        let q = searchQuery
        guard !q.isEmpty else { return activeDownloads }
        return activeDownloads.filter { download in
            downloadDisplayName(for: download, entries: catalogEntries).lowercased().contains(q)
                || download.filename.lowercased().contains(q)
                || download.key.lowercased().contains(q)
                || (download.repo?.lowercased().contains(q) ?? false)
                || (download.quant?.lowercased().contains(q) ?? false)
        }
    }

    private var visibleModelRows: [ModelListRow] {
        var downloadsByModel = Dictionary(grouping: filteredActiveDownloads) { download in
            downloadModelKey(for: download)
        }
        var rows: [ModelListRow] = []
        let filteredGroupIDs = Set(filteredDownloadedGroups.map(\.id))

        for group in downloadedGroups where filteredGroupIDs.contains(group.id) || downloadsByModel[group.key] != nil {
            let downloads = downloadsByModel.removeValue(forKey: group.key) ?? []
            rows.append(.group(group, downloads: downloads))
        }

        for (key, downloads) in downloadsByModel {
            let name = downloads.first.map { downloadDisplayName(for: $0, entries: catalogEntries) } ?? key
            let entry = downloads.first?.modelId.flatMap { catalogEntries[$0] }
            rows.append(.pendingDownloads(key: key, downloads: downloads, name: name,
                                          family: CatalogEntry.family(of: entry, name: name)))
        }

        return rows
    }

    private var searchQuery: String {
        searchText.searchNormalized
    }

    private var activeDownloadKeys: [String] {
        downloadCoordinator.downloads.keys.sorted()
    }

    private var visibleModelFamilies: [ModelFamily<ModelListRow>] {
        ModelFamily.grouping(visibleModelRows, logoDigests: familyLogoDigests)
    }

    private enum ModelListRow: FamilyMember {
        case group(ModelGroup, downloads: [DownloadCoordinator.Download])
        case pendingDownloads(key: String, downloads: [DownloadCoordinator.Download], name: String,
                              family: String)

        var modelKey: String {
            switch self {
            case .group(let group, _): group.key
            case .pendingDownloads(let key, _, _, _): key
            }
        }

        var name: String {
            switch self {
            case .group(let group, _): group.name
            case .pendingDownloads(_, _, let name, _): name
            }
        }

        var family: String {
            switch self {
            case .group(let group, _):
                group.family
            case .pendingDownloads(_, _, _, let family):
                family
            }
        }

        var brand: ModelBrand {
            switch self {
            case .group(let group, _): group.brand
            case .pendingDownloads(_, let downloads, let name, _):
                ModelBrand.detect(name: name, hfRepo: downloads.first?.repo)
            }
        }

        var id: String {
            switch self {
            case .group(let group, _):
                return "group-\(group.id)"
            case .pendingDownloads(let key, _, _, _):
                return "pending-\(key)"
            }
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {
                    header
                    ModelSearchField(text: $searchText, placeholder: "Search your downloaded models")
                    if !downloadedModelFiles.isEmpty {
                        manageToolbar
                    }

                    if pausedDownloadCount > 0 {
                        resumeAllBar
                    }

                    if downloadedGroups.isEmpty && activeDownloads.isEmpty {
                        emptyModelsCard
                    } else if visibleModelRows.isEmpty {
                        noSearchResultsCard
                    } else {
                        modelsCard
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 18 + pillTabBarReservedHeight)
            }
            .environment(\.catalogLogoOwner, catalogLogoOwner)
            .refreshable {
                do {
                    try await ModelShortList.refreshExplicitly(ws: ws)
                } catch {
                    AppLog.storage.warning("models tab catalog refresh failed: error=\(error)")
                }
                refreshModels()
            }
            .background(Color(.systemBackground))
            .navigationBarHidden(true)
            .onAppear {
                reloadCatalog()
                refreshModels()
            }
            .onReceive(NotificationCenter.default.publisher(
                for: ModelShortList.didChange, object: ws.modelShortList
            ).receive(on: DispatchQueue.main)) { _ in
                reloadCatalog()
            }
            .onChange(of: ObjectIdentifier(ws.modelShortList)) { _, _ in
                reloadCatalog()
            }
            .onChange(of: downloadCoordinator.completedVersion) { _, _ in
                refreshModels()
            }
            .onChange(of: activeDownloadKeys) { _, _ in
                refreshModels()
            }
            .onChange(of: downloadCoordinator.errorMessage) { _, new in
                if let new {
                    downloadError = new
                    downloadCoordinator.errorMessage = nil
                }
            }
            .fullScreenCover(isPresented: $showAddModels) {
                AddModelsView(downloadedKeys: downloadedKeys) {
                    refreshModels()
                }
                .environment(downloadCoordinator)
            }
            .alert(
                "Download Error",
                isPresented: Binding<Bool>(
                    get: { downloadError != nil },
                    set: { if !$0 { downloadError = nil } }
                ),
                presenting: downloadError
            ) { _ in
                Button("OK", role: .cancel) {}
            } message: { message in
                Text(message)
            }
            .confirmationDialog(
                deleteConfirmationTitle,
                isPresented: Binding<Bool>(
                    get: { !pendingDeleteModels.isEmpty },
                    set: { if !$0 { pendingDeleteModels.removeAll() } }
                ),
                titleVisibility: .visible
            ) {
                Button(deleteConfirmationActionTitle, role: .destructive) {
                    deleteModels(pendingDeleteModels)
                    pendingDeleteModels.removeAll()
                }
                Button("Cancel", role: .cancel) { pendingDeleteModels.removeAll() }
            } message: {
                Text(deleteConfirmationMessage)
            }
        }
    }

    private var selectedModels: [DiscoveredModel] {
        downloadedModelFiles.filter { selectedModelIDs.contains($0.id) }
    }

    private var deleteConfirmationTitle: String {
        if pendingDeleteModels.count == 1 {
            return "Delete \(ModelCatalog.displayLabel(for: pendingDeleteModels[0]))?"
        }
        return "Delete \(pendingDeleteModels.count) model files?"
    }

    private var deleteConfirmationActionTitle: String {
        pendingDeleteModels.count == 1 ? "Delete" : "Delete \(pendingDeleteModels.count) Files"
    }

    private var deleteConfirmationMessage: String {
        if pendingDeleteModels.count == 1 {
            return "This will permanently delete the model file (\(pendingDeleteModels[0].sizeFormatted)) from this device."
        }
        return "This will permanently delete the selected model files from this device."
    }

    private var header: some View {
        HStack(alignment: .center) {
            Text("Your models")
                .pageHeaderLarge()
                .foregroundStyle(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.85)

            Spacer()

            if isManagingModels {
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isManagingModels = false
                        selectedModelIDs.removeAll()
                    }
                } label: {
                    Text("Done")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(Color(.systemBackground))
                        .padding(.horizontal, 18)
                        .frame(height: 35)
                        .background(Color.primary, in: Capsule())
                }
                .buttonStyle(.plain)
            } else {
                Button {
                    showAddModels = true
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 14, weight: .medium))
                        Text("Add models")
                            .font(.system(size: 13, weight: .medium))
                    }
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 14)
                    .frame(height: 35)
                    .background(Color(.systemBackground), in: Capsule())
                    .overlay(
                        Capsule()
                            .strokeBorder(Color.primary.opacity(0.12), lineWidth: 1)
                    )
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Add models")
            }
        }
        .padding(.top, 4)
    }

    private var manageToolbar: some View {
        HStack {
            Text(isManagingModels ? "\(selectedModelIDs.count) selected" : "\(downloadedModelFiles.count) downloaded quant\(downloadedModelFiles.count == 1 ? "" : "s")")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)

            Spacer()

            if isManagingModels {
                Button {
                    if selectedModelIDs.count == downloadedModelFiles.count {
                        selectedModelIDs.removeAll()
                    } else {
                        selectedModelIDs = Set(downloadedModelFiles.map(\.id))
                    }
                } label: {
                    Text(selectedModelIDs.count == downloadedModelFiles.count ? "Clear all" : "Select all")
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(.primary)
                        .padding(.horizontal, 16)
                        .frame(height: 44)
                        .background(Color(.systemBackground), in: Capsule())
                        .overlay(
                            Capsule()
                                .strokeBorder(Color.primary.opacity(0.12), lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
            } else {
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isManagingModels = true
                    }
                } label: {
                    Text("Manage")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.primary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Manage downloaded models")
            }
        }
        .padding(.horizontal, 4)
        .padding(.top, -2)
    }

    private var modelsCard: some View {
        ModelListCard(cornerRadius: 18) {
            ForEach(Array(visibleModelFamilies.enumerated()), id: \.element.id) { index, family in
                if index > 0 {
                    Divider()
                }
                modelFamilySection(family)
            }

            // Not `filteredDownloadedGroups`, which includes the built-in AFM row.
            if isManagingModels && !downloadedModelFiles.isEmpty {
                manageDeleteFooter
            }
        }
    }

    /// A family with one model shows the model row alone. The chevron column of the header
    /// has the width of the chevron and spacing of a model row. The models of a family have
    /// one more indent step.
    @ViewBuilder
    private func modelFamilySection(_ family: ModelFamily<ModelListRow>) -> some View {
        if family.members.count == 1, let row = family.members.first {
            modelListRow(row, logoDigest: family.logoDigest)
        } else {
            let isOpen = !searchQuery.isEmpty || !closedFamilies.contains(family.id)
            VStack(spacing: 0) {
                ModelFamilyHeader(
                    name: family.name,
                    count: family.members.count,
                    brand: family.brand,
                    logoDigest: family.logoDigest,
                    isOpen: isOpen,
                    chevronColumnWidth: downloadedRowChevronColumn,
                    titleFont: .system(size: 17, weight: .medium),
                    subtitleFont: .system(size: 14),
                    horizontalPadding: 18,
                    verticalPadding: 15,
                    onToggleOpen: { toggleFamilyOpen(family.id) }
                )
                if isOpen {
                    ForEach(family.members) { row in
                        Divider().padding(.leading, 18 + downloadedRowChevronColumn + ModelFamilyHeader.memberIndent)
                        modelListRow(row, logoDigest: family.logoDigest)
                            .padding(.leading, ModelFamilyHeader.memberIndent)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func modelListRow(_ row: ModelListRow, logoDigest: String?) -> some View {
        switch row {
        case .group(let group, let downloads):
            DownloadedModelGroupBlock(
                group: group,
                logoDigest: logoDigest,
                activeDownloads: downloads,
                isExpanded: expandedDownloadedKeys.contains(group.key),
                isManaging: isManagingModels,
                selectedModelIDs: selectedModelIDs,
                onToggle: { toggleExpanded(group.key) },
                onToggleModelSelection: toggleModelSelection,
                onDeleteModels: { pendingDeleteModels = $0 }
            )
        case .pendingDownloads(_, let downloads, _, _):
            PendingDownloadGroupBlock(downloads: downloads, logoDigest: logoDigest)
                .environment(downloadCoordinator)
        }
    }

    private var manageDeleteFooter: some View {
        VStack(spacing: 10) {
            Divider()
            Button {
                pendingDeleteModels = selectedModels
            } label: {
                Text(selectedModelIDs.isEmpty ? "Select quants to delete" : "Delete \(selectedModelIDs.count) quant\(selectedModelIDs.count == 1 ? "" : "s")")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(height: 44)
                    .background(selectedModelIDs.isEmpty ? Color(.systemGray3) : Color.red, in: Capsule())
            }
            .buttonStyle(.plain)
            .disabled(selectedModelIDs.isEmpty)
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
        }
        .background(Color(.systemBackground))
    }

    private var emptyModelsCard: some View {
        VStack(spacing: 16) {
            Spacer(minLength: 0)
            EmptyModelsPrompt(
                message: "No models downloaded. Download\nmodels to select for a benchmarking job.",
                actionTitle: "Search models",
                actionSystemImage: "magnifyingglass"
            ) {
                showAddModels = true
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity)
        .frame(minHeight: 420)
        .padding(.vertical, 34)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.10), lineWidth: 1)
        )
    }

    private var noSearchResultsCard: some View {
        VStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 28, weight: .light))
                .foregroundStyle(.tertiary)
            Text("No matching models")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Try a different model name or quant.")
                .font(.subheadline)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity)
        .frame(minHeight: 220)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(Color.primary.opacity(0.10), lineWidth: 1)
        )
    }

    private var pausedDownloadCount: Int {
        downloadCoordinator.downloads.values.reduce(into: 0) { count, download in
            if case .paused = download.state { count += 1 }
        }
    }

    /// Manual fallback for paused downloads; most resume on their own on reconnect.
    @ViewBuilder private var resumeAllBar: some View {
        Button {
            downloadCoordinator.resumeAll()
        } label: {
            Label("Resume \(pausedDownloadCount) paused download\(pausedDownloadCount == 1 ? "" : "s")",
                  systemImage: "arrow.clockwise")
                .frame(maxWidth: .infinity)
                .padding(.vertical, 4)
        }
        .buttonStyle(.bordered)
    }

    private var activeDownloads: [DownloadCoordinator.Download] {
        downloadCoordinator.downloads.values.sorted { lhs, rhs in
            let lhsRank = downloadStateSortRank(lhs.state)
            let rhsRank = downloadStateSortRank(rhs.state)
            if lhsRank != rhsRank {
                return lhsRank < rhsRank
            }
            return lhs.filename.localizedCaseInsensitiveCompare(rhs.filename) == .orderedAscending
        }
    }

    private func downloadStateSortRank(_ state: DownloadCoordinator.Download.State) -> Int {
        switch state {
        case .queued, .connecting, .downloading, .resuming: return 0
        case .pausing, .paused, .cancelling: return 1
        case .failed: return 2
        }
    }

    private func downloadModelKey(for download: DownloadCoordinator.Download) -> String {
        download.modelId ?? LocalStorage.normalizedModelStem(download.filename)
    }

    private func reloadCatalog() {
        let snapshot = ModelShortList.snapshot(ws: ws)
        catalogEntries = CatalogEntry.firstEntries(in: snapshot.entries)
        catalogLogoOwner = snapshot.owner
        familyLogoDigests = CatalogEntry.familyLogoDigests(in: snapshot.entries)
    }

    private func refreshModels() {
        modelStore.reload()
        reconcileListUIState()
    }

    /// Prune selection and expansion against the current models, and reveal models
    /// that have newly started downloads.
    private func reconcileListUIState() {
        let activeModelKeys = Set(activeDownloads.map(downloadModelKey))
        let modelKeysToReveal = lastActiveDownloadModelKeys.union(activeModelKeys)

        let currentKeys = Set(downloadedGroups.map(\.key))
        expandedDownloadedKeys.formIntersection(currentKeys)
        expandedDownloadedKeys.formUnion(modelKeysToReveal.intersection(currentKeys))
        selectedModelIDs.formIntersection(Set(downloadedModelFiles.map(\.id)))
        // Not `downloadedGroups`: its built-in AFM row would keep Manage mode alive
        // after the last real model is deleted.
        if downloadedModelFiles.isEmpty {
            isManagingModels = false
            selectedModelIDs.removeAll()
        }
        if expandedDownloadedKeys.isEmpty, let first = downloadedGroups.first {
            expandedDownloadedKeys.insert(first.key)
        }
        let newDownloadKeys = activeModelKeys.subtracting(lastActiveDownloadModelKeys)
        for family in visibleModelFamilies
        where family.members.contains(where: { newDownloadKeys.contains($0.modelKey) }) {
            closedFamilies.remove(family.id)
        }

        lastActiveDownloadModelKeys = activeModelKeys
    }

    private func toggleFamilyOpen(_ id: String) {
        withAnimation(.easeInOut(duration: 0.18)) {
            if closedFamilies.remove(id) == nil { closedFamilies.insert(id) }
        }
    }

    private func toggleExpanded(_ key: String) {
        withAnimation(.easeInOut(duration: 0.18)) {
            if expandedDownloadedKeys.contains(key) {
                expandedDownloadedKeys.remove(key)
            } else {
                expandedDownloadedKeys.insert(key)
            }
        }
    }

    private func toggleModelSelection(_ model: DiscoveredModel) {
        withAnimation(.easeInOut(duration: 0.12)) {
            if selectedModelIDs.contains(model.id) {
                selectedModelIDs.remove(model.id)
            } else {
                selectedModelIDs.insert(model.id)
            }
        }
    }

    private func deleteModels(_ models: [DiscoveredModel]) {
        modelStore.delete(models)
        reconcileListUIState()
    }
}

// MARK: - Models UI

/// The chevron of a downloaded model row and the space after it. A family header and a
/// row for a model still downloading use this width as well, so all logos line up.
private let downloadedRowChevronWidth: CGFloat = 18
private let downloadedRowChevronSpacing: CGFloat = 10
private let downloadedRowChevronColumn = downloadedRowChevronWidth + downloadedRowChevronSpacing

private struct DownloadedModelGroupBlock: View {
    let group: ModelGroup
    let logoDigest: String?
    let activeDownloads: [DownloadCoordinator.Download]
    let isExpanded: Bool
    let isManaging: Bool
    let selectedModelIDs: Set<String>
    let onToggle: () -> Void
    let onToggleModelSelection: (DiscoveredModel) -> Void
    let onDeleteModels: ([DiscoveredModel]) -> Void

    /// A built-in system model (AFM): no file, so no size, no delete, no quant rows.
    private var isBuiltIn: Bool {
        group.files.contains { if case .appleFoundationText = $0.source { true } else { false } }
    }

    private var subtitle: String {
        // "Built in" is already the trailing badge, so the subtitle says something else.
        if isBuiltIn { return "Apple Intelligence" }
        return group.files.first?.sizeFormatted ?? group.sizeLabel
    }

    var body: some View {
        VStack(spacing: 0) {
            Button(action: { if !isBuiltIn { onToggle() } }) {
                HStack(spacing: downloadedRowChevronSpacing) {
                    Image(systemName: isBuiltIn ? "sparkles" : (isExpanded ? "chevron.down" : "chevron.right"))
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(.secondary)
                        .frame(width: downloadedRowChevronWidth)

                    ModelRow(
                        title: group.name,
                        subtitle: subtitle,
                        brand: group.brand,
                        logoDigest: logoDigest,
                        titleFont: .system(size: 17, weight: .medium),
                        subtitleFont: .system(size: 14),
                        titleLineLimit: 2
                    ) {
                        if isBuiltIn {
                            Text("Built in")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 11)
                                .frame(height: 25)
                                .background(Color(.secondarySystemBackground), in: Capsule())
                        } else {
                            Text(group.quantCountLabel)
                                .font(.system(size: 14))
                                .foregroundStyle(.primary)
                                .padding(.horizontal, 11)
                                .frame(height: 25)
                                .background(Color(.secondarySystemBackground), in: Capsule())
                        }
                    }
                }
                .contentShape(Rectangle())
                .padding(.horizontal, 18)
                .padding(.vertical, 15)
            }
            .buttonStyle(.plain)
            .disabled(isBuiltIn)
            .contextMenu {
                if !isBuiltIn {
                    Button(role: .destructive) {
                        onDeleteModels(group.files)
                    } label: {
                        Label("Delete All", systemImage: "trash")
                    }
                }
            }

            if !activeDownloads.isEmpty {
                Divider()
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(DownloadTransferGroup.groups(activeDownloads)) { transfer in
                        DownloadProgressSubrow(transfer: transfer)
                    }
                }
                .padding(.vertical, 7)
            }

            if isExpanded && !isBuiltIn {
                Divider()
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(sortedFiles) { model in
                        Button {
                            if isManaging {
                                onToggleModelSelection(model)
                            }
                        } label: {
                            HStack(spacing: 10) {
                                ModelQuantBadge(text: model.quant ?? "unknown")
                                Divider()
                                    .frame(height: 18)
                                Text(model.sizeFormatted)
                                    .font(.system(size: 14))
                                    .foregroundStyle(.secondary)
                                if group.spansRepos,
                                   let owner = model.hfRepo.split(separator: "/").first {
                                    Divider()
                                        .frame(height: 18)
                                    Text(String(owner))
                                        .font(.system(size: 13))
                                        .foregroundStyle(.secondary)
                                }
                                Spacer(minLength: 0)
                                if isManaging {
                                    WizardCheckbox(isOn: selectedModelIDs.contains(model.id), size: 22)
                                }
                            }
                            .frame(maxWidth: .infinity, minHeight: 34, alignment: .leading)
                            .padding(.leading, 58)
                            .padding(.trailing, 16)
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button(role: .destructive) {
                                onDeleteModels([model])
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                        .padding(.vertical, 7)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 9)
            }
        }
    }

    private var sortedFiles: [DiscoveredModel] {
        group.files.sorted { lhs, rhs in
            (lhs.quant ?? lhs.name) < (rhs.quant ?? rhs.name)
        }
    }
}

private func downloadDisplayName(
    for download: DownloadCoordinator.Download,
    entries: [String: CatalogEntry] = CatalogEntry.firstEntryByModelId
) -> String {
    if let modelId = download.modelId,
       let name = entries[modelId]?.name {
        return name
    }
    return LocalStorage.modelStem(from: download.filename)
}

private struct PendingDownloadGroupBlock: View {
    let downloads: [DownloadCoordinator.Download]
    let logoDigest: String?

    private var firstDownload: DownloadCoordinator.Download? {
        downloads.first
    }

    var body: some View {
        VStack(spacing: 0) {
            ModelRow(
                title: title,
                subtitle: subtitle,
                brand: brand,
                logoDigest: logoDigest,
                titleFont: .system(size: 17, weight: .medium),
                subtitleFont: .system(size: 14),
                titleLineLimit: 2
            ) {
                Text(quantCountLabel)
                    .font(.system(size: 14))
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 11)
                    .frame(height: 25)
                    .background(Color(.secondarySystemBackground), in: Capsule())
            }
            .padding(.leading, 18 + downloadedRowChevronColumn)
            .padding(.trailing, 18)
            .padding(.vertical, 15)

            Divider()

            VStack(alignment: .leading, spacing: 0) {
                ForEach(transfers) { transfer in
                    DownloadProgressSubrow(transfer: transfer)
                }
            }
            .padding(.vertical, 7)
        }
    }

    private var title: String {
        firstDownload.map { downloadDisplayName(for: $0) } ?? "Model"
    }

    private var subtitle: String {
        "Downloading"
    }

    private var transfers: [DownloadTransferGroup] {
        DownloadTransferGroup.groups(downloads)
    }

    private var quantCountLabel: String {
        "\(transfers.count) quant\(transfers.count == 1 ? "" : "s")"
    }

    private var brand: ModelBrand {
        guard let firstDownload else { return .unknown }
        return ModelBrand.detect(name: title, hfRepo: firstDownload.repo)
    }
}

/// One model's transfers: one file, or the weights, projector and draft files of a vision
/// or audio model. Each file downloads as a separate transfer.
struct DownloadTransferGroup: Identifiable {
    /// The pause or resume control that the row shows for its files.
    enum Control: Equatable { case pause, resume, none }

    let id: String
    /// The weights first, then the rest in their download order.
    let files: [DownloadCoordinator.Download]
    /// The file whose state the badge shows. Priority order: failed, then stopping or
    /// active, then queued or paused.
    let lead: DownloadCoordinator.Download

    /// The bytes of every file. While a file has no known size, the lead file's progress.
    var progress: Double? {
        guard files.count > 1 else { return lead.progress }
        let totals = files.compactMap(\.totalBytes)
        guard totals.count == files.count, totals.allSatisfy({ $0 > 0 }) else { return lead.progress }
        return Double(files.reduce(0) { $0 + $1.bytesDownloaded }) / Double(totals.reduce(0, +))
    }

    /// Pause while a file transfers, else resume while a file is paused. A failed file
    /// does not hide the control of the other files.
    var control: Control {
        guard case .file = lead.kind else { return .none }
        if files.contains(where: { Self.isActive($0.state) }) { return .pause }
        if files.contains(where: { $0.state == .paused }) { return .resume }
        return .none
    }

    static func isActive(_ state: DownloadCoordinator.Download.State) -> Bool {
        switch state {
        case .connecting, .downloading, .resuming: true
        case .queued, .pausing, .paused, .cancelling, .failed: false
        }
    }

    /// Groups transfers by their model coordinate, in first-seen order.
    static func groups(_ downloads: [DownloadCoordinator.Download]) -> [DownloadTransferGroup] {
        var order: [String] = []
        var byModel: [String: [DownloadCoordinator.Download]] = [:]
        for download in downloads {
            let key = download.source?.description ?? download.id
            if byModel[key] == nil { order.append(key) }
            byModel[key, default: []].append(download)
        }
        return order.compactMap { key in
            guard let files = byModel[key],
                  let lead = files.min(by: { rank($0.state) < rank($1.state) }) else { return nil }
            let weightsFirst = files.filter { $0.fileRole == .weights } + files.filter { $0.fileRole != .weights }
            return DownloadTransferGroup(id: key, files: weightsFirst, lead: lead)
        }
    }

    private static func rank(_ state: DownloadCoordinator.Download.State) -> Int {
        switch state {
        case .failed: 0
        case .cancelling: 1
        case .pausing: 2
        case .downloading: 3
        case .resuming: 4
        case .connecting: 5
        case .queued: 6
        case .paused: 7
        }
    }
}

extension DownloadCoordinator.Download {
    /// The part of a vision or audio model that a file holds.
    enum FileRole: Equatable {
        case weights, projector, draft

        var label: String {
            switch self {
            case .weights: "Weights"
            case .projector: "Projector"
            case .draft: "Draft"
            }
        }
    }

    /// The file's part of a vision or audio model, or nil for a model with one file.
    var fileRole: FileRole? {
        func leaf(_ reference: String) -> String {
            reference.split(whereSeparator: { $0 == "/" || $0 == ":" }).last.map(String.init) ?? reference
        }
        switch source {
        case let .ggufVision(m):
            return leaf(m.source.weightsReference) == filename ? .weights : .projector
        case let .ggufVisionSpeculative(m):
            if leaf(m.source.vision.weightsReference) == filename { return .weights }
            if let draft = m.source.draftReference, leaf(draft) == filename { return .draft }
            return .projector
        case .ggufText, .mlx, .onnxGenai, .appleFoundationText, nil:
            return nil
        }
    }
}

private struct DownloadProgressSubrow: View {
    @Environment(DownloadCoordinator.self) private var downloadCoordinator
    let transfer: DownloadTransferGroup

    private var lead: DownloadCoordinator.Download { transfer.lead }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                ModelQuantBadge(text: lead.quant ?? "unknown")
                    .fixedSize(horizontal: true, vertical: false)

                DownloadStateBadge(state: lead.state, progress: transfer.progress)
                    .fixedSize(horizontal: true, vertical: false)

                Spacer(minLength: 8)

                downloadControls
            }

            if transfer.files.count == 1 {
                if let detail = statusDetail(lead) {
                    Text(detail)
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                DownloadProgressTrack(progress: lead.progress, state: lead.state)
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(transfer.files) { file in
                        VStack(alignment: .leading, spacing: 3) {
                            HStack(spacing: 8) {
                                Text(file.fileRole?.label ?? file.filename)
                                    .font(.system(size: 12, weight: .medium))
                                Spacer(minLength: 8)
                                if let detail = statusDetail(file) {
                                    Text(detail)
                                        .lineLimit(1)
                                        .truncationMode(.middle)
                                }
                            }
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                            DownloadProgressTrack(progress: file.progress, state: file.state)
                        }
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, minHeight: 34, alignment: .leading)
        .padding(.leading, 58)
        .padding(.trailing, 16)
        .padding(.vertical, 9)
    }

    /// One set of controls for all of the model's files.
    @ViewBuilder
    private var downloadControls: some View {
        HStack(spacing: 4) {
            // `.directory` pulls go through HubApi, which cannot pause or resume.
            switch transfer.control {
            case .pause:
                DownloadIconButton(
                    systemName: "pause.fill",
                    accessibilityLabel: "Pause \(lead.filename)"
                ) {
                    // Model-level pause when possible, so a queued file does not start in
                    // the slot that the pause frees.
                    if let source = lead.source {
                        downloadCoordinator.pauseTransfers(for: source)
                    } else {
                        for file in transfer.files where DownloadTransferGroup.isActive(file.state) {
                            downloadCoordinator.pause(key: file.id)
                        }
                    }
                }
            case .resume:
                DownloadIconButton(
                    systemName: "play.fill",
                    accessibilityLabel: "Resume \(lead.filename)"
                ) {
                    for file in transfer.files where file.state == .paused {
                        downloadCoordinator.resume(key: file.id)
                    }
                }
            case .none:
                EmptyView()
            }

            DownloadIconButton(
                systemName: "xmark",
                tint: .red,
                isDestructive: true,
                accessibilityLabel: "Cancel \(lead.filename)"
            ) {
                // Model-level cancel when possible, so a caller waiting on these weights
                // fails now rather than a poll later, and every file of the model goes too.
                if let source = lead.source {
                    downloadCoordinator.cancelTransfers(for: source)
                } else {
                    for file in transfer.files { downloadCoordinator.cancel(key: file.id) }
                }
            }
        }
    }

    /// Nil when the badge alone says everything (a plain `.paused`).
    private func statusDetail(_ download: DownloadCoordinator.Download) -> String? {
        switch download.state {
        case .queued:
            return "Queued"
        case .connecting:
            return "Connecting…"
        case .downloading:
            if let total = download.totalBytes, total > 0 {
                return DownloadByteFormat.progress(written: download.bytesDownloaded, total: total)
            }
            if download.bytesDownloaded > 0 {
                return "\(DownloadByteFormat.byteString(download.bytesDownloaded)) downloaded"
            }
            if let fraction = download.explicitFraction {
                return "Downloading… \(Int((fraction * 100).rounded()))%"
            }
            return "Downloading…"
        case .cancelling:
            return "Cancelling…"
        case .pausing:
            return "Pausing…"
        case .paused:
            return download.interruptedByNetwork ? "Waiting for network…" : nil
        case .resuming:
            return "Resuming…"
        case .failed(let reason):
            return reason
        }
    }
}

/// Byte sizes for the download detail line: decimal units, two decimals, GB from
/// 1 GB else MB, matching `ByteFormat.fileSize`.
private enum DownloadByteFormat {
    private enum Unit {
        case gb, mb
        var divisor: Double { self == .gb ? 1_000_000_000 : 1_000_000 }
        var suffix: String { self == .gb ? "GB" : "MB" }
    }

    /// e.g. `1.20 GB`, `512.00 MB`.
    static func byteString(_ bytes: Int64) -> String {
        format(bytes, in: bytes >= 1_000_000_000 ? .gb : .mb)
    }

    /// Both sides in the total's unit, e.g. `2.34 / 5.60 GB`.
    static func progress(written: Int64, total: Int64) -> String {
        let unit: Unit = total >= 1_000_000_000 ? .gb : .mb
        return "\(format(written, in: unit, withSuffix: false)) / \(format(total, in: unit))"
    }

    private static func format(_ bytes: Int64, in unit: Unit, withSuffix: Bool = true) -> String {
        let value = String(format: "%.2f", Double(bytes) / unit.divisor)
        return withSuffix ? "\(value) \(unit.suffix)" : value
    }
}

private struct DownloadStateBadge: View {
    let state: DownloadCoordinator.Download.State
    let progress: Double?

    var body: some View {
        Text(label)
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(foregroundColor)
            .lineLimit(1)
            .padding(.horizontal, 10)
            .frame(height: 25)
            .background(backgroundColor, in: Capsule())
    }

    private var label: String {
        switch state {
        case .queued:
            return "Queued"
        case .connecting:
            return "Starting"
        case .downloading, .resuming:
            guard let progress else { return "Starting" }
            return "\(Int((progress * 100).rounded(.down)))%"
        case .pausing, .paused:
            return "Paused"
        case .cancelling:
            return "Cancelling"
        case .failed:
            return "Failed"
        }
    }

    private var foregroundColor: Color {
        switch state {
        case .queued, .connecting, .downloading, .pausing, .paused, .cancelling,
             .resuming: return .primary
        case .failed: return .red
        }
    }

    private var backgroundColor: Color {
        switch state {
        case .queued, .connecting, .downloading, .pausing, .paused, .cancelling,
             .resuming: return Color(.secondarySystemBackground)
        case .failed: return Color.red.opacity(0.10)
        }
    }
}

private struct DownloadIconButton: View {
    let systemName: String
    var tint: Color = .primary
    var isDestructive = false
    let accessibilityLabel: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 44, height: 44)
                .background(Color(.systemBackground), in: Circle())
                .overlay(
                    Circle()
                        .strokeBorder(borderColor, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
        .contentShape(Circle())
        .accessibilityLabel(accessibilityLabel)
    }

    private var borderColor: Color {
        isDestructive ? Color.red.opacity(0.24) : Color.primary.opacity(0.12)
    }
}

private struct DownloadProgressTrack: View {
    let progress: Double?
    let state: DownloadCoordinator.Download.State

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.primary.opacity(0.08))

                Capsule()
                    .fill(fillColor)
                    .frame(width: progressWidth(in: geo.size.width))
            }
        }
        .frame(height: 5)
    }

    private var isActive: Bool {
        switch state {
        case .connecting, .downloading, .resuming: return true
        case .queued, .pausing, .paused, .cancelling, .failed: return false
        }
    }

    private var fillColor: Color {
        switch state {
        case .connecting, .downloading, .resuming: return .primary
        case .queued, .pausing, .paused, .cancelling: return Color.primary.opacity(0.35)
        case .failed: return .red
        }
    }

    private func progressWidth(in width: CGFloat) -> CGFloat {
        guard let progress else {
            return isActive ? max(18, width * 0.16) : 0
        }
        let clamped = CGFloat(min(max(progress, 0), 1))
        return max(clamped > 0 ? 6 : 0, width * clamped)
    }
}

private struct AddModelsView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.ws) private var ws
    @Environment(DownloadCoordinator.self) private var downloadCoordinator

    let downloadedKeys: Set<ModelStorageKey>
    let onDownloadStarted: () -> Void

    /// The offered models within the device's size limits. The app refreshes these offers
    /// when you open the sheet or pull down.
    @State private var catalog: [CatalogEntry] = []
    @State private var catalogLogoOwner: ModelShortList.Owner?
    @State private var familyLogoDigests: [String: String] = [:]
    @State private var searchText = ""
    @State private var expandedModelIDs: Set<String> = []
    /// The families opened by hand. A search shows every family with a match open as well.
    @State private var openFamilies: Set<String> = []
    /// All offers for the same artifact share one selection.
    @State private var selectedKeys: Set<ModelStorageKey> = []
    @State private var pendingDownload: (entries: [CatalogEntry], warning: CatalogEntry.DownloadWarning)?

    private let largeDownloadThreshold: Int64 = 200 * 1024 * 1024

    private var families: [CatalogFamily] {
        let query = searchText.searchNormalized
        return ModelFamily.grouping(
            PresetModelGroup.groups(from: catalog).filter { $0.matches(query) }, logoDigests: familyLogoDigests)
    }

    private var unavailableDownloadKeys: Set<ModelStorageKey> {
        downloadedKeys.union(downloadCoordinator.downloads.values.compactMap(\.storageKey))
    }

    private var allAvailableKeys: Set<ModelStorageKey> {
        Set(catalog.compactMap(\.storageKey)).subtracting(unavailableDownloadKeys)
    }

    private var selectedDownloadPresets: [CatalogEntry] {
        CatalogEntry.selectedDownloads(catalog, selected: selectedKeys, unavailable: unavailableDownloadKeys)
    }

    var body: some View {
        VStack(spacing: 0) {
            addHeader
            VStack(alignment: .leading, spacing: 18) {
                ModelSearchField(text: $searchText, placeholder: "Search models")
                presetList
                    .refreshable { await refreshCatalog() }
                    .frame(maxHeight: .infinity)
            }
            .padding(.horizontal, 24)
            .padding(.top, 18)
            .padding(.bottom, 22)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .scrollDismissesKeyboard(.interactively)
            .background(Color(.systemBackground))

            Divider()
            downloadFooter
        }
        .environment(\.catalogLogoOwner, catalogLogoOwner)
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .task(id: ObjectIdentifier(ws.modelShortList)) { await refreshCatalog() }
        .onReceive(NotificationCenter.default.publisher(
            for: ModelShortList.didChange, object: ws.modelShortList
        ).receive(on: DispatchQueue.main)) { _ in
            reloadCatalog()
        }
        .alert("Confirm download", isPresented: Binding(
            get: { pendingDownload != nil },
            set: { if !$0 { pendingDownload = nil } }
        ), presenting: pendingDownload) { pending in
            Button("Cancel", role: .cancel) {}
            Button("Download") { startDownload(pending.entries) }
        } message: { pending in
            switch pending.warning {
            case .unknownSize:
                Text("The size of at least one selected model is unknown. The download may be large.")
            case .largeDownload:
                Text("The combined download is over 200 MiB.")
            }
        }
    }

    private var addHeader: some View {
        VStack(spacing: 0) {
            HStack {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 15, weight: .regular))
                        .foregroundStyle(.primary)
                        .frame(width: 44, height: 44, alignment: .leading)
                }
                .buttonStyle(.plain)

                Spacer()
                Text("Download models")
                    .pageHeaderSmall()
                Spacer()
                Color.clear
                    .frame(width: 44, height: 44)
            }
            .padding(.horizontal, 24)
            .padding(.top, 12)
            .frame(height: 62)

            Divider()
        }
    }

    private func reloadCatalog() {
        let snapshot = ModelShortList.snapshot(ws: ws)
        catalog = CatalogEntry.fitting(snapshot.entries, quotaBytes: ws.storageQuotaBytes)
        catalogLogoOwner = snapshot.owner
        familyLogoDigests = CatalogEntry.familyLogoDigests(in: snapshot.entries)
    }

    private func refreshCatalog() async {
        // The current list first, so the sheet does not wait on the network.
        reloadCatalog()
        do {
            try await ModelShortList.refresh(ws: ws)
        } catch {
            AppLog.storage.warning("add-models catalog refresh failed: error=\(error)")
        }
        reloadCatalog()
    }

    private var presetList: some View {
        ModelListCard(cornerRadius: 16) {
            ScrollView {
                LazyVStack(spacing: 0) {
                    if families.isEmpty {
                        Text("No models match \"\(searchText)\".")
                            .font(.system(size: 15))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 34)
                    } else {
                        ForEach(Array(families.enumerated()), id: \.element.id) { index, family in
                            if index > 0 {
                                Divider().padding(.leading, 16)
                            }
                            familySection(family)
                        }
                    }
                }
            }
            .scrollDismissesKeyboard(.interactively)
        }
    }

    /// A family with one model shows the model row alone.
    @ViewBuilder
    private func familySection(_ family: CatalogFamily) -> some View {
        if family.members.count == 1, let model = family.members.first {
            presetRow(model, logoDigest: family.logoDigest)
        } else {
            let isOpen = isFamilyOpen(family)
            VStack(spacing: 0) {
                familyHeader(family, isOpen: isOpen)
                if isOpen {
                    ForEach(family.members) { model in
                        Divider().padding(.leading, 16 + ModelFamilyHeader.chevronWidth + ModelFamilyHeader.memberIndent)
                        presetRow(model, logoDigest: family.logoDigest)
                            .padding(.leading, ModelFamilyHeader.memberIndent)
                    }
                }
            }
        }
    }

    private func familyHeader(_ family: CatalogFamily, isOpen: Bool) -> some View {
        let available = availableKeys(in: family)
        let allSelected = !available.isEmpty && available.isSubset(of: selectedKeys)
        return ModelFamilyHeader(
            name: family.name,
            count: family.members.count,
            brand: family.brand,
            logoDigest: family.logoDigest,
            isOpen: isOpen,
            isSelected: allSelected,
            isMixed: !allSelected && !available.isDisjoint(with: selectedKeys),
            titleFont: .system(size: 17, weight: .semibold),
            subtitleFont: .system(size: 14),
            checkboxSize: 22,
            horizontalPadding: 18,
            verticalPadding: 15,
            onToggleOpen: { toggleFamilyOpen(family.id) },
            onToggleSelection: { toggleFamily(family) }
        )
    }

    private func presetRow(_ group: PresetModelGroup, logoDigest: String?) -> some View {
        let isExpanded = expandedModelIDs.contains(group.id)
        let rowLeading = 18 + ModelFamilyHeader.chevronWidth
        return VStack(spacing: 0) {
            HStack(spacing: 0) {
                Button { toggleExpanded(group.id) } label: {
                    ModelRow(
                        title: group.name,
                        subtitle: quantSummary(group),
                        brand: group.brand,
                        logoDigest: logoDigest,
                        titleFont: .system(size: 17, weight: .semibold),
                        subtitleFont: .system(size: 14),
                        titleLineLimit: 2
                    ) {
                        Image(systemName: "chevron.down")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .rotationEffect(.degrees(isExpanded ? 0 : -90))
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                Button { toggleAllQuants(group) } label: {
                    WizardCheckbox(isOn: allQuantsSelected(group),
                                   isMixed: !allQuantsSelected(group)
                                       && !availableKeys(in: group).isDisjoint(with: selectedKeys),
                                   size: 22)
                }
                .buttonStyle(.plain)
                .padding(.leading, 12)
            }
            .padding(.leading, rowLeading)
            .padding(.trailing, 18)
            .padding(.vertical, 15)

            if isExpanded {
                quantChips(group)
                    .padding(.leading, rowLeading + ModelRowLayout.logoSize + ModelRowLayout.spacing)
                    .padding(.trailing, 18)
                    .padding(.bottom, 14)
            }
        }
    }

    private func quantChips(_ group: PresetModelGroup) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(group.quantSections, id: \.sourceLabel) { section in
                VStack(alignment: .leading, spacing: 7) {
                    Text(section.sourceLabel)
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    ChipFlowLayout(horizontalSpacing: 8, verticalSpacing: 8) {
                        ForEach(section.entries, id: \.offerID) { quantChip($0) }
                    }
                }
            }
        }
    }

    private func quantChip(_ entry: CatalogEntry) -> some View {
        let unavailable = isUnavailableForDownload(entry)
        let selected = entry.storageKey.map(selectedKeys.contains) ?? false
        return Button { toggleEntry(entry) } label: {
            HStack(spacing: 5) {
                if unavailable {
                    Image(systemName: "checkmark").font(.system(size: 10, weight: .bold))
                }
                Text(entry.quant ?? "—").font(.system(size: 13, weight: .medium))
                if entry.sizeBytes > 0 {
                    Text(ByteFormat.fileSize(entry.sizeBytes))
                        .font(.system(size: 12, weight: .regular))
                        .opacity(0.7)
                }
            }
            .padding(.horizontal, 12)
            .frame(height: 32)
            .background(chipFill(selected: selected, unavailable: unavailable), in: Capsule())
            .overlay(Capsule().strokeBorder(chipStroke(selected: selected, unavailable: unavailable), lineWidth: 1))
            .foregroundStyle(chipText(selected: selected, unavailable: unavailable))
        }
        .buttonStyle(.plain)
        .disabled(unavailable)
    }

    private func chipFill(selected: Bool, unavailable: Bool) -> Color {
        if unavailable { return Color(.secondarySystemBackground) }
        return selected ? Color.primary : Color(.systemBackground)
    }

    private func chipStroke(selected: Bool, unavailable: Bool) -> Color {
        (unavailable || selected) ? .clear : Color.primary.opacity(0.18)
    }

    private func chipText(selected: Bool, unavailable: Bool) -> Color {
        if unavailable { return .secondary }
        return selected ? Color(.systemBackground) : .primary
    }

    /// The selected count once anything is picked, else the available formats.
    private func quantSummary(_ group: PresetModelGroup) -> String {
        let selected = availableKeys(in: group).intersection(selectedKeys).count
        if selected > 0 { return "\(selected) selected" }
        var formats: [String] = []
        for variant in group.variants {
            let label: String
            switch variant.source {
            case .mlx: label = "MLX"
            case .onnxGenai: label = "ONNX"
            case .ggufText, .ggufVision, .ggufVisionSpeculative: label = "GGUF"
            case .appleFoundationText: label = "Apple Foundation"
            }
            if !formats.contains(label) { formats.append(label) }
        }
        return formats.joined(separator: " · ")
    }

    private var downloadFooter: some View {
        Button {
            attemptDownload()
        } label: {
            HStack(spacing: 9) {
                Image(systemName: "arrow.down.to.line")
                    .font(.system(size: 13, weight: .medium))
                Text("Download \(selectedDownloadPresets.count) models")
                    .font(.system(size: 16, weight: .medium))
            }
            .foregroundStyle(Color(.systemBackground))
            .frame(maxWidth: .infinity)
            .frame(height: 42)
            .background(
                selectedDownloadPresets.isEmpty ? Color(.systemGray3) : Color.primary,
                in: Capsule()
            )
        }
        .buttonStyle(.plain)
        .disabled(selectedDownloadPresets.isEmpty)
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
    }

    private func attemptDownload() {
        let entries = selectedDownloadPresets
        guard !entries.isEmpty else { return }
        if let warning = CatalogEntry.downloadWarning(for: entries, largeDownloadThreshold: largeDownloadThreshold) {
            pendingDownload = (entries, warning)
        } else {
            startDownload(entries)
        }
    }

    private func isFamilyOpen(_ family: CatalogFamily) -> Bool {
        !searchText.searchNormalized.isEmpty || openFamilies.contains(family.id)
    }

    private func toggleFamilyOpen(_ id: String) {
        if openFamilies.remove(id) == nil { openFamilies.insert(id) }
    }

    private func toggleExpanded(_ id: String) {
        if expandedModelIDs.contains(id) { expandedModelIDs.remove(id) } else { expandedModelIDs.insert(id) }
    }

    private func toggleEntry(_ entry: CatalogEntry) {
        guard let key = entry.storageKey, !unavailableDownloadKeys.contains(key) else { return }
        if selectedKeys.remove(key) == nil {
            selectedKeys.insert(key)
        }
    }

    private func availableKeys(in group: PresetModelGroup) -> Set<ModelStorageKey> {
        Set(group.variants.compactMap(\.storageKey)).subtracting(unavailableDownloadKeys)
    }

    /// The keys of the family's visible models only, so a search narrows the checkbox.
    private func availableKeys(in family: CatalogFamily) -> Set<ModelStorageKey> {
        family.members.reduce(into: []) { $0.formUnion(availableKeys(in: $1)) }
    }

    private func toggleFamily(_ family: CatalogFamily) {
        let available = availableKeys(in: family)
        if available.isSubset(of: selectedKeys) {
            selectedKeys.subtract(available)
        } else {
            selectedKeys.formUnion(available)
            openFamilies.insert(family.id)
        }
    }

    private func allQuantsSelected(_ group: PresetModelGroup) -> Bool {
        let available = availableKeys(in: group)
        return !available.isEmpty && available.isSubset(of: selectedKeys)
    }

    private func toggleAllQuants(_ group: PresetModelGroup) {
        let available = availableKeys(in: group)
        if available.isSubset(of: selectedKeys) {
            selectedKeys.subtract(available)
        } else {
            selectedKeys.formUnion(available)
            expandedModelIDs.insert(group.id)
        }
    }

    private func isUnavailableForDownload(_ entry: CatalogEntry) -> Bool {
        guard let key = entry.storageKey else { return true }
        return unavailableDownloadKeys.contains(key)
    }

    private func startDownload(_ confirmedEntries: [CatalogEntry]) {
        let available = allAvailableKeys
        let entries = confirmedEntries.filter { $0.storageKey.map(available.contains) ?? false }
        guard !entries.isEmpty else { return }
        // Through `ensureModel`, so installed weights get their last-used stamp
        // refreshed instead of a second transfer. Progress and errors surface through
        // the coordinator, so nothing here awaits before dismissing.
        for entry in entries {
            Task {
                do {
                    _ = try await ensureModel(entry.source, ws: downloadCoordinator.ws,
                                              coordinator: downloadCoordinator,
                                              modelId: entry.modelId, quant: entry.quant,
                                              // Admission resolves an unknown size from the model's source.
                                              declaredSizeBytes: entry.sizeBytes > 0 ? entry.sizeBytes : nil)
                } catch let ModelStoreError.admission(refusal) {
                    // Refused before any transfer existed, so the coordinator never saw it:
                    // without this the tap would show neither a download row nor an alert.
                    downloadCoordinator.errorMessage = refusal.localizedDescription
                } catch {
                    // Already reported through the coordinator.
                }
            }
        }
        onDownloadStarted()
        dismiss()
    }
}

typealias CatalogFamily = ModelFamily<PresetModelGroup>

/// One catalog row per model, keyed by display name so a model's GGUF and MLX builds
/// share it.
struct PresetModelGroup: FamilyMember {
    let id: String
    let name: String
    let repo: String
    let variants: [CatalogEntry]
    /// The family of the model's first catalog entry.
    let family: String

    var brand: ModelBrand {
        ModelBrand.detect(name: name, hfRepo: repo)
    }

    /// True for an empty `query`, or one found in the name, a repo or a quant.
    func matches(_ query: String) -> Bool {
        query.isEmpty
            || name.lowercased().contains(query)
            || variants.contains { $0.repoIdentifier.lowercased().contains(query) }
            || variants.contains { ($0.quant ?? "").lowercased().contains(query) }
    }

    /// Variants split by Hugging Face repo or Vault, in catalog order.
    var quantSections: [(sourceLabel: String, entries: [CatalogEntry])] {
        var order: [String] = []
        var bySource: [String: [CatalogEntry]] = [:]
        for variant in variants {
            let label = variant.repoIdentifier.isEmpty ? "Vault" : variant.repoIdentifier
            if bySource[label] == nil { order.append(label) }
            bySource[label, default: []].append(variant)
        }
        return order.map { ($0, bySource[$0] ?? []) }
    }

    /// Preserves first-occurrence order.
    static func groups(from entries: [CatalogEntry]) -> [PresetModelGroup] {
        var order: [String] = []
        var byName: [String: [CatalogEntry]] = [:]
        for entry in entries {
            if byName[entry.name] == nil { order.append(entry.name) }
            byName[entry.name, default: []].append(entry)
        }
        return order.compactMap { name in
            guard let first = byName[name]?.first else { return nil }
            return PresetModelGroup(id: name, name: name, repo: first.repoIdentifier,
                                    variants: byName[name] ?? [], family: first.family)
        }
    }
}

#Preview {
    ModelsView()
        .environment(ModelStore(ws: PipetteWorkspace.production))
        .environment(DownloadCoordinator.shared)
}
