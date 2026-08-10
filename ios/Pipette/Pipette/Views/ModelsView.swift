import SwiftUI

/// Tab view for managing downloaded models — GGUF files and MLX repo directories.
struct ModelsView: View {
    @Environment(\.pillTabBarReservedHeight) private var pillTabBarReservedHeight
    @Environment(ModelStore.self) private var modelStore
    @Environment(DownloadCoordinator.self) private var downloadCoordinator

    @State private var searchText: String = ""
    @State private var expandedDownloadedKeys: Set<String> = []
    @State private var showAddModels = false
    @State private var isManagingModels = false
    @State private var selectedModelIDs: Set<String> = []
    @State private var pendingDeleteModels: [DiscoveredModel] = []
    @State private var downloadError: String?
    @State private var lastActiveDownloadFamilyKeys: Set<String> = []

    private var models: [DiscoveredModel] { modelStore.models }

    // Keyed by `<hf_repo>/<filename>` so a model already downloaded from one
    // repo doesn't hide the same-named preset from a different repo.
    private var downloadedKeys: Set<String> {
        Set(models.map { LocalStorage.modelRelativePath(repo: $0.hfRepo, filename: $0.name) })
    }

    /// Models shown as rows in the Models tab, including the built-in AFM model (which
    /// has no file but is selectable like an MLX model). A vision model's projector is
    /// part of its entry rather than a row of its own, so the mmproj filter is now
    /// only a guard against a sideloaded bare projector.
    private var displayableModelFiles: [DiscoveredModel] {
        models.filter { $0.name.range(of: "mmproj", options: .caseInsensitive) == nil }
    }

    /// The delete/manage pool: downloaded files only. AFM is built-in — it has no file
    /// to delete and no on-disk footprint to count — so it's excluded from the "N
    /// downloaded quants" count, select-all, and the multi-select delete flow.
    private var downloadedModelFiles: [DiscoveredModel] {
        displayableModelFiles.filter { if case .appleFoundationText = $0.source { false } else { true } }
    }

    private var downloadedGroups: [ModelGroup] {
        ModelCatalog.groups(from: displayableModelFiles)
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
            downloadDisplayName(for: download).lowercased().contains(q)
                || download.filename.lowercased().contains(q)
                || download.key.lowercased().contains(q)
                || (download.repo?.lowercased().contains(q) ?? false)
                || (download.quant?.lowercased().contains(q) ?? false)
        }
    }

    private var visibleModelRows: [ModelListRow] {
        var downloadsByFamily = Dictionary(grouping: filteredActiveDownloads) { download in
            downloadFamilyKey(for: download)
        }
        var rows: [ModelListRow] = []
        let filteredGroupIDs = Set(filteredDownloadedGroups.map(\.id))

        for group in downloadedGroups where filteredGroupIDs.contains(group.id) || downloadsByFamily[group.key] != nil {
            let downloads = downloadsByFamily.removeValue(forKey: group.key) ?? []
            rows.append(.group(group, downloads: downloads))
        }

        let pendingKeys = downloadsByFamily.keys.sorted { lhs, rhs in
            let lhsName = downloadsByFamily[lhs]?.first.map { downloadDisplayName(for: $0) } ?? lhs
            let rhsName = downloadsByFamily[rhs]?.first.map { downloadDisplayName(for: $0) } ?? rhs
            return lhsName.localizedCaseInsensitiveCompare(rhsName) == .orderedAscending
        }
        for key in pendingKeys {
            rows.append(.pendingDownloads(key: key, downloads: downloadsByFamily[key] ?? []))
        }

        return rows
    }

    private var searchQuery: String {
        searchText.searchNormalized
    }

    private var activeDownloadKeys: [String] {
        downloadCoordinator.downloads.keys.sorted()
    }

    private enum ModelListRow: Identifiable {
        case group(ModelGroup, downloads: [DownloadCoordinator.Download])
        case pendingDownloads(key: String, downloads: [DownloadCoordinator.Download])

        var id: String {
            switch self {
            case .group(let group, _):
                return "group-\(group.id)"
            case .pendingDownloads(let key, _):
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
                    // Manage/delete is for downloaded files only; hide it when the sole
                    // model is the built-in AFM (nothing deletable).
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
            .background(Color(.systemBackground))
            .navigationBarHidden(true)
            .onAppear { refreshModels() }
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
            ForEach(Array(visibleModelRows.enumerated()), id: \.element.id) { index, row in
                if index > 0 {
                    Divider()
                }
                modelListRow(row)
            }

            // Gate on the deletable pool (file-backed models), not
            // `filteredDownloadedGroups` (which includes the built-in AFM row), so the
            // footer doesn't linger with nothing selectable.
            if isManagingModels && !downloadedModelFiles.isEmpty {
                manageDeleteFooter
            }
        }
    }

    @ViewBuilder
    private func modelListRow(_ row: ModelListRow) -> some View {
        switch row {
        case .group(let group, let downloads):
            DownloadedModelGroupBlock(
                group: group,
                activeDownloads: downloads,
                isExpanded: expandedDownloadedKeys.contains(group.key),
                isManaging: isManagingModels,
                selectedModelIDs: selectedModelIDs,
                onToggle: { toggleExpanded(group.key) },
                onToggleModelSelection: toggleModelSelection,
                onDeleteModels: { pendingDeleteModels = $0 }
            )
        case .pendingDownloads(_, let downloads):
            PendingDownloadGroupBlock(downloads: downloads)
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

    /// Bulk resume for downloads a connectivity drop (or the user) paused — most
    /// come back on their own via reconnect, but this is the manual fallback.
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

    /// In-flight transfers sort first, then paused (incl. the transitional
    /// pausing), then failed — the ordering the old phase rank produced.
    private func downloadStateSortRank(_ state: DownloadCoordinator.Download.State) -> Int {
        switch state {
        case .queued, .connecting, .downloading, .resuming: return 0
        case .pausing, .paused: return 1
        case .failed: return 2
        }
    }

    private func downloadFamilyKey(for download: DownloadCoordinator.Download) -> String {
        download.familyId ?? LocalStorage.normalizedModelStem(download.filename)
    }

    private func refreshModels() {
        modelStore.reload()
        reconcileListUIState()
    }

    /// Prune the selection/expansion state against the current models list
    /// and reveal families that have newly started downloads.
    private func reconcileListUIState() {
        let activeFamilyKeys = Set(activeDownloads.map(downloadFamilyKey))
        let familyKeysToReveal = lastActiveDownloadFamilyKeys.union(activeFamilyKeys)

        let currentKeys = Set(downloadedGroups.map(\.key))
        expandedDownloadedKeys.formIntersection(currentKeys)
        expandedDownloadedKeys.formUnion(familyKeysToReveal.intersection(currentKeys))
        selectedModelIDs.formIntersection(Set(downloadedModelFiles.map(\.id)))
        // Gate on the DELETABLE pool (file-backed models), not `downloadedGroups`:
        // the built-in AFM row is always in the groups when available, so keying off
        // groups would keep Manage mode alive after the last real model is deleted.
        if downloadedModelFiles.isEmpty {
            isManagingModels = false
            selectedModelIDs.removeAll()
        }
        if expandedDownloadedKeys.isEmpty, let first = downloadedGroups.first {
            expandedDownloadedKeys.insert(first.key)
        }

        lastActiveDownloadFamilyKeys = activeFamilyKeys
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

private struct DownloadedModelGroupBlock: View {
    let group: ModelGroup
    let activeDownloads: [DownloadCoordinator.Download]
    let isExpanded: Bool
    let isManaging: Bool
    let selectedModelIDs: Set<String>
    let onToggle: () -> Void
    let onToggleModelSelection: (DiscoveredModel) -> Void
    let onDeleteModels: ([DiscoveredModel]) -> Void

    /// A built-in system model (AFM): no file, so no size, no download, no delete, and
    /// nothing to expand into per-quant rows. Rendered like an MLX model otherwise — a
    /// selectable model with a "Built in" indicator in place of the size/quant chips.
    private var isBuiltIn: Bool {
        group.files.contains { if case .appleFoundationText = $0.source { true } else { false } }
    }

    private var subtitle: String {
        // "Built in" is the trailing badge (where other rows show their size / quant
        // count), so the subtitle carries a distinct descriptor rather than repeating it.
        if isBuiltIn { return "Apple Intelligence" }
        return group.files.first?.sizeFormatted ?? group.sizeLabel
    }

    var body: some View {
        VStack(spacing: 0) {
            Button(action: { if !isBuiltIn { onToggle() } }) {
                HStack(spacing: 10) {
                    // Built-in models have no per-quant rows to reveal, so no disclosure.
                    Image(systemName: isBuiltIn ? "sparkles" : (isExpanded ? "chevron.down" : "chevron.right"))
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(.secondary)
                        .frame(width: 18)

                    ModelFamilyRow(
                        title: group.name,
                        subtitle: subtitle,
                        brand: group.brand,
                        logoSize: 17,
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
            // No delete affordance for a built-in model — there's nothing on disk.
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
                    ForEach(activeDownloads) { download in
                        DownloadProgressSubrow(download: download)
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
                                ModelQuantBadge(text: model.quant?.lowercased() ?? "unknown")
                                Divider()
                                    .frame(height: 18)
                                Text(model.sizeFormatted)
                                    .font(.system(size: 14))
                                    .foregroundStyle(.secondary)
                                // When a family mixes repos, show each quant's
                                // source org so the provenance is legible.
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

private func downloadDisplayName(for download: DownloadCoordinator.Download) -> String {
    if let familyId = download.familyId,
       let name = CatalogEntry.familyIdToName[familyId] {
        return name
    }
    return LocalStorage.modelStem(from: download.filename)
}

private struct PendingDownloadGroupBlock: View {
    let downloads: [DownloadCoordinator.Download]

    private var firstDownload: DownloadCoordinator.Download? {
        downloads.first
    }

    var body: some View {
        VStack(spacing: 0) {
            ModelFamilyRow(
                title: title,
                subtitle: subtitle,
                brand: brand,
                logoSize: 17,
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
            .padding(.horizontal, 18)
            .padding(.vertical, 15)

            Divider()

            VStack(alignment: .leading, spacing: 0) {
                ForEach(downloads) { download in
                    DownloadProgressSubrow(download: download)
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

    private var quantCountLabel: String {
        "\(downloads.count) quant\(downloads.count == 1 ? "" : "s")"
    }

    private var brand: ModelBrand {
        guard let firstDownload else { return .unknown }
        return ModelBrand.detect(name: title, hfRepo: firstDownload.repo)
    }
}

private struct DownloadProgressSubrow: View {
    @Environment(DownloadCoordinator.self) private var downloadCoordinator
    let download: DownloadCoordinator.Download

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                ModelQuantBadge(text: (download.quant ?? "unknown").lowercased())
                    .fixedSize(horizontal: true, vertical: false)

                DownloadStateBadge(download: download)
                    .fixedSize(horizontal: true, vertical: false)

                Spacer(minLength: 8)

                downloadControls
            }

            if let statusDetail {
                Text(statusDetail)
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            DownloadProgressTrack(progress: download.progress, state: download.state)
        }
        .frame(maxWidth: .infinity, minHeight: 34, alignment: .leading)
        .padding(.leading, 58)
        .padding(.trailing, 16)
        .padding(.vertical, 9)
    }

    @ViewBuilder
    private var downloadControls: some View {
        HStack(spacing: 4) {
            // Pause/resume only apply to `.file` (GGUF) transfers. MLX `.directory`
            // pulls go through HubApi with no pause/resume — only cancel below.
            if case .file = download.kind {
                switch download.state {
                case .connecting, .downloading, .resuming:
                    DownloadIconButton(
                        systemName: "pause.fill",
                        accessibilityLabel: "Pause \(download.filename)"
                    ) {
                        downloadCoordinator.pause(key: download.id)
                    }
                case .paused:
                    DownloadIconButton(
                        systemName: "play.fill",
                        accessibilityLabel: "Resume \(download.filename)"
                    ) {
                        downloadCoordinator.resume(key: download.id)
                    }
                case .queued, .pausing, .failed:
                    EmptyView()  // queued/transient/terminal — only cancel remains
                }
            }

            DownloadIconButton(
                systemName: "xmark",
                tint: .red,
                isDestructive: true,
                accessibilityLabel: "Cancel \(download.filename)"
            ) {
                downloadCoordinator.cancel(key: download.id)
            }
        }
    }

    /// The single home for the download's supplementary status line. Switches on
    /// `state` (with numeric progress for the active-transfer detail) to produce the
    /// text directly — it is no longer reverse-engineered from a free-form string.
    /// Returns nil when the badge alone says everything (a plain `.paused`).
    private var statusDetail: String? {
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

/// Byte-size formatting for the download detail line: decimal (1000-based), two
/// decimals, GB at/above 1 GB else MB (matching `ByteFormat.fileSize`). Lives here
/// because the download's status text has a single home in this view.
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

    /// `written / total` in the *total's* unit so both sides share one unit and
    /// decimal width, e.g. `2.34 / 5.60 GB`.
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
    let download: DownloadCoordinator.Download

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
        switch download.state {
        case .queued:
            return "Queued"
        case .connecting:
            return "Starting"
        case .downloading, .resuming:
            guard let progress = download.progress else { return "Starting" }
            return "\(Int((progress * 100).rounded(.down)))%"
        case .pausing, .paused:
            return "Paused"
        case .failed:
            return "Failed"
        }
    }

    private var foregroundColor: Color {
        switch download.state {
        case .queued, .connecting, .downloading, .pausing, .paused, .resuming: return .primary
        case .failed: return .red
        }
    }

    private var backgroundColor: Color {
        switch download.state {
        case .queued, .connecting, .downloading, .pausing, .paused, .resuming: return Color(.secondarySystemBackground)
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

    /// Whether the transfer is actively moving — drives the full-strength fill and
    /// the indeterminate stub when there's no numeric progress yet.
    private var isActive: Bool {
        switch state {
        case .connecting, .downloading, .resuming: return true
        case .queued, .pausing, .paused, .failed: return false
        }
    }

    private var fillColor: Color {
        switch state {
        case .connecting, .downloading, .resuming: return .primary
        case .queued, .pausing, .paused: return Color.primary.opacity(0.35)
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
    @Environment(DownloadCoordinator.self) private var downloadCoordinator

    let downloadedKeys: Set<String>
    let onDownloadStarted: () -> Void

    @State private var searchText = ""
    /// Which model rows are expanded to reveal their quant chips.
    @State private var expandedModelIDs: Set<String> = []
    /// Selected quant chips, by `CatalogEntry.id` — selection is per-quant, not
    /// per-family, so a model's GGUF and MLX quants are picked individually.
    @State private var selectedEntryIDs: Set<String> = []
    @State private var showLargeDownloadWarning = false

    private let largeDownloadThreshold: Int64 = 200 * 1024 * 1024

    private var groups: [PresetModelGroup] {
        PresetModelGroup.groups(from: CatalogEntry.catalog)
    }

    private var filteredGroups: [PresetModelGroup] {
        let q = searchText.searchNormalized
        guard !q.isEmpty else { return groups }
        return groups.filter { group in
            group.name.lowercased().contains(q)
                || group.variants.contains { $0.repoIdentifier.lowercased().contains(q) }
                || group.variants.contains { ($0.quant ?? "").lowercased().contains(q) }
        }
    }

    private var unavailableDownloadKeys: Set<String> {
        downloadedKeys.union(downloadCoordinator.downloads.keys)
    }

    /// Every catalog quant not already downloaded or downloading — the pool the
    /// header checkbox and "Select all" operate over.
    private var allAvailableEntries: [CatalogEntry] {
        CatalogEntry.catalog.filter { !isUnavailableForDownload($0) }
    }

    private var allSelected: Bool {
        let available = allAvailableEntries
        return !available.isEmpty && available.allSatisfy { selectedEntryIDs.contains($0.id) }
    }

    private var selectedDownloadPresets: [CatalogEntry] {
        CatalogEntry.catalog.filter { selectedEntryIDs.contains($0.id) && !isUnavailableForDownload($0) }
    }

    private var totalSelectedDownloadBytes: Int64 {
        selectedDownloadPresets.reduce(0) { $0 + $1.sizeBytes }
    }

    var body: some View {
        VStack(spacing: 0) {
            addHeader
            VStack(alignment: .leading, spacing: 18) {
                introHeader
                ModelSearchField(text: $searchText, placeholder: "Search models")
                presetList
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
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .alert("Large download detected", isPresented: $showLargeDownloadWarning) {
            Button("Cancel", role: .cancel) {}
            Button("Proceed") { startDownload() }
        } message: {
            Text("Download size over 200MB.")
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
                Text("Add models")
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

    private var introHeader: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 7) {
                Text("Download models")
                    .font(.serif(19))
                Text("Select the models to download for benchmarking.")
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button(action: toggleSelectAll) {
                HStack(spacing: 6) {
                    if allSelected {
                        Image(systemName: "checkmark")
                            .font(.system(size: 12, weight: .semibold))
                    }
                    Text(allSelected ? "Selected all" : "Select all")
                        .font(.system(size: 15, weight: .semibold))
                }
                .foregroundStyle(.primary)
                .padding(.horizontal, 16)
                .frame(height: 44)
                .background(Color(.systemBackground), in: Capsule())
                .overlay(
                    Capsule()
                        .strokeBorder(Color.primary.opacity(0.12), lineWidth: 1)
                )
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
        }
    }

    private var presetList: some View {
        ModelListCard(cornerRadius: 16) {
            ScrollView {
                LazyVStack(spacing: 0) {
                    if filteredGroups.isEmpty {
                        Text("No models match \"\(searchText)\".")
                            .font(.system(size: 15))
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 34)
                    } else {
                        ForEach(Array(filteredGroups.enumerated()), id: \.element.id) { index, group in
                            if index > 0 {
                                Divider().padding(.leading, 16)
                            }
                            presetRow(group)
                        }
                    }
                }
            }
            .scrollDismissesKeyboard(.interactively)
        }
    }

    private func presetRow(_ group: PresetModelGroup) -> some View {
        let isExpanded = expandedModelIDs.contains(group.id)
        return VStack(spacing: 0) {
            HStack(spacing: 0) {
                // Header tap expands/collapses; the checkbox is a separate control
                // that selects or clears every available quant of the model.
                Button { toggleExpanded(group.id) } label: {
                    ModelFamilyRow(
                        title: group.name,
                        subtitle: quantSummary(group),
                        brand: group.brand,
                        logoSize: 17,
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
                    WizardCheckbox(isOn: allQuantsSelected(group), size: 22)
                }
                .buttonStyle(.plain)
                .padding(.leading, 12)
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 15)

            if isExpanded {
                quantChips(group)
                    .padding(.leading, 47)
                    .padding(.trailing, 18)
                    .padding(.bottom, 14)
            }
        }
    }

    /// Quant chips grouped by source repo: a highlighted repo header, then that
    /// repo's quant chips beneath it.
    private func quantChips(_ group: PresetModelGroup) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(group.quantSections, id: \.repo) { section in
                VStack(alignment: .leading, spacing: 7) {
                    Text(section.repo)
                        .font(.system(size: 12, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    ChipFlowLayout(horizontalSpacing: 8, verticalSpacing: 8) {
                        ForEach(section.entries) { quantChip($0) }
                    }
                }
            }
        }
    }

    private func quantChip(_ entry: CatalogEntry) -> some View {
        let unavailable = isUnavailableForDownload(entry)
        let selected = selectedEntryIDs.contains(entry.id)
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

    /// Header subtitle: selected-count once anything is picked, else the available
    /// formats (GGUF / MLX) so a collapsed row still shows what it offers.
    private func quantSummary(_ group: PresetModelGroup) -> String {
        let selected = group.variants.filter { selectedEntryIDs.contains($0.id) }.count
        if selected > 0 { return "\(selected) selected" }
        // Distinct file formats present (GGUF before MLX), by switching on each source.
        var formats: [String] = []
        for variant in group.variants {
            let label: String
            switch variant.source {
            case .mlx: label = "MLX"
            case .ggufText, .ggufVision: label = "GGUF"
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
        guard !selectedDownloadPresets.isEmpty else { return }
        if totalSelectedDownloadBytes > largeDownloadThreshold {
            showLargeDownloadWarning = true
        } else {
            startDownload()
        }
    }

    private func toggleSelectAll() {
        if allSelected {
            selectedEntryIDs.removeAll()
        } else {
            selectedEntryIDs = Set(allAvailableEntries.map(\.id))
        }
    }

    private func toggleExpanded(_ id: String) {
        if expandedModelIDs.contains(id) { expandedModelIDs.remove(id) } else { expandedModelIDs.insert(id) }
    }

    private func toggleEntry(_ entry: CatalogEntry) {
        guard !isUnavailableForDownload(entry) else { return }
        if selectedEntryIDs.contains(entry.id) { selectedEntryIDs.remove(entry.id) } else { selectedEntryIDs.insert(entry.id) }
    }

    private func availableVariants(_ group: PresetModelGroup) -> [CatalogEntry] {
        group.variants.filter { !isUnavailableForDownload($0) }
    }

    private func allQuantsSelected(_ group: PresetModelGroup) -> Bool {
        let available = availableVariants(group)
        return !available.isEmpty && available.allSatisfy { selectedEntryIDs.contains($0.id) }
    }

    /// Select every available quant of the model (revealing them), or clear them
    /// all when they're already selected.
    private func toggleAllQuants(_ group: PresetModelGroup) {
        let available = availableVariants(group)
        if allQuantsSelected(group) {
            available.forEach { selectedEntryIDs.remove($0.id) }
        } else {
            available.forEach { selectedEntryIDs.insert($0.id) }
            expandedModelIDs.insert(group.id)
        }
    }

    private func isUnavailableForDownload(_ entry: CatalogEntry) -> Bool {
        // Only single-file GGUF entries have a weight filename to key the
        // unavailable set on; MLX is a repo directory and never blocks here.
        // The key is a HuggingFace download key, so only that arm can be in the set — a
        // catalog entry is always one, and any other arm has no download to block.
        let filename: String
        switch entry.source {
        case let .ggufText(m):
            guard case let .huggingFace(_, path, _) = m.source else { return false }
            filename = path.value
        case let .ggufVision(m):
            guard case let .huggingFace(_, weights, _, _, _) = m.source else { return false }
            filename = weights.value
        case .mlx, .appleFoundationText: return false  // no single weight file to key on
        }
        let key = LocalStorage.modelRelativePath(repo: entry.repoIdentifier, filename: filename)
        return unavailableDownloadKeys.contains(key)
    }

    private func startDownload() {
        let entries = selectedDownloadPresets
        guard !entries.isEmpty else { return }
        // Each selection goes through `ensureModel`, so a quant already in the store is
        // answered from it — tapping Download on installed weights refreshes their
        // last-used stamp instead of starting a second transfer. Progress and errors keep
        // surfacing through the coordinator, which is what the sheet below renders, so
        // nothing here awaits before dismissing.
        for entry in entries {
            Task {
                _ = try? await ensureModel(entry.source, storage: downloadCoordinator.storage,
                                           coordinator: downloadCoordinator,
                                           familyId: entry.familyId, quant: entry.quant,
                                           declaredSizeBytes: entry.sizeBytes)
            }
        }
        onDownloadStarted()
        dismiss()
    }
}

/// One catalog row per model, keyed by display name so a model's GGUF and MLX
/// builds unify into a single expandable entry whose quants are chosen individually.
private struct PresetModelGroup: Identifiable {
    let id: String
    let name: String
    let repo: String
    let variants: [CatalogEntry]

    var brand: ModelBrand {
        ModelBrand.detect(name: name, hfRepo: repo)
    }

    /// Variants split by source repo, one section per repo present, preserving
    /// catalog order. The repo is the section's identity — a model whose quants span
    /// repos (e.g. Ministral GGUF: Q4_0 from unsloth, K-quants from mistralai; or a
    /// GGUF repo plus an MLX repo) shows each repo as its own labeled group.
    var quantSections: [(repo: String, entries: [CatalogEntry])] {
        var order: [String] = []
        var byRepo: [String: [CatalogEntry]] = [:]
        for variant in variants {
            if byRepo[variant.repoIdentifier] == nil { order.append(variant.repoIdentifier) }
            byRepo[variant.repoIdentifier, default: []].append(variant)
        }
        return order.map { ($0, byRepo[$0] ?? []) }
    }

    /// Groups the catalog by model id (display name), preserving first-occurrence
    /// order, so GGUF and MLX variants of one model share a single row.
    static func groups(from entries: [CatalogEntry]) -> [PresetModelGroup] {
        var order: [String] = []
        var byName: [String: [CatalogEntry]] = [:]
        for entry in entries {
            if byName[entry.name] == nil { order.append(entry.name) }
            byName[entry.name, default: []].append(entry)
        }
        return order.compactMap { name in
            guard let first = byName[name]?.first else { return nil }
            return PresetModelGroup(id: name, name: name, repo: first.repoIdentifier, variants: byName[name] ?? [])
        }
    }
}

#Preview {
    ModelsView()
        .environment(ModelStore(storage: FileStorage.production))
        .environment(DownloadCoordinator.shared)
}
