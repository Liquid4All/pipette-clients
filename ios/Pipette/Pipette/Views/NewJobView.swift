import Combine
import SwiftUI
import UIKit

/// Creates and runs a new job via a four-step wizard: runtime, models, benchmarks,
/// review. The runtime comes first and filters both lists, so a job targets a single
/// runtime.
struct NewJobView: View {
    @Environment(JobRunner.self) private var jobRunner
    @Environment(JobStore.self) private var jobStore
    @Environment(ModelStore.self) private var modelStore
    @Environment(\.ws) private var ws
    @Environment(\.dismiss) private var dismiss

    /// Invoked after the run starts; the host pushes the job's detail page.
    var onStarted: (JobId) -> Void = { _ in }

    /// Invoked from the empty state's "Go to Models" button.
    var onGoToModels: () -> Void = {}

    enum Step: Int, CaseIterable {
        case runtime, models, benchmarks, review
    }

    @State private var step: Step = .runtime
    /// Nil until picked. Changing it re-seeds the downstream selections.
    @State private var selectedRuntime: RuntimeKind?

    @State private var benchmarks: [BenchmarkItem] = []
    @State private var models: [DiscoveredModel] = []
    @State private var catalogEntries: [String: CatalogEntry] = [:]
    @State private var catalogLogoOwner: ModelShortList.Owner?
    @State private var familyLogoDigests: [String: String] = [:]
    @State private var mmprojFiles: [DiscoveredModel] = []
    @State private var isSyncingBenchmarks: Bool = false

    @State private var selectedModelKeys: Set<String> = []  // keyed by group key
    @State private var selectedQuants: Set<QuantPill> = QuantPill.allSelection()
    @State private var modelSearch: String = ""
    /// The model families closed by hand. Families start open, as the downloaded models
    /// are few. A search opens every family with a match.
    @State private var closedModelFamilies: Set<String> = []
    /// Adds the catalog models that are not on the device. A job downloads them before
    /// it runs.
    @State private var showsUndownloadedModels = false
    @State private var benchmarkSearch: String = ""
    @State private var expandedBenchmarkTypes: Set<String> = []
    @State private var selectedBenchmarks: Set<String> = []
    @State private var selectedMmproj: Set<String> = []  // keyed by path
    // Only takes effect with `canSubmitResults` (see `shouldAutoSubmitResults`).
    @State private var contributeResults: Bool = LocalStorage.defaultContributeResults
    @State private var didApplyInitialSelections: Bool = false
    @State private var startBlockedMessage: String?
    @State private var lowPowerWarningPending: Bool = false
    private let stepHorizontalPadding: CGFloat = 24
    private let stepTopPadding: CGFloat = 16
    private let stepBottomPadding: CGFloat = 24
    private let benchmarkLeadingColumnWidth: CGFloat = 44

    #if targetEnvironment(simulator)
    @State private var nGpuLayers: Int = 0
    #else
    @State private var nGpuLayers: Int = 99
    #endif
    // llama's n_ubatch and MLX's prefill chunk.
    @State private var prefillBatch: Int = 512
    private let prefillBatchOptions = [256, 512, 1024, 2048]

    // MARK: - VL helpers

    nonisolated private static func isVlBenchmark(_ item: BenchmarkItem) -> Bool {
        item.type?.isVisionLanguage == true
    }

    nonisolated private static func declaresProjector(_ source: Model) -> Bool {
        switch source {
        case .ggufVision, .ggufVisionSpeculative: true
        default: false
        }
    }

    /// Independent of the VL pairing gate. An unrecognized benchmark type counts as
    /// runnable; the runtime decides at execution time.
    private func runtimeSupports(_ benchmark: BenchmarkItem, on model: DiscoveredModel) -> Bool {
        guard let bt = benchmark.type else { return true }
        return isBenchmarkSupported(bt, on: RuntimeKind(model.source))
    }

    private var isVlSelected: Bool {
        benchmarks.contains { selectedBenchmarks.contains($0.benchmarkId) && Self.isVlBenchmark($0) }
    }

    private var canSelectVisionBenchmarks: Bool {
        Self.allModelsHaveVisionPairings(resolvedModelFiles, mmprojFiles: mmprojFiles)
    }

    private var selectedProjectors: [DiscoveredModel] {
        mmprojFiles.filter { selectedMmproj.contains($0.path) }
    }

    private var visionSelectionIsReady: Bool {
        !isVlSelected || Self.allModelsHaveVisionPairings(resolvedModelFiles, mmprojFiles: selectedProjectors)
    }

    // MARK: - Model grouping & quant resolution

    /// Runtimes with at least one offerable model, on device or in the catalog. AFM
    /// appears whenever available, since `availableModels()` injects it as a model.
    private var availableRuntimes: [RuntimeKind] {
        let kinds = Set(models.map { RuntimeKind($0.source) })
        return RuntimeKind.allCases.filter { kinds.contains($0) }
    }

    /// What is on disk plus the rest of the catalog: `RunCell.prepare` fetches an
    /// absent model before the cell runs.
    ///
    /// Downloaded rows win a collision, since only they know the real size and path.
    /// Collision is judged by `description` (the crate's `Display`), not `Model`
    /// equality, which also compares a plan's `sha256` that the catalog never states.
    nonisolated static func offerable(downloaded: [DiscoveredModel],
                                      catalog: [CatalogEntry] = CatalogEntry.catalog) -> [DiscoveredModel] {
        var byReference: [String: DiscoveredModel] = [:]
        for entry in catalog {
            byReference[entry.source.description] = DiscoveredModel(catalog: entry)
        }
        for model in downloaded { byReference[model.source.description] = model }
        return byReference.values.sorted {
            ($0.modelId, $0.quant ?? "", $0.source.description)
                < ($1.modelId, $1.quant ?? "", $1.source.description)
        }
    }

    /// The VL coordinates to run `model` as, one per cell.
    ///
    /// Declared vision coordinates already include their projector. A standalone GGUF
    /// weight needs a downloaded projector from the same repo and revision, or a matching
    /// filename stem when one of the files has no repo.
    nonisolated static func visionSources(
        for model: DiscoveredModel, mmprojFiles: [DiscoveredModel]
    ) -> [Model] {
        if declaresProjector(model.source) { return [model.source] }
        guard case .ggufText = model.source, model.isDownloaded,
              let weights = try? AbsolutePath(model.path)
        else { return [] }
        return mmprojFiles.compactMap { mmproj in
            guard mmproj.isDownloaded,
                  matchesProjector(mmproj, to: model),
                  let projector = try? AbsolutePath(mmproj.path)
            else { return nil }
            return .ggufVision(.init(source: .absoluteFiles(model: weights, mmproj: projector)))
        }
    }

    nonisolated private static func matchesProjector(_ projector: DiscoveredModel, to model: DiscoveredModel) -> Bool {
        if let modelRepo = model.source.repo, let projectorRepo = projector.source.repo {
            return modelRepo == projectorRepo
        }
        return LocalStorage.normalizedModelStem(projector.name) == LocalStorage.normalizedModelStem(model.name)
    }

    nonisolated static func allModelsHaveVisionPairings(
        _ models: [DiscoveredModel], mmprojFiles: [DiscoveredModel]
    ) -> Bool {
        !models.isEmpty && models.allSatisfy { !visionSources(for: $0, mmprojFiles: mmprojFiles).isEmpty }
    }

    private func sources(
        for model: DiscoveredModel, benchmark: BenchmarkItem, projectors: [DiscoveredModel]
    ) -> [Model] {
        guard runtimeSupports(benchmark, on: model) else { return [] }
        guard Self.isVlBenchmark(benchmark) else { return [model.source] }
        if case let .ggufVisionSpeculative(speculative) = model.source,
           benchmark.type != .vlSpecDecode {
            return [.ggufVision(.init(source: speculative.source.vision))]
        }
        return Self.visionSources(for: model, mmprojFiles: projectors)
    }

    /// Every offer for the selected runtime: the catalog plus what is on disk.
    private var runtimeModels: [DiscoveredModel] {
        guard let rt = selectedRuntime else { return [] }
        return models.filter { RuntimeKind($0.source) == rt }
    }

    private var eligibleModels: [DiscoveredModel] {
        Self.listedModels(runtimeModels, showsUndownloaded: showsUndownloadedModels)
    }

    /// The downloaded models, with the rest of `offered` added on request. With nothing
    /// downloaded, `offered` is the whole list.
    nonisolated static func listedModels(_ offered: [DiscoveredModel],
                                         showsUndownloaded: Bool) -> [DiscoveredModel] {
        guard !showsUndownloaded, offered.contains(where: \.isDownloaded) else { return offered }
        return offered.filter(\.isDownloaded)
    }

    /// True when a selected model is still offered but the list of downloaded models would
    /// hide it. The step then shows the models not downloaded, so the selection stays visible.
    nonisolated static func selectionNeedsUndownloaded(_ selected: Set<String>,
                                                       offered: [DiscoveredModel]) -> Bool {
        let listed = Set(listedModels(offered, showsUndownloaded: false).map(\.modelId))
        let offeredIds = Set(offered.map(\.modelId))
        return selected.contains { !listed.contains($0) && offeredIds.contains($0) }
    }

    /// The switch matters only when the runtime has models both on and off the device.
    private var offersUndownloadedModelsSwitch: Bool {
        let offered = runtimeModels
        return offered.contains(where: \.isDownloaded) && offered.contains { !$0.isDownloaded }
    }

    /// VL is additionally gated per model by a projector. An unrecognized type is kept.
    private var runtimeBenchmarks: [BenchmarkItem] {
        guard let rt = selectedRuntime else { return [] }
        let runtime = rt
        return benchmarks.filter { item in
            guard let bt = item.type else { return true }
            return isBenchmarkSupported(bt, on: runtime)
        }
    }

    private var modelGroups: [ModelGroup] {
        ModelCatalog.groups(from: eligibleModels, entries: catalogEntries)
    }

    /// Search filters only the model list, never the quant pills.
    private var filteredModelGroups: [ModelGroup] {
        let q = modelSearch.searchNormalized
        guard !q.isEmpty else { return modelGroups }
        return modelGroups.filter {
            $0.name.lowercased().contains(q) || $0.key.lowercased().contains(q)
        }
    }

    private var modelFamilies: [ModelFamily<ModelGroup>] {
        ModelFamily.grouping(filteredModelGroups, logoDigests: familyLogoDigests)
    }

    /// Quant pills with no matching file among the selected models (or all of them
    /// while nothing is selected), so the filter only names quants the job could run.
    private var unavailableQuants: Set<QuantPill> {
        let relevantGroups = selectedModelKeys.isEmpty
            ? modelGroups
            : modelGroups.filter { selectedModelKeys.contains($0.key) }
        let downloadedQuants = relevantGroups.flatMap(\.files).compactMap(\.quant)
        return Set(QuantPill.specificCases.filter { pill in
            !downloadedQuants.contains { pill.matches($0) }
        })
    }

    private func quantMatchesFilter(_ quant: String?) -> Bool {
        if selectedQuants.contains(.all) { return true }
        return selectedQuants.contains { $0.matches(quant) }
    }

    /// Selected groups narrowed by the quant filter.
    private var resolvedModelFiles: [DiscoveredModel] {
        modelGroups
            .filter { selectedModelKeys.contains($0.key) }
            .flatMap { group in group.files.filter { quantMatchesFilter($0.quant) } }
    }

    /// Selected models with no file matching the chosen quant, surfaced as a warning.
    private var modelsMissingQuant: [ModelGroup] {
        guard !selectedQuants.contains(.all) else { return [] }
        return modelGroups
            .filter { selectedModelKeys.contains($0.key) }
            .filter { group in !group.files.contains { quantMatchesFilter($0.quant) } }
    }

    private var cellCount: Int { countCells { _ in true } }

    private var memoryCellCount: Int {
        countCells { MemoryFreshness.isMemory(BenchmarkType(rawValue: $0.benchmarkType)) }
    }

    private func countCells(_ include: (BenchmarkItem) -> Bool) -> Int {
        let selectedItems = benchmarks.filter {
            selectedBenchmarks.contains($0.benchmarkId) && include($0)
        }
        let resolved = resolvedModelFiles
        let projectors = selectedProjectors
        return resolved.reduce(0) { total, model in
            total + selectedItems.reduce(0) { sub, benchmark in
                sub + sources(for: model, benchmark: benchmark, projectors: projectors).count
            }
        }
    }

    private var groupedBenchmarks: [(type: String, items: [BenchmarkItem])] {
        let grouped = Dictionary(grouping: runtimeBenchmarks) { $0.benchmarkType }
        return grouped
            .map { (type: $0.key, items: sortBenchmarkItems($0.value)) }
            .sorted { a, b in
                let ra = BenchmarkCatalog.typeRank(for: a.type), rb = BenchmarkCatalog.typeRank(for: b.type)
                if ra != rb { return ra < rb }
                return a.type < b.type
            }
    }

    private var filteredGroupedBenchmarks: [(type: String, items: [BenchmarkItem])] {
        let query = benchmarkSearch.searchNormalized
        guard !query.isEmpty else { return groupedBenchmarks }

        return groupedBenchmarks.compactMap { group in
            let groupMatches = BenchmarkCatalog.displayName(for: group.type).lowercased().contains(query)
                || BenchmarkCatalog.description(for: group.type).lowercased().contains(query)
                || group.type.lowercased().contains(query)
            let items = group.items.filter { item in
                groupMatches || benchmarkItemMatchesSearch(item, query: query)
            }
            guard !items.isEmpty else { return nil }
            return (type: group.type, items: items)
        }
    }

    private var selectedReviewModelGroups: [ModelGroup] {
        modelGroups.filter { selectedModelKeys.contains($0.key) }
    }

    private var selectedReviewBenchmarkTypes: [String] {
        groupedBenchmarks
            .compactMap { group in
                group.items.contains { selectedBenchmarks.contains($0.benchmarkId) } ? group.type : nil
            }
            .sorted { BenchmarkCatalog.typeRank(for: $0) < BenchmarkCatalog.typeRank(for: $1) }
    }

    // MARK: - Step gating

    private var canAdvance: Bool {
        switch step {
        case .runtime:
            return selectedRuntime != nil
        case .models:
            return !selectedModelKeys.isEmpty
        case .benchmarks:
            return !selectedBenchmarks.isEmpty && visionSelectionIsReady
        case .review:
            return visionSelectionIsReady && cellCount > 0
        }
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            wizardHeader
            stepContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            if showFooter {
                Divider()
                wizardFooter
            }
        }
        .environment(\.catalogLogoOwner, catalogLogoOwner)
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            loadData()
            syncBenchmarksOnVisit()
        }
        .onReceive(NotificationCenter.default.publisher(
            for: ModelShortList.didChange, object: ws.modelShortList
        ).receive(on: DispatchQueue.main)) { _ in
            loadModels()
        }
        .onChange(of: ObjectIdentifier(ws.modelShortList)) { _, _ in
            loadModels()
        }
        .onChange(of: selectedRuntime) { _, _ in seedSelectionsForRuntime() }
        .alert("Job Already Running", isPresented: Binding<Bool>(
            get: { startBlockedMessage != nil },
            set: { if !$0 { startBlockedMessage = nil } }
        )) {
            Button("OK", role: .cancel) { startBlockedMessage = nil }
        } message: {
            Text(startBlockedMessage ?? "")
        }
        .lowPowerModeWarning(isPresented: $lowPowerWarningPending) { start() }
    }

    // MARK: - Header / footer

    private var wizardHeader: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: "chevron.left")
                    .opacity(0)
                    .accessibilityHidden(true)
                Spacer()
                Text("Create a job")
                    .font(.custom("IowanOldStyle-Bold", size: 20))
                Spacer()
                Button { dismiss() } label: {
                    Image(systemName: "xmark")
                }
            }
            .foregroundStyle(.primary)
            .padding(.horizontal)
            .padding(.top, 12)

            HStack(spacing: 0) {
                ForEach(Step.allCases, id: \.self) { s in
                    Rectangle()
                        .fill(s.rawValue <= step.rawValue ? Color.primary : Color(.systemGray5))
                        .frame(height: 2)
                }
            }
        }
    }

    /// The empty states have nothing to advance to, so "Go to Models" stands alone.
    private var showFooter: Bool {
        switch step {
        case .runtime: return !availableRuntimes.isEmpty
        case .models: return !modelGroups.isEmpty
        case .benchmarks, .review: return true
        }
    }

    @ViewBuilder
    private var wizardFooter: some View {
        HStack(spacing: 10) {
            if step != .runtime {
                Button { back() } label: {
                    Text("Back")
                        .font(.system(size: 17, weight: .medium))
                        .foregroundStyle(.primary)
                        .frame(width: 80, height: 42)
                        .background(
                            Capsule().strokeBorder(Color(.systemGray3), lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
            }
            Button {
                if step == .review { attemptStart() } else { next() }
            } label: {
                Group {
                    if step == .review {
                        HStack(spacing: 10) {
                            Image(systemName: "play")
                                .font(.system(size: 17, weight: .medium))
                            Text("Run job")
                        }
                    } else {
                        Text("Next")
                    }
                }
                .font(.system(size: 17, weight: .medium))
                .foregroundStyle(Color(.systemBackground))
                .frame(maxWidth: .infinity)
                .frame(height: 42)
                .background(
                    canAdvance ? Color.primary : Color(.systemGray4),
                    in: Capsule()
                )
            }
            .buttonStyle(.plain)
            .disabled(!canAdvance)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
    }

    private func next() {
        guard let nextStep = Step(rawValue: step.rawValue + 1) else { return }
        withAnimation { step = nextStep }
    }

    private func back() {
        guard let prevStep = Step(rawValue: step.rawValue - 1) else { return }
        withAnimation { step = prevStep }
    }

    // MARK: - Step content

    @ViewBuilder
    private var stepContent: some View {
        switch step {
        case .runtime:
            runtimeStep
        case .models:
            modelStep
        case .benchmarks:
            benchmarkStep
        case .review:
            reviewStep
        }
    }

    // MARK: - Step 1: Runtime selection

    private var runtimeStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            WizardStepHeader(
                title: "Runtime selection",
                description: "Choose the inference engine. Models and benchmarks are filtered to what this runtime can run."
            )

            if availableRuntimes.isEmpty {
                Spacer()
                modelsEmptyState
                Spacer()
            } else {
                ModelListCard(cornerRadius: 16) {
                    ForEach(Array(availableRuntimes.enumerated()), id: \.element) { index, rt in
                        if index > 0 {
                            Divider().padding(.leading, 16)
                        }
                        runtimeRow(rt)
                    }
                }
            }
        }
        .padding(.horizontal, stepHorizontalPadding)
        .padding(.top, stepTopPadding)
        .padding(.bottom, stepBottomPadding)
    }

    private func runtimeRow(_ rt: RuntimeKind) -> some View {
        let isSelected = selectedRuntime == rt
        // Every offer for the runtime: the catalog plus what is on disk. Counted per model,
        // as the model step lists one row per model.
        let offered = models.filter { RuntimeKind($0.source) == rt }
        let available = Set(offered.map(\.modelId)).count
        let downloaded = Set(offered.filter(\.isDownloaded).map(\.modelId)).count
        return Button {
            withAnimation { selectedRuntime = rt }
        } label: {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(rt.label)
                        .font(.system(size: 17, weight: .medium))
                        .foregroundStyle(.primary)
                    // AFM has no downloads, so a count would be meaningless there.
                    Text(rt == .afm ? rt.detail
                         : "\(rt.detail) · \(available) available · \(downloaded) downloaded")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 12)
                Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                    .font(.system(size: 22))
                    .foregroundStyle(isSelected ? Color.primary : Color(.systemGray3))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 16)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Step 2: Model selection

    private var modelStep: some View {
        VStack(alignment: .leading, spacing: 20) {
            WizardStepHeader(
                title: "Model selection",
                description: "Select the models to run benchmarks on."
            )

            if modelGroups.isEmpty {
                Spacer()
                modelsEmptyState
                Spacer()
            } else {
                searchField
                if offersUndownloadedModelsSwitch {
                    undownloadedModelsSwitch
                }
                modelList
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                // Only llama.cpp ships several quants per model; elsewhere the filter
                // stays at "all".
                if selectedRuntime == .llamaCpp {
                    quantSection
                }
            }
        }
        .padding(.horizontal, stepHorizontalPadding)
        .padding(.top, stepTopPadding)
        .padding(.bottom, stepBottomPadding)
    }

    private var searchField: some View {
        ModelSearchField(text: $modelSearch, placeholder: "Search models")
    }

    private var undownloadedModelsSwitch: some View {
        Button {
            showsUndownloadedModels.toggle()
            // A hidden model must not stay in the job.
            if !showsUndownloadedModels {
                selectedModelKeys.formIntersection(Set(modelGroups.map(\.key)))
            }
        } label: {
            HStack(spacing: 10) {
                WizardCheckbox(isOn: showsUndownloadedModels, size: 18)
                Text("Show models not downloaded")
                    .font(.system(size: 15))
                    .foregroundStyle(Color.primary.opacity(0.78))
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var modelList: some View {
        let families = modelFamilies
        return ScrollView {
            if families.isEmpty {
                Text("No models match \u{201C}\(modelSearch)\u{201D}.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 32)
            } else {
                ModelListCard(cornerRadius: 16) {
                    ForEach(Array(families.enumerated()), id: \.element.id) { index, family in
                        if index > 0 {
                            Divider().padding(.leading, 16)
                        }
                        modelFamilySection(family)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    /// A family with one model shows the model row alone.
    @ViewBuilder
    private func modelFamilySection(_ family: ModelFamily<ModelGroup>) -> some View {
        if family.members.count == 1, let group = family.members.first {
            modelRow(group, logoDigest: family.logoDigest)
        } else {
            let isOpen = !modelSearch.searchNormalized.isEmpty || !closedModelFamilies.contains(family.id)
            // The visible models only, so a search narrows the checkbox.
            let keys = Set(family.members.map(\.key))
            let allSelected = keys.isSubset(of: selectedModelKeys)
            VStack(spacing: 0) {
                ModelFamilyHeader(
                    name: family.name,
                    count: family.members.count,
                    brand: family.brand,
                    logoDigest: family.logoDigest,
                    isOpen: isOpen,
                    isSelected: allSelected,
                    isMixed: !allSelected && !keys.isDisjoint(with: selectedModelKeys),
                    checkboxSize: 26,
                    onToggleOpen: {
                        if closedModelFamilies.remove(family.id) == nil { closedModelFamilies.insert(family.id) }
                    },
                    onToggleSelection: {
                        if allSelected {
                            selectedModelKeys.subtract(keys)
                        } else {
                            selectedModelKeys.formUnion(keys)
                            closedModelFamilies.remove(family.id)
                        }
                    }
                )
                if isOpen {
                    ForEach(family.members) { group in
                        Divider().padding(.leading, 16 + ModelFamilyHeader.chevronWidth
                                              + ModelFamilyHeader.memberIndent)
                        modelRow(group, logoDigest: family.logoDigest)
                            .padding(.leading, ModelFamilyHeader.memberIndent)
                    }
                }
            }
        }
    }

    private func modelRow(_ group: ModelGroup, logoDigest: String?) -> some View {
        let isSelected = selectedModelKeys.contains(group.key)
        return SelectableModelGroupRow(group: group, isSelected: isSelected, logoDigest: logoDigest, checkboxSize: 26,
                                       leadingInset: ModelFamilyHeader.chevronWidth,
                                       showsDownloadedHint: showsUndownloadedModels) {
            if isSelected {
                selectedModelKeys.remove(group.key)
            } else {
                selectedModelKeys.insert(group.key)
            }
        }
    }

    private var quantSection: some View {
        QuantizationSelector(
            selectedQuants: $selectedQuants,
            subtitle: "Select the level of compression to use",
            disabledQuants: unavailableQuants,
            // An empty quant filter would resolve zero model files.
            allowsEmptySelection: false
        )
    }

    private var modelsEmptyState: some View {
        EmptyModelsPrompt.needModelsForJobs {
            onGoToModels()
        }
    }

    // MARK: - Step 3: Benchmark selection

    private var benchmarkStep: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                WizardStepHeader(
                    title: "Benchmark selection",
                    description: "Select benchmarks to track model performance on.\nExpand to select context size."
                )
                benchmarkSearchField
                benchmarkSelectionCard

                if isVlSelected && resolvedModelFiles.contains(where: { !Self.declaresProjector($0.source) }) {
                    mmprojSelectionCard
                }
            }
            .padding(.horizontal, stepHorizontalPadding)
            .padding(.top, stepTopPadding)
            .padding(.bottom, stepBottomPadding)
        }
        .background(Color(.systemBackground))
    }

    private var benchmarkSearchField: some View {
        HStack(spacing: 14) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(.secondary)
            TextField("Search benchmarks", text: $benchmarkSearch)
                .font(.system(size: 16))
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            if !benchmarkSearch.isEmpty {
                Button { benchmarkSearch = "" } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .frame(height: 38)
        .padding(.horizontal, 18)
        .appCard(cornerRadius: 8)
        .shadow(color: .black.opacity(0.07), radius: 2, y: 1)
    }

    private var benchmarkSelectionCard: some View {
        VStack(spacing: 0) {
            if benchmarks.isEmpty {
                Text("No benchmarks yet. Check your connection or sync in Settings to load the catalog.")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 32)
            } else if filteredGroupedBenchmarks.isEmpty {
                Text("No benchmarks match \"\(benchmarkSearch)\".")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 32)
            } else {
                ForEach(Array(filteredGroupedBenchmarks.enumerated()), id: \.element.type) { index, group in
                    if index > 0 {
                        Divider()
                    }
                    benchmarkGroupBlock(group: group)
                }
            }
        }
        .appCard(cornerRadius: 22)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
    }

    private func benchmarkGroupBlock(group: (type: String, items: [BenchmarkItem])) -> some View {
        VStack(spacing: 0) {
            benchmarkGroupHeader(group: group)
            if expandedBenchmarkTypes.contains(group.type) {
                benchmarkGroupOptions(group: group)
            }
        }
    }

    private func benchmarkGroupHeader(group: (type: String, items: [BenchmarkItem])) -> some View {
        let enabledItems = group.items.filter { !isBenchmarkSelectionDisabled($0) }
        let groupDisabled = enabledItems.isEmpty
        let selectedCount = group.items.filter { selectedBenchmarks.contains($0.benchmarkId) }.count
        let allGroupSelected = !enabledItems.isEmpty
            && enabledItems.allSatisfy { selectedBenchmarks.contains($0.benchmarkId) }
        let isExpanded = expandedBenchmarkTypes.contains(group.type)

        return HStack(spacing: 0) {
            Button { toggleBenchmarkGroupExpansion(group.type) } label: {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: benchmarkLeadingColumnWidth)
                    .frame(maxHeight: .infinity)
            }
            .buttonStyle(.plain)

            Button { toggleBenchmarkGroupExpansion(group.type) } label: {
                VStack(alignment: .leading, spacing: 8) {
                    Text(BenchmarkCatalog.displayName(for: group.type))
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(groupDisabled ? .secondary : .primary)
                    Text(BenchmarkCatalog.description(for: group.type))
                        .font(.system(size: 14.5))
                        .foregroundStyle(.secondary)
                        .lineSpacing(5)
                        .fixedSize(horizontal: false, vertical: true)
                    if !BuildFlavor.hasPrivateMemory,
                       MemoryFreshness.isMemory(BenchmarkType(rawValue: group.type)) {
                        Label {
                            Text(MemoryFreshness.pickerNote)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        } icon: {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.orange)
                        }
                        .font(.system(size: 13))
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Button {
                toggleBenchmarkGroup(items: enabledItems, allSelected: allGroupSelected)
            } label: {
                WizardCheckbox(
                    isOn: allGroupSelected,
                    isMixed: selectedCount > 0 && !allGroupSelected,
                    size: 22
                )
            }
            .buttonStyle(.plain)
            .disabled(groupDisabled)
            .padding(.leading, 12)
            .padding(.trailing, 22)
        }
        .frame(minHeight: 92)
        .opacity(groupDisabled ? 0.65 : 1)
    }

    private func benchmarkGroupOptions(group: (type: String, items: [BenchmarkItem])) -> some View {
        let groupDisabled = group.items.allSatisfy(Self.isVlBenchmark) && !canSelectVisionBenchmarks

        return ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(Color(.secondarySystemBackground))

            HStack(alignment: .top, spacing: 0) {
                Color.clear
                    .frame(width: benchmarkLeadingColumnWidth)

                benchmarkOptionsList(group: group, groupDisabled: groupDisabled)
                    .padding(.trailing, 28)
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .overlay(alignment: .top) {
            BenchmarkDetailSeparator()
        }
    }

    @ViewBuilder
    private func benchmarkOptionsList(
        group: (type: String, items: [BenchmarkItem]),
        groupDisabled: Bool
    ) -> some View {
        benchmarkOptionsStack(group: group, groupDisabled: groupDisabled)
    }

    private func benchmarkOptionsStack(
        group: (type: String, items: [BenchmarkItem]),
        groupDisabled: Bool
    ) -> some View {
        VStack(spacing: 20) {
            if groupDisabled {
                Text("Every selected model needs a compatible projector for vision benchmarks.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            ForEach(group.items) { item in
                benchmarkOptionRow(item)
            }
        }
        .padding(.vertical, 22)
    }

    private func benchmarkOptionRow(_ item: BenchmarkItem) -> some View {
        let itemDisabled = isBenchmarkSelectionDisabled(item)
        let isSelected = selectedBenchmarks.contains(item.benchmarkId)

        return Button {
            toggleBenchmark(item)
        } label: {
            HStack(spacing: 12) {
                Text(benchmarkOptionLabel(item))
                    .font(.system(size: 16))
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
                    .padding(.horizontal, 13)
                    .frame(height: 28)
                    .background(Color(.systemBackground), in: Capsule())
                    .capsuleBorder()
                if item.source == .local {
                    LocalResultBadge()
                }
                Spacer(minLength: 8)
                WizardCheckbox(isOn: isSelected, size: 22)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(itemDisabled)
        .opacity(itemDisabled ? 0.45 : 1)
    }

    private var mmprojSelectionCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("MMProjectors")
                .font(.system(size: 17, weight: .semibold))
            Text("Select a compatible projector for each model that needs one.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            if mmprojFiles.isEmpty {
                Text("No mmproj files. Download one from the Models tab.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 4)
            } else {
                let allSelected = mmprojFiles.allSatisfy { selectedMmproj.contains($0.path) }
                Button {
                    if allSelected {
                        selectedMmproj.removeAll()
                    } else {
                        selectedMmproj = Set(mmprojFiles.map { $0.path })
                    }
                } label: {
                    Text(allSelected ? "Deselect All" : "Select All")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.plain)

                ForEach(mmprojFiles) { mmproj in
                    Button {
                        if selectedMmproj.contains(mmproj.path) {
                            selectedMmproj.remove(mmproj.path)
                        } else {
                            selectedMmproj.insert(mmproj.path)
                        }
                    } label: {
                        HStack(spacing: 12) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(mmproj.name)
                                    .font(.subheadline)
                                    .foregroundStyle(.primary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Text(mmproj.sizeFormatted)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            WizardCheckbox(isOn: selectedMmproj.contains(mmproj.path), size: 22)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(20)
        .appCard(cornerRadius: 22)
    }

    private func toggleBenchmark(_ item: BenchmarkItem) {
        guard !isBenchmarkSelectionDisabled(item) else { return }
        if selectedBenchmarks.contains(item.benchmarkId) {
            selectedBenchmarks.remove(item.benchmarkId)
        } else {
            selectedBenchmarks.insert(item.benchmarkId)
        }
    }

    private func isBenchmarkSelectionDisabled(_ item: BenchmarkItem) -> Bool {
        Self.isVlBenchmark(item) && !selectedBenchmarks.contains(item.benchmarkId) && !canSelectVisionBenchmarks
    }

    private func toggleBenchmarkGroup(items: [BenchmarkItem], allSelected: Bool) {
        if allSelected {
            for item in items { selectedBenchmarks.remove(item.benchmarkId) }
        } else {
            for item in items { selectedBenchmarks.insert(item.benchmarkId) }
        }
    }

    private func toggleBenchmarkGroupExpansion(_ type: String) {
        withAnimation(.easeInOut(duration: 0.18)) {
            if expandedBenchmarkTypes.contains(type) {
                expandedBenchmarkTypes.remove(type)
            } else {
                expandedBenchmarkTypes.insert(type)
            }
        }
    }

    private func sortBenchmarkItems(_ items: [BenchmarkItem]) -> [BenchmarkItem] {
        items.sorted { a, b in
            let ak = benchmarkSortKey(a)
            let bk = benchmarkSortKey(b)
            if ak.prefill != bk.prefill { return ak.prefill < bk.prefill }
            if ak.decode != bk.decode { return ak.decode < bk.decode }
            if ak.width != bk.width { return ak.width < bk.width }
            if ak.height != bk.height { return ak.height < bk.height }
            return a.benchmarkId < b.benchmarkId
        }
    }

    private func benchmarkSortKey(_ item: BenchmarkItem) -> (prefill: Int, decode: Int, width: Int, height: Int) {
        (
            intParameter(item, "parameter_prefill_tokens")
                ?? intParameter(item, "parameter_text_tokens")
                ?? Int.max,
            intParameter(item, "parameter_decode_tokens") ?? Int.max,
            intParameter(item, "parameter_image_width") ?? Int.max,
            intParameter(item, "parameter_image_height") ?? Int.max
        )
    }

    private func intParameter(_ item: BenchmarkItem, _ key: String) -> Int? {
        if let value = item.rawJson[key] as? Int { return value }
        return (item.rawJson[key] as? NSNumber)?.intValue
    }

    private func benchmarkOptionLabel(_ item: BenchmarkItem) -> String {
        if let width = intParameter(item, "parameter_image_width"),
           let height = intParameter(item, "parameter_image_height") {
            let text = intParameter(item, "parameter_text_tokens")
            let decode = intParameter(item, "parameter_decode_tokens")
            if let text = text, let decode = decode {
                return "\(width)x\(height) · \(text)tok text · \(decode) tok out"
            }
            return "\(width)x\(height)"
        }

        let prefill = intParameter(item, "parameter_prefill_tokens")
        let decode = intParameter(item, "parameter_decode_tokens")
        switch (prefill, decode) {
        case let (.some(prefill), .some(decode)):
            return "\(prefill)tok in · \(decode) tok out"
        case let (.some(prefill), .none) where item.type == .maxMemoryUsage:
            return "\(prefill)tok context"
        case let (.some(prefill), .none):
            return "\(prefill)tok in"
        default:
            return item.benchmarkId
        }
    }

    private func benchmarkItemMatchesSearch(_ item: BenchmarkItem, query: String) -> Bool {
        item.benchmarkId.lowercased().contains(query)
            || item.benchmarkType.lowercased().contains(query)
            || benchmarkOptionLabel(item).lowercased().contains(query)
    }

    // MARK: - Step 4: Review & run

    private var reviewStep: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                WizardStepHeader(title: "Review & run")

                reviewSummaryCard
                // Hidden, so every job runs at batch 512, until the llama.cpp executables
                // honor the batch too (#407) and iOS and executable runs stay comparable.
                // prefillBatchControl
                if canSubmitResults {
                    reviewContributionToggle
                }

                if let notice = MemoryFreshness.runNotice(
                    .newJob, publicBuild: !BuildFlavor.hasPrivateMemory,
                    memoryCells: memoryCellCount, totalCells: cellCount,
                    ranEngineWork: ProcessMemory.hasRunEngineWork
                ) {
                    NoticeCallout(message: notice.message, isWarning: notice.isWarning,
                                  font: .caption)
                }

                if !modelsMissingQuant.isEmpty {
                    NoticeCallout(
                        message: "\(modelsMissingQuant.map(\.name).joined(separator: ", ")) \(modelsMissingQuant.count == 1 ? "has" : "have") no \(quantSummary) build downloaded and will be skipped.",
                        font: .caption
                    )
                }
            }
            .padding(.horizontal, stepHorizontalPadding)
            .padding(.top, stepTopPadding)
            .padding(.bottom, stepBottomPadding)
        }
        .background(Color(.systemBackground))
    }

    private var prefillBatchControl: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Prefill batch")
                    .font(.system(size: 15, weight: .medium))
                Text(MlxFeatureFlag.visibleInUI ? "llama n_ubatch · MLX prefill chunk" : "llama n_ubatch")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Picker("Prefill batch", selection: $prefillBatch) {
                ForEach(prefillBatchOptions, id: \.self) { Text("\($0)").tag($0) }
            }
            .pickerStyle(.menu)
            .labelsHidden()
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
        .appCard(cornerRadius: 22)
    }

    private var reviewSummaryCard: some View {
        VStack(spacing: 0) {
            Text(reviewSummaryTitle)
                .font(.serif(21))
                .lineLimit(1)
                .minimumScaleFactor(0.78)
                .allowsTightening(true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 24)
                .padding(.vertical, 18)

            Divider()

            VStack(alignment: .leading, spacing: 18) {
                reviewModelsSection
                reviewBenchmarksSection
                reviewQuantsSection
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 20)
        }
        .appCard(cornerRadius: 22)
    }

    private var reviewContributionToggle: some View {
        Button {
            contributeResults.toggle()
        } label: {
            HStack(alignment: .top, spacing: 14) {
                WizardCheckbox(isOn: contributeResults, size: 18)
                    .padding(.top, 5)
                Text("Auto-submit benchmark results to the public dataset when the job finishes. Only performance metrics are shared, never personal or device data.")
                    .font(.system(size: 15))
                    .foregroundStyle(Color.primary.opacity(0.78))
                    .lineSpacing(5)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var reviewSummaryTitle: String {
        let models = selectedReviewModelGroups.count
        let benchmarks = selectedReviewBenchmarkTypes.count
        return "\(JobDateFormat.shortDate.string(from: Date())) · " +
            "\(models) model\(models == 1 ? "" : "s") · " +
            "\(benchmarks) benchmark\(benchmarks == 1 ? "" : "s")"
    }

    private var canSubmitResults: Bool {
        ResultSubmissionFeatureGate.canSubmitResults(registration: ws.registration.get())
    }

    private var reviewQuantLabels: [String] {
        if selectedQuants.contains(.all) { return ["All quants"] }
        return QuantPill.allCases
            .filter { $0 != .all && selectedQuants.contains($0) }
            .map(reviewQuantLabel)
    }

    private var reviewModelsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            reviewChipSectionTitle("Models")

            ChipFlowLayout(horizontalSpacing: 8, verticalSpacing: 9) {
                ForEach(selectedReviewModelGroups) { group in
                    reviewModelChip(group, maxWidth: 170)
                }
            }
        }
    }

    private var reviewBenchmarksSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            reviewChipSectionTitle("Benchmarks")

            ChipFlowLayout(horizontalSpacing: 8, verticalSpacing: 9) {
                ForEach(selectedReviewBenchmarkTypes, id: \.self) { type in
                    reviewTextChip(BenchmarkCatalog.displayName(for: type), maxWidth: 240)
                }
            }
        }
    }

    private var reviewQuantsSection: some View {
        reviewChipSection("Quants") {
            ForEach(reviewQuantLabels, id: \.self) { label in
                reviewTextChip(label, maxWidth: 140)
            }
        }
    }

    private func reviewChipSectionTitle(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 16))
            .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private func reviewChipSection<Content: View>(
        _ title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            reviewChipSectionTitle(title)
            ChipFlowLayout(horizontalSpacing: 8, verticalSpacing: 9) {
                content()
            }
        }
    }

    private func reviewModelChip(
        _ group: ModelGroup,
        width: CGFloat? = nil,
        maxWidth: CGFloat? = nil
    ) -> some View {
        let brand = ModelBrand.detect(name: group.name, hfRepo: group.files.first?.hfRepo)

        return AppModelChip(
            text: group.name,
            brand: brand,
            logoSize: 22,
            font: .system(size: 16),
            width: width,
            maxWidth: maxWidth
        )
    }

    private func reviewTextChip(
        _ text: String,
        width: CGFloat? = nil,
        maxWidth: CGFloat? = nil
    ) -> some View {
        AppTextChip(
            text: text,
            font: .system(size: 16),
            width: width,
            maxWidth: maxWidth
        )
    }

    private func reviewQuantLabel(_ pill: QuantPill) -> String {
        pill.label
    }

    private var quantSummary: String {
        if selectedQuants.contains(.all) { return "All quants" }
        return QuantPill.allCases
            .filter { $0 != .all && selectedQuants.contains($0) }
            .map(\.label)
            .joined(separator: ", ")
    }

    // MARK: - Data loading

    private func loadData() {
        loadModels()
        benchmarks = BenchmarkCatalog.selectable(store: ws.benchmarks)

        guard !didApplyInitialSelections else { return }
        // Model and benchmark selections are seeded when a runtime is picked. Every
        // projector starts selected so VL pairing works as soon as a VL benchmark is.
        selectedMmproj = Set(mmprojFiles.map(\.path))
        didApplyInitialSelections = true
    }

    private func loadModels() {
        let all = modelStore.models
        // mmproj files are regular .gguf files, distinguished only by filename.
        let quota = ws.storageQuotaBytes
        let ram = ProcessInfo.processInfo.physicalMemory
        let snapshot = ModelShortList.snapshot(ws: ws)
        catalogEntries = CatalogEntry.firstEntries(in: snapshot.entries)
        catalogLogoOwner = snapshot.owner
        familyLogoDigests = CatalogEntry.familyLogoDigests(in: snapshot.entries)
        models = Self.offerable(downloaded: all.filter {
            $0.name.range(of: "mmproj", options: .caseInsensitive) == nil
        }, catalog: snapshot.entries).filter {
            CatalogEntry.sizeFits($0.sizeBytes, quotaBytes: quota, ramBytes: ram)
        }
        mmprojFiles = all.filter { $0.name.range(of: "mmproj", options: .caseInsensitive) != nil }

        guard !selectedModelKeys.isEmpty else { return }
        // A finished download narrows the list; keep a selected catalog model visible.
        if !showsUndownloadedModels,
           Self.selectionNeedsUndownloaded(selectedModelKeys, offered: runtimeModels) {
            showsUndownloadedModels = true
        }
        let available = Set(eligibleModels.map(\.modelId))
        let selection = Self.reconcileModelSelection(selectedModelKeys, available: available, step: step)
        selectedModelKeys = selection.keys
        if step != selection.step {
            step = selection.step
            lowPowerWarningPending = false
        }
        if step == .review && !visionSelectionIsReady {
            step = .benchmarks
            lowPowerWarningPending = false
        }
    }

    static func reconcileModelSelection(
        _ selected: Set<String>, available: Set<String>, step: Step
    ) -> (keys: Set<String>, step: Step) {
        let kept = selected.intersection(available)
        if !selected.isEmpty, kept.isEmpty, step == .benchmarks || step == .review {
            return (kept, .models)
        }
        return (kept, step)
    }

    private func syncBenchmarksOnVisit() {
        guard !isSyncingBenchmarks else { return }
        guard let registration = ws.registration.get() else { return }

        isSyncingBenchmarks = true
        let serverUrl = registration.serverUrl
        let ws = ws
        let shouldSeedAfterSync = benchmarks.isEmpty

        Task.detached {
            do {
                _ = try await BenchmarkSyncCoordinator.shared.sync(
                    serverUrl: serverUrl,
                    ws: ws
                )
                await MainActor.run {
                    loadData()
                    if shouldSeedAfterSync, selectedRuntime != nil, selectedBenchmarks.isEmpty {
                        seedBenchmarksForSelectedRuntime()
                    }
                }
            } catch {
                AppLog.benchmarkSync.error("new job sync failed: \(error)")
            }
            await MainActor.run {
                isSyncingBenchmarks = false
            }
        }
    }

    private func seedSelectionsForRuntime() {
        selectedModelKeys = []
        showsUndownloadedModels = false
        closedModelFamilies = []
        selectedQuants = QuantPill.allSelection()
        selectedMmproj = Set(mmprojFiles.map(\.path))
        seedBenchmarksForSelectedRuntime()
    }

    private func seedBenchmarksForSelectedRuntime() {
        let visionAvailable = canSelectVisionBenchmarks
        selectedBenchmarks = Set(runtimeBenchmarks.compactMap { item in
            Self.isVlBenchmark(item) && !visionAvailable ? nil : item.benchmarkId
        })
    }

    // MARK: - Actions

    private func attemptStart() {
        if detectPowerSaveMode() {
            lowPowerWarningPending = true
        } else {
            start()
        }
    }

    private func start() {
        guard visionSelectionIsReady, cellCount > 0 else { return }
        let selectedBenchmarkItems = benchmarks.filter { selectedBenchmarks.contains($0.benchmarkId) }
        let selectedModelItems = resolvedModelFiles
        let projectors = selectedProjectors

        // Model outer, benchmark inner: the runner loads each model once and runs
        // all of its benchmarks before unloading.
        var cells: [JobCell] = []
        for model in selectedModelItems {
            for benchmark in selectedBenchmarkItems {
                for source in sources(for: model, benchmark: benchmark, projectors: projectors) {
                    cells.append(JobCell(
                        cellId: CellId(UUID().uuidString),
                        benchmarkId: benchmark.benchmarkId,
                        benchmarkType: benchmark.type,
                        runStatus: .pending,
                        serverJobId: nil,
                        errorMessage: nil,
                        spec: ClientRunSpec.authored(
                            benchmarkId: benchmark.benchmarkId,
                            benchmarkType: benchmark.type, model: source,
                            numberGpuLayers: UInt32(max(0, nGpuLayers)),
                            nUbatch: UInt32(max(1, prefillBatch))),
                        benchmarkSource: benchmark.source
                    ))
                }
            }
        }

        // The largest window any selected cell asks for, for the record only: each cell
        // is sized from its own benchmark at load.
        let jobContextSize = selectedBenchmarkItems
            .compactMap { $0.definition.map { Int(LlamaRuntimeFlags.ctxSize(for: $0)) } }
            .max() ?? 0

        let shouldAutoSubmitResults = canSubmitResults && contributeResults

        guard let jobId = JobLauncher.launch(
            cells: cells,
            nGpuLayers: nGpuLayers,
            contextSize: jobContextSize,
            prefillBatch: prefillBatch,
            contributeResults: shouldAutoSubmitResults,
            jobRunner: jobRunner,
            jobStore: jobStore,
            ws: ws
        ) else {
            startBlockedMessage = "Pause or finish the current job before starting another one."
            return
        }

        // JobDetailView presents Pocket Mode when it appears for this job.
        jobRunner.pocketModeRequestedJobId = jobId

        onStarted(jobId)
    }
}

private struct BenchmarkDetailSeparator: View {
    var body: some View {
        Color.primary.opacity(0.08)
            .frame(height: 1)
    }
}

struct ChipFlowLayout: Layout {
    var horizontalSpacing: CGFloat = 8
    var verticalSpacing: CGFloat = 8

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        let rows = rows(for: subviews, maxWidth: maxWidth)
        let height = rows.reduce(CGFloat.zero) { total, row in
            total + row.height
        } + CGFloat(max(0, rows.count - 1)) * verticalSpacing
        let width = proposal.width ?? rows.map(\.width).max() ?? 0
        return CGSize(width: width, height: height)
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        let rows = rows(for: subviews, maxWidth: bounds.width)
        var y = bounds.minY

        for row in rows {
            var x = bounds.minX
            for item in row.items {
                subviews[item.index].place(
                    at: CGPoint(x: x, y: y),
                    anchor: .topLeading,
                    proposal: ProposedViewSize(item.size)
                )
                x += item.size.width + horizontalSpacing
            }
            y += row.height + verticalSpacing
        }
    }

    private func rows(for subviews: Subviews, maxWidth: CGFloat) -> [FlowRow] {
        var rows: [FlowRow] = []
        var current = FlowRow()
        let wraps = maxWidth.isFinite && maxWidth > 0

        for index in subviews.indices {
            let size = subviews[index].sizeThatFits(.unspecified)
            let nextWidth = current.items.isEmpty
                ? size.width
                : current.width + horizontalSpacing + size.width

            if wraps, !current.items.isEmpty, nextWidth > maxWidth {
                rows.append(current)
                current = FlowRow()
            }

            current.append(index: index, size: size, spacing: horizontalSpacing)
        }

        if !current.items.isEmpty {
            rows.append(current)
        }

        return rows
    }

    private struct FlowRow {
        var items: [FlowItem] = []
        var width: CGFloat = 0
        var height: CGFloat = 0

        mutating func append(index: Int, size: CGSize, spacing: CGFloat) {
            if !items.isEmpty {
                width += spacing
            }
            items.append(FlowItem(index: index, size: size))
            width += size.width
            height = max(height, size.height)
        }
    }

    private struct FlowItem {
        let index: Int
        let size: CGSize
    }
}

#if DEBUG
#Preview("New Job") {
    NewJobView()
        .environment(JobRunner())
        .environment(JobStore(ws: PipetteWorkspace.production))
        .environment(ModelStore(ws: PipetteWorkspace.production))
}

/// The simulator always reports Low Power Mode off, so this is the only way to see
/// the warning in the canvas.
#Preview("Low Power Mode Warning") {
    struct WarningPreview: View {
        @State private var isPresented = true
        var body: some View {
            Color(.systemBackground)
                .ignoresSafeArea()
                .lowPowerModeWarning(isPresented: $isPresented) {}
        }
    }
    return WarningPreview()
}
#endif
