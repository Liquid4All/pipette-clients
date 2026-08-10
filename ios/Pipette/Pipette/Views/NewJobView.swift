import SwiftUI
import UIKit

/// Creates and runs a new job via a four-step wizard: pick a runtime, then the
/// models (+ quant filter), the benchmarks, then review & run. The runtime is chosen
/// first and filters both downstream lists — only that engine's models and only the
/// benchmarks it can run are offered — so a job targets a single runtime. Cells are
/// the cartesian product of the resolved model files × selected benchmarks, executed
/// sequentially.
struct NewJobView: View {
    @Environment(JobRunner.self) private var jobRunner
    @Environment(JobStore.self) private var jobStore
    @Environment(ModelStore.self) private var modelStore
    @Environment(\.storage) private var storage
    @Environment(\.dismiss) private var dismiss

    /// Invoked after the run loop is launched. The host navigation stack uses
    /// this to replace NewJobView with the JobDetailView for the new job so
    /// the user sees live progress in the detail screen instead of inline.
    var onStarted: (JobId) -> Void = { _ in }

    /// Invoked from the empty state's "Go to Models" button. The host pops this
    /// screen and switches to the Models tab so the user can download models.
    var onGoToModels: () -> Void = {}

    enum Step: Int, CaseIterable {
        case runtime, models, benchmarks, review
    }

    @State private var step: Step = .runtime
    /// The engine this job targets. Chosen on step 1; nil until picked. Changing it
    /// re-seeds the downstream selections (`seedSelectionsForRuntime`).
    @State private var selectedRuntime: RuntimeKind?

    @State private var benchmarks: [BenchmarkItem] = []
    @State private var models: [DiscoveredModel] = []
    @State private var mmprojFiles: [DiscoveredModel] = []
    @State private var isSyncingBenchmarks: Bool = false

    @State private var selectedModelKeys: Set<String> = []  // keyed by group key
    @State private var selectedQuants: Set<QuantPill> = QuantPill.allSelection()
    @State private var modelSearch: String = ""
    @State private var benchmarkSearch: String = ""
    @State private var expandedBenchmarkTypes: Set<String> = []
    @State private var selectedBenchmarks: Set<String> = []
    @State private var selectedMmproj: Set<String> = []  // keyed by path
    // Opt-in default; only ever used when `canSubmitResults` (registration) also
    // holds — see `shouldAutoSubmitResults`, which ANDs the two.
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
    // Prefill batch applied to both runtimes (llama n_ubatch / MLX prefill chunk).
    @State private var prefillBatch: Int = 512
    private let prefillBatchOptions = [256, 512, 1024, 2048]

    // MARK: - VL helpers

    nonisolated private static func isVlBenchmark(_ item: BenchmarkItem) -> Bool {
        item.type?.isVisionLanguage == true
    }

    /// A base model is VL-capable iff it names a projector or one can be paired with it.
    /// A `.ggufVision` coordinate names both files itself — discovery returns the two as
    /// a single entry, so there is no separate mmproj row to match. Otherwise VL-ness is
    /// a pairing of two separately-discovered files: primary match is shared `hfRepo`,
    /// fallback is a normalized filename-stem match for sideloads.
    nonisolated private static func isVlCompatible(_ model: DiscoveredModel, mmprojFiles: [DiscoveredModel]) -> Bool {
        if case .ggufVision = model.source { return true }
        let stem = LocalStorage.normalizedModelStem(model.name)
        for mmproj in mmprojFiles {
            if model.hfRepo == mmproj.hfRepo { return true }
            if LocalStorage.normalizedModelStem(mmproj.name) == stem { return true }
        }
        return false
    }

    /// Whether `model`'s runtime can run `benchmark` at all — the enum-driven
    /// capability check (`isBenchmarkSupported`), independent of the VL model-pairing
    /// gate. Support is config-independent, so the exact knobs don't matter; we pass
    /// the wizard's current config for a faithful `Runtime`. A benchmark whose type we
    /// don't recognize is treated as runnable (the runtime decides at execution time).
    private func runtimeSupports(_ benchmark: BenchmarkItem, on model: DiscoveredModel) -> Bool {
        guard let bt = BenchmarkType(type: benchmark.benchmarkType) else { return true }
        // A capability query, so the engine alone answers it — no cell settings exist
        // yet at this point in the wizard.
        return isBenchmarkSupported(bt, on: RuntimeKind(model.source))
    }

    private var isVlSelected: Bool {
        benchmarks.contains { selectedBenchmarks.contains($0.benchmarkId) && Self.isVlBenchmark($0) }
    }

    private var hasVlModelSelected: Bool {
        resolvedModelFiles.contains { Self.isVlCompatible($0, mmprojFiles: mmprojFiles) }
    }

    // MARK: - Model grouping & quant resolution

    /// Runtimes with at least one model the run screen can offer — on device or in the
    /// catalog, since a benchmark may be started against either. AFM appears whenever
    /// it's available, since `Storage.availableModels` injects it as a model file.
    private var availableRuntimes: [RuntimeKind] {
        let kinds = Set(models.map { RuntimeKind($0.source) })
        return RuntimeKind.allCases.filter { kinds.contains($0) }
    }

    /// Everything the run screen can offer: what is on disk, plus the rest of the
    /// catalog. A benchmark may be started against a model that has not been fetched —
    /// `RunCell.prepare` calls `ensureModel` before the cell runs — so restricting the
    /// list to downloads was a UI rule the run path never needed.
    ///
    /// Downloaded rows win a collision, since only they know the real on-disk size and
    /// path; the catalog's figure is the declared download size.
    ///
    /// Collision is judged by `reference` — the crate's identity for a coordinate — not
    /// by the whole `Model`. Equality there also covers the integrity metadata a plan may
    /// carry (`sha256`), which the catalog never states, so a model downloaded for a
    /// claimed job would otherwise fail to match its own catalog row and list twice.
    nonisolated static func offerable(downloaded: [DiscoveredModel]) -> [DiscoveredModel] {
        var byReference: [String: DiscoveredModel] = [:]
        for entry in CatalogEntry.catalog {
            byReference[entry.source.reference] = DiscoveredModel(catalog: entry)
        }
        for model in downloaded { byReference[model.source.reference] = model }
        // A stable order the list can rely on: family, then quant, then the coordinate.
        return byReference.values.sorted {
            ($0.familyId, $0.quant ?? "", $0.source.reference)
                < ($1.familyId, $1.quant ?? "", $1.source.reference)
        }
    }

    /// The VL coordinates to run `model` as — one per cell.
    ///
    /// A model discovered as `.ggufVision` already names its projector, and that
    /// coordinate is the portable declared form `RunCell.prepare` binds; re-deriving it
    /// from host paths would throw that away. It is one cell, not one per selected
    /// projector, since the pairing is not the user's to choose.
    ///
    /// A sideloaded base has no such coordinate — its projector was discovered as a
    /// separate row — so the pair is named by the two host paths, which exist only once
    /// both are downloaded. An undownloaded pair yields nothing rather than a cell that
    /// cannot name its own files.
    nonisolated static func visionSources(
        for model: DiscoveredModel, mmprojFiles: [DiscoveredModel]
    ) -> [Model] {
        if case .ggufVision = model.source { return [model.source] }
        return mmprojFiles.compactMap { mmproj in
            guard model.isDownloaded, mmproj.isDownloaded,
                  let weights = try? AbsolutePath(model.path),
                  let projector = try? AbsolutePath(mmproj.path)
            else { return nil }
            return .ggufVision(.init(source: .absoluteFiles(model: weights, mmproj: projector)))
        }
    }

    /// Models belonging to the selected runtime — the pool the model step draws from.
    /// Empty until a runtime is picked, so a job targets a single engine.
    private var eligibleModels: [DiscoveredModel] {
        // Every discovered model carries a manifest (typed provenance), so all are
        // eligible — discovery already rejected anything manifest-less.
        guard let rt = selectedRuntime else { return [] }
        return models.filter { RuntimeKind($0.source) == rt }
    }

    /// Benchmarks the selected runtime can run — `isBenchmarkSupported` applied up
    /// front so the benchmark step only offers runnable options (VL is additionally
    /// gated per-model by an mmproj for llama.cpp). An unrecognized type is kept; the
    /// runtime decides at run time.
    private var runtimeBenchmarks: [BenchmarkItem] {
        guard let rt = selectedRuntime else { return [] }
        let runtime = rt
        return benchmarks.filter { item in
            guard let bt = item.type else { return true }
            return isBenchmarkSupported(bt, on: runtime)
        }
    }

    private var modelGroups: [ModelGroup] {
        // All downloaded models, regardless of engine — there is no runtime picker;
        // a job's cells run on whatever engine each selected model implies.
        ModelCatalog.groups(from: eligibleModels)
    }

    /// Search filters only the model list, never the quant pills.
    private var filteredModelGroups: [ModelGroup] {
        let q = modelSearch.searchNormalized
        guard !q.isEmpty else { return modelGroups }
        return modelGroups.filter {
            $0.name.lowercased().contains(q) || $0.key.lowercased().contains(q)
        }
    }

    /// Quant pills with no matching downloaded file among the selected models
    /// (or all of the runtime's models while nothing is selected yet). Grayed
    /// out and unselectable so the filter can only name quants the job could
    /// actually run. Search never affects this — it filters the model list,
    /// not the pills.
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

    /// Concrete GGUF files to run: selected groups × the active quant filter.
    private var resolvedModelFiles: [DiscoveredModel] {
        modelGroups
            .filter { selectedModelKeys.contains($0.key) }
            .flatMap { group in group.files.filter { quantMatchesFilter($0.quant) } }
    }

    /// Selected models that have no on-disk file matching the chosen quant.
    /// Surfaced as a pre-run warning; they are skipped at run time.
    private var modelsMissingQuant: [ModelGroup] {
        guard !selectedQuants.contains(.all) else { return [] }
        return modelGroups
            .filter { selectedModelKeys.contains($0.key) }
            .filter { group in !group.files.contains { quantMatchesFilter($0.quant) } }
    }

    /// Non-VL cells = non_vl_benchmarks × model files.
    /// VL cells = vl_benchmarks × vl_compatible_model_files × mmproj; plain LLMs
    /// are skipped because they cannot run VL benchmarks.
    private var cellCount: Int {
        let selectedItems = benchmarks.filter { selectedBenchmarks.contains($0.benchmarkId) }
        let resolved = resolvedModelFiles
        // Mirror `start()`'s per-(model, benchmark) expansion: drop pairs the model's
        // runtime can't run, then count 1 cell per non-VL pair and one per (VL pair ×
        // compatible model × mmproj). Keeping this in lockstep with the loop is what
        // makes the review step's zero-cell gate honest.
        return resolved.reduce(0) { total, model in
            total + selectedItems.reduce(0) { sub, benchmark in
                guard runtimeSupports(benchmark, on: model) else { return sub }
                if Self.isVlBenchmark(benchmark) {
                    guard Self.isVlCompatible(model, mmprojFiles: mmprojFiles) else { return sub }
                    return sub + selectedMmproj.count
                }
                return sub + 1
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
            return !selectedBenchmarks.isEmpty && !(isVlSelected && selectedMmproj.isEmpty)
        case .review:
            return cellCount > 0
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
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            loadData()
            syncBenchmarksOnVisit()
        }
        .onChange(of: selectedRuntime) { _, _ in seedSelectionsForRuntime() }
        .onChange(of: selectedModelKeys) { _, _ in pruneVlBenchmarksIfNeeded() }
        .onChange(of: selectedQuants) { _, _ in pruneVlBenchmarksIfNeeded() }
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

    /// When the selected models lose their last VL-compatible file, auto-deselect
    /// any VL benchmarks so the user isn't left with an invalid configuration.
    private func pruneVlBenchmarksIfNeeded() {
        if !hasVlModelSelected {
            let vlIds = Set(benchmarks.filter(Self.isVlBenchmark).map { $0.benchmarkId })
            selectedBenchmarks.subtract(vlIds)
        }
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

    /// The empty runtime/model states have no advanceable action, so we drop the
    /// Back/Next footer there and let the "Go to Models" button stand alone.
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
        let count = models.filter { RuntimeKind($0.source) == rt }.count
        return Button {
            withAnimation { selectedRuntime = rt }
        } label: {
            HStack(spacing: 14) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(rt.label)
                        .font(.system(size: 17, weight: .medium))
                        .foregroundStyle(.primary)
                    // AFM has no downloads, so a count would be meaningless there.
                    Text(rt == .afm ? rt.detail : "\(rt.detail) · \(count) downloaded")
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
                modelList
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                // Quant filtering only applies to GGUF/llama.cpp, which ships several
                // quants per family. MLX is uniformly 4-bit and AFM has none, so the
                // filter stays at "all" (matching every file) and the section is hidden.
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

    private var modelList: some View {
        let groups = filteredModelGroups
        return ScrollView {
            if groups.isEmpty {
                Text("No models match \u{201C}\(modelSearch)\u{201D}.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 32)
            } else {
                ModelListCard(cornerRadius: 16) {
                    ForEach(Array(groups.enumerated()), id: \.element.id) { index, group in
                        if index > 0 {
                            Divider().padding(.leading, 16)
                        }
                        modelRow(group)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private func modelRow(_ group: ModelGroup) -> some View {
        let isSelected = selectedModelKeys.contains(group.key)
        return SelectableModelGroupRow(group: group, isSelected: isSelected, checkboxSize: 26) {
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
            // An empty quant filter would resolve zero model files, so
            // deselecting here falls back to "All quants" instead.
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

                // MMProjectors multi-select — only visible when a VL benchmark is
                // selected. VL cells require pairing a base model with a projector.
                if isVlSelected {
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
        let groupDisabled = group.items.allSatisfy(Self.isVlBenchmark) && !hasVlModelSelected
        let enabledItems = group.items.filter { !(Self.isVlBenchmark($0) && !hasVlModelSelected) }
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
        let groupDisabled = group.items.allSatisfy(Self.isVlBenchmark) && !hasVlModelSelected

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
        // The secondary-fill background already groups the options under their
        // benchmark, so a leading hierarchy line here would be duplicative.
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
                Text("Requires a VL model whose HF repo also contains an mmproj file.")
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
        let itemDisabled = Self.isVlBenchmark(item) && !hasVlModelSelected
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
            Text("Required for VL benchmarks. Each selected mmproj is paired with each selected model.")
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
        guard !(Self.isVlBenchmark(item) && !hasVlModelSelected) else { return }
        if selectedBenchmarks.contains(item.benchmarkId) {
            selectedBenchmarks.remove(item.benchmarkId)
        } else {
            selectedBenchmarks.insert(item.benchmarkId)
        }
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
                // Prefill batch picker hidden for now — every job runs at the default
                // batch (512). Re-enable by uncommenting `prefillBatchControl` once the
                // llama.cpp executables honor the batch too (#407), so iOS and executable
                // runs stay comparable (the control + state are kept).
                // prefillBatchControl
                if canSubmitResults {
                    reviewContributionToggle
                }

                if !modelsMissingQuant.isEmpty {
                    reviewWarning(
                        "\(modelsMissingQuant.map(\.name).joined(separator: ", ")) \(modelsMissingQuant.count == 1 ? "has" : "have") no \(quantSummary) build downloaded and will be skipped."
                    )
                }
            }
            .padding(.horizontal, stepHorizontalPadding)
            .padding(.top, stepTopPadding)
            .padding(.bottom, stepBottomPadding)
        }
        .background(Color(.systemBackground))
    }

    /// Prefill batch knob — sets llama's `n_ubatch` and MLX's prefill chunk for
    /// every cell in the job. Applies to both runtimes.
    private var prefillBatchControl: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Prefill batch")
                    .font(.system(size: 15, weight: .medium))
                Text(MLXFeatureFlag.visibleInUI ? "llama n_ubatch · MLX prefill chunk" : "llama n_ubatch")
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
        ResultSubmissionFeatureGate.canSubmitResults(registration: storage.identity.getRegistration())
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

    private func reviewWarning(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
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
        let all = modelStore.models
        // mmproj files are regular .gguf files, distinguished only by filename.
        // Upstream HF repos publish them as `mmproj-*.gguf` alongside the base
        // model. Split so users pick them from a dedicated section when VL
        // benchmarks are selected.
        models = Self.offerable(
            downloaded: all.filter { $0.name.range(of: "mmproj", options: .caseInsensitive) == nil })
        mmprojFiles = all.filter { $0.name.range(of: "mmproj", options: .caseInsensitive) != nil }
        benchmarks = BenchmarkCatalog.selectable(store: storage.benchmarks)

        guard !didApplyInitialSelections else { return }
        // No runtime is chosen yet, so there's nothing to pre-select — the model and
        // benchmark selections are seeded when a runtime is picked
        // (`seedSelectionsForRuntime`). mmproj defaults to all-selected so VL pairing
        // works the moment a VL benchmark is picked.
        selectedMmproj = Set(mmprojFiles.map(\.path))
        didApplyInitialSelections = true
    }

    private func syncBenchmarksOnVisit() {
        guard !isSyncingBenchmarks else { return }
        guard let registration = storage.identity.getRegistration() else { return }

        isSyncingBenchmarks = true
        let serverUrl = registration.serverUrl
        let storage = storage
        let shouldSeedAfterSync = benchmarks.isEmpty

        Task.detached {
            do {
                _ = try await BenchmarkSyncCoordinator.shared.sync(
                    serverUrl: serverUrl,
                    storage: storage
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

    /// Re-seed the downstream selection whenever the runtime changes (or is first
    /// chosen): clear the model picks, restore the quant filter to "all", and
    /// pre-select every benchmark the new runtime supports — pruning VL when the
    /// runtime has no VL-capable model (llama.cpp only).
    private func seedSelectionsForRuntime() {
        selectedModelKeys = []
        selectedQuants = QuantPill.allSelection()
        selectedMmproj = Set(mmprojFiles.map(\.path))
        seedBenchmarksForSelectedRuntime()
    }

    private func seedBenchmarksForSelectedRuntime() {
        let hasVlModel = eligibleModels.contains { Self.isVlCompatible($0, mmprojFiles: mmprojFiles) }
        selectedBenchmarks = Set(runtimeBenchmarks.compactMap { item in
            Self.isVlBenchmark(item) && !hasVlModel ? nil : item.benchmarkId
        })
    }

    // MARK: - Actions

    /// Gate the run on Low Power Mode: it throttles CPU/GPU clocks and skews
    /// results, so warn before starting. `start()` proceeds directly when it's off.
    private func attemptStart() {
        if DeviceProbe.detectPowerSaveMode() {
            lowPowerWarningPending = true
        } else {
            start()
        }
    }

    private func start() {
        // Build cells as cartesian product
        let selectedBenchmarkItems = benchmarks.filter { selectedBenchmarks.contains($0.benchmarkId) }
        let selectedModelItems = resolvedModelFiles
        let selectedMmprojItems = mmprojFiles.filter { selectedMmproj.contains($0.path) }

        // Model outer, benchmark inner: the job runner loads each model once
        // and runs all of its benchmarks in a row before unloading. Grouping
        // by model here is what makes that reuse kick in.
        var cells: [JobCell] = []
        for model in selectedModelItems {
            for benchmark in selectedBenchmarkItems {
                // Skip pairs this model's runtime can't run (e.g. AFM prefill / max-memory
                // / VL) — mirrors the VL model-pairing skip below; no unsupported cell is
                // created, so the run doesn't fail late at execution time.
                guard runtimeSupports(benchmark, on: model) else { continue }
                let isVl = Self.isVlBenchmark(benchmark)
                let benchmarkType = benchmark.type
                if isVl {
                    // Skip base models without a paired mmproj — plain LLMs
                    // cannot run VL benchmarks even if the user selected them.
                    guard Self.isVlCompatible(model, mmprojFiles: mmprojFiles) else { continue }
                    // The projector is part of the cell's source: `GgufVisionSource` names
                    // both files, so a VL cell is one coordinate rather than a text source
                    // carrying a projector path beside it.
                    for source in Self.visionSources(for: model, mmprojFiles: selectedMmprojItems) {
                        cells.append(JobCell(
                            cellId: CellId(UUID().uuidString),
                            benchmarkId: benchmark.benchmarkId,
                            benchmarkType: benchmarkType,
                            runStatus: .pending,
                            serverJobId: nil,
                            errorMessage: nil,
                            spec: ClientRunSpec.authored(
                                benchmarkId: benchmark.benchmarkId,
                                benchmarkType: benchmarkType, model: source,
                                numberGpuLayers: UInt32(max(0, nGpuLayers)),
                                nUbatch: UInt32(max(1, prefillBatch))),
                            benchmarkSource: benchmark.source
                        ))
                    }
                } else {
                    cells.append(JobCell(
                        cellId: CellId(UUID().uuidString),
                        benchmarkId: benchmark.benchmarkId,
                        benchmarkType: benchmarkType,
                        runStatus: .pending,
                        serverJobId: nil,
                        errorMessage: nil,
                        spec: ClientRunSpec.authored(
                            benchmarkId: benchmark.benchmarkId,
                            benchmarkType: benchmarkType, model: model.source,
                            numberGpuLayers: UInt32(max(0, nGpuLayers)),
                            nUbatch: UInt32(max(1, prefillBatch))),
                        benchmarkSource: benchmark.source
                    ))
                }
            }
        }

        // The largest window any selected cell will load with, as the engine sizes them —
        // a job-level summary for the record. No cell reads it: each is sized from its own
        // benchmark at load.
        let jobContextSize = selectedBenchmarkItems
            .compactMap { $0.definition.map { Int(LlamaRuntimeFlags.contextSize(for: $0)) } }
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
            storage: storage
        ) else {
            startBlockedMessage = "Pause or finish the current job before starting another one."
            return
        }

        // Starting a job drops straight into Pocket Mode; JobDetailView presents
        // it when it appears for this job.
        jobRunner.pocketModeRequestedJobId = jobId

        // Hand off to the host navigation stack, which replaces this view
        // with JobDetailView(jobId:) so the user watches progress there.
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
        .environment(JobStore(storage: FileStorage.production))
        .environment(ModelStore(storage: FileStorage.production))
}

/// Standalone preview of the Low Power Mode warning — the simulator always
/// reports Low Power Mode off, so this is the only way to see the dialog in the
/// canvas. Run the preview live (▶) to interact with it.
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
