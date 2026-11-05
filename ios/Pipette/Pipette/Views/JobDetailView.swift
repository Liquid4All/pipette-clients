import Combine
import SwiftUI

/// Shows a job's cells with run/submission status; provides retry, rerun, and delete.
struct JobDetailView: View {
    let jobId: JobId

    @Environment(JobRunner.self) private var jobRunner
    @Environment(JobStore.self) private var jobStore
    @Environment(\.ws) private var ws
    @State private var inFlight: InFlight = .idle
    @State private var retryFlag: CancelFlag?
    @State private var showDeleteConfirmation = false
    @State private var showSubmitConfirmation = false
    @State private var showPocketMode = false
    @State private var showRenameDialog = false
    @State private var renameText = ""
    @State private var selectedCellIds: Set<CellId> = []
    @State private var activeAlert: ActiveAlert?
    @State private var showLowPowerWarning = false
    @State private var pendingRunScope: RunScope?
    /// The memory warning for `pendingRunScope`: its own alert, or appended to the Low Power one.
    @State private var memoryWarning: String?
    @State private var showMemoryWarning = false
    @State private var now: Date = Date()
    @Environment(\.dismiss) private var dismiss

    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    /// Read through the store, so changes from anywhere (including the executor's
    /// background saves) show without a manual reload.
    private var manifest: JobManifest? {
        jobStore.job(id: jobId)
    }

    private var isActiveRunningPage: Bool {
        jobRunner.runningJobId == jobId
    }

    // Every job page uses the run-detail chrome (empty nav title, hidden pill tab
    // bar); only the "Job Not Found" fallback keeps the standard one.
    private var usesRunDetailChrome: Bool {
        manifest != nil
    }

    @ViewBuilder private var jobContent: some View {
        Group {
            if let manifest {
                if isActiveRunningPage {
                    runningJobPage(manifest)
                } else if shouldUsePausedRunPage(manifest) {
                    pausedJobPage(manifest)
                } else {
                    // A paused run stays on the progress page only while nothing
                    // has completed yet.
                    completedRunPage(manifest)
                }
            } else {
                ContentUnavailableView("Job Not Found", systemImage: "questionmark.circle")
            }
        }
    }

    private var contentWithToolbar: some View {
        jobContent
        // The run-detail title lives in the page body, but the system navigation bar
        // stays so the back button and swipe-back work everywhere.
        .navigationTitle(usesRunDetailChrome ? "" : manifest?.displayTitle ?? "Job")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button {
                        renameText = manifest?.title ?? ""
                        showRenameDialog = true
                    } label: {
                        Label("Rename", systemImage: "pencil")
                    }
                    Button(role: .destructive) {
                        showDeleteConfirmation = true
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                } label: {
                    Label("More", systemImage: "ellipsis.circle")
                }
            }
        }
    }

    var body: some View {
        contentWithToolbar
        .alert("Rename Job", isPresented: $showRenameDialog) {
            TextField("Name", text: $renameText)
                .autocorrectionDisabled()
            Button("Save") { renameJob(to: renameText) }
            Button("Reset to Default", role: .destructive) { renameJob(to: "") }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Leave empty to reset to the default name.")
        }
        .confirmationDialog("Delete Job?", isPresented: $showDeleteConfirmation, titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                jobStore.delete(jobId: jobId)
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will delete the job and all its results from the device.")
        }
        .confirmationDialog(
            "Submit Results?",
            isPresented: $showSubmitConfirmation,
            titleVisibility: .visible
        ) {
            Button("Submit \(currentUnsubmittedCount) \(currentUnsubmittedCount == 1 ? "Result" : "Results")") {
                submitAll()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will upload \(currentUnsubmittedCount) completed results to the server.")
        }
        .alert(
            activeAlert?.title ?? "",
            isPresented: Binding<Bool>(
                get: { activeAlert != nil },
                set: { if !$0 { activeAlert = nil } }
            ),
            presenting: activeAlert
        ) { _ in
            Button("OK", role: .cancel) {}
        } message: { alert in
            Text(alert.message)
        }
        .lowPowerModeWarning(isPresented: $showLowPowerWarning, alsoNote: memoryWarning) {
            runPendingScope()
        }
        .alert(MemoryFreshness.alertTitle, isPresented: $showMemoryWarning) {
            Button("Run Anyway") { runPendingScope() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(memoryWarning ?? "")
        }
        .fullScreenCover(isPresented: $showPocketMode) {
            PocketModeView(jobId: jobId)
        }
        // `now` only feeds the running page's time-left estimate, so other pages
        // skip the 1 Hz re-render.
        .onReceive(timer) { tick in
            guard isActiveRunningPage else { return }
            now = tick
        }
        // A user-initiated start asks for Pocket Mode, either before this page is
        // pushed or while it is visible. The request is consumed so it fires once.
        .onAppear { presentPocketModeIfRequested() }
        .onChange(of: jobRunner.pocketModeRequestedJobId) { _, _ in presentPocketModeIfRequested() }
        .preference(key: PillTabBarHiddenPreferenceKey.self, value: usesRunDetailChrome)
    }

    private func presentPocketModeIfRequested() {
        guard jobRunner.pocketModeRequestedJobId == jobId else { return }
        jobRunner.pocketModeRequestedJobId = nil
        showPocketMode = true
    }

    // MARK: - Running layout

    private func runningJobPage(_ manifest: JobManifest) -> AnyView {
        AnyView(RunningJobDetailPage(
            manifest: manifest,
            dateTitle: runningDateTitle(manifest),
            modelChips: runningModelChips(manifest),
            benchmarkChips: runningBenchmarkChips(manifest),
            quantChips: runningQuantChips(manifest),
            mode: .running,
            progressFraction: progressFraction(manifest),
            estimatedTimeLeft: estimatedTimeLeftText(manifest),
            currentCellLabel: jobRunner.currentCellLabel,
            progressText: jobRunner.currentProgressText,
            cooling: jobRunner.coolingState,
            temperatureC: jobRunner.deviceTemperatureC,
            isResumeDisabled: false,
            unsubmittedCount: 0,
            isSubmitting: false,
            onPocketMode: { showPocketMode = true },
            onPause: { jobRunner.cancel() },
            onResume: {},
            onSubmit: {},
            canAutoSubmit: canSubmitResults,
            onAutoSubmitChanged: { setAutoSubmit($0) }
        ))
    }

    private func shouldUsePausedRunPage(_ manifest: JobManifest) -> Bool {
        manifest.cancelledCells > 0 && manifest.completedCells == 0 && manifest.failedCells == 0
    }

    private func pausedJobPage(_ manifest: JobManifest) -> AnyView {
        AnyView(RunningJobDetailPage(
            manifest: manifest,
            dateTitle: runningDateTitle(manifest),
            modelChips: runningModelChips(manifest),
            benchmarkChips: runningBenchmarkChips(manifest),
            quantChips: runningQuantChips(manifest),
            mode: .paused,
            progressFraction: progressFraction(manifest),
            estimatedTimeLeft: nil,
            currentCellLabel: "",
            progressText: "",
            cooling: nil,
            temperatureC: nil,
            isResumeDisabled: inFlight != .idle,
            unsubmittedCount: canSubmitResults ? unsubmittedResultCount(manifest) : 0,
            isSubmitting: inFlight == .submitting,
            onPocketMode: {},
            onPause: {},
            onResume: { resumePaused() },
            onSubmit: { showSubmitConfirmation = true },
            canAutoSubmit: canSubmitResults,
            onAutoSubmitChanged: { setAutoSubmit($0) }
        ))
    }

    private func completedRunPage(_ manifest: JobManifest) -> AnyView {
        let unsubmittedCount = canSubmitResults ? unsubmittedResultCount(manifest) : 0
        let failedCount = manifest.cells.filter { $0.runStatus == .failed }.count
        return AnyView(CompletedRunDetailPage(
            manifest: manifest,
            dateTitle: runningDateTitle(manifest),
            modelChips: runningModelChips(manifest),
            benchmarkChips: runningBenchmarkChips(manifest),
            quantChips: runningQuantChips(manifest),
            resultColumns: CompletedResultsCSVExporter.resultColumns(for: manifest, in: CompletedResultsCSVExporter.catalogById(store: ws.benchmarks)),
            resultGroups: CompletedResultsCSVExporter.resultGroups(for: manifest, ws: ws),
            unsubmittedCount: unsubmittedCount,
            failedCount: failedCount,
            resumableCount: manifest.cancelledCells,
            isSubmitting: inFlight == .submitting,
            isRetrying: inFlight == .retrying,
            selectedCellIds: $selectedCellIds,
            onSubmit: { showSubmitConfirmation = true },
            onResumePaused: { resumePaused() },
            onRetryFailed: { retryFailed() },
            onRerunSelected: { rerunSelected() },
            csvExport: csvExportFile(manifest)
        ))
    }

    /// Cells whose result is still waiting to go up; the store owns the predicate.
    private func unsubmittedResultCount(_ manifest: JobManifest) -> Int {
        ws.results.unsubmittedResultCount(manifest)
    }

    private var currentUnsubmittedCount: Int {
        guard canSubmitResults, let manifest else { return 0 }
        return unsubmittedResultCount(manifest)
    }

    private func renameJob(to newName: String) {
        guard var updated = manifest else { return }
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        updated.title = trimmed.isEmpty ? nil : trimmed
        jobStore.save(updated)
    }

    private func setAutoSubmit(_ isEnabled: Bool) {
        guard var updated = manifest else { return }
        updated.contributeResults = canSubmitResults && isEnabled
        jobStore.save(updated)
    }

    private var canSubmitResults: Bool {
        ResultSubmissionFeatureGate.canSubmitResults(registration: ws.registration.get())
    }

    private func progressFraction(_ manifest: JobManifest) -> Double {
        guard manifest.totalCells > 0 else { return 0 }
        let done = Double(manifest.completedCells)
        // Fold in the running cell's fraction so the bar moves during a long cell.
        let within: Double = (jobRunner.runningJobId == jobId)
            ? max(0, min(1, jobRunner.currentCellFraction))
            : 0
        return min(1, (done + within) / Double(manifest.totalCells))
    }

    private func runningDateTitle(_ manifest: JobManifest) -> String {
        guard let date = manifest.createdDate else {
            return String(manifest.createdAt.prefix(10))
        }
        return JobDateFormat.shortDate.string(from: date)
    }

    private func runningModelChips(_ manifest: JobManifest) -> [String] {
        var modelOrder: [String] = []
        var byModel: [String: [JobCell]] = [:]
        for cell in manifest.cells {
            let modelKey = CompletedResultsCSVExporter.resultModelGroupKey(for: cell)
            if byModel[modelKey] == nil { modelOrder.append(modelKey) }
            byModel[modelKey, default: []].append(cell)
        }
        return orderedUnique(modelOrder.map { key in
            CompletedResultsCSVExporter.resultModelDisplayName(for: key, cells: byModel[key] ?? [])
        })
    }

    private func runningBenchmarkChips(_ manifest: JobManifest) -> [String] {
        let typesById = Dictionary(uniqueKeysWithValues: BenchmarkCatalog.all(store: ws.benchmarks).map { ($0.benchmarkId, $0.benchmarkType) })
        let types = orderedUnique(
            manifest.cells.map { cell in
                cell.benchmarkType?.rawValue ?? typesById[cell.benchmarkId] ?? cell.benchmarkId
            }
        )
        return types.map { BenchmarkCatalog.displayName(for: $0) }
    }

    private func runningQuantChips(_ manifest: JobManifest) -> [String] {
        let quants = orderedUnique(
            manifest.cells.compactMap(\.source.quant)
        )
        let labels = quants.map { quant in
            quant
                .lowercased()
                .replacingOccurrences(of: "_k_m", with: "_km")
        }
        return labels.isEmpty ? ["unknown"] : labels
    }

    private func orderedUnique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for value in values where seen.insert(value).inserted {
            result.append(value)
        }
        return result
    }

    /// Building the CSV reads every payload from disk, so rows are generated only
    /// once a share destination is picked.
    private func csvExportFile(_ manifest: JobManifest) -> ResultsCSVFile {
        let ws = ws
        return ResultsCSVFile(
            filename: CompletedResultsCSVExporter.filename(
                for: manifest,
                dateTitle: runningDateTitle(manifest)
            )
        ) {
            CompletedResultsCSVExporter.csv(for: manifest, ws: ws)
        }
    }

    private func estimatedTimeLeftText(_ manifest: JobManifest) -> String? {
        jobRunner.estimatedTimeLeft(jobId: jobId, now: now)
    }

    // MARK: - Job execution

    /// One state, so submitting and retrying at once is unrepresentable.
    private enum InFlight: Equatable {
        case idle
        case submitting
        case retrying
    }

    /// The one alert this page can show at a time.
    private enum ActiveAlert: Identifiable {
        case submitError(String)
        case runBlocked(String)

        var id: String {
            switch self {
            case .submitError: return "submitError"
            case .runBlocked: return "runBlocked"
            }
        }

        var title: String {
            switch self {
            case .submitError: return "Submit Error"
            case .runBlocked: return "Job Already Running"
            }
        }

        var message: String {
            switch self {
            case .submitError(let message),
                 .runBlocked(let message):
                return message
            }
        }
    }

    private enum RunScope {
        case cancelled
        case failed
        case selected(Set<CellId>)

        /// The cells `runCells(scope:)` resets, which the memory warning counts.
        func includes(_ cell: JobCell) -> Bool {
            switch self {
            case .cancelled: cell.runStatus == .cancelled
            case .failed: cell.runStatus == .failed
            case .selected(let ids): ids.contains(cell.cellId)
            }
        }
    }

    private func resumePaused() {
        attemptRunCells(scope: .cancelled)
    }

    private func retryFailed() {
        attemptRunCells(scope: .failed)
    }

    private func rerunSelected() {
        guard inFlight != .retrying else { return }
        let ids = selectedCellIds
        guard !ids.isEmpty else { return }
        attemptRunCells(scope: .selected(ids))
    }

    /// Warns before a rerun under Low Power Mode, or with Max Memory cells a public build
    /// cannot measure accurately now (`NewJobView` shows that on its review step instead).
    private func attemptRunCells(scope: RunScope) {
        memoryWarning = manifest.flatMap {
            MemoryFreshness.runWarning(.rerun, cells: $0.cells.filter(scope.includes))
        }
        if detectPowerSaveMode() {
            pendingRunScope = scope
            showLowPowerWarning = true
        } else if memoryWarning != nil {
            pendingRunScope = scope
            showMemoryWarning = true
        } else {
            runCells(scope: scope)
        }
    }

    private func runPendingScope() {
        if let scope = pendingRunScope { runCells(scope: scope) }
        pendingRunScope = nil
    }

    private func runCells(scope: RunScope) {
        guard inFlight == .idle else { return }
        guard var manifest = self.manifest else { return }
        let jobId = manifest.jobId
        let flag = CancelFlag()
        guard jobRunner.start(jobId: jobId, flag: flag) else {
            activeAlert = .runBlocked("Pause or finish the current job before starting another one.")
            return
        }
        // Resuming or retrying is a start too, so it opens Pocket Mode.
        jobRunner.pocketModeRequestedJobId = jobId

        for i in manifest.cells.indices where scope.includes(manifest.cells[i]) {
            manifest.cells[i].runStatus = .pending
            manifest.cells[i].errorMessage = nil
        }
        if case .selected = scope {
            // Forget the prior submission so the new payload goes up as a fresh
            // result (the server keeps the old one). The on-disk record must go too:
            // the submit sweep re-adopts it, so if the app dies before the new
            // upload, a stale record would pass the old upload off as this one.
            for i in manifest.cells.indices where scope.includes(manifest.cells[i]) {
                manifest.cells[i].serverJobId = nil
                ws.results.deleteSubmission(manifest.cells[i].cellId)
            }
            // Cleared here, not at the tap site: a cancelled Low Power warning
            // should leave the selection intact.
            selectedCellIds.removeAll()
        }
        manifest.status = .running
        manifest.pausedReason = nil
        jobStore.save(manifest)

        retryFlag = flag
        inFlight = .retrying

        // Three intents, reported apart to keep the `job_started` funnel meaningful.
        let source: RunSource = switch scope {
        case .cancelled: .resume
        case .failed: .retryFailed
        case .selected: .rerun
        }

        JobExecutor.run(
            manifest: manifest, jobRunner: jobRunner, jobStore: jobStore,
            flag: flag, ws: ws, source: source
        ) { _ in
            inFlight = .idle
            retryFlag = nil
        }
    }

    private func submitAll() {
        guard inFlight == .idle else { return }
        guard jobRunner.runningJobId != jobId else { return }
        guard currentUnsubmittedCount > 0 else { return }
        guard ws.registration.signingIdentity() != nil else {
            activeAlert = .submitError("Result submission requires registration.")
            return
        }

        inFlight = .submitting

        let ws = ws
        Task.detached {
            // The shared uploader serializes this with any other drain, so a
            // double-fire can't double-submit the same cells.
            let outcome = await ResultUploader.shared.drainJob(jobId: jobId)

            await MainActor.run {
                inFlight = .idle
                // The drain saved serverJobIds to disk; sync them into the store.
                if let updated = ws.loadJobManifest(jobId: jobId) {
                    jobStore.apply(updated)
                }
                if !outcome.errors.isEmpty {
                    activeAlert = .submitError("Submitted \(outcome.submitted), \(outcome.errors.count) failed:\n"
                        + outcome.errors.joined(separator: "\n"))
                }
            }
        }
    }

}

#if DEBUG
/// Seeds a finished job (completed cells plus one failure) into a fresh `JobStore`,
/// which is where JobDetailView reads its manifest.
private struct JobDetailPreviewHost: View {
    private static let jobId = JobId("preview-job-detail")

    @State private var jobStore: JobStore

    init() {
        let store = JobStore(ws: PipetteWorkspace.production)
        store.save(Self.sampleManifest)
        _jobStore = State(initialValue: store)
    }

    var body: some View {
        NavigationStack {
            JobDetailView(jobId: Self.jobId)
        }
        .environment(JobRunner())
        .environment(jobStore)
    }

    private static var sampleManifest: JobManifest {
        let cells: [JobCell] = [
            JobCell(
                cellId: CellId("preview-cell-0"),
                benchmarkId: "decode_throughput_512_100",
                benchmarkType: .decodeThroughput,
                runStatus: .completed,
                serverJobId: nil,
                errorMessage: nil,
                source: .previewSample
            ),
            JobCell(
                cellId: CellId("preview-cell-1"),
                benchmarkId: "prefill_throughput_512",
                benchmarkType: .prefillThroughput,
                runStatus: .completed,
                serverJobId: nil,
                errorMessage: nil,
                source: .previewSample
            ),
            JobCell(
                cellId: CellId("preview-cell-2"),
                benchmarkId: "end_to_end_latency_512_100",
                benchmarkType: .endToEndLatency,
                runStatus: .failed,
                serverJobId: nil,
                errorMessage: "Engine error: decode rc=-3.",
                source: .previewSample
            )
        ]
        return JobManifest(
            jobId: jobId,
            createdAt: JobDateFormat.iso8601.string(from: Date()),
            nGpuLayers: 99,
            contextSize: 8192,
            cells: cells,
            status: .completed,
            title: "Preview Job"
        )
    }
}

#Preview("Job Detail") {
    JobDetailPreviewHost()
}
#endif
