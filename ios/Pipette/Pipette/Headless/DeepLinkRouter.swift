import Foundation
import SwiftUI

/// Live-app handler for `pipette://run/…` deep links.
///
/// Unlike `HeadlessRunner`, which exits when done, this drives the running app's shared
/// `JobRunner`/`JobStore`, so a link-triggered run shows up in the Jobs UI. Only commands
/// with `allowedViaDeepLink` reach here; destructive and identity verbs stay CLI-only.
///
/// Models resolve from what is already on device; a link never starts a download.
@Observable @MainActor
final class DeepLinkRouter {
    /// Consumed and cleared by `MainTabView.onChange`.
    var requestedTab: MainTab?
    /// Set with `requestedTab = .jobs` so the user lands on the live run, not the list.
    /// Consumed and cleared by `JobsView`.
    var pendingJobId: JobId?
    var alert: DeepLinkAlert?
    /// An action waiting on the user: `job submit` (off-device egress), or a bench or job run
    /// whose Max Memory a public build cannot measure accurately.
    var confirmation: DeepLinkConfirmation?
    var export: DeepLinkExport?

    struct DeepLinkAlert: Identifiable {
        let id = UUID()
        var title: String
        var message: String
    }

    struct DeepLinkConfirmation: Identifiable {
        let id = UUID()
        var title: String
        var message: String
        var confirmTitle: String
        var perform: () -> Void
    }

    struct DeepLinkExport: Identifiable {
        let id = UUID()
        var jobId: String
        var csv: String
    }

    func handle(_ url: URL, ws: PipetteWorkspace, jobRunner: JobRunner, jobStore: JobStore) {
        let command: HeadlessCommand
        switch HeadlessCommand.parse(url: url) {
        case let .success(parsed):
            command = parsed
        case let .failure(error):
            // Say which token was wrong. A link is typed once and pasted many times, so
            // "couldn't read it" leaves the reader guessing at a URL they cannot debug.
            // `notHeadless` is the exception: on this path it means the URL is not a
            // `pipette://run` link at all, and the argv wording would be nonsense here.
            let detail = error == .notHeadless
                ? "“\(url.absoluteString)” is not a pipette://run link"
                : error.message
            alert = .init(title: "Unrecognized link", message: "\(detail).")
            return
        }
        guard command.allowedViaDeepLink else {
            alert = .init(title: "Not permitted",
                          message: "This command can only be run from the command line, "
                              + "not a link.")
            return
        }
        // The shared controllers exist before the auth gate, so without this a link could
        // start a benchmark behind the sign-in screen whose results could not be submitted.
        guard ws.registration.isRegistered else {
            alert = .init(title: "Register this device first",
                          message: "Register in the app before running benchmarks from a link.")
            return
        }

        // `submit=` and `claim=` are ignored for bench links: uploading is off-device
        // egress, so it happens only through the confirmed `pipette://run/job/submit` verb.
        switch command {
        // Refused above by `allowedViaDeepLink`: `benchmarks run` carries `sync=`, and
        // fetching or deleting a model is a side effect a URL must not cause.
        case .benchmarksRun, .modelsPull, .modelsDelete, .storageGc, .version:
            return
        case let .bench(spec, model, quant, runtime, batch, runtimeFlags, benchmarks, metrics,
                        offsets, _, benchmarkFlags, _, llamaPin):
            guard runsThisLlamaBuild(llamaPin) else { return }
            launchBench(spec: spec, modelHint: model, quant: quant, runtime: runtime, batch: batch,
                        runtimeFlags: runtimeFlags, benchmarkFlags: benchmarkFlags,
                        benchmarks: benchmarks, metrics: metrics, offsets: offsets,
                        ws: ws, jobRunner: jobRunner, jobStore: jobStore)

        case let .bareBench(runtime, batch, metrics, offsets, benchmarks, model, _, llamaPin):
            guard runsThisLlamaBuild(llamaPin) else { return }
            launchBench(bareSweep: true,
                        spec: nil, modelHint: model, quant: nil, runtime: runtime, batch: batch,
                        runtimeFlags: nil, benchmarkFlags: nil,
                        benchmarks: benchmarks, metrics: metrics, offsets: offsets,
                        ws: ws, jobRunner: jobRunner, jobStore: jobStore)

        case let .afm(metrics, benchmarks, offsets, _, _, benchmarkFlags):
            launchAFM(metrics: metrics, benchmarks: benchmarks, offsets: offsets,
                      benchmarkFlags: benchmarkFlags,
                      jobRunner: jobRunner, jobStore: jobStore, ws: ws)

        case let .runJob(id, scope):
            runJob(id: id, scope: scope, ws: ws, jobRunner: jobRunner, jobStore: jobStore)

        case let .submitJob(id):
            confirmSubmit(jobId: id)

        case let .exportJob(id):
            exportJob(id: id, ws: ws)

        case .status:
            requestedTab = .settings

        // Also refused above. Exhaustive on purpose: a newly allowed command is a compile
        // error here until it is handled.
        case .authRegister, .diagMemseq, .modelsList, .modelsRm, .listJobs, .removeJob,
             .worker, .runtimesList, .benchmarksList, .benchmarksShow,
             .benchmarksInitLocal,
             .storageStatus, .sync,
             .authMe, .authSetDevice, .authReset, .authExport, .authImport, .diagProbe,
             .resultsList, .resultsShow, .resultsDelete:
            alert = .init(title: "Not permitted", message: "This command is command-line only.")
        }
    }

    // MARK: - Bench

    /// Every on-device model a bench link's selector matches, in scan order.
    ///
    /// A list rather than a first hit so the caller can refuse "several match", as headless
    /// `bench match=` does, instead of benching whichever file the directory yields first.
    ///
    /// Rows named exactly `modelHint` win, so a filename that another filename contains
    /// stays reachable. `quant=` narrows first. Same tiebreak as
    /// `BenchmarksCommands.resolveCatalogRows`, but over the on-device filename only, so
    /// `match=org/repo` resolves headlessly and matches nothing here.
    ///
    /// A `model://sha256=` prefix short enough to cover two artifacts is ambiguous too.
    static func matchingModels(_ models: [DiscoveredModel], spec: HeadlessCommand.ModelSelector?,
                               runtime: RuntimeType, modelHint: String?,
                               quant: String?) -> [DiscoveredModel] {
        if let spec {
            switch spec {
            case let .model(named):
                // As the link named it, so the report keeps the link's vault label.
                return models.filter { $0.source == named }.map { $0.asRequested(named) }
            case let .digest(prefix):
                // A digest names no label, so the manifest's stale one is not reported.
                return models.filter {
                    ((try? modelDescriptorDigest($0.source)) ?? "").hasPrefix(prefix)
                }.map { $0.asRequested($0.source.withoutCustomLabel) }
            }
        }
        let hint = modelHint ?? ""
        let wantedQuant = (quant?.isEmpty == false) ? quant : nil
        let candidates = models.filter { m in
            let nameOK = hint.isEmpty || m.name.localizedCaseInsensitiveContains(hint)
            let quantOK = wantedQuant.map { m.quant?.caseInsensitiveCompare($0) == .orderedSame } ?? true
            return isCompatible(m.source, runtime) && nameOK && quantOK
        }
        // Localized, matching the filter above (see `resolveCatalogRows`).
        let exact = candidates.filter {
            $0.name.localizedCaseInsensitiveCompare(hint) == .orderedSame
        }
        return exact.isEmpty ? candidates : exact
    }

    /// Whether a resolved match set is refused rather than run.
    ///
    /// Ambiguity is refused except for the bare sweep form naming no model, whose contract
    /// is "whatever matching model is on device" (its CLI twin `BenchmarksCommands.runSweep`
    /// takes the first match); refusing it would break runbook links that pin only a runtime.
    /// Both halves matter: the bare form carries `match=` too, and a `bench` link naming
    /// nothing is refused by `BenchmarksCommands.run`.
    ///
    /// Separate from `launchBench` so the rule can be asserted directly in tests.
    static func refusesAsAmbiguous(bareSweep: Bool, spec: HeadlessCommand.ModelSelector?,
                                   modelHint: String?, quant: String?,
                                   matchCount: Int) -> Bool {
        let named = spec != nil || !(modelHint ?? "").isEmpty || !(quant ?? "").isEmpty
        return !(bareSweep && !named) && matchCount > 1
    }

    /// A link runs in the resident app. That app keeps one llama.cpp build, the build that
    /// its worker advertises. A link can name that build, but it cannot switch to another.
    private func runsThisLlamaBuild(_ pin: LlamaPin?) -> Bool {
        guard let pin, pin != LlamaPins.current else { return true }
        alert = .init(title: "Not permitted",
                      message: "This app runs llama.cpp \(LlamaPins.current.name); a link cannot "
                          + "switch it to \(pin.name). Run that build from the command line.")
        return false
    }

    private func launchBench(bareSweep: Bool = false,
                             spec: HeadlessCommand.ModelSelector?, modelHint: String?,
                             quant: String?,
                             runtime: RuntimeType, batch: Int, runtimeFlags: RuntimeFlags?,
                             benchmarkFlags: BenchmarkFlags?,
                             benchmarks: [String], metrics: [String], offsets: [Int],
                             ws: PipetteWorkspace, jobRunner: JobRunner, jobStore: JobStore) {
        if runtime == .appleFoundation {
            launchAFM(metrics: metrics, benchmarks: benchmarks, offsets: offsets,
                      benchmarkFlags: benchmarkFlags,
                      jobRunner: jobRunner, jobStore: jobStore, ws: ws)
            return
        }
        let ids = BenchmarksCommands.resolveBenchmarkIds(metrics: metrics, offsets: offsets, explicit: benchmarks)
        guard !ids.isEmpty else {
            alert = .init(title: "No benchmarks", message: "The link didn't name any benchmarks to run.")
            return
        }

        let matches = Self.matchingModels(ws.availableModels(), spec: spec,
                                          runtime: runtime, modelHint: modelHint, quant: quant)
        if matches.isEmpty {
            requestedTab = .models
            alert = .init(title: "Model not on device",
                          message: "Download the model on the Models tab, then open the link again.")
            return
        }
        if Self.refusesAsAmbiguous(bareSweep: bareSweep, spec: spec, modelHint: modelHint,
                                   quant: quant, matchCount: matches.count) {
            requestedTab = .models
            let names = matches.map(\.name).joined(separator: ", ")
            // Exact naming works on every form; `quant=` is carried by the `bench` verb
            // alone, so it is offered second rather than led with.
            alert = .init(title: "Link matches several models",
                          message: "\(names). Name the model exactly, or narrow it with "
                              + "quant= on a bench link.")
            return
        }
        let model = matches[0]

        let ctx = (offsets.max() ?? 4096) + 300
        // The flags ride on each cell, as in the headless run, so a link's `threads` or
        // `ctx_size` reaches the run.
        let cells = JobCell.pending(benchmarkIds: ids, for: model,
                                    store: ws.benchmarks,
                                    runtimeFlags: runtimeFlags.map(RuntimeFlagRef.init),
                                    benchmarkFlags: benchmarkFlags.map(BenchmarkFlagRef.init))
        launch(cells: cells, contextSize: ctx, batch: batch, runtimeFlags: runtimeFlags,
               jobRunner: jobRunner, jobStore: jobStore, ws: ws)
    }

    private func launchAFM(metrics: [String], benchmarks: [String], offsets: [Int],
                           benchmarkFlags: BenchmarkFlags?,
                           jobRunner: JobRunner, jobStore: JobStore, ws: PipetteWorkspace) {
        let ids = BenchmarksCommands.resolveBenchmarkIds(metrics: metrics, offsets: offsets, explicit: benchmarks)
        let cells = JobCell.pending(benchmarkIds: ids, for: .appleFoundation,
                                    store: ws.benchmarks,
                                    benchmarkFlags: benchmarkFlags.map(BenchmarkFlagRef.init))
        // AFM has no load knobs: ngl/ctx/batch are inert placeholders.
        launch(cells: cells, contextSize: 4096, batch: 512, runtimeFlags: nil,
               jobRunner: jobRunner, jobStore: jobStore, ws: ws)
    }

    /// Goes through `JobLauncher` like the New Job screen, so a link-created job is
    /// identical to a UI-created one.
    private func launch(cells: [JobCell], contextSize: Int, batch: Int, runtimeFlags: RuntimeFlags?,
                        jobRunner: JobRunner, jobStore: JobStore, ws: PipetteWorkspace) {
        guard !cells.isEmpty else {
            alert = .init(title: "No benchmarks",
                          message: "None of the requested benchmarks were recognized.")
            return
        }
        confirmingMemory(cells: cells, kind: .newJob, jobRunner: jobRunner) { [weak self] in
            self?.start(cells: cells, contextSize: contextSize, batch: batch,
                        runtimeFlags: runtimeFlags, jobRunner: jobRunner, jobStore: jobStore,
                        ws: ws)
        }
    }

    /// Run `perform` now, or after confirming when a public build cannot measure these Max
    /// Memory cells accurately (the New Job review step's notice, as an alert). A busy runner
    /// skips the question, since `perform` refuses the run.
    private func confirmingMemory(cells: [JobCell], kind: MemoryFreshness.RunKind,
                                  jobRunner: JobRunner, perform: @escaping () -> Void) {
        guard !jobRunner.isRunning, let warning = MemoryFreshness.runWarning(kind, cells: cells)
        else { return perform() }
        confirmation = .init(title: MemoryFreshness.alertTitle, message: warning,
                             confirmTitle: "Run Anyway", perform: perform)
    }

    private func start(cells: [JobCell], contextSize: Int, batch: Int, runtimeFlags: RuntimeFlags?,
                       jobRunner: JobRunner, jobStore: JobStore, ws: PipetteWorkspace) {
        // A link never auto-uploads; results go up only via the confirmed `job submit`.
        let jobId = JobLauncher.launch(
            cells: cells,
            nGpuLayers: Int(runtimeFlags?.numberGpuLayers ?? HeadlessCommand.defaultGpuLayers),
            contextSize: contextSize, prefillBatch: batch,
            threads: runtimeFlags?.threads.map(Int.init),
            contributeResults: false, jobRunner: jobRunner, jobStore: jobStore, ws: ws,
            onFinish: { _ in jobStore.reload() })
        guard let jobId else {
            alert = .init(title: "A job is already running",
                          message: "Wait for the current job to finish, then open the link again.")
            return
        }
        jobRunner.pocketModeRequestedJobId = jobId
        requestedTab = .jobs
        pendingJobId = jobId
    }

    // MARK: - Job lifecycle

    /// `job run`: the same reset and re-run as `JobDetailView.runCells(scope:)` and
    /// `JobCommands.run`.
    private func runJob(id: String, scope: HeadlessCommand.RunScope,
                        ws: PipetteWorkspace, jobRunner: JobRunner, jobStore: JobStore) {
        guard let manifest = ws.loadJobManifest(jobId: JobId(id)) else {
            alert = .init(title: "No such job", message: "No job with id \(id) on this device.")
            return
        }
        let target = scope.resetTarget
        let cells = manifest.cells.filter { $0.runStatus == target }
        confirmingMemory(cells: cells, kind: .rerun, jobRunner: jobRunner) { [weak self] in
            self?.rerun(manifest, scope: scope, jobRunner: jobRunner, jobStore: jobStore,
                        ws: ws)
        }
    }

    private func rerun(_ manifest: JobManifest, scope: HeadlessCommand.RunScope,
                       jobRunner: JobRunner, jobStore: JobStore, ws: PipetteWorkspace) {
        guard let jobId = JobLauncher.rerun(
            manifest, resetting: scope.resetTarget, jobRunner: jobRunner, jobStore: jobStore,
            ws: ws, onFinish: { _ in jobStore.reload() })
        else {
            alert = .init(title: "A job is already running",
                          message: "Wait for the current job to finish, then open the link again.")
            return
        }
        jobRunner.pocketModeRequestedJobId = jobId
        requestedTab = .jobs
        pendingJobId = jobId
    }

    /// The one allowed command with off-device egress, so it confirms first.
    private func confirmSubmit(jobId id: String) {
        confirmation = .init(
            title: "Submit results?",
            message: "Upload this job's completed results to the collector?",
            confirmTitle: "Submit"
        ) { [weak self] in
            Task { @MainActor in
                let outcome = await ResultUploader.shared.drainJob(jobId: JobId(id))
                self?.alert = .init(
                    title: outcome.errors.isEmpty ? "Submitted" : "Submitted with errors",
                    message: "Uploaded \(outcome.submitted) result(s)"
                        + (outcome.errors.isEmpty ? "." : "; \(outcome.errors.count) error(s)."))
            }
        }
    }

    private func exportJob(id: String, ws: PipetteWorkspace) {
        guard let manifest = ws.loadJobManifest(jobId: JobId(id)) else {
            alert = .init(title: "No such job", message: "No job with id \(id) on this device.")
            return
        }
        export = .init(jobId: id, csv: CompletedResultsCSVExporter.csv(for: manifest, ws: ws))
    }

}

// MARK: - Presentation

extension View {
    /// Attached at the scene root so the router's alerts and sheets present over any tab.
    func deepLinkPresentations(_ router: DeepLinkRouter) -> some View {
        modifier(DeepLinkPresentationModifier(router: router))
    }
}

private struct DeepLinkPresentationModifier: ViewModifier {
    @Bindable var router: DeepLinkRouter

    func body(content: Content) -> some View {
        content
            .alert(router.alert?.title ?? "",
                   isPresented: Binding(get: { router.alert != nil },
                                        set: { if !$0 { router.alert = nil } }),
                   presenting: router.alert) { _ in
                Button("OK", role: .cancel) {}
            } message: { Text($0.message) }
            .alert(router.confirmation?.title ?? "",
                   isPresented: Binding(get: { router.confirmation != nil },
                                        set: { if !$0 { router.confirmation = nil } }),
                   presenting: router.confirmation) { confirmation in
                Button(confirmation.confirmTitle) { confirmation.perform() }
                Button("Cancel", role: .cancel) {}
            } message: { Text($0.message) }
            .sheet(item: $router.export) { payload in
                DeepLinkExportSheet(payload: payload)
            }
    }
}

private struct DeepLinkExportSheet: View {
    let payload: DeepLinkRouter.DeepLinkExport
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView([.horizontal, .vertical]) {
                Text(payload.csv)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .padding()
            }
            .navigationTitle("Export")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    ShareLink(item: csvFile, preview: SharePreview(csvFilename))
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var csvFilename: String { "job-\(payload.jobId).csv" }

    /// A file, not a `String`: a shared string is saved as `.txt` whatever the preview says.
    private var csvFile: ResultsCSVFile {
        let csv = payload.csv
        return ResultsCSVFile(filename: csvFilename) { csv }
    }
}
