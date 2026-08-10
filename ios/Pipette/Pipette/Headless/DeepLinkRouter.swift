import Foundation
import SwiftUI

/// Live-app handler for `pipette://run/…` deep links.
///
/// Where `HeadlessRunner` is a one-shot process (prints `[HEADLESS] …` to
/// stdout, then `exit`s), this drives the *running* app's shared controllers —
/// the same `JobRunner`/`JobStore` the SwiftUI views bind — so a link-triggered
/// run shows up in the normal Jobs UI and never terminates the user's session.
/// The URL is parsed by the shared `HeadlessCommand.parse(url:)` grammar and
/// gated on `allowedViaDeepLink`, so only run/observe commands reach here;
/// destructive and identity verbs stay CLI-only.
///
/// v1 resolves benchmark models from what's already on device — it does not
/// download. A link naming an absent model routes to the Models tab with an
/// explanatory alert rather than kicking off a silent multi-gigabyte fetch.
@Observable @MainActor
final class DeepLinkRouter {
    /// Tab `MainTabView` should switch to. Set by a handler, consumed and
    /// cleared by `MainTabView.onChange`.
    var requestedTab: MainTab?
    /// A job the Jobs screen should open (push its running/detail page). A
    /// run-starting link sets this alongside `requestedTab = .jobs` so the user
    /// lands on the live run, not just the list. Consumed and cleared by `JobsView`.
    var pendingJobId: JobId?
    /// A message to surface: a parse/allow-list failure, a missing model/job, or
    /// the outcome of a submit.
    var alert: DeepLinkAlert?
    /// A pending action awaiting confirmation. `job submit` uploads results off
    /// device, so it clears this gate before running (the Q3 confirm-egress gate).
    var confirmation: DeepLinkConfirmation?
    /// CSV to present in a share sheet (`job export`) — the UI equivalent of the
    /// CLI's stdout dump.
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

    /// Parse, gate, and dispatch a deep link against the live app's controllers.
    func handle(_ url: URL, storage: Storage, jobRunner: JobRunner, jobStore: JobStore) {
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
        // The shared controllers exist before the auth gate, so without this a
        // link could start a benchmark behind the sign-in screen — and its
        // results couldn't be submitted anyway. The provisioned/fleet devices
        // this targets are already registered.
        guard storage.identity.isRegistered else {
            alert = .init(title: "Register this device first",
                          message: "Register in the app before running benchmarks from a link.")
            return
        }

        // `submit=` is intentionally ignored for bench links: uploading results
        // is off-device egress, so it happens only through the confirmed
        // `pipette://run/job/submit` verb, never silently as a bench parameter.
        switch command {
        // Refused above as CLI-only — `benchmarks run` carries `sync=`, and fetching or
        // deleting a model is a side effect a URL must not cause. The arms exist because
        // the switch is exhaustive by design.
        case .benchmarksRun, .pullModel, .deleteModel, .storageGc, .version:
            return
        case let .bench(spec, model, quant, runtime, batch, nGpuLayers, _, benchmarks, metrics,
                        offsets, _):
            launchBench(spec: spec, modelHint: model, quant: quant, runtime: runtime, batch: batch,
                        nGpuLayers: nGpuLayers,
                        benchmarks: benchmarks, metrics: metrics, offsets: offsets,
                        storage: storage, jobRunner: jobRunner, jobStore: jobStore)

        case let .bareBench(runtime, batch, nGpuLayers, _, metrics, offsets, benchmarks, model, _):
            launchBench(spec: nil, modelHint: model, quant: nil, runtime: runtime, batch: batch,
                        nGpuLayers: nGpuLayers,
                        benchmarks: benchmarks, metrics: metrics, offsets: offsets,
                        storage: storage, jobRunner: jobRunner, jobStore: jobStore)

        case let .afm(metrics, benchmarks, offsets, _):
            launchAFM(metrics: metrics, benchmarks: benchmarks, offsets: offsets,
                      jobRunner: jobRunner, jobStore: jobStore, storage: storage)

        case let .runJob(id, scope):
            runJob(id: id, scope: scope, storage: storage, jobRunner: jobRunner, jobStore: jobStore)

        case let .submitJob(id):
            confirmSubmit(jobId: id)

        case let .exportJob(id):
            exportJob(id: id, storage: storage)

        case .status:
            requestedTab = .settings

        // Filtered out by `allowedViaDeepLink` above; the switch stays exhaustive
        // so a newly-allowed command is a compile error here until it's handled.
        case .register, .memSeq, .listModels, .removeModel, .listJobs, .removeJob,
             .settings, .listRuntimes, .listBenchmarks, .showBenchmark, .initLocalBenchmarks,
             .storageStatus, .sync,
             .authMe, .authReset, .diagProbe,
             .listResults, .showResult, .deleteResult:
            alert = .init(title: "Not permitted", message: "This command is command-line only.")
        }
    }

    // MARK: - Bench

    private func launchBench(spec: HeadlessCommand.ModelSelector?, modelHint: String?,
                             quant: String?,
                             runtime: RuntimeType, batch: Int, nGpuLayers: UInt32?,
                             benchmarks: [String], metrics: [String], offsets: [Int],
                             storage: Storage, jobRunner: JobRunner, jobStore: JobStore) {
        // `runtime=afm` benches Apple's on-device model — no file to resolve.
        if runtime == .appleFoundation {
            launchAFM(metrics: metrics, benchmarks: benchmarks, offsets: offsets,
                      jobRunner: jobRunner, jobStore: jobStore, storage: storage)
            return
        }
        let ids = HeadlessRunner.resolveBenchmarkIds(metrics: metrics, offsets: offsets, explicit: benchmarks)
        guard !ids.isEmpty else {
            alert = .init(title: "No benchmarks", message: "The link didn't name any benchmarks to run.")
            return
        }

        // Resolve an on-device model (v1 never downloads). An explicit `spec=`
        // is authoritative — it names the exact file and does not fall back to a
        // name search; otherwise match runtime + name(+quant).
        let models = storage.availableModels()
        let model: DiscoveredModel?
        if let spec {
            switch spec {
            case let .model(named):
                model = models.first { $0.source == named }
            case let .digest(prefix):
                model = models.first {
                    ((try? Descriptor.digest($0.source.withoutAuthToken)) ?? "").hasPrefix(prefix)
                }
            }
        } else {
            model = models.first { m in
                let engineOK = runtime.matches(m.source)
                let nameOK = (modelHint ?? "").isEmpty || m.name.localizedCaseInsensitiveContains(modelHint!)
                let quantOK = (quant ?? "").isEmpty
                    || (m.quant?.caseInsensitiveCompare(quant!) == .orderedSame)
                return engineOK && nameOK && quantOK
            }
        }
        guard let model else {
            requestedTab = .models
            alert = .init(title: "Model not on device",
                          message: "Download the model on the Models tab, then open the link again.")
            return
        }

        let ctx = (offsets.max() ?? 4096) + 300
        let cells = JobCell.pending(benchmarkIds: ids, for: model)
        launch(cells: cells, contextSize: ctx, batch: batch, nGpuLayers: nGpuLayers,
               jobRunner: jobRunner, jobStore: jobStore, storage: storage)
    }

    private func launchAFM(metrics: [String], benchmarks: [String], offsets: [Int],
                           jobRunner: JobRunner, jobStore: JobStore, storage: Storage) {
        let ids = HeadlessRunner.resolveBenchmarkIds(metrics: metrics, offsets: offsets, explicit: benchmarks)
        let cells = JobCell.pending(benchmarkIds: ids, for: .appleFoundation)
        // AFM has no load knobs — ngl/ctx/batch are inert placeholders, matching
        // `HeadlessRunner.runAFM`.
        launch(cells: cells, contextSize: 4096, batch: 512, nGpuLayers: nil,
               jobRunner: jobRunner, jobStore: jobStore, storage: storage)
    }

    /// Build a one-model job through the same `JobLauncher` sequence the New Job
    /// screen uses, so a link-created job is identical to a UI-created one, and
    /// switch to the Jobs tab. A `nil` job id means the runner was already busy.
    private func launch(cells: [JobCell], contextSize: Int, batch: Int, nGpuLayers: UInt32?,
                        jobRunner: JobRunner, jobStore: JobStore, storage: Storage) {
        guard !cells.isEmpty else {
            alert = .init(title: "No benchmarks",
                          message: "None of the requested benchmarks were recognized.")
            return
        }
        // `contributeResults: false` — a link never auto-uploads; results are
        // submitted only via the confirmed `job submit` verb.
        let jobId = JobLauncher.launch(
            cells: cells, nGpuLayers: Int(nGpuLayers ?? HeadlessCommand.defaultGpuLayers),
            contextSize: contextSize, prefillBatch: batch,
            contributeResults: false, jobRunner: jobRunner, jobStore: jobStore, storage: storage,
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

    /// `job run`: flip the scoped terminal cells back to `.pending` and re-run on
    /// the shared runner — the same reset `JobDetailView.runCells(scope:)` and
    /// `JobCommands.run` perform.
    private func runJob(id: String, scope: HeadlessCommand.RunScope,
                        storage: Storage, jobRunner: JobRunner, jobStore: JobStore) {
        guard let manifest = storage.loadJobManifest(jobId: JobId(id)) else {
            alert = .init(title: "No such job", message: "No job with id \(id) on this device.")
            return
        }
        guard let jobId = JobLauncher.rerun(
            manifest, resetting: scope.resetTarget, jobRunner: jobRunner, jobStore: jobStore,
            storage: storage, onFinish: { _ in jobStore.reload() })
        else {
            alert = .init(title: "A job is already running",
                          message: "Wait for the current job to finish, then open the link again.")
            return
        }
        jobRunner.pocketModeRequestedJobId = jobId
        requestedTab = .jobs
        pendingJobId = jobId
    }

    /// `job submit`: confirm before uploading (the one allowed command with
    /// off-device egress), then drain the job's unsubmitted results.
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

    /// `job export`: build the same CSV the export button saves and present it in
    /// a share sheet.
    private func exportJob(id: String, storage: Storage) {
        guard let manifest = storage.loadJobManifest(jobId: JobId(id)) else {
            alert = .init(title: "No such job", message: "No job with id \(id) on this device.")
            return
        }
        export = .init(jobId: id, csv: CompletedResultsCSVExporter.csv(for: manifest, storage: storage))
    }

}

// MARK: - Presentation

extension View {
    /// Attach the deep-link router's alert / confirm / export surfaces at the
    /// scene root so they present over any tab. Keeps `PipetteApp` to just the
    /// `.onOpenURL` wiring.
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

/// Read-only CSV preview with a system share sheet — the export destination for
/// a `pipette://run/job/export` link.
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

    /// Shares the CSV as a file, not as a `String` — a shared string is saved as
    /// a `.txt` no matter what the preview is titled.
    private var csvFile: ResultsCSVFile {
        let csv = payload.csv
        return ResultsCSVFile(filename: csvFilename) { csv }
    }
}
