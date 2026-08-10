import SwiftUI

/// Displays local jobs with their execution and submission state.
enum JobsRoute: Hashable {
    case detail(JobId)
}

struct JobsView: View {
    @Environment(JobRunner.self) private var jobRunner
    @Environment(JobStore.self) private var jobStore
    @Environment(ModelStore.self) private var modelStore
    @Environment(DeepLinkRouter.self) private var deepLinkRouter
    @Environment(\.pillTabBarReservedHeight) private var pillTabBarReservedHeight
    @Binding var selectedTab: MainTab
    @State private var path = NavigationPath()
    @State private var searchText: String = ""
    @State private var pendingDelete: JobManifest?
    // The creation wizard is a modal (fullScreenCover) so it covers the global
    // pill tab bar and presents its own Back/Next footer, matching the design.
    @State private var showNewJob = false

    private var jobs: [JobManifest] { jobStore.jobs }

    private var hasDownloadedModels: Bool {
        modelStore.models.contains { model in
            model.name.range(of: "mmproj", options: .caseInsensitive) == nil
        }
    }

    private var filteredJobs: [JobManifest] {
        let q = searchText.searchNormalized
        guard !q.isEmpty else { return jobs }
        return jobs.filter { job in
            if job.displayTitle.lowercased().contains(q) { return true }
            if job.modelNames.joined(separator: " ").lowercased().contains(q) { return true }
            if job.benchmarkIds.joined(separator: " ").lowercased().contains(q) { return true }
            return false
        }
    }

    var body: some View {
        NavigationStack(path: $path) {
            content
            .background(Color(.systemBackground))
            .navigationBarHidden(true)
            .navigationDestination(for: JobsRoute.self) { route in
                switch route {
                case .detail(let jobId):
                    JobDetailView(jobId: jobId)
                }
            }
            // A run-starting deep link opens the job's live page, same as the
            // wizard's onStarted push above.
            .onChange(of: deepLinkRouter.pendingJobId) { _, jobId in
                guard let jobId else { return }
                path.append(JobsRoute.detail(jobId))
                deepLinkRouter.pendingJobId = nil
            }
            .fullScreenCover(isPresented: $showNewJob) {
                NewJobView(
                    onStarted: { jobId in
                        // Dismiss the wizard and push JobDetailView so the user
                        // lands on the detail screen and can see live progress.
                        showNewJob = false
                        path.append(JobsRoute.detail(jobId))
                    },
                    onGoToModels: {
                        // Empty-state escape hatch: close the wizard and jump to
                        // the Models tab so the user can download models.
                        showNewJob = false
                        selectedTab = .models
                    }
                )
                .environment(jobRunner)
                .environment(jobStore)
                .environment(modelStore)
            }
            .confirmationDialog(
                "Delete Job?",
                isPresented: Binding<Bool>(
                    get: { pendingDelete != nil },
                    set: { if !$0 { pendingDelete = nil } }
                ),
                titleVisibility: .visible,
                presenting: pendingDelete
            ) { job in
                Button("Delete", role: .destructive) {
                    jobStore.delete(jobId: job.jobId)
                    pendingDelete = nil
                }
                Button("Cancel", role: .cancel) {
                    pendingDelete = nil
                }
            } message: { _ in
                Text("This will delete this job and all results from the device. This cannot be undone.")
            }
        }
    }

    // MARK: - Subviews

    @ViewBuilder
    private var content: some View {
        if hasDownloadedModels, jobs.isEmpty {
            VStack(spacing: 20) {
                header
                AppSearchField(text: $searchText, placeholder: "Search jobs")

                Spacer(minLength: 0)
                noJobsState
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 12 + pillTabBarReservedHeight)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollView {
                VStack(spacing: 20) {
                    header

                    if !hasDownloadedModels {
                        noModelsState
                            .padding(.top, 134)
                    } else {
                        AppSearchField(text: $searchText, placeholder: "Search jobs")

                        if filteredJobs.isEmpty {
                            noSearchResultsState
                                .padding(.top, 52)
                        } else {
                            jobsCard
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 12 + pillTabBarReservedHeight)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            Text("Your jobs")
                .pageHeaderLarge()
                .foregroundStyle(.primary)
            Spacer()
            Button {
                showNewJob = true
            } label: {
                ZStack {
                    Circle()
                        .fill(Color.primary)
                        .frame(width: 44, height: 44)
                    Image(systemName: "plus")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundStyle(Color(.systemBackground))
                }
            }
            .accessibilityLabel("New Job")
        }
        .padding(.top, 4)
    }

    private var jobsCard: some View {
        AppListCard(cornerRadius: 18) {
            ForEach(Array(filteredJobs.enumerated()), id: \.element.jobId) { index, job in
                NavigationLink(value: JobsRoute.detail(job.jobId)) {
                    JobRow(job: job)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 16)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .contextMenu {
                    Button(role: .destructive) {
                        pendingDelete = job
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }

                if index < filteredJobs.count - 1 {
                    Divider()
                        .padding(.leading, 20)
                }
            }
        }
    }

    private var noJobsState: some View {
        JobsEmptyPrompt(
            title: "No jobs yet",
            message: "Tap below to add your first benchmarking job",
            actionTitle: "Create a job",
            actionSystemImage: "plus"
        ) {
            showNewJob = true
        }
    }

    private var noSearchResultsState: some View {
        VStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 28, weight: .light))
                .foregroundStyle(.tertiary)
            Text("No matching jobs")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Try a different job, model, or benchmark name.")
                .font(.subheadline)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
    }

    private var noModelsState: some View {
        EmptyModelsPrompt.needModelsForJobs {
            selectedTab = .models
        }
    }

}

private struct JobsEmptyPrompt: View {
    let title: String
    let message: String
    let actionTitle: String
    let actionSystemImage: String
    let action: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            JobsEmptyIllustration()
                .padding(.bottom, 22)

            Text(title)
                .font(.serif(24))
                .foregroundStyle(.primary)

            Text(message)
                .font(.system(size: 16))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineSpacing(5)

            Button(action: action) {
                HStack(spacing: 8) {
                    Image(systemName: actionSystemImage)
                        .font(.system(size: 14, weight: .semibold))
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

private struct JobsEmptyIllustration: View {
    var body: some View {
        Image("jobs-empty-state")
            .resizable()
            .scaledToFit()
            .frame(width: 233, height: 192)
            .accessibilityHidden(true)
    }
}

// MARK: - Job row

struct JobRow: View {
    let job: JobManifest
    @Environment(JobRunner.self) private var jobRunner

    private var isRunning: Bool {
        job.status == .running || jobRunner.runningJobId == job.jobId
    }

    private var progressFraction: Double {
        guard job.totalCells > 0 else { return 0 }
        let done = Double(job.completedCells)
        let within: Double = (jobRunner.runningJobId == job.jobId)
            ? max(0, min(1, jobRunner.currentCellFraction))
            : 0
        return min(1, (done + within) / Double(job.totalCells))
    }

    private var relativeCreated: String {
        guard let date = job.createdDate else {
            return String(job.createdAt.prefix(16)).replacingOccurrences(of: "T", with: " ")
        }
        return JobDateFormat.relative.localizedString(for: date, relativeTo: Date())
    }

    /// Rough wall-clock estimate for the active run. See `JobRunner.estimatedTimeLeft`.
    private var estimatedTimeLeft: String? {
        jobRunner.estimatedTimeLeft(jobId: job.jobId, now: Date())
    }

    /// The engine(s) this job's cells run on — one badge each. New jobs are
    /// single-runtime; a legacy mixed-runtime job surfaces every engine present.
    private var runtimeKinds: [RuntimeKind] {
        let kinds = Set(job.cells.map { RuntimeKind($0.source) })
        return RuntimeKind.allCases.filter { kinds.contains($0) }
    }

    private var primaryMeta: String {
        if isRunning {
            return "\(job.completedCells)/\(job.totalCells) cells done"
        }
        return "\(job.totalCells) cell\(job.totalCells == 1 ? "" : "s")"
    }

    private var secondaryMeta: String {
        if isRunning {
            if let left = estimatedTimeLeft {
                return "In progress · \(left)"
            }
            return "In progress"
        }
        switch job.status {
        case .completed:
            return "Completed \(relativeCreated)"
        case .cancelled:
            return "Cancelled"
        case .paused:
            return "Paused"
        case .planned, .running:
            return "Created \(relativeCreated)"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 8) {
                Text(job.displayTitle)
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.primary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)

                ForEach(runtimeKinds) { kind in
                    RuntimeBadge(kind: kind)
                }
            }

            if isRunning {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule()
                            .fill(Color.primary.opacity(0.08))
                            .frame(height: 4)
                        Capsule()
                            .fill(Color.primary)
                            .frame(width: max(0, geo.size.width * progressFraction), height: 4)
                    }
                }
                .frame(height: 4)
            }

            HStack {
                Text(primaryMeta)
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(secondaryMeta)
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
            }
        }
    }
}

/// Compact, color-coded runtime tag for a job row — makes the engine a job ran on
/// scannable at a glance (distinct hue per runtime).
private struct RuntimeBadge: View {
    let kind: RuntimeKind

    var body: some View {
        Text(kind.badgeLabel)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(tint)
            .padding(.horizontal, 8)
            .frame(height: 22)
            .background(tint.opacity(0.14), in: Capsule())
            .overlay(Capsule().strokeBorder(tint.opacity(0.30), lineWidth: 1))
            .fixedSize()
    }

    private var tint: Color {
        switch kind {
        case .llamaCpp: return .blue
        case .mlx: return .purple
        case .afm: return Color(.systemGray)
        }
    }
}

#if DEBUG
#Preview("Jobs") {
    let store = JobStore(storage: FileStorage.production)
    store.reload()
    return JobsView(selectedTab: .constant(.jobs))
        .environment(JobRunner())
        .environment(store)
        .environment(ModelStore(storage: FileStorage.production))
        .environment(DeepLinkRouter())
}
#endif
