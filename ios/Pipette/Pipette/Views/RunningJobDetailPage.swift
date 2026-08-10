import SwiftUI

enum RunningJobDetailPageMode: Equatable {
    case running
    case paused
}

struct RunningJobDetailPage: View {
    let manifest: JobManifest
    let dateTitle: String
    let modelChips: [String]
    let benchmarkChips: [String]
    let quantChips: [String]
    let mode: RunningJobDetailPageMode
    let progressFraction: Double
    let estimatedTimeLeft: String?
    let currentCellLabel: String
    let progressText: String
    let cooling: JobCoolingState?
    let temperatureC: Double?
    let isResumeDisabled: Bool
    let unsubmittedCount: Int
    let isSubmitting: Bool
    let onPocketMode: () -> Void
    let onPause: () -> Void
    let onResume: () -> Void
    let onSubmit: () -> Void
    let canAutoSubmit: Bool
    let onAutoSubmitChanged: (Bool) -> Void

    private var isCooling: Bool { cooling != nil }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                RunDetailHeaderView(
                    manifest: manifest,
                    dateTitle: dateTitle,
                    modelChips: modelChips,
                    benchmarkChips: benchmarkChips,
                    quantChips: quantChips
                )

                Divider()
                    .padding(.top, 36)

                progressSection
                    .padding(.top, 42)

                actions
                    .padding(.top, 48)

                if canAutoSubmit {
                    contributionRow
                        .padding(.top, 42)
                        .padding(.bottom, 28)
                } else {
                    Spacer(minLength: 28)
                }
            }
            .padding(.horizontal, 24)
        }
        .background(Color(.systemBackground))
    }

    private var progressSection: some View {
        VStack(alignment: .leading, spacing: 24) {
            Text(progressTitle)
                .font(.serif(28))
                .foregroundStyle(.primary)

            VStack(spacing: 12) {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule()
                            .fill(Color(.systemGray4))
                            .frame(height: 4)
                        Capsule()
                            .fill(Color.primary)
                            .frame(width: max(0, geo.size.width * progressFraction), height: 4)
                    }
                }
                .frame(height: 4)
                // No implicit width tween on progress updates — this page is live while
                // the benchmark runs, and UI animation shares the GPU with inference.
                .animation(nil, value: progressFraction)

                HStack {
                    Text("\(manifest.completedCells)/\(manifest.totalCells) cells done")
                    Spacer()
                    Text(progressDetail)
                }
                .font(.system(size: 18))
                .foregroundStyle(.secondary)
                .monospacedDigit()
            }

            // The same live indicators Pocket Mode shows, so watching a running
            // job on this page is no less informative than the full-screen cover.
            if mode == .running {
                deviceTemperatureRow

                JobLiveActivityView(
                    currentCellLabel: currentCellLabel,
                    progressText: progressText,
                    cooling: cooling,
                    palette: Self.systemActivityPalette
                )
            }

            if mode == .paused, let reason = manifest.pausedReason {
                Text(reason)
                    .font(.system(size: 15))
                    .foregroundStyle(Color.primary.opacity(0.78))
                    .lineSpacing(5)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        // Ambient cool wash while the readiness gate cools the device — matches
        // Pocket Mode. Drawn behind the section (extended past the content so it
        // doesn't hug the text) so it adds no layout.
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.blue.opacity(isCooling ? 0.08 : 0))
                .padding(.horizontal, -16)
                .padding(.vertical, -12)
                .animation(.easeInOut(duration: 0.45), value: isCooling)
        )
    }

    /// System-styled palette for the shared live-activity block (Pocket Mode
    /// supplies its own dark palette).
    private static let systemActivityPalette = JobActivityPalette(
        primaryText: .primary,
        secondaryText: .secondary,
        accent: .blue
    )

    private var deviceTemperatureRow: some View {
        let display = DeviceThermalDisplay(
            temperatureC: temperatureC, cooling: cooling,
            state: ProcessInfo.processInfo.thermalState)
        return HStack {
            Text("Device temperature")
                .foregroundStyle(.secondary)
            Spacer(minLength: 8)
            HStack(spacing: 8) {
                Image(systemName: display.iconName)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(display.color)
                Text(display.text)
                    .foregroundStyle(.primary)
                    .monospacedDigit()
            }
        }
        .font(.system(size: 16))
    }

    private var actions: some View {
        VStack(alignment: .leading, spacing: 10) {
            switch mode {
            case .running:
                Button(action: onPocketMode) {
                    Label("Open in Pocket Mode", systemImage: "lock.shield")
                        .font(.system(size: 18))
                        .foregroundStyle(Color(.systemBackground))
                        .frame(maxWidth: .infinity)
                        .frame(height: 42)
                        .background(Color.primary, in: Capsule())
                }
                .buttonStyle(.plain)

                Button(action: onPause) {
                    Label("Pause job", systemImage: "pause")
                        .font(.system(size: 18))
                        .foregroundStyle(.primary)
                        .frame(maxWidth: .infinity)
                        .frame(height: 42)
                        .background(Color(.systemBackground), in: Capsule())
                        .overlay(
                            Capsule()
                                .strokeBorder(Color(.systemGray4), lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)

            case .paused:
                if unsubmittedCount > 0 {
                    submitButton
                }

                Button(action: onResume) {
                    Label("Resume job", systemImage: "play.fill")
                        .font(.system(size: 18))
                        .foregroundStyle(Color(.systemBackground))
                        .frame(maxWidth: .infinity)
                        .frame(height: 42)
                        .background(Color.primary, in: Capsule())
                }
                .buttonStyle(.plain)
                .disabled(isResumeDisabled)
            }
        }
    }

    private var submitButton: some View {
        Button(action: onSubmit) {
            HStack {
                if isSubmitting {
                    ProgressView()
                        .controlSize(.small)
                        .tint(Color(.systemBackground))
                    Text("Submitting...")
                } else {
                    Image(systemName: "paperplane.fill")
                    Text("Submit \(unsubmittedCount) \(unsubmittedCount == 1 ? "Result" : "Results")")
                }
            }
            .font(.system(size: 18))
            .foregroundStyle(Color(.systemBackground))
            .frame(maxWidth: .infinity)
            .frame(height: 42)
            .background(Color.primary, in: Capsule())
        }
        .buttonStyle(.plain)
        .disabled(isSubmitting || isResumeDisabled)
    }

    private var contributionRow: some View {
        Button {
            onAutoSubmitChanged(!(manifest.contributeResults == true))
        } label: {
            HStack(alignment: .top, spacing: 14) {
                WizardCheckbox(isOn: manifest.contributeResults == true, size: 18)
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
        .accessibilityLabel("Auto-submit results when this job finishes")
        .accessibilityValue(manifest.contributeResults == true ? "On" : "Off")
    }

    private var progressTitle: String {
        switch mode {
        case .running: return "In progress"
        case .paused: return "Paused"
        }
    }

    private var progressDetail: String {
        if let estimatedTimeLeft {
            return estimatedTimeLeft
        }
        guard mode == .paused else {
            return ""
        }
        let remaining = max(0, manifest.totalCells - manifest.completedCells)
        return "\(remaining) cells left"
    }

}

#if DEBUG
private struct RunningJobDetailPreviewHost: View {
    let mode: RunningJobDetailPageMode

    private var manifest: JobManifest {
        switch mode {
        case .running: return Self.previewManifest(status: .running)
        case .paused: return Self.previewManifest(status: .paused)
        }
    }

    var body: some View {
        RunningJobDetailPage(
            manifest: manifest,
            dateTitle: "2026-05-29",
            modelChips: ["LFM2.5-350M"],
            benchmarkChips: [
                "Decode Throughput",
                "Time to First Token (TTFT)",
                "End-to-End Latency",
                "Max Memory Usage"
            ],
            quantChips: ["q4_0"],
            mode: mode,
            progressFraction: mode == .running ? (1.0 + 0.42) / 28.0 : 1.0 / 28.0,
            estimatedTimeLeft: mode == .running ? "4 min left" : nil,
            currentCellLabel: mode == .running ? "Decode Throughput · LFM2.5-350M" : "",
            progressText: mode == .running ? "Measurement 3/5" : "",
            cooling: mode == .running
                ? JobCoolingState(since: Date(timeIntervalSinceNow: -20), deadline: 300, targetC: 36)
                : nil,
            temperatureC: mode == .running ? 42 : nil,
            isResumeDisabled: false,
            unsubmittedCount: mode == .paused ? 1 : 0,
            isSubmitting: false,
            onPocketMode: {},
            onPause: {},
            onResume: {},
            onSubmit: {},
            canAutoSubmit: true,
            onAutoSubmitChanged: { _ in }
        )
    }

    private static func previewManifest(status: JobStatus) -> JobManifest {
        let benchmarkTypes: [BenchmarkType] = [
            .decodeThroughput,
            .prefillThroughput,
            .endToEndLatency,
            .maxMemoryUsage
        ]

        let cells: [JobCell] = (0..<28).map { index in
            let type = benchmarkTypes[index % benchmarkTypes.count]
            return JobCell(
                cellId: CellId("preview-cell-\(index)"),
                benchmarkId: "\(type.rawValue)_\(index)",
                benchmarkType: type,
                runStatus: previewCellStatus(index: index, jobStatus: status),
                serverJobId: nil,
                errorMessage: nil,
                source: .previewSample
            )
        }

        return JobManifest(
            jobId: JobId("preview-running-job"),
            createdAt: JobDateFormat.iso8601.string(from: Date()),
            nGpuLayers: 99,
            contextSize: 8192,
            cells: cells,
            status: status,
            title: nil
        )
    }

    private static func previewCellStatus(index: Int, jobStatus: JobStatus) -> CellRunStatus {
        if index == 0 {
            return .completed
        }
        if jobStatus == .paused {
            return .cancelled
        }
        return index == 1 ? .running : .pending
    }
}

#Preview("Running Job Detail") {
    RunningJobDetailPreviewHost(mode: .running)
}

#Preview("Paused Job Detail") {
    RunningJobDetailPreviewHost(mode: .paused)
}
#endif
