import Combine
import SwiftUI
import UIKit

/// Full-screen Pocket Mode for a running job. Presented as a fullScreenCover so
/// the tab bar is hidden and exit remains intentional while the benchmark runs.
struct PocketModeView: View {
    let jobId: JobId
    private let previewManifest: JobManifest?

    @Environment(JobRunner.self) private var jobRunner
    @Environment(JobStore.self) private var jobStore
    @Environment(\.dismiss) private var dismiss

    @State private var now: Date = Date()
    @State private var sliderOffset: CGFloat = 0
    @State private var isDragging = false

    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    init(jobId: JobId, previewManifest: JobManifest? = nil) {
        self.jobId = jobId
        self.previewManifest = previewManifest
    }

    private var manifest: JobManifest? {
        previewManifest ?? jobStore.job(id: jobId)
    }

    var body: some View {
        ZStack {
            PocketPalette.background.ignoresSafeArea()

            VStack(spacing: 0) {
                Spacer(minLength: 128)

                PocketBrandMark()
                    .frame(width: 44, height: 44)

                Text("Benchmarking in progress...")
                    .font(.serif(24))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [PocketPalette.secondaryText, PocketPalette.primaryText],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .minimumScaleFactor(0.75)
                    .padding(.horizontal, 48)
                    .padding(.top, 40)

                PocketThermalRow(
                    label: thermalDisplay.text,
                    icon: thermalDisplay.iconName,
                    color: thermalDisplay.color
                )
                .padding(.horizontal, 48)
                .padding(.top, 30)

                PocketProgressCard(
                    dateTitle: dateTitle,
                    summary: summaryText,
                    progress: progressFraction,
                    completedCells: manifest?.completedCells ?? 0,
                    totalCells: manifest?.totalCells ?? 0,
                    timeLeft: estimatedTimeLeft ?? "--",
                    currentCellLabel: jobRunner.currentCellLabel,
                    progressText: jobRunner.currentProgressText,
                    cooling: jobRunner.coolingState
                )
                .padding(.horizontal, 48)
                .padding(.top, 16)

                Spacer(minLength: 48)

                SlideToExitControl(
                    offset: $sliderOffset,
                    isDragging: $isDragging,
                    onExit: dismiss.callAsFunction
                )
                .padding(.horizontal, 24)
                .padding(.bottom, 28)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .interactiveDismissDisabled(true)
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
        .defersSystemGestures(on: .all)
        .onReceive(timer) { now = $0 }
        .onChange(of: jobRunner.runningJobId) { _, newValue in
            if newValue == nil || newValue != jobId {
                dismiss()
            }
        }
    }

    // MARK: - Derived values

    private var progressFraction: Double {
        guard let manifest, manifest.totalCells > 0 else { return 0 }
        let done = Double(manifest.completedCells)
        let within = max(0, min(1, jobRunner.currentCellFraction))
        return min(1, (done + within) / Double(manifest.totalCells))
    }

    private var dateTitle: String {
        guard let manifest else { return "--" }
        if let date = manifest.createdDate {
            return JobDateFormat.shortDate.string(from: date)
        }
        return String(manifest.createdAt.prefix(10))
    }

    private var summaryText: String {
        guard let manifest else { return "Loading benchmark details" }
        return "\(manifest.modelNames.count) \("model".pluralized(manifest.modelNames.count)) - \(manifest.benchmarkIds.count) \("benchmark".pluralized(manifest.benchmarkIds.count))"
    }

    private var estimatedTimeLeft: String? {
        guard let manifest else { return nil }
        return jobRunner.estimatedTimeLeft(jobId: manifest.jobId, now: now)
    }

    /// Device-temperature indicator: the real reading (with the cooldown target
    /// while cooling) when a sensor is available, else the OS thermal-state label.
    private var thermalDisplay: DeviceThermalDisplay {
        DeviceThermalDisplay(
            temperatureC: jobRunner.deviceTemperatureC,
            cooling: jobRunner.coolingState,
            state: ProcessInfo.processInfo.thermalState)
    }
}

// MARK: - Pocket Mode Components

private enum PocketPalette {
    static let background = Color(red: 10 / 255, green: 10 / 255, blue: 10 / 255)
    static let card = Color(red: 23 / 255, green: 23 / 255, blue: 23 / 255)
    static let divider = Color.white.opacity(0.10)
    static let track = Color(red: 64 / 255, green: 64 / 255, blue: 64 / 255)
    static let primaryText = Color(red: 250 / 255, green: 250 / 255, blue: 250 / 255)
    static let secondaryText = Color(red: 163 / 255, green: 163 / 255, blue: 163 / 255)
    static let green = Color(red: 34 / 255, green: 197 / 255, blue: 94 / 255)
    // Cool wash laid over the card while the readiness gate cools the device — an
    // ambient "waiting" cue that adds no layout, only color.
    static let coolWash = Color(red: 56 / 255, green: 122 / 255, blue: 210 / 255).opacity(0.16)
    static let coolBorder = Color(red: 104 / 255, green: 162 / 255, blue: 230 / 255).opacity(0.5)
    // Emphasis color for the cooldown caption — a light cool blue that reads on
    // the dark card and ties to the wash.
    static let coolText = Color(red: 137 / 255, green: 186 / 255, blue: 247 / 255)
}

private struct PocketBrandMark: View {
    // A transparent "P" silhouette tinted (template mode) with the light gradient
    // so it reads on the dark pocket background. The shipped `pipette-logo-mark`
    // has an opaque white background, so template mode fills the whole square —
    // this dedicated alpha-only silhouette is what makes the glyph show.
    var body: some View {
        Image("pipette-mark-silhouette")
            .resizable()
            .renderingMode(.template)
            .aspectRatio(contentMode: .fit)
            .foregroundStyle(
                LinearGradient(
                    colors: [PocketPalette.primaryText, Color(red: 115 / 255, green: 115 / 255, blue: 115 / 255)],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .accessibilityHidden(true)
    }
}

private struct PocketThermalRow: View {
    let label: String
    let icon: String
    let color: Color

    var body: some View {
        HStack(alignment: .center) {
            Text("Device temperature")
                .font(.system(size: 16))
                .foregroundStyle(PocketPalette.secondaryText)

            Spacer(minLength: 14)

            HStack(spacing: 10) {
                Image(systemName: icon)
                    .font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(color)
                    .frame(width: 8, height: 8)

                Text(label)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(PocketPalette.primaryText)
                    .lineLimit(1)
            }
            .padding(.horizontal, 13)
            .frame(height: 28)
            .background(PocketPalette.background, in: Capsule())
            .overlay(Capsule().strokeBorder(PocketPalette.divider, lineWidth: 1))
        }
    }
}

private struct PocketProgressCard: View {
    let dateTitle: String
    let summary: String
    let progress: Double
    let completedCells: Int
    let totalCells: Int
    let timeLeft: String
    let currentCellLabel: String
    let progressText: String
    let cooling: JobCoolingState?

    private var isCooling: Bool { cooling != nil }

    private var activityPalette: JobActivityPalette {
        JobActivityPalette(
            primaryText: PocketPalette.primaryText,
            secondaryText: PocketPalette.secondaryText,
            accent: PocketPalette.coolText
        )
    }

    var body: some View {
        VStack(spacing: 0) {
            VStack(spacing: 10) {
                Text(dateTitle)
                    .font(.serif(24))
                    .foregroundStyle(PocketPalette.primaryText)
                    .monospacedDigit()
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)

                Text(summary)
                    .font(.system(size: 16))
                    .foregroundStyle(PocketPalette.secondaryText)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)
            }
            .padding(.top, 28)

            VStack(spacing: 12) {
                PocketProgressBar(progress: progress)
                    .frame(height: 8)

                HStack {
                    Text("\(completedCells)/\(totalCells) cells done")
                    Spacer(minLength: 8)
                    Text(timeLeft)
                }
                .font(.system(size: 16))
                .foregroundStyle(PocketPalette.secondaryText)
                .monospacedDigit()
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            }
            .padding(.top, 34)

            JobLiveActivityView(
                currentCellLabel: currentCellLabel,
                progressText: progressText,
                cooling: cooling,
                palette: activityPalette
            )
            .padding(.top, 14)

            Spacer()
                .frame(height: 24)
        }
        .padding(.horizontal, 32)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(PocketPalette.card)
                .overlay(
                    // Cross-fade only the wash opacity at the cooling boundaries —
                    // scoped to this shape so it can't tween the progress bar. Safe
                    // for the measurement: the gate holds the GPU idle while cooling.
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(PocketPalette.coolWash)
                        .opacity(isCooling ? 1 : 0)
                        .animation(.easeInOut(duration: 0.45), value: isCooling)
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .strokeBorder(isCooling ? PocketPalette.coolBorder : PocketPalette.divider, lineWidth: 1)
                .animation(.easeInOut(duration: 0.45), value: isCooling)
        )
    }
}

private struct PocketProgressBar: View {
    let progress: Double

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(PocketPalette.track)

                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [PocketPalette.secondaryText, PocketPalette.primaryText],
                            startPoint: .leading,
                            endPoint: .trailing
                        )
                    )
                    .frame(width: max(8, geo.size.width * max(0, min(1, progress))))
            }
        }
        .clipShape(Capsule())
        // Progress ticks must not animate: no implicit width tween competing with
        // the GPU-bound benchmark while pocket mode is on screen.
        .animation(nil, value: progress)
    }
}

private struct SlideToExitControl: View {
    @Binding var offset: CGFloat
    @Binding var isDragging: Bool
    let onExit: () -> Void

    var body: some View {
        GeometryReader { geo in
            let trackWidth = geo.size.width
            let thumbSize: CGFloat = 42
            let maxOffset = max(0, trackWidth - thumbSize)
            let threshold = maxOffset * 0.72

            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.white.opacity(0.05))
                    .overlay(Capsule().strokeBorder(PocketPalette.track, lineWidth: 1))

                Text("Slide to exit pocket mode")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(PocketPalette.primaryText.opacity(isDragging ? 0.35 : 1))
                    .frame(maxWidth: .infinity)
                    .allowsHitTesting(false)

                ZStack {
                    Image(systemName: "rectangle.portrait.and.arrow.right")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundStyle(PocketPalette.primaryText)
                }
                .frame(width: thumbSize, height: thumbSize)
                .offset(x: offset)
                // The gesture lives on the whole capsule (below), not the thumb.
                .allowsHitTesting(false)
            }
            // Drag anywhere on the capsule, not just the 42pt thumb — the thumb-only
            // hit target made the control feel unresponsive (most touches landed on the
            // label, which ignores hits). Tracking `translation` keeps a tap (no
            // movement) from ever crossing the threshold and exiting accidentally.
            .contentShape(Capsule())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        isDragging = true
                        offset = max(0, min(maxOffset, value.translation.width))
                    }
                    .onEnded { _ in
                        isDragging = false
                        if offset >= threshold {
                            UINotificationFeedbackGenerator()
                                .notificationOccurred(.success)
                            onExit()
                        } else {
                            // Snap back with no animation — pocket mode is on screen
                            // while the benchmark runs, and UI animation shares the GPU
                            // with inference, skewing the measurement.
                            offset = 0
                        }
                    }
            )
        }
        .frame(height: 42)
        .accessibilityLabel("Slide to exit pocket mode")
    }
}

#if DEBUG
private struct PocketModePreviewHost: View {
    @State private var runner = previewRunner

    var body: some View {
        PocketModeView(
            jobId: Self.previewJobId,
            previewManifest: Self.previewManifest
        )
        .environment(runner)
        .environment(JobStore(storage: FileStorage.production))
        .previewLayout(.fixed(width: 440, height: 988))
    }

    private static let previewJobId = JobId("preview-pocket-job")

    private static var previewRunner: JobRunner {
        let runner = JobRunner()
        // Fresh run over all 28 preview cells (1 done + 1 running at 42%),
        // backdated ~90s so the ETA label shows a realistic estimate.
        runner.start(jobId: previewJobId, flag: CancelFlag(), completedAtStart: 0, totalToRun: 28)
        runner.startedAt = Date().addingTimeInterval(-90)
        runner.currentCellLabel = "decode_throughput_512_100 / LFM2.5-350M"
        runner.currentProgressText = "Prefill 42%"
        runner.currentCellFraction = 0.42
        // 1 cell done + 42% into the next, so the ETA label shows an estimate.
        runner.anchorETA(completedCells: 1)
        return runner
    }

    private static var previewManifest: JobManifest {
        let benchmarkTypes: [BenchmarkType] = [
            .decodeThroughput,
            .prefillThroughput,
            .endToEndLatency,
            .maxMemoryUsage
        ]

        let cells: [JobCell] = (0..<28).map { index in
            let type = benchmarkTypes[index % benchmarkTypes.count]
            return JobCell(
                cellId: CellId("preview-pocket-cell-\(index)"),
                benchmarkId: "\(type.rawValue)_\(index)",
                benchmarkType: type,
                runStatus: index == 0 ? .completed : (index == 1 ? .running : .pending),
                serverJobId: nil,
                errorMessage: nil,
                source: .previewSample
            )
        }

        return JobManifest(
            jobId: previewJobId,
            createdAt: JobDateFormat.iso8601.string(from: Date()),
            nGpuLayers: 99,
            contextSize: 8192,
            cells: cells,
            status: .running,
            title: nil
        )
    }
}

#Preview("Pocket Mode") {
    PocketModePreviewHost()
}
#endif
