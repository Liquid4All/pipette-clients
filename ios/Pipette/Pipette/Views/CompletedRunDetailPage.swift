import SwiftUI

struct CompletedRunResultColumn: Identifiable {
    let id: String
    let title: String
    let subtitle: String?
}

struct CompletedRunResultValue: Identifiable {
    let columnId: String
    let displayValue: String?
    let intensity: Double
    let cell: JobCell?
    /// A coordinate with no metric whose cell failed — rendered distinctly and
    /// kept tappable/selectable. Derived once here so the view never re-infers
    /// "did this cell fail?" from the nested optional.
    let isFailed: Bool
    let isCancelled: Bool

    var id: String { columnId }

    init(columnId: String, displayValue: String?, intensity: Double, cell: JobCell? = nil) {
        self.columnId = columnId
        self.displayValue = displayValue
        self.intensity = intensity
        self.cell = cell
        self.isFailed = displayValue == nil && cell?.runStatus == .failed
        self.isCancelled = displayValue == nil && cell?.runStatus == .cancelled
    }
}

struct CompletedRunQuantResultRow: Identifiable {
    let quant: String
    let values: [CompletedRunResultValue]

    var id: String { quant }
}

struct CompletedRunModelResultGroup: Identifiable {
    let id: String
    let modelName: String
    let brand: ModelBrand
    let rows: [CompletedRunQuantResultRow]

    init(id: String? = nil, modelName: String, brand: ModelBrand, rows: [CompletedRunQuantResultRow]) {
        self.id = id ?? modelName
        self.modelName = modelName
        self.brand = brand
        self.rows = rows
    }
}

struct CompletedRunDetailPage: View {
    @Environment(\.pillTabBarReservedHeight) private var pillTabBarReservedHeight

    let manifest: JobManifest
    let dateTitle: String
    let modelChips: [String]
    let benchmarkChips: [String]
    let quantChips: [String]
    let resultColumns: [CompletedRunResultColumn]
    let resultGroups: [CompletedRunModelResultGroup]
    let unsubmittedCount: Int
    let failedCount: Int
    let resumableCount: Int
    let isSubmitting: Bool
    let isRetrying: Bool
    @Binding var selectedCellIds: Set<CellId>
    let onSubmit: () -> Void
    let onResumePaused: () -> Void
    let onRetryFailed: () -> Void
    let onRerunSelected: () -> Void
    let csvExport: ResultsCSVFile
    @State private var collapsedModelIds = Set<String>()
    @State private var isSelecting = false

    /// Page-level horizontal inset. Kept as a constant so the results table can
    /// negate it and scroll edge to edge (see `resultsSection`).
    private static let horizontalInset: CGFloat = 24

    private let firstColumnWidth: CGFloat = 130
    private let resultColumnWidth: CGFloat = 148
    private let rowHeight: CGFloat = 44
    private let groupHeight: CGFloat = 46

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

                controls
                    .padding(.top, 28)

                resultsSection
                    .padding(.top, 32)
            }
            .padding(.horizontal, Self.horizontalInset)
            .padding(.bottom, 28 + pillTabBarReservedHeight)
        }
        .background(Color(.systemBackground))
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Results")
                        .font(.serif(24))
                        .foregroundStyle(.primary)

                    Text(isSelecting
                         ? "Tap cells to select, then re-run the chosen subset."
                         : "Tap a cell to view its details.")
                        .foregroundStyle(.secondary)
                        .font(.system(size: 16))
                }

                Spacer(minLength: 16)

                if canExportResults && !isSelecting {
                    ResultsCSVExportButton(file: csvExport)
                        .padding(.top, 22)
                }
            }

            if resultColumns.isEmpty || resultGroups.isEmpty {
                Text("No saved result payloads were found for this run.")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 16)
            } else {
                ScrollView(.horizontal, showsIndicators: true) {
                    VStack(alignment: .leading, spacing: 0) {
                        tableHeader
                        ForEach(resultGroups) { group in
                            modelGroup(group)
                        }
                    }
                    .background(Color(.systemBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .cardBorder(cornerRadius: 12)
                    // Resting inset that keeps the card aligned with the padded
                    // header; the negative padding below lets it scroll past.
                    .padding(.horizontal, Self.horizontalInset)
                }
                // Cancel the page's horizontal inset so the scroll viewport spans
                // edge to edge — the table now scrolls into the screen edges
                // instead of clipping short of them.
                .padding(.horizontal, -Self.horizontalInset)

                if hasSubmittedResults {
                    submittedLegend
                }
            }
        }
    }

    private var submittedLegend: some View {
        HStack(spacing: 6) {
            Image(systemName: "checkmark.icloud.fill")
                .font(.system(size: 13))
                .foregroundStyle(Self.submittedColor)
            Text("Already submitted")
                .font(.system(size: 14))
                .foregroundStyle(.secondary)
        }
        .padding(.top, 4)
        .accessibilityElement(children: .combine)
    }

    private var canExportResults: Bool {
        resultGroups.contains { group in
            group.rows.contains { row in
                row.values.contains { $0.cell != nil }
            }
        }
    }

    private var hasSubmittedResults: Bool {
        resultGroups.contains { group in
            group.rows.contains { row in
                row.values.contains { $0.cell?.serverJobId != nil }
            }
        }
    }

    private var tableHeader: some View {
        HStack(spacing: 0) {
            Text("Model")
                .font(.system(size: 16))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .padding(.horizontal, 16)
                .frame(width: firstColumnWidth, height: 52, alignment: .leading)
                .background(Color(.systemBackground))
                .overlay(alignment: .trailing) {
                    Divider()
                }

            ForEach(resultColumns) { column in
                VStack(alignment: .leading, spacing: 3) {
                    Text(column.title)
                        .font(.system(size: 16))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .minimumScaleFactor(0.75)
                    if let subtitle = column.subtitle {
                        Text(subtitle)
                            .font(.system(size: 14))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                .padding(.horizontal, 14)
                .frame(width: resultColumnWidth, height: 52, alignment: .leading)
                .background(Color(.systemBackground))
                .overlay(alignment: .trailing) {
                    Divider()
                }
            }
        }
        .overlay(alignment: .bottom) {
            Divider()
        }
    }

    @ViewBuilder
    private func modelGroup(_ group: CompletedRunModelResultGroup) -> some View {
        let isCollapsed = collapsedModelIds.contains(group.id)

        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.snappy(duration: 0.2)) {
                    toggleModel(group)
                }
            } label: {
                HStack(spacing: 14) {
                    Image(systemName: isCollapsed ? "chevron.right" : "chevron.down")
                        .font(.system(size: 14, weight: .regular))
                        .foregroundStyle(.secondary)
                        .frame(width: 16)
                    BrandLogoView(brand: group.brand, size: 20)
                    Text(group.modelName)
                        .font(.system(size: 16))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Spacer()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(group.modelName)
            .accessibilityHint(isCollapsed ? "Shows results for this model" : "Hides results for this model")
            .padding(.horizontal, 18)
            .frame(width: tableWidth, height: groupHeight, alignment: .leading)
            .background(Color(.systemBackground))
            .overlay(alignment: .bottom) {
                Divider()
            }

            if !isCollapsed {
                ForEach(group.rows) { row in
                    quantRow(row)
                }
            }
        }
    }

    private func toggleModel(_ group: CompletedRunModelResultGroup) {
        if collapsedModelIds.contains(group.id) {
            collapsedModelIds.remove(group.id)
        } else {
            collapsedModelIds.insert(group.id)
        }
    }

    private func quantRow(_ row: CompletedRunQuantResultRow) -> some View {
        let rowCellIds = row.values.compactMap { $0.cell?.cellId }
        let rowSelected = rowCellIds.filter { selectedCellIds.contains($0) }.count
        return HStack(spacing: 0) {
            firstColumn(row: row, rowCellIds: rowCellIds, rowSelected: rowSelected)

            ForEach(row.values) { value in
                cellView(value)
            }
        }
        .overlay(alignment: .bottom) {
            Divider()
        }
    }

    @ViewBuilder
    private func firstColumn(
        row: CompletedRunQuantResultRow,
        rowCellIds: [CellId],
        rowSelected: Int
    ) -> some View {
        let label = HStack(spacing: 8) {
            if isSelecting {
                Image(systemName: selectionIconName(selected: rowSelected, total: rowCellIds.count))
                    .font(.system(size: 16))
                    .foregroundStyle(rowSelected == 0 ? Color.secondary : Color.accentColor)
            }
            AppTextChip(text: row.quant, font: .system(size: 14), height: 28)
        }
        .padding(.leading, 14)
        .frame(width: firstColumnWidth, height: rowHeight, alignment: .leading)
        .background(Color(.systemBackground))
        .overlay(alignment: .trailing) {
            Divider()
        }

        if isSelecting && !rowCellIds.isEmpty {
            Button { toggleAll(rowCellIds) } label: { label }
                .buttonStyle(.plain)
                .accessibilityLabel("Select all \(row.quant) results")
        } else {
            label
        }
    }

    @ViewBuilder
    private func cellView(_ value: CompletedRunResultValue) -> some View {
        if isSelecting, let cell = value.cell {
            Button { toggle(cell.cellId) } label: {
                resultCell(value, selected: selectedCellIds.contains(cell.cellId))
            }
            .buttonStyle(.plain)
        } else if let cell = value.cell, value.displayValue != nil || value.isFailed || value.isCancelled {
            NavigationLink(destination: CellDetailView(jobId: manifest.jobId, cell: cell)) {
                resultCell(value, selected: false)
            }
            .buttonStyle(.plain)
            .accessibilityHint("Shows cell details")
        } else {
            resultCell(value, selected: false)
        }
    }

    private func resultCell(_ value: CompletedRunResultValue, selected: Bool) -> some View {
        let hasValue = value.displayValue != nil
        let isFailed = value.isFailed
        let isCancelled = value.isCancelled
        let isSubmitted = value.cell?.serverJobId != nil
        return Text(value.displayValue ?? (isFailed ? "Failed" : (isCancelled ? "Paused" : "-")))
            .font(.system(size: 16))
            .foregroundStyle(statusColor(hasValue: hasValue, isFailed: isFailed, isCancelled: isCancelled))
            .monospacedDigit()
            .padding(.horizontal, 14)
            .frame(width: resultColumnWidth, height: rowHeight, alignment: .leading)
            .background(cellBackground(
                hasValue: hasValue,
                isFailed: isFailed,
                isCancelled: isCancelled,
                intensity: value.intensity
            ))
            .overlay(alignment: .topTrailing) {
                if selected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(Color.accentColor)
                        .padding(3)
                } else if isSubmitted && !isSelecting {
                    // Hidden while selecting so the green submission badge can't
                    // be mistaken for the blue selection checkmark.
                    Image(systemName: "checkmark.icloud.fill")
                        .font(.system(size: 12))
                        .foregroundStyle(Self.submittedColor)
                        .padding(3)
                        .accessibilityLabel("Submitted")
                }
            }
            .overlay(alignment: .trailing) {
                Divider()
                    .background(Color.black.opacity(0.05))
            }
    }

    private func statusColor(hasValue: Bool, isFailed: Bool, isCancelled: Bool) -> Color {
        if isFailed { return .red }
        if isCancelled { return .orange }
        return hasValue ? .primary : .secondary
    }

    private func cellBackground(
        hasValue: Bool,
        isFailed: Bool,
        isCancelled: Bool,
        intensity: Double
    ) -> Color {
        if hasValue { return resultCellColor(intensity) }
        if isFailed { return Color.red.opacity(0.10) }
        if isCancelled { return Color.orange.opacity(0.10) }
        return Color(.systemBackground)
    }

    @ViewBuilder
    private var controls: some View {
        VStack(spacing: 10) {
            if unsubmittedCount > 0 {
                submitButton
            }
            if isSelecting {
                primaryButton(
                    "Re-run \(selectedCellIds.count) selected",
                    systemImage: "arrow.clockwise.circle",
                    disabled: selectedCellIds.isEmpty || isRetrying || isSubmitting
                ) {
                    onRerunSelected()
                    isSelecting = false
                }
                borderedButton(selectedCellIds.isEmpty ? "Cancel" : "Clear selection") {
                    if selectedCellIds.isEmpty {
                        isSelecting = false
                    } else {
                        selectedCellIds.removeAll()
                    }
                }
            } else {
                if resumableCount > 0 {
                    primaryButton(
                        "Resume \(resumableCount) \(resumableCount == 1 ? "cell" : "cells")",
                        systemImage: "play.fill",
                        disabled: isRetrying || isSubmitting
                    ) {
                        onResumePaused()
                    }
                }
                if failedCount > 0 {
                    borderedButton(
                        "Retry \(failedCount) failed \(failedCount == 1 ? "cell" : "cells")",
                        systemImage: "arrow.clockwise",
                        disabled: isRetrying || isSubmitting
                    ) {
                        onRetryFailed()
                    }
                }
                if !resultGroups.isEmpty {
                    borderedButton("Select cells", systemImage: "checkmark.circle") {
                        isSelecting = true
                    }
                }
            }
        }
    }

    private func primaryButton(
        _ title: String,
        systemImage: String,
        disabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.system(size: 18))
                .foregroundStyle(Color(.systemBackground))
                .frame(maxWidth: .infinity)
                .frame(height: 42)
                .background(Color.primary, in: Capsule())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
    }

    private func borderedButton(
        _ title: String,
        systemImage: String? = nil,
        disabled: Bool = false,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Group {
                if let systemImage {
                    Label(title, systemImage: systemImage)
                } else {
                    Text(title)
                }
            }
            .font(.system(size: 18))
            .foregroundStyle(.primary)
            .frame(maxWidth: .infinity)
            .frame(height: 42)
            .background(Color(.systemBackground), in: Capsule())
            .overlay(
                Capsule().strokeBorder(Color(.systemGray4), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .disabled(disabled)
    }

    private func toggle(_ id: CellId) {
        if selectedCellIds.contains(id) {
            selectedCellIds.remove(id)
        } else {
            selectedCellIds.insert(id)
        }
    }

    private func toggleAll(_ ids: [CellId]) {
        let allSelected = !ids.isEmpty && ids.allSatisfy { selectedCellIds.contains($0) }
        if allSelected {
            ids.forEach { selectedCellIds.remove($0) }
        } else {
            ids.forEach { selectedCellIds.insert($0) }
        }
    }

    private func selectionIconName(selected: Int, total: Int) -> String {
        if selected == 0 { return "circle" }
        if selected == total { return "checkmark.circle.fill" }
        return "minus.circle.fill"
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
        .disabled(isSubmitting || isRetrying)
    }

    /// Accent green shared by the heatmap fill and the "submitted" badge so a
    /// submitted cell reads as part of the same visual family.
    static let submittedColor = Color(red: 0.06, green: 0.72, blue: 0.50)

    private func resultCellColor(_ intensity: Double) -> Color {
        Self.submittedColor
            .opacity(max(0.12, min(0.55, intensity)))
    }

    private var tableWidth: CGFloat {
        firstColumnWidth + (CGFloat(resultColumns.count) * resultColumnWidth)
    }

}

#if DEBUG
private struct CompletedRunDetailPreviewHost: View {
    var withFailures = false
    @State private var selected: Set<CellId> = []

    var body: some View {
        CompletedRunDetailPage(
            manifest: Self.previewManifest,
            dateTitle: "2026-05-28",
            modelChips: ["LFM2.5-Instruct-2.5B", "Qwen4.0-120M", "Granite-1B"],
            benchmarkChips: ["Decode Throughput", "Time to First Token (TTFT)", "Max Memory Usage"],
            quantChips: ["q4_0", "q_4km", "q5_km"],
            resultColumns: [
                CompletedRunResultColumn(id: "decode", title: "Decode Throughput", subtitle: "100-100"),
                CompletedRunResultColumn(id: "prefill", title: "TTFT", subtitle: "512"),
                CompletedRunResultColumn(id: "memory", title: "Max Memory", subtitle: "512")
            ],
            resultGroups: withFailures ? Self.previewGroupsWithFailure : Self.previewGroups,
            unsubmittedCount: withFailures ? 0 : 4,
            failedCount: withFailures ? 1 : 0,
            resumableCount: 0,
            isSubmitting: false,
            isRetrying: false,
            selectedCellIds: $selected,
            onSubmit: {},
            onResumePaused: {},
            onRetryFailed: {},
            onRerunSelected: {},
            csvExport: ResultsCSVFile(filename: "pipette-results-preview.csv") { "" }
        )
    }

    private static var previewGroupsWithFailure: [CompletedRunModelResultGroup] {
        [
            CompletedRunModelResultGroup(
                modelName: "LFM2.5-Instruct-2.5B",
                brand: .liquid,
                rows: [
                    CompletedRunQuantResultRow(quant: "q4_0", values: [
                        CompletedRunResultValue(columnId: "decode", displayValue: "25", intensity: 0.50),
                        CompletedRunResultValue(columnId: "prefill", displayValue: "25", intensity: 0.50),
                        CompletedRunResultValue(columnId: "memory", displayValue: "4.1 GB", intensity: 0.32)
                    ]),
                    CompletedRunQuantResultRow(quant: "q_4km", values: [
                        failedValue(columnId: "decode"),
                        CompletedRunResultValue(columnId: "prefill", displayValue: "100", intensity: 0.14),
                        CompletedRunResultValue(columnId: "memory", displayValue: "3.8 GB", intensity: 0.45)
                    ])
                ]
            )
        ]
    }

    private static func failedValue(columnId: String) -> CompletedRunResultValue {
        CompletedRunResultValue(
            columnId: columnId,
            displayValue: nil,
            intensity: 0,
            cell: JobCell(
                cellId: CellId("failed-\(columnId)"),
                benchmarkId: columnId,
                benchmarkType: .decodeThroughput,
                runStatus: .failed,
                serverJobId: nil,
                errorMessage: "Engine error: decode rc=-3.",
                source: .previewSample
            )
        )
    }

    private static var previewGroups: [CompletedRunModelResultGroup] {
        [
            group("LFM2.5-Instruct-2.5B", brand: .liquid, rows: [
                row("q4_0", values: [("decode", "25", 0.50), ("prefill", "25", 0.50), ("memory", "4.1 GB", 0.32)],
                    submitted: ["decode", "prefill", "memory"]),
                row("q_4km", values: [("decode", "50", 0.25), ("prefill", "100", 0.14), ("memory", "3.8 GB", 0.45)],
                    submitted: ["decode"]),
                row("q5_km", values: [("decode", "50", 0.28), ("prefill", "50", 0.31), ("memory", "4.4 GB", 0.20)])
            ]),
            group("Qwen4.0-120M", brand: .qwen, rows: [
                row("q4_0", values: [("decode", "50", 0.31), ("prefill", "50", 0.31), ("memory", "2.2 GB", 0.50)]),
                row("q4_km", values: [("decode", "25", 0.50), ("prefill", "25", 0.50), ("memory", "2.3 GB", 0.46)]),
                row("q5_km", values: [("decode", "50", 0.31), ("prefill", "50", 0.31), ("memory", "2.5 GB", 0.38)])
            ])
        ]
    }

    private static func group(
        _ modelName: String,
        brand: ModelBrand,
        rows: [CompletedRunQuantResultRow]
    ) -> CompletedRunModelResultGroup {
        CompletedRunModelResultGroup(modelName: modelName, brand: brand, rows: rows)
    }

    private static func row(
        _ quant: String,
        values: [(String, String, Double)],
        submitted: Set<String> = []
    ) -> CompletedRunQuantResultRow {
        CompletedRunQuantResultRow(
            quant: quant,
            values: values.map { value in
                CompletedRunResultValue(
                    columnId: value.0,
                    displayValue: value.1,
                    intensity: value.2,
                    cell: submitted.contains(value.0)
                        ? previewSubmittedCell(quant: quant, columnId: value.0)
                        : nil
                )
            }
        )
    }

    private static func previewSubmittedCell(quant: String, columnId: String) -> JobCell {
        JobCell(
            cellId: CellId("\(quant)-\(columnId)"),
            benchmarkId: columnId,
            benchmarkType: .decodeThroughput,
            runStatus: .completed,
            serverJobId: "server-\(quant)-\(columnId)",
            errorMessage: nil,
            source: .previewSample
        )
    }

    private static var previewManifest: JobManifest {
        let models = [
            ("LiquidAI/LFM2.5-Instruct-2.5B-GGUF", "/tmp/LFM2.5-Instruct-2.5B-Q4_0.gguf"),
            ("Qwen/Qwen4.0-120M-GGUF", "/tmp/Qwen4.0-120M-Q4_K_M.gguf")
        ]
        let benchmarkIds = [
            "decode_throughput_100_100",
            "prefill_throughput_512",
            "max_memory_usage_512"
        ]

        let cells = models.flatMap { model in
            benchmarkIds.enumerated().map { offset, benchmarkId in
                JobCell(
                    cellId: CellId("\(model.0)-\(benchmarkId)"),
                    benchmarkId: benchmarkId,
                    benchmarkType: [.decodeThroughput, .prefillThroughput, .maxMemoryUsage][offset],
                    runStatus: .completed,
                    serverJobId: nil,
                    errorMessage: nil,
                    source: .previewSample
                )
            }
        }

        return JobManifest(
            jobId: JobId("preview-completed-job"),
            createdAt: "2026-05-28T16:41:00Z",
            nGpuLayers: 99,
            contextSize: 8192,
            cells: cells,
            status: .completed,
            title: nil
        )
    }
}

#Preview("Completed Run Detail") {
    CompletedRunDetailPreviewHost()
}

#Preview("Completed Run · Failures") {
    NavigationStack {
        CompletedRunDetailPreviewHost(withFailures: true)
    }
}
#endif
