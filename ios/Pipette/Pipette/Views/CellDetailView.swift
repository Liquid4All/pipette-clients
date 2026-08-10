import SwiftUI

struct CellDetailRow: Identifiable {
    let id = UUID()
    let title: String
    let value: String
}

struct CellDetailView: View {
    let jobId: JobId?
    let cell: JobCell

    @Environment(\.pillTabBarReservedHeight) private var pillTabBarReservedHeight
    @Environment(\.storage) private var storage
    @State private var payload: [String: Any]?
    @State private var metrics: [[String: Any]]?

    init(
        jobId: JobId? = nil,
        cell: JobCell,
        previewPayload: [String: Any]? = nil,
        previewMetrics: [[String: Any]]? = nil
    ) {
        self.jobId = jobId
        self.cell = cell
        _payload = State(initialValue: previewPayload)
        _metrics = State(initialValue: previewMetrics)
    }

    private let labelColumnWidth: CGFloat = 168

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                properties
                    .padding(.horizontal, 24)
                    .padding(.top, 54)
                    .padding(.bottom, 34)

                // Why this result has no submission state: it was never going to have one.
                if !cell.isSubmittable {
                    LocalResultBadge(showsCaption: true)
                        .padding(.horizontal, 24)
                        .padding(.bottom, 28)
                }

                if cell.runStatus == .failed, let error = cell.errorMessage {
                    errorCallout(error)
                        .padding(.horizontal, 24)
                        .padding(.bottom, 28)
                }

                Divider()

                resultsTable
                    .padding(.horizontal, 10)
                    .padding(.top, 20)
                    .padding(.bottom, 32 + pillTabBarReservedHeight)
            }
        }
        .background(Color(.systemBackground))
        .navigationTitle("Cell details")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Color(.systemBackground), for: .navigationBar)
        .task {
            // File reads + JSON parsing happen off the main thread; only the
            // resulting state assignment hops back onto it.
            guard payload == nil else { return }
            guard let jobId else { return }
            let storage = storage
            let result = await Task.detached(priority: .userInitiated) {
                Self.readPayload(jobId: jobId, for: cell, storage: storage)
            }.value
            payload = result.payload
            metrics = result.metrics
        }
    }

    private var properties: some View {
        VStack(alignment: .leading, spacing: PropertyList.rowGap) {
            PropertyChipRow(title: "Models", values: [ModelCatalog.displayName(for: cell.modelName)]) {
                PropertyChip.model($0, brandSource: cell.modelName)
            }
            PropertyChipRow(title: "Quant", values: [quantLabel]) {
                PropertyChip.text($0)
            }
            PropertyChipRow(title: "Benchmark", values: [benchmarkTitle]) {
                PropertyChip.text($0)
            }
            PropertyChipRow(title: "Status", values: [cell.runStatus.label]) {
                PropertyChip.text($0)
            }
            if let runtimeFlags {
                PropertyRow(title: "Runtime") {
                    Text(runtimeFlags)
                        .font(.system(size: 16))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
    }

    private func errorCallout(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 15))
                .foregroundStyle(.red)
            Text(message)
                .font(.system(size: 15))
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(14)
        .background(Color.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var resultsTable: some View {
        Group {
            if resultRows.isEmpty {
                Text("No saved result payload was found for this cell.")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                VStack(spacing: 0) {
                    ForEach(Array(resultRows.enumerated()), id: \.element.id) { index, row in
                        resultRow(row)
                            .overlay(alignment: .bottom) {
                                if index < resultRows.count - 1 {
                                    Divider()
                                }
                            }
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                .cardBorder(cornerRadius: 20)
            }
        }
    }

    private func resultRow(_ row: CellDetailRow) -> some View {
        HStack(spacing: 0) {
            Text(row.title)
                .font(.system(size: 16))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(width: labelColumnWidth, alignment: .leading)

            Text(row.value)
                .font(.system(size: 16))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.tail)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
        .padding(.horizontal, 22)
        .frame(height: 64)
        .background(Color(.systemBackground))
    }

    private var resultRows: [CellDetailRow] {
        guard let payload else { return [] }

        var rows: [CellDetailRow] = []
        if let primaryMetric {
            rows.append(primaryMetric)
        }

        let hiddenKeys: Set<String> = [
            "benchmark_id",
            "cell_id",
            "completions",
            "device_name",
            "job_id",
            "runtime_flags",
            "submitted_at"
        ]

        let fields = payload
            .filter { key, value in
                !hiddenKeys.contains(key) && PayloadScalars.string(value) != nil
            }
            .sorted { $0.key < $1.key }

        for (key, value) in fields {
            rows.append(CellDetailRow(title: humanizedKey(key), value: PayloadScalars.string(value) ?? ""))
        }

        if let metrics {
            for metric in metrics {
                let name = metric["metric"] as? String ?? "Metric"
                let value = PayloadScalars.string(metric["value"] ?? "")
                let unit = metric["unit"] as? String
                rows.append(CellDetailRow(
                    title: humanizedKey(name),
                    value: [value, unit].compactMap { $0 }.joined(separator: " ")
                ))
            }
        }

        return rows
    }

    private var primaryMetric: CellDetailRow? {
        guard let payload, let type = BenchmarkType(rawValue: benchmarkType) else { return nil }
        do {
            guard let metric = try BenchmarkMetrics.compute(
                payload: payload, params: benchmarkCatalogParams, type: type
            ) else { return nil }
            return CellDetailRow(title: metric.name, value: detailValue(for: metric))
        } catch {
            AppLog.results.error("metric parse failed for cell \(cell.cellId.value) (\(cell.benchmarkId)): \(error)")
            return nil
        }
    }

    /// Detail-row rendering for a computed metric: fixed-precision throughput and
    /// latency, or a human byte size. (CSV export renders the same metric compactly.)
    private func detailValue(for metric: BenchmarkMetric) -> String {
        switch metric.unit {
        case "ms": return String(format: "%.0f ms", metric.numericValue)
        case "bytes": return ByteFormat.memory(Int64(metric.numericValue))
        default: return String(format: "%.1f %@", metric.numericValue, metric.unit)
        }
    }

    /// Reads a cell's payload (and optional metrics) from disk off the main
    /// thread, returning the parsed JSON rather than mutating view state. Parse
    /// failures are logged and surface as nil; an absent file is a normal nil.
    private nonisolated static func readPayload(
        jobId: JobId,
        for cell: JobCell,
        storage: Storage
    ) -> (payload: [String: Any]?, metrics: [[String: Any]]?) {
        var payload: [String: Any]?
        do {
            payload = try storage.results.payloadPath(of: cell.cellId)
                .flatMap { try ResultPayload.readObject(at: $0) }
        } catch {
            AppLog.results.error("payload parse failed for cell \(cell.cellId.value): \(error)")
        }

        var metrics: [[String: Any]]?
        do {
            let object = try storage.results.metricsPath(of: cell.cellId)
                .flatMap { try ResultPayload.readObject(at: $0) }
            metrics = object?["metrics"] as? [[String: Any]]
        } catch {
            AppLog.results.error("metrics parse failed for cell \(cell.cellId.value): \(error)")
        }

        return (payload, metrics)
    }

    private var benchmarkTitle: String {
        BenchmarkCatalog.displayName(for: benchmarkType)
    }

    /// This cell.s catalog entry. Resolved per access — both readers below re-run it,
    /// so it takes the single-id lookup rather than reading the whole catalog.
    private var catalogItem: BenchmarkItem? {
        BenchmarkCatalog.item(forId: cell.benchmarkId, store: storage.benchmarks)
    }

    private var benchmarkType: String {
        cell.benchmarkType?.rawValue ?? catalogItem?.benchmarkType ?? cell.benchmarkId
    }

    private var benchmarkCatalogParams: [String: Any] {
        catalogItem?.rawJson ?? [:]
    }

    private var quantLabel: String {
        // The cell's typed source, the only place an MLX bit-width is recoverable.
        normalizedQuant(cell.source.quant ?? "unknown")
    }

    private func normalizedQuant(_ quant: String) -> String {
        quant.lowercased().replacingOccurrences(of: "_k_m", with: "_km")
    }

    private var runtimeFlags: String? {
        payload?["runtime_flags"] as? String
    }

    private func humanizedKey(_ key: String) -> String {
        key
            .replacingOccurrences(of: "_ms", with: " ms")
            .replacingOccurrences(of: "_bytes", with: " bytes")
            .split(separator: "_")
            .map { $0.capitalized }
            .joined(separator: " ")
    }
}

#if DEBUG
#Preview("Cell Details") {
    NavigationStack {
        CellDetailView(
            cell: JobCell(
                cellId: CellId("preview-cell"),
                benchmarkId: "decode_throughput_100_100",
                benchmarkType: .decodeThroughput,
                runStatus: .completed,
                serverJobId: nil,
                errorMessage: nil,
                source: .previewSample
            ),
            previewPayload: [
                "benchmark_id": "decode_throughput_100_100",
                "decode_time_ms": 4000.0,
                "stddev": 1.6,
                "sample_count": 10,
                "runtime_flags": "-ngl 99 -c 8192",
                "device_name": "iPhone"
            ],
            previewMetrics: [
                ["metric": "type", "value": "Value"],
                ["metric": "type", "value": "Value"],
                ["metric": "type", "value": "Value"],
                ["metric": "type", "value": "Value"],
                ["metric": "type", "value": "Value"]
            ]
        )
    }
}

#Preview("Cell Details · Failed") {
    NavigationStack {
        CellDetailView(
            cell: JobCell(
                cellId: CellId("preview-cell-failed"),
                benchmarkId: "decode_throughput_100_100",
                benchmarkType: .decodeThroughput,
                runStatus: .failed,
                serverJobId: nil,
                errorMessage: "Engine error: decode rc=-3.",
                source: .previewSample
            )
        )
    }
}
#endif
