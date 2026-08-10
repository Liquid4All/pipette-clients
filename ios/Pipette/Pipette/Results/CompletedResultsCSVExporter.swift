import Foundation

struct CompletedRunMetric {
    let name: String
    let unit: String
    let displayValue: String
    let numericValue: Double
    let higherIsBetter: Bool
}

enum CompletedResultsCSVExporter {
    static let headers = [
        "job_id",
        "job_title",
        "created_at",
        "cell_id",
        "model_name",
        "model_display_name",
        "benchmark_id",
        "benchmark_type",
        "benchmark_name",
        "benchmark_parameters",
        "status",
        "metric_name",
        "metric_value",
        "metric_unit",
        "metric_display_value",
        "submitted_at",
        "server_job_id",
        "model_descriptor",
        "runtime_descriptor",
        "runtime_flags",
        "runtime_cpu_variant",
        "device_name",
        "device_form_factor",
        "device_os_name",
        "device_os_version",
        "device_os_build",
        "device_chip_model",
        "device_ram_bytes",
        "device_battery_level",
        "device_power_state",
        "device_power_save_mode",
        "error_message"
    ]

    static func filename(for manifest: JobManifest, dateTitle: String) -> String {
        let jobPrefix = String(manifest.jobId.value.prefix(8))
        return "pipette-results-\(dateTitle)-\(jobPrefix).csv"
    }

    static func csv(for manifest: JobManifest, storage: Storage) -> String {
        csv(for: manifest, payloadsByCellId: payloadsByCellId(for: manifest, storage: storage),
            store: storage.benchmarks)
    }

    static func csv(
        for manifest: JobManifest,
        payloadsByCellId: [CellId: [String: Any]],
        store: BenchmarkStore
    ) -> String {
        let catalog = catalogById(store: store)
        let metricsByCellId = metricsByCellId(
            for: manifest,
            payloadsByCellId: payloadsByCellId,
            in: catalog
        )
        let rows = csvCells(in: manifest, catalog: catalog).map { cell -> [String] in
            let payload = payloadsByCellId[cell.cellId]
            let metric = metricsByCellId[cell.cellId]
            let benchmarkType = benchmarkType(for: cell, in: catalog)

            return [
                manifest.jobId.value,
                manifest.displayTitle,
                manifest.createdAt,
                cell.cellId.value,
                cell.modelName,
                ModelCatalog.displayName(for: cell.modelName),
                cell.benchmarkId,
                benchmarkType,
                BenchmarkCatalog.displayName(for: benchmarkType),
                parameterSummary(cell.benchmarkId, in: catalog) ?? "",
                cell.runStatus.rawValue,
                metric?.name ?? "",
                metric.map { PayloadScalars.number($0.numericValue) } ?? "",
                metric?.unit ?? "",
                metric?.displayValue ?? "",
                payloadString(payload, "submitted_at"),
                cell.serverJobId ?? "",
                payloadString(payload, "model_descriptor"),
                payloadString(payload, "runtime_descriptor"),
                payloadString(payload, "runtime_flags"),
                payloadString(payload, "runtime_cpu_variant"),
                payloadString(payload, "device_name"),
                payloadString(payload, "device_form_factor"),
                payloadString(payload, "device_os_name"),
                payloadString(payload, "device_os_version"),
                payloadString(payload, "device_os_build"),
                payloadString(payload, "device_chip_model"),
                payloadString(payload, "device_ram_bytes"),
                payloadString(payload, "device_battery_level"),
                payloadString(payload, "device_power_state"),
                payloadString(payload, "device_power_save_mode"),
                cell.errorMessage ?? ""
            ]
        }

        return ([headers] + rows)
            .map { csvLine($0) }
            .joined(separator: "\n") + "\n"
    }

    static func resultColumns(
        for manifest: JobManifest, in catalog: [String: BenchmarkItem] = [:]
    ) -> [CompletedRunResultColumn] {
        let ids = orderedUnique(manifest.cells.map(\.benchmarkId))
        return ids.map { id in
            let type = manifest.cells.first { $0.benchmarkId == id }
                .map { benchmarkType(for: $0, in: catalog) }
                ?? id
            return CompletedRunResultColumn(
                id: id,
                title: BenchmarkCatalog.shortName(for: type),
                subtitle: parameterSummary(id, in: catalog)
            )
        }
    }

    static func resultGroups(for manifest: JobManifest, storage: Storage) -> [CompletedRunModelResultGroup] {
        let payloadsByCellId = payloadsByCellId(for: manifest, storage: storage)
        return resultGroups(for: manifest, payloadsByCellId: payloadsByCellId, store: storage.benchmarks)
    }

    static func resultGroups(
        for manifest: JobManifest,
        payloadsByCellId: [CellId: [String: Any]],
        store: BenchmarkStore
    ) -> [CompletedRunModelResultGroup] {
        let catalog = catalogById(store: store)
        let columns = resultColumns(for: manifest, in: catalog)
        let metrics = metricsByCellId(for: manifest, payloadsByCellId: payloadsByCellId, in: catalog)

        var byColumnValues: [String: [Double]] = [:]
        for cell in manifest.cells {
            guard let metric = metrics[cell.cellId] else { continue }
            byColumnValues[cell.benchmarkId, default: []].append(metric.numericValue)
        }

        var modelOrder: [String] = []
        var byModel: [String: [JobCell]] = [:]
        for cell in manifest.cells {
            let modelKey = resultModelGroupKey(for: cell)
            if byModel[modelKey] == nil { modelOrder.append(modelKey) }
            byModel[modelKey, default: []].append(cell)
        }

        return modelOrder.map { modelKey in
            let modelCells = byModel[modelKey] ?? []
            var quantOrder: [String] = []
            var byQuant: [String: [JobCell]] = [:]
            for cell in modelCells {
                let quant = quantLabel(for: cell)
                if byQuant[quant] == nil { quantOrder.append(quant) }
                byQuant[quant, default: []].append(cell)
            }

            let rows = quantOrder.map { quant in
                let quantCells = byQuant[quant] ?? []
                let values = columns.map { column -> CompletedRunResultValue in
                    let cell = quantCells.first(where: { $0.benchmarkId == column.id })
                    guard let cell, let metric = metrics[cell.cellId] else {
                        return CompletedRunResultValue(
                            columnId: column.id,
                            displayValue: nil,
                            intensity: 0,
                            cell: cell
                        )
                    }
                    return CompletedRunResultValue(
                        columnId: column.id,
                        displayValue: metric.displayValue,
                        intensity: heatmapIntensity(
                            metric.numericValue,
                            values: byColumnValues[column.id] ?? [],
                            higherIsBetter: metric.higherIsBetter
                        ),
                        cell: cell
                    )
                }
                return CompletedRunQuantResultRow(quant: quant, values: values)
            }

            let displayName = resultModelDisplayName(for: modelKey, cells: modelCells)
            return CompletedRunModelResultGroup(
                id: modelKey,
                modelName: displayName,
                brand: resultModelBrand(displayName: displayName, cells: modelCells),
                rows: rows
            )
        }
    }

    static func metricsByCellId(
        for manifest: JobManifest,
        payloadsByCellId: [CellId: [String: Any]],
        in catalog: [String: BenchmarkItem] = [:]
    ) -> [CellId: CompletedRunMetric] {
        manifest.cells.reduce(into: [CellId: CompletedRunMetric]()) { result, cell in
            if let metric = metric(for: cell, payload: payloadsByCellId[cell.cellId], in: catalog) {
                result[cell.cellId] = metric
            }
        }
    }

    static func metric(
        for cell: JobCell, payload: [String: Any]?, in catalog: [String: BenchmarkItem] = [:]
    ) -> CompletedRunMetric? {
        guard let payload else { return nil }
        // Resolve the cell's typed kind; an unknown/legacy type yields no metric.
        guard let type = BenchmarkType(rawValue: benchmarkType(for: cell, in: catalog)) else { return nil }
        let params = benchmarkParams(for: cell, in: catalog)
        do {
            guard let metric = try BenchmarkMetrics.compute(payload: payload, params: params, type: type) else { return nil }
            return CompletedRunMetric(
                name: metric.name,
                unit: metric.unit,
                displayValue: displayValue(for: metric),
                numericValue: metric.numericValue,
                higherIsBetter: metric.higherIsBetter
            )
        } catch {
            AppLog.results.error("metric parse failed for cell \(cell.cellId.value) (\(cell.benchmarkId)): \(error)")
            return nil
        }
    }

    /// CSV/heatmap display string for a computed metric — compact throughput,
    /// millisecond latency, or a human byte size, keyed off the metric's unit.
    private static func displayValue(for metric: BenchmarkMetric) -> String {
        switch metric.unit {
        case "ms": return formatMilliseconds(metric.numericValue)
        case "bytes": return ByteFormat.memory(Int64(metric.numericValue))
        default: return formatNumber(metric.numericValue)
        }
    }

    static func payloadsByCellId(for manifest: JobManifest, storage: Storage) -> [CellId: [String: Any]] {
        var payloads: [CellId: [String: Any]] = [:]

        for cell in manifest.cells {
            if let payload = payloadAtKnownPath(for: cell, jobId: manifest.jobId, storage: storage) {
                payloads[cell.cellId] = payload
            }
        }

        return payloads
    }

    static func quantLabel(for cell: JobCell) -> String {
        // The typed source is the only place an MLX bit-width is recoverable; a GGUF
        // source may still yield nil quant for an odd filename, hence "unknown".
        let raw = cell.source.quant
        return (raw ?? "unknown")
            .lowercased()
            .replacingOccurrences(of: "_k_m", with: "_km")
    }

    static func resultModelGroupKey(for cell: JobCell) -> String {
        // The typed source's family id unifies a model's GGUF and MLX quants and is
        // correct for MLX (where the dir name isn't a GGUF filename).
        cell.source.familyId
    }

    static func resultModelDisplayName(for modelKey: String, cells: [JobCell]) -> String {
        if let familyName = CatalogEntry.familyIdToName[modelKey] {
            return familyName
        }
        guard let first = cells.first else { return modelKey }
        let displayName = ModelCatalog.displayName(for: first.modelName)
        if displayName == first.modelName,
           displayName.lowercased().hasSuffix(".gguf") {
            return LocalStorage.modelStem(from: first.source.artifactName)
        }
        return displayName
    }

    static func resultModelBrand(displayName: String, cells: [JobCell]) -> ModelBrand {
        // The typed source's repo names the vendor. (Brand still keys off text — a
        // redistributor's org must not override the packaged model's vendor.)
        let hfRepo = cells.first?.source.repo?.description
        return ModelBrand.detect(name: displayName, hfRepo: hfRepo)
    }

    /// A by-id snapshot of `BenchmarkCatalog.all`, taken once per export/render.
    /// `all` is recomputed on each access (re-reading the synced catalog), so the
    /// per-cell helpers below take this snapshot rather than hitting `all` in a loop.
    static func catalogById(store: BenchmarkStore) -> [String: BenchmarkItem] {
        Dictionary(BenchmarkCatalog.all(store: store).map { ($0.benchmarkId, $0) }, uniquingKeysWith: { first, _ in first })
    }

    static func parameterSummary(
        _ benchmarkId: String, in catalog: [String: BenchmarkItem] = [:]
    ) -> String? {
        // Synced catalog first, then a structured-id parse, so a historical result
        // whose benchmark left the catalog still resolves its parameters.
        guard let item = resolvedItem(benchmarkId, in: catalog), let type = item.type else { return nil }
        let params = item.rawJson
        switch type {
        case .prefillThroughput, .maxMemoryUsage:
            let prefill = PayloadScalars.uint(params, "parameter_prefill_tokens")
            return prefill > 0 ? "\(prefill)" : nil
        case .decodeThroughput, .endToEndLatency:
            let prefill = PayloadScalars.uint(params, "parameter_prefill_tokens")
            let decode = PayloadScalars.uint(params, "parameter_decode_tokens")
            return prefill > 0 && decode > 0 ? "\(prefill)-\(decode)" : nil
        case .vlThroughput:
            let width = PayloadScalars.uint(params, "parameter_image_width")
            let height = PayloadScalars.uint(params, "parameter_image_height")
            return width > 0 && height > 0 ? "\(width)x\(height)" : nil
        case .eval:
            return nil
        }
    }

    static func benchmarkType(
        for cell: JobCell, in catalog: [String: BenchmarkItem] = [:]
    ) -> String {
        cell.benchmarkType?.rawValue
            ?? resolvedItem(cell.benchmarkId, in: catalog)?.benchmarkType
            ?? cell.benchmarkId
    }

    private static func benchmarkParams(
        for cell: JobCell, in catalog: [String: BenchmarkItem] = [:]
    ) -> [String: Any] {
        resolvedItem(cell.benchmarkId, in: catalog)?.rawJson ?? [:]
    }

    /// Resolve an id to an item for export: the synced catalog or a parsed ladder
    /// definition (`BenchmarkCatalog.item(forId:in:)`), then a parsed `vl_throughput`
    /// item for historical VL results — VL isn't on the run ladder, so its id parse
    /// lives here rather than in the shared resolver.
    private static func resolvedItem(
        _ benchmarkId: String, in catalog: [String: BenchmarkItem] = [:]
    ) -> BenchmarkItem? {
        BenchmarkCatalog.item(forId: benchmarkId, in: catalog) ?? vlItem(benchmarkId)
    }

    /// A `vl_throughput` item reconstructed from its structured id, for historical
    /// VL results whose params the synced catalog no longer lists.
    private static func vlItem(_ benchmarkId: String) -> BenchmarkItem? {
        guard let raw = parseVlParams(benchmarkId) else { return nil }
        return BenchmarkItem(
            benchmarkId: benchmarkId, benchmarkType: "vl_throughput",
            sampleCount: nil, rawJson: raw, definition: nil)
    }

    /// Parse `vl_throughput_<W>x<H>_<text>_<decode>` into its `parameter_*` dict.
    /// Returns nil for any non-VL or malformed id.
    private static func parseVlParams(_ benchmarkId: String) -> [String: Any]? {
        let prefix = "vl_throughput_"
        guard benchmarkId.hasPrefix(prefix) else { return nil }
        let parts = benchmarkId.dropFirst(prefix.count).split(separator: "_", omittingEmptySubsequences: false)
        guard parts.count == 3 else { return nil }
        let dims = parts[0].split(separator: "x", omittingEmptySubsequences: false)
        guard dims.count == 2,
              let w = Int(dims[0]), let h = Int(dims[1]),
              let text = Int(parts[1]), let decode = Int(parts[2])
        else { return nil }
        return [
            "parameter_image_width": w, "parameter_image_height": h,
            "parameter_text_tokens": text, "parameter_decode_tokens": decode,
        ]
    }

    private static func payloadAtKnownPath(for cell: JobCell, jobId: JobId, storage: Storage) -> [String: Any]? {
        guard let path = storage.results.payloadPath(of: cell.cellId) else { return nil }
        do {
            return try ResultPayload.readObject(at: path)
        } catch {
            AppLog.results.error("payload parse failed for cell \(cell.cellId.value): \(error)")
            return nil
        }
    }

    private static func heatmapIntensity(_ value: Double, values: [Double], higherIsBetter: Bool) -> Double {
        guard let min = values.min(), let max = values.max(), max > min else {
            return 0.32
        }
        let normalized = (value - min) / (max - min)
        let score = higherIsBetter ? normalized : 1 - normalized
        return 0.16 + (Swift.max(0, Swift.min(1, score)) * 0.34)
    }

    private static func formatNumber(_ value: Double) -> String {
        if value >= 100 {
            return "\(Int(value.rounded()))"
        }
        return String(format: "%.1f", value)
    }

    private static func formatMilliseconds(_ value: Double) -> String {
        if value >= 100 {
            return "\(Int(value.rounded())) ms"
        }
        return String(format: "%.1f ms", value)
    }

    private static func csvCells(in manifest: JobManifest, catalog: [String: BenchmarkItem]) -> [JobCell] {
        let columns = resultColumns(for: manifest, in: catalog)

        var modelOrder: [String] = []
        var byModel: [String: [JobCell]] = [:]
        for cell in manifest.cells {
            let modelKey = resultModelGroupKey(for: cell)
            if byModel[modelKey] == nil { modelOrder.append(modelKey) }
            byModel[modelKey, default: []].append(cell)
        }

        var ordered: [JobCell] = []
        for modelKey in modelOrder {
            let modelCells = byModel[modelKey] ?? []
            var quantOrder: [String] = []
            var byQuant: [String: [JobCell]] = [:]
            for cell in modelCells {
                let quant = quantLabel(for: cell)
                if byQuant[quant] == nil { quantOrder.append(quant) }
                byQuant[quant, default: []].append(cell)
            }

            for quant in quantOrder {
                let quantCells = byQuant[quant] ?? []
                for column in columns {
                    ordered.append(contentsOf: quantCells.filter { $0.benchmarkId == column.id })
                }
            }
        }

        return ordered
    }

    private static func orderedUnique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for value in values where seen.insert(value).inserted {
            result.append(value)
        }
        return result
    }

    private static func csvLine(_ values: [String]) -> String {
        values.map { csvEscape($0) }.joined(separator: ",")
    }

    private static func csvEscape(_ value: String) -> String {
        if value.contains("\"") || value.contains(",") || value.contains("\n") || value.contains("\r") {
            return "\"\(value.replacingOccurrences(of: "\"", with: "\"\""))\""
        }
        return value
    }

    private static func payloadString(_ payload: [String: Any]?, _ key: String) -> String {
        PayloadScalars.string(payload?[key]) ?? ""
    }

}
