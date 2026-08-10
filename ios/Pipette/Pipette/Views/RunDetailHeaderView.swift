import SwiftUI

struct RunDetailHeaderView: View {
    let manifest: JobManifest
    let dateTitle: String
    let modelChips: [String]
    let benchmarkChips: [String]
    let quantChips: [String]

    // Back and the Rename/Delete menu live in the host's system navigation bar
    // (see JobDetailView) so the standard back button — and its swipe-back
    // gesture — work like the rest of the app.
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            summaryHeader
                .padding(.top, 8)

            properties
                .padding(.top, 50)
        }
    }

    private var summaryHeader: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(dateTitle)
                .font(.serif(24))
                .foregroundStyle(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.85)

            Text(summaryText)
                .font(.serif(18))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
    }

    private var properties: some View {
        VStack(alignment: .leading, spacing: PropertyList.rowGap) {
            PropertyChipRow(title: "Models", values: modelChips) {
                PropertyChip.model($0)
            }
            PropertyChipRow(title: "Benchmarks", values: benchmarkChips) {
                PropertyChip.text($0)
            }
            PropertyChipRow(title: "Quants", values: quantChips) {
                PropertyChip.text($0)
            }
            PropertyChipRow(title: "GPU", values: [manifest.nGpuLayers > 0 ? "On" : "Off"]) {
                PropertyChip.text($0)
            }
        }
    }

    private var summaryText: String {
        "\(modelChips.count) \("model".pluralized(modelChips.count)) • \(manifest.benchmarkIds.count) \("benchmark".pluralized(manifest.benchmarkIds.count))"
    }
}

#if DEBUG
#Preview("Run Detail Header") {
    RunDetailHeaderView(
        manifest: JobManifest(
            jobId: JobId("preview-header-job"),
            createdAt: JobDateFormat.iso8601.string(from: Date()),
            nGpuLayers: 99,
            contextSize: 8192,
            cells: [
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
                    runStatus: .pending,
                    serverJobId: nil,
                    errorMessage: nil,
                    source: .previewSample
                )
            ],
            status: .running
        ),
        dateTitle: "May 30, 2026",
        modelChips: ["LFM2.5-350M", "gemma4.0-1.2b"],
        benchmarkChips: ["decode_throughput", "prefill_throughput"],
        quantChips: ["Q4_0", "Q4_K_M"]
    )
    .padding()
}
#endif
