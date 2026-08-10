import SwiftUI
import UniformTypeIdentifiers

/// A run's results CSV, packaged for the system share sheet.
///
/// Generation is deferred to the moment a share destination is picked: the
/// results page rebuilds this value on every render, while `makeCSV` re-reads
/// every result payload from disk.
nonisolated struct ResultsCSVFile: Transferable {
    let filename: String
    let makeCSV: @MainActor @Sendable () -> String

    /// Shared as a file rather than as bytes: a data representation only carries
    /// a *suggested* name, leaving the destination to derive the extension from
    /// the content type — and `.commaSeparatedText` conforms to
    /// `public.plain-text`, so the CSV can land as a `.txt`. A file on disk
    /// carries its own name, so every destination writes `…​.csv`.
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .commaSeparatedText) { file in
            SentTransferredFile(try await file.writeToTemporaryFile())
        }
        .suggestedFileName { $0.filename }
    }

    /// The share sheet copies the file it is handed, so this only has to outlive
    /// the transfer. One reused directory, emptied first, leaves at most a single
    /// stale CSV behind between exports.
    @MainActor
    private func writeToTemporaryFile() throws -> URL {
        let directory = URL.temporaryDirectory.appending(
            path: "results-csv-export",
            directoryHint: .isDirectory
        )
        try? FileManager.default.removeItem(at: directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let url = directory.appending(path: filename, directoryHint: .notDirectory)
        try Data(makeCSV().utf8).write(to: url)
        return url
    }
}

#if !os(iOS)
struct CSVExportDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.commaSeparatedText] }
    static var writableContentTypes: [UTType] { [.commaSeparatedText] }

    var text: String

    init(text: String = "") {
        self.text = text
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents,
              let text = String(data: data, encoding: .utf8)
        else {
            self.text = ""
            return
        }
        self.text = text
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}
#endif

/// The results-export affordance on the completed-run page. iOS hands the CSV to
/// the native share sheet — AirDrop, Mail, Messages, Save to Files — which is a
/// superset of what a save panel offers. macOS keeps the file exporter, where
/// picking a destination folder is the expected way to export.
struct ResultsCSVExportButton: View {
    let file: ResultsCSVFile

    var body: some View {
        control
            .buttonStyle(.plain)
            .accessibilityLabel("Export results CSV")
    }

    private var icon: some View {
        Image(systemName: "square.and.arrow.up")
            .font(.system(size: 22, weight: .regular))
            .foregroundStyle(.secondary)
            .frame(width: 44, height: 44)
            .contentShape(Rectangle())
    }

#if os(iOS)
    private var control: some View {
        ShareLink(item: file, preview: SharePreview(file.filename)) { icon }
    }
#else
    @State private var isExporterPresented = false
    @State private var document = CSVExportDocument()
    @State private var exportError: String?

    private var control: some View {
        Button {
            // The save panel needs the document up front, so this path pays for
            // the payload reads on tap rather than on export.
            document = CSVExportDocument(text: file.makeCSV())
            isExporterPresented = true
        } label: {
            icon
        }
        .fileExporter(
            isPresented: $isExporterPresented,
            document: document,
            contentType: .commaSeparatedText,
            defaultFilename: file.filename
        ) { result in
            if case .failure(let error) = result {
                exportError = error.localizedDescription
            }
        }
        .alert(
            "Export Error",
            isPresented: Binding<Bool>(
                get: { exportError != nil },
                set: { if !$0 { exportError = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(exportError ?? "")
        }
    }
#endif
}
