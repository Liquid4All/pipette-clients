import Foundation

/// The `runtimes` group, the crate's `commands/runtimes.rs`. A phone installs no runtimes,
/// so `list` is the only leaf: the CLI's `pull`/`remove`/`catalog`/`flavors` act on a store
/// that does not exist here.
enum RuntimesCommands {
    /// How `runtimes list` renders the runtime column, the crate's `--format`.
    nonisolated enum ListFormat: String, Sendable {
        /// The runtime's `cli_ref`: a llama.cpp build as `version:flavor`.
        case name
        /// plan-types gives the on-device runtimes no URI form, so this renders the
        /// `cli_ref`, as the crate does for any runtime whose URI cannot be written.
        case uri
    }

    /// `runtimes list [format=name|uri]`: each llama.cpp build the app carries and the other
    /// engines compiled in, the phone's analogue of the crate's installed runtimes. False
    /// when a llama.cpp build does not load.
    static func list(format: ListFormat = .name) -> Bool {
        let rows: [String]
        do {
            rows = try runtimeRows(format: format)
        } catch {
            HeadlessRunner.log("runtimes list ERROR \(error.localizedDescription)")
            return false
        }
        HeadlessRunner.log("runtimes count=\(rows.count)")
        for row in rows { HeadlessRunner.log(row) }
        return true
    }

    /// No `PULLED` column: these engines were never pulled. `descriptor=` is this client's:
    /// the `Runtime` JSON `benchmarks run runtime=` accepts, since an on-device runtime has
    /// no URI to pass back. `runtime=` goes last because an MLX `cli_ref` carries spaces.
    ///
    /// Every llama.cpp build is loaded first, so a listing proves each one loads here.
    static func runtimeRows(format: ListFormat) throws -> [String] {
        // Sorted keys, so two captures diff cleanly.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let others = [RuntimeType.mlxIosPipette, .onnxGenaiIosPipette, .appleFoundation]
            .compactMap { Runtime.thisBuild(for: $0) }
        return try (LlamaPins.probeAll().map(Runtime.carried) + others)
            .map { runtime -> (column: String, runtime: Runtime) in
                switch format {
                case .name, .uri: (runtime.cliRef, runtime)
                }
            }
            .sorted { $0.column < $1.column }
            .map { column, runtime in
                let digest = (try? descriptorDigest(runtime)).map(shortDigest) ?? "-"
                let descriptor = (try? encoder.encode(runtime))
                    .map { String(decoding: $0, as: UTF8.self) } ?? "-"
                return "runtime type=\(RuntimeType.of(runtime).rawValue) digest=\(digest) "
                    + "descriptor=\(descriptor) runtime=\(column)"
            }
    }
}
