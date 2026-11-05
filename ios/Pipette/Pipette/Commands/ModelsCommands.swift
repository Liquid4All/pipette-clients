import Foundation

/// The `models` group, as in the crate's `commands/models.rs`, plus this client's `rm`.
enum ModelsCommands {
    /// How `models list` renders the model column, the crate's `--format`.
    enum ListFormat: String {
        /// Human identity, `org/repo:path`.
        case name
        /// The importable URI that round-trips through `models pull`.
        case uri
    }

    /// `models list [format=name|uri]`: the discovered models as ModelsView sees them.
    static func list(format: ListFormat = .name, ws: PipetteWorkspace) async {
        let models = await MainActor.run { ws.availableModels() }
        HeadlessRunner.log("models count=\(models.count)")
        for m in models {
            let source = m.source.withoutAuthToken
            let digest = (try? modelDescriptorDigest(source)).map(shortDigest) ?? "-"
            let fetched = await MainActor.run { (try? ws.models.find(m.source))?.fetchedAt }
            // A model no URI can name falls back to its identity, as in the crate.
            let rendered = switch format {
            case .name: source.description
            case .uri: (try? modelToUri(source)) ?? source.description
            }
            HeadlessRunner.log("model model=\(rendered) "
                + "type=\(ModelType.of(source).rawValue) digest=\(digest) "
                + "fetched=\(fetched.map(JobDateFormat.iso8601.string(from:)) ?? "-") "
                // Not in the crate's columns: `storage status` reports only the totals.
                + "size_bytes=\(m.sizeBytes) path=\(m.path)")
        }
    }

    /// `models pull`: a model already held is a no-op that reports where it is.
    @MainActor
    static func pull(_ model: Model, ws: PipetteWorkspace) async -> Bool {
        let uri = (try? modelToUri(model)) ?? model.artifactName
        let transfer = DownloadProgressLog(label: "models pull", what: uri)
        do {
            let bound = try await ensureModel(model, ws: ws, coordinator: .shared,
                                              progress: transfer.report)
            transfer.finish()
            HeadlessRunner.log("models pull fetched model=\(uri) "
                + "path=\(bound.boundPaths?.payload ?? "-")")
            return true
        } catch {
            HeadlessRunner.log("models pull ERROR \(error.localizedDescription)")
            return false
        }
    }

    /// `models delete`: by coordinate, so it names one model exactly (`rm` selects by file
    /// or slug). A model never pulled is an error, not a silent no-op, as in the crate.
    @MainActor
    static func delete(_ model: Model, ws: PipetteWorkspace) -> Bool {
        guard let discovered = ws.availableModels().first(where: { $0.source == model }) else {
            HeadlessRunner.log("models delete ERROR model "
                + "`\((try? modelToUri(model)) ?? model.artifactName)` is not in the local store")
            return false
        }
        ws.deleteModel(discovered)
        HeadlessRunner.log("models delete deleted model=\((try? modelToUri(model)) ?? model.artifactName) "
            + "path=\(discovered.path)")
        return true
    }

    /// `models rm`: by exact file name or repo slug. `name=` wins when both are given; a
    /// repo can hold several files (GGUF quants), so a repo match deletes them all.
    static func remove(name: String?, repo: String?, ws: PipetteWorkspace) async -> Bool {
        await MainActor.run {
            let models = ws.availableModels()
            let matches = models.filter { m in
                if let name { return m.name == name }
                if let repo { return m.hfRepo == repo }
                return false
            }
            guard !matches.isEmpty else {
                let query = name.map { "name=\($0)" } ?? "repo=\(repo ?? "")"
                HeadlessRunner.log("models rm ERROR no model matches \(query); have: "
                    + models.map { "\($0.name)[\($0.engineLabel)]" }.joined(separator: ", "))
                return false
            }
            for m in matches {
                ws.deleteModel(m)
                HeadlessRunner.log("models rm deleted name=\(m.name) engine=\(m.engineLabel) path=\(m.path)")
            }
            return true
        }
    }
}
