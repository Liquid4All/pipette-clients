import Foundation

/// Models-tab CLI verbs (`models` / `models rm`): thin handlers over the same
/// `storage.availableModels()` / `storage.deleteModel` calls the UI binds.
/// Dispatched from `HeadlessRunner.startIfRequested`.
enum ModelCommands {
    /// `models`: list the discovered models exactly as ModelsView sees them
    /// (`storage.availableModels()`), one line per model.
    /// How `models` renders the model column — the crate's `--format`.
    enum ListFormat: String {
        /// Human identity, `org/repo:path`.
        case name
        /// The importable URI that round-trips through `models pull`.
        case uri
    }

    static func list(format: ListFormat = .name, storage: Storage) async {
        let models = await MainActor.run { storage.availableModels() }
        HeadlessRunner.log("models count=\(models.count)")
        for m in models {
            // The crate's `models list` columns: MODEL, TYPE, DIGEST, FETCHED. `model=`
            // is the URI form, which is what makes a listing line feed `--model`; a model
            // no URI can name falls back to its artifact name, as the crate falls back to
            // the declared identity.
            let source = m.source.withoutAuthToken
            let digest = (try? Descriptor.digest(source)).map(Descriptor.shortDigest) ?? "-"
            let fetched = await MainActor.run { storage.modelStore.find(m.source)?.fetchedAt }
            // `name` is the default the crate defaults to; `uri` is what round-trips
            // through `models pull`, and what a caller pastes into `--model`.
            let rendered = switch format {
            case .name: source.reference
            case .uri: ModelUri.uri(for: source) ?? source.reference
            }
            HeadlessRunner.log("model model=\(rendered) "
                + "type=\(ModelType.of(source).rawValue) digest=\(digest) "
                + "fetched=\(fetched.map(JobDateFormat.iso8601.string(from:)) ?? "-") "
                // Not in the crate's columns and not derivable from the coordinate: where
                // the bytes are and how many. `storage status` reports only the totals.
                + "sizeBytes=\(m.sizeBytes) path=\(m.path)")
        }
    }

    /// `models pull`: fetch the named model into the store — the crate's `models pull
    /// --model`, which takes a self-contained reference rather than a repo and a filename.
    ///
    /// `ensureModel` decides whether anything is fetched, so naming a model already held
    /// is a no-op that reports where it is.
    @MainActor
    static func pull(_ model: Model, storage: Storage) async -> Bool {
        let uri = ModelUri.uri(for: model) ?? model.artifactName
        let transfer = DownloadProgressLog(label: "models pull", what: uri)
        do {
            let bound = try await ensureModel(model, storage: storage, coordinator: .shared,
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

    /// `models delete`: drop the named model — the crate's `models delete --model`.
    ///
    /// Addressed by the coordinate, so it names one model exactly; `rm name=`/`repo=`
    /// remains for selecting by file or by slug.
    @MainActor
    static func delete(_ model: Model, storage: Storage) -> Bool {
        guard let discovered = storage.availableModels().first(where: { $0.source == model }) else {
            HeadlessRunner.log("models delete ERROR no such model "
                + "\(ModelUri.uri(for: model) ?? model.artifactName)")
            return false
        }
        storage.deleteModel(discovered)
        HeadlessRunner.log("models delete deleted model=\(ModelUri.uri(for: model) ?? model.artifactName) "
            + "path=\(discovered.path)")
        return true
    }

    /// `models rm`: delete by exact file name or by repo slug through the same
    /// `storage.deleteModel` call `ModelStore.delete` makes for the UI.
    /// `name=` wins when both are given; a repo can hold several downloaded
    /// files (GGUF quants), so a repo match deletes them all.
    static func remove(name: String?, repo: String?, storage: Storage) async -> Bool {
        await MainActor.run {
            let models = storage.availableModels()
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
                storage.deleteModel(m)
                HeadlessRunner.log("models rm deleted name=\(m.name) engine=\(m.engineLabel) path=\(m.path)")
            }
            return true
        }
    }
}
