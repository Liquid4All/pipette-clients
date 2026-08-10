import Foundation

/// Serializes the `model_descriptor` / `runtime_descriptor` submission specs:
/// the full, lossless typed spec for the model and runtime that produced a run.
///
/// Both travel on the wire as opaque JSON **strings** (not nested objects). The
/// management server never interprets the schema — it only canonicalizes each
/// string (object keys sorted, whitespace stripped) before storing, and rejects a
/// present-but-invalid descriptor with `400`. That canonicalization is why nothing here
/// sorts: two clients emitting the same descriptor in different field orders still reduce
/// to one stored value and one `*_descriptor_sha256`. Shapes mirror the reshaped
/// `pipette-plan-types::{Model, Runtime}` families (PIP-340) exactly, so a stored
/// descriptor round-trips into the warehouse `model_descriptor` /
/// `runtime_descriptor` columns.
nonisolated enum SubmissionRef {
    /// Keys come out in each type's own field order, as the crate's
    /// `serde_json::to_string(&declared)` gives them — no sorting pass. The server
    /// canonicalizes every descriptor before storing it, and the cross-client identity is
    /// the digest over that canonical form, never these bytes.
    private static func jsonString(_ value: some Encodable) throws -> String {
        String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
    }

    /// The Apple Silicon build target iOS runs on — the single `ios-arm64` member
    /// of the plan-types `LlamacppIosPipetteFlavor` / `MlxIosPipetteFlavor` enums
    /// (both `#[serde(rename_all = "kebab-case")]`).
    static let iosFlavor = "ios-arm64"

    /// The canonical upstream llama.cpp repo, scheme-less per the plan-types
    /// `RepositoryUrl` sanitizer — matches `pipette_plan_types::default_repository_url`.
    /// Also read by `headlessrun runtimes`, which reports the same build coordinates a
    /// descriptor carries, so the two cannot disagree about what compiled.
    static let llamaCppRepositoryUrl = "github.com/ggml-org/llama.cpp"

    /// `model_descriptor`: the model coordinate, encoded by `Model` itself.
    ///
    /// Was a hand-written `ModelRef` mirror, which is how `revision` came to be missing
    /// from all three fetchable arms — the warehouse could not tell two pinned revisions
    /// apart. `Model`'s own `Codable` is the wire shape, so there is one implementation
    /// and a `sha256` or a store-relative arm cannot go missing from it either. Same move
    /// `RuntimeRef` got.
    static func model(_ model: Model) throws -> String {
        try jsonString(model.withoutAuthToken)
    }

    /// `runtime_descriptor`: the engine identity, which `Runtime` now *is* — load
    /// settings are not part of it and ship separately as the `runtime_flags` string.
    static func runtime(_ runtime: Runtime) throws -> String {
        try jsonString(runtime)
    }
}
