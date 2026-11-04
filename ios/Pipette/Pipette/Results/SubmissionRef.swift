import Foundation

/// Serializes the `model_descriptor` / `runtime_descriptor` submission specs, which travel
/// as opaque JSON strings. The server canonicalizes each (keys sorted, whitespace
/// stripped) before storing and hashing, which is why nothing here sorts. Shapes must
/// match pipette-mgmt's `pipette-plan-types::{Model, Runtime}` exactly.
nonisolated enum SubmissionRef {
    private static func jsonString(_ value: some Encodable) throws -> String {
        String(decoding: try JSONEncoder().encode(value), as: UTF8.self)
    }

    /// Encoded by `Model`'s own `Codable` so no field can go missing from a hand mirror.
    static func model(_ model: Model) throws -> String {
        try jsonString(model.withoutAuthToken)
    }

    /// The engine identity only: load settings ship separately as `runtime_flags`.
    static func runtime(_ runtime: Runtime) throws -> String {
        try jsonString(runtime)
    }
}
