import Foundation

// Bound-model check for a prepared `RunRequest`, named for its engine as the llama and MLX
// files beside it are (`LlamaModels`, `MLXModels`).
//
// There is no path to project: Apple Foundation ships with the OS, so this only asserts the
// pairing plan-types states for the runtime — "loads only `Model::AppleFoundationText`".
// Kept beside the engine that would have loaded the model, so a mismatched pair fails by
// name rather than against a central compatibility table.

nonisolated enum AFMModels {
    /// The bound model, refused unless it is Apple Foundation's.
    static func requireAppleFoundation(_ req: RunRequest) throws {
        guard case .appleFoundationText = req.model.bound else {
            throw UnexpectedBoundModel(expected: "AppleFoundationText",
                                       got: req.model.declared.artifactName)
        }
    }
}
