import Foundation

// Apple Foundation ships with the OS, so there is no path to project; this only asserts
// the plan-types pairing ("loads only `Model::AppleFoundationText`").

nonisolated enum AFMModels {
    static func requireAppleFoundation(_ model: Model) throws {
        guard case .appleFoundationText = model else {
            throw UnexpectedBoundModel(expected: "AppleFoundationText", got: model.description)
        }
    }
}
