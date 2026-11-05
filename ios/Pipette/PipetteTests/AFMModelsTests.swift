import Foundation
import Testing

@testable import Pipette

/// `AFMModels`: Apple Foundation has no path to project, so this pairing check is all it
/// has, and without it the runtime-routed dispatch would hand Apple Foundation a
/// downloaded model.
struct AFMModelsTests {
    @Test func aDownloadedModelIsRefusedByName() throws {
        let gguf = try boundRequest(try ggufTextSpec("org/a-GGUF", "a.gguf"), "/tmp/a.gguf")

        #expect(throws: UnexpectedBoundModel.self) {
            try AFMModels.requireAppleFoundation(gguf.model.bound)
        }
    }

    @Test func appleFoundationIsAccepted() throws {
        #expect(throws: Never.self) {
            try AFMModels.requireAppleFoundation(.appleFoundationText)
        }
    }
}
