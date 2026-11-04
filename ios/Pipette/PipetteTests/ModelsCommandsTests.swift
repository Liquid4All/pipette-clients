import Foundation
import Testing

@testable import Pipette

/// Ports the crate's `commands/models.rs` tests.
@MainActor struct ModelsCommandsTests {

    /// Deleting a model that was never pulled is an error, not a silent no-op.
    @Test func deleteAbsentModelReportsNotFound() throws {
        let ws = makeTemporaryWorkspace()
        defer { removeWorkspace(ws) }
        let model = try parseModelUri("gguf-text://repo=org/repo-GGUF&path=model-Q4_K_M.gguf")

        #expect(ModelsCommands.delete(model, ws: ws) == false)
    }
}
