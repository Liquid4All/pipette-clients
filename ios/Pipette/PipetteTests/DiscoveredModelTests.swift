import Foundation
import Testing

@testable import Pipette

/// `DiscoveredModel.appleFoundation`: the built-in, non-deletable row for Apple's on-device
/// model. It has no file, so path and size read as empty.
struct DiscoveredModelTests {
    @Test func sentinelFactoryExposesBuiltInIdentity() {
        let file = DiscoveredModel.appleFoundation
        #expect(file.source == .appleFoundationText)
        // AFM has no HF repo, so the row reports its fixed slug rather than "".
        #expect(file.hfRepo == AFMRuntime.submissionModelName)
        #expect(file.quant == nil)
        #expect(file.familyId == "apple-foundation")
        #expect(file.name == "Apple Foundation")
        #expect(file.displayName == "Apple Foundation")
        #expect(file.path == "")
        #expect(file.sizeBytes == 0)
    }

    /// AFM has no file, so deleting it is a no-op that must not trap or touch storage
    /// (the UI also hides the delete affordance for it).
    @Test @MainActor func deleteModelIsANoOpForBuiltIn() {
        makeTemporaryWorkspace().deleteModel(.appleFoundation)
    }
}
