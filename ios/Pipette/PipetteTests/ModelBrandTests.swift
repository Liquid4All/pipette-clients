import Testing

@testable import Pipette

/// Brand detection keys off the model family in the name or repo, not the HF org, so a
/// redistributor's repackaging is labelled by whoever trained the model.
struct ModelBrandTests {
    @Test func brandDetectionKeysOffModelFamilyNotJustRepoOwner() {
        #expect(ModelBrand.detect(name: "unsloth/gemma-3n", hfRepo: "unsloth/gemma-3n-GGUF") == .google)
        #expect(ModelBrand.detect(name: "Granite Tiny", hfRepo: nil) == .ibm)
        #expect(ModelBrand.detect(name: "Qwen Test", hfRepo: nil) == .qwen)
        #expect(ModelBrand.detect(name: "Meta Llama", hfRepo: nil) == .meta)
        #expect(ModelBrand.detect(name: "Ministral", hfRepo: nil) == .mistral)
        #expect(ModelBrand.detect(name: "Phi 4", hfRepo: nil) == .microsoft)
        #expect(ModelBrand.detect(name: "DeepSeek", hfRepo: nil) == .deepseek)
        #expect(ModelBrand.detect(name: "Unknown", hfRepo: nil) == .unknown)
    }
}
