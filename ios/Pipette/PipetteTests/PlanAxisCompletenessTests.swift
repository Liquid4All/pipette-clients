import Testing

@testable import Pipette

/// The two axis enums must spell every variant plan-types does, so an unrecognized value
/// fails to decode here exactly where it fails there — and a *recognized* one this client
/// cannot run is refused by variant instead of dying as an unreadable payload.
///
/// Both had drifted: the crate gained `openvino` and `uv_openvino` and these did not, so
/// an OpenVINO cell decoded on the Rust side and failed to parse at all here.
struct PlanAxisCompletenessTests {
    @Test(arguments: ["gguf_text", "gguf_vision", "mlx", "torch", "openvino",
                      "apple_foundation_text"])
    func modelTypeSpellsEveryCrateVariant(raw: String) {
        #expect(ModelType(rawValue: raw) != nil)
    }

    @Test(arguments: ["llamacpp_cli_stock_tools", "llamacpp_apk_pipette",
                      "llamacpp_ios_pipette", "mlx_macos_pipette", "mlx_ios_pipette",
                      "docker_vllm", "docker_sglang", "uv_vllm", "uv_sglang",
                      "uv_openvino", "apple_foundation"])
    func runtimeTypeSpellsEveryCrateVariant(raw: String) {
        #expect(RuntimeType(rawValue: raw) != nil)
    }

    /// A spelling neither side knows still fails, which is what makes the above a
    /// completeness claim rather than a permissive decoder.
    @Test func anInventedSpellingStillFails() {
        #expect(ModelType(rawValue: "gguf_quantum") == nil)
        #expect(RuntimeType(rawValue: "uv_banana") == nil)
    }
}
