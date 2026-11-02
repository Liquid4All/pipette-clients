// swift-tools-version:6.0
import PackageDescription

// pipette-coreai sidecar: a small HTTP server that drives an Apple Core AI
// `.aimodel` bundle over the pipette benchmark HTTP contract (token-count
// prefill/decode/e2e/memory). Built and cached by pipette-coreai's Rust
// sidecar module; the Rust side spawns it and drives the endpoints.
//
// # Official Apple dependency
//
// Apple PR #227 fixes descriptor-driven prefill and logits allocation for S=1
// decode bundles. Pin its merged commit so the benchmark runtime is reproducible.
// The complete resolved Swift stack is also recorded in the Rust runtime identity.
// https://github.com/apple/coreai-models/pull/227
//
// # Why not Apple's `llm-server`
//
// `llm-server` is OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`).
// Pipette's timing cells need a raw-token generate with a prefill/decode split
// (`prompt_tps`/`generation_tps`) and no chat template; the high-level
// `CoreAILanguageModel` + `LanguageModelSession` path applies a template and
// rejects decode-only zoo bundles (`missingChatTemplate`). The sidecar uses the
// same low-level `LanguageBundle` + `EngineFactory` path as Apple's
// `llm-benchmark`. Revisit `llm-server` if it grows a raw-token generate that
// reports the split we need.
let package = Package(
    name: "pipette-coreai-sidecar",
    platforms: [.macOS("27.0")],
    dependencies: [
        .package(url: "https://github.com/apple/coreai-models", revision: "27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60"),
    ],
    targets: [
        .executableTarget(
            name: "pipette-coreai-sidecar",
            dependencies: [
                .product(name: "CoreAILM", package: "coreai-models"),
            ],
            path: "Sources/pipette-coreai-sidecar"
        )
    ]
)
