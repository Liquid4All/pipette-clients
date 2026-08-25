// swift-tools-version:6.0
import PackageDescription

// pipette-coreai sidecar: a small HTTP server that drives an Apple Core AI
// `.aimodel` bundle over the pipette benchmark HTTP contract (token-count
// prefill/decode/e2e/memory). Built and cached by pipette-coreai's Rust
// sidecar module; the Rust side spawns it and drives the endpoints.
//
// # Why john-rocky/coreai-models (zoo fork), not apple/coreai-models
//
// The maintainer review (PR #13) asked us to depend on Apple's upstream. We
// tried and had to come back, and this comment is the record of why.
//
// - `apple/coreai-models@main` (as of 2026-08-25) does NOT compile against the
//   macOS 27 / Xcode-beta 27 SDK in use here: `CoreAILanguageModel` still uses
//   `LanguageModelCapabilities(capabilities:)`, which dropped its label in the
//   beta 5 SDK, and `KVCache+CoreAI` assigns through a get-only `shape`. The
//   fix for the label is committed in the fork but not yet upstream.
//   **As of Xcode 27.0 Beta 6 (27A5252f), upstream `main` compiles**, so the
//   SDK-drift half of this is resolved.
// - Even on beta 6, upstream `main` still crashes on the S=1 decode-only
//   bundles pipette benchmarks: its `CoreAIStaticShapeEngine` cannot chunk a
//   multi-token prefill into the single-token static graph ("Shape at
//   dimension 1 of 256 is not a valid substitution for source shape 1"). The
//   fork's static-shape engine has the chunked-prefill
//   (`gather_embeddings_<batch>` + per-batch stepping) that Apple's lacks.
//
// The fork (`0.2.2-zoo`) is Apple's code plus a small set of zoo patches
// (see `git log upstream/main..HEAD` in john-rocky/coreai-models). It is
// required for S=1 decode bundles until upstream grows chunked prefill.
// Revisit upstream on each SDK bump; the delta is expected to shrink.
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
        .package(url: "https://github.com/john-rocky/coreai-models", exact: "0.2.2-zoo"),
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
