// swift-tools-version:6.0
import PackageDescription

// pipette-coreai sidecar: a small HTTP server that drives an Apple Core AI
// `.aimodel` bundle over the pipette benchmark HTTP contract. Built and cached
// by `pipette-coreai`'s Rust `runtimes` module; the Rust side spawns it and
// drives the benchmark endpoints.
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
