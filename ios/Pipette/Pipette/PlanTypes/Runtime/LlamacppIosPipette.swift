import Foundation

// The llama.cpp runtime's iOS half: pipette-mgmt's
// `crates/pipette-plan-types/src/runtime/llamacpp.rs`. The engine is `LlamaCpp.swift`.

/// Build target for the llama.cpp iOS app. Its own type so a second ABI is an added case
/// rather than a reinterpreted string.
nonisolated enum LlamacppIosPipetteFlavor: String, Sendable, Codable {
    case iosArm64 = "ios-arm64"
}

/// The crate's `LlamacppLoadMode`, upstream's `llama_load_mode`.
///
/// The raw value is the plan wire word, which differs from the stock tools' spelling for
/// two of the five (`mmap+mlock`, `dio`). The mapping to llama's discriminant is in
/// `LlamaCpp.swift`, since this layer does not import the C headers.
nonisolated enum LlamacppLoadMode: String, Codable, Sendable, CaseIterable {
    /// `LLAMA_LOAD_MODE_NONE`: read the weights into anonymous memory. This client's
    /// default. Not `.none`, which would read as `Optional.none`; the wire word is `none`.
    case read = "none"
    /// `LLAMA_LOAD_MODE_MMAP`: memory-map the file. llama.cpp's own default.
    case mmap
    /// `LLAMA_LOAD_MODE_MLOCK`: read in, then lock the pages against eviction.
    case mlock
    /// `LLAMA_LOAD_MODE_MMAP_MLOCK`: map the file and lock the mapping.
    case mmapMlock = "mmap_mlock"
    /// `LLAMA_LOAD_MODE_DIRECT_IO`: bypass the page cache where the platform supports it.
    case directIo = "direct_io"
}
