import Foundation

/// Which half of the catalog a benchmark came from — the crate's `BenchmarkSource`
/// (`pipette-plan-types/src/benchmark/mod.rs:25`), lowercase on the wire.
///
/// The distinction is not cosmetic: a definition the server sent is submittable, so a
/// result from it waits for the next sync; one this device generated is not, so its
/// result stays on the device. The crate encodes exactly that in
/// `BenchmarkResultLocation::from` (`pipette-cli/src/results/types.rs:22`).
///
/// TODO: review — mirrors `benchmark/mod.rs:25`; two variants and the lowercase wire
/// spelling checked against it.
nonisolated enum BenchmarkSource: String, Sendable, Codable, CaseIterable {
    /// Generated on this device by ``StandardBenchmarks``; never submitted.
    case local
    /// Synced from the management server; submittable.
    case remote
}
