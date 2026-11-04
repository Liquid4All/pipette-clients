import Foundation

/// What advanced within a cell; `JobExecutor` decides how to show it. Counted as the Rust
/// kernel counts: throughput in measured attempts, eval in dataset samples. No crate
/// counterpart: the CLI reports progress through its log alone.
enum BenchmarkProgress: Equatable {
    case attempt(completed: Int, total: Int)
    case sample(completed: Int, total: Int)
    case corpusDownload
}
