import Foundation

/// The `runtimes` group. A phone installs no runtimes, so `list` is the only leaf — the
/// CLI's `pull`/`remove`/`catalog`/`flavors` act on a store that does not exist here.
enum RuntimeCommands {
    /// `runtimes`: the engines compiled into this build, with the build ids they report in
    /// a submitted descriptor. The CLI's `runtimes list` enumerates *installed* runtimes;
    /// on a phone the set is fixed at compile time, so this is the honest analogue rather
    /// than a store query.
    static func list() {
        // Rows first, count derived: the `count=N` header has to be a fact about what
        // follows, not a number maintained beside it.
        // Encoded, so the line is what `benchmarks run --runtime` takes and what a
        // descriptor records — the same value, not a second rendering of it. A hand-built
        // string drifted once already: it advertised the commit where a result recorded
        // the tag.
        let rows = [RuntimeType.llamacppIosPipette, .mlxIosPipette, .appleFoundation]
            .compactMap { Runtime.thisBuild(for: $0) }
            .compactMap { try? SubmissionRef.runtime($0) }
            .map { "runtime=\($0)" }
        HeadlessRunner.log("runtimes count=\(rows.count)")
        for row in rows { HeadlessRunner.log("runtime \(row)") }
    }
}
