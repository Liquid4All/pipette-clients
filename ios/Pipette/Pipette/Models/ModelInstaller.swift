import Foundation

/// The completion seam every finished download funnels through: validate the
/// downloaded artifact, relocate it into the models bucket, and record its
/// provenance manifest. One protocol so the GGUF (single-file) and MLX
/// (directory) paths install through the same contract instead of each open-coding
/// its own move + manifest step.
@MainActor
protocol ModelInstaller {
    /// Relocate the artifact at `downloaded` into its models-bucket destination and
    /// write its provenance manifest. Throws on a filesystem failure, and — for
    /// formats that validate completeness — when the artifact isn't a usable model.
    func install(from downloaded: URL) throws
}
