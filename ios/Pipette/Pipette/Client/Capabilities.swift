import Foundation

/// The capability flags this client reports to the management server, sent as
/// `capabilities` on both `POST /clients/register` and `PATCH /clients/me`.
///
/// The planner compares each flag as a whole, opaque string, so every level we
/// support is advertised: the general `runtime:mlx` *and* each versioned
/// `runtime:mlx:<pin>`. A client that reported only a versioned flag would match
/// jobs pinned to that exact build and nothing else — the versioned flag does
/// not imply the general one (mgmt `planner.md` §Client Matching Rules).
///
/// Levels are reported generously rather than minimally. Matching is set
/// containment (`effective_capabilities ⊇ requires`) and the server caps neither
/// the number of flags nor their length, so an extra flag can only widen what
/// this client matches — never narrow it. That lets a plan author pin as coarsely
/// or as precisely as they like without the client having to guess which
/// granularity they'll choose.
///
/// Flags must be **canonical** — lowercase, no whitespace — or the server rejects
/// the entire request with `400`, failing the whole profile update rather than
/// dropping the one bad flag. They must also stay clear of the server-owned
/// reserved namespaces (`os:`, `os_version:`, `device:`, `chip:`, `form_factor:`,
/// `ram_bytes:`, `gpu:`, `gpu_vram_bytes:`, `npu:`, `npu_vram_bytes:`), which the
/// server derives from the `device_*` profile itself. Reporting only `runtime:`
/// flags keeps us clear of both rules (mgmt `httpapi.md` §2.2.1).
///
/// Every runtime here is compiled into the app rather than installed at runtime,
/// so this set is a build-time property of the binary — there is no on-disk
/// runtime inventory to enumerate and no availability to probe.
///
/// No `job_schema:<n>` flag is reported, matching the Rust client: the mechanism
/// exists for rolling out an incompatible job-body change (mgmt
/// `plan-ingestion.md` §7), but no plan emits one, and a flag no job names does
/// nothing. Add it on both sides in the same change that starts emitting it.
nonisolated enum Capabilities {
    /// `LlamaCppBuildInfo.commit` when the vendored checkout carried no git
    /// metadata: `ios/build-llama.sh` substitutes this literal rather than
    /// failing the build.
    private static let unknownBuild = "unknown"

    /// Flags for this build.
    ///
    /// Sorted so the reported set is byte-stable across launches — the server
    /// voids the client's queue standing whenever the matching input *changes*
    /// (`capabilities != stored`), so a set that reordered itself would cost a
    /// reindex and a lease relinquish on every launch.
    ///
    /// The build ids default to the generated constants and are parameters only
    /// so the canonicalization and drop-the-placeholder branches are testable
    /// without regenerating `LlamaCppBuildInfo`. llama.cpp arrives as one `Build`
    /// rather than a commit/release pair, so a test cannot construct a
    /// combination the generator never emits.
    static func flags(
        llamaCpp: LlamaCppBuildInfo.Build = LlamaCppBuildInfo.build,
        mlxSwiftVersion: String = MLXBuildInfo.mlxSwiftVersion,
        mlxSwiftLMRevision: String = MLXBuildInfo.mlxSwiftLMRevision,
        swiftTransformersVersion: String = MLXBuildInfo.swiftTransformersVersion
    ) -> [String] {
        var flags: Set<String> = [
            "runtime:llama_cpp",
            "runtime:mlx",
            // Apple Foundation Models ship with the OS and expose no build id,
            // so there is no versioned level to advertise. The Rust counterpart
            // makes the same call (`runtime_capability_flags`, AppleFoundation
            // → no version).
            "runtime:apple_foundation",
        ]
        // The commit is this build's identity, but every desktop runtime is
        // pinned by git tag (`repository_version = "b10216"`). Advertising
        // both levels is what lets one plan pin iOS the same way it pins the
        // rest of the fleet; by the containment rule above, the extra flag only
        // widens what this client matches.
        [llamaCpp.commit, llamaCpp.tag]
            .compactMap { $0 }
            .compactMap(canonicalBuild)
            .forEach { flags.insert("runtime:llama_cpp:\($0)") }
        // MLX is pinned by a three-package stack, and all three affect output —
        // swift-transformers changes tokenization, and so the prompt encoding —
        // so the mlx-swift version alone does not identify a build. Each package
        // gets its own flag rather than one composite string: by the containment
        // rule above an author requiring several is equivalent, and the split
        // spelling is short enough to read and write by hand.
        if let build = canonicalBuild(mlxSwiftVersion) {
            // Bare form: what `runtime_capability_flags` derives for an
            // `mlx_ios_pipette` cell in `pipette-ops`, so a plan generated from a
            // runtime spec matches without the author writing anything.
            flags.insert("runtime:mlx:\(build)")
            // Named form: self-describing, and consistent with the sibling
            // package flags below, which need their name to mean anything.
            flags.insert("runtime:mlx:mlx-swift=\(build)")
        }
        if let build = canonicalBuild(mlxSwiftLMRevision) {
            flags.insert("runtime:mlx:mlx-swift-lm=\(build)")
        }
        if let build = canonicalBuild(swiftTransformersVersion) {
            flags.insert("runtime:mlx:swift-transformers=\(build)")
        }
        return flags.sorted()
    }

    /// `build` canonicalized for the wire, or nil when there is nothing worth
    /// pinning to. Blank and the `unknown` placeholder are both dropped rather
    /// than advertised: a versioned flag built from a non-build-id matches no
    /// real job, and would churn the server's stored set — and with it the
    /// client's queue standing — every time the value changed.
    private static func canonicalBuild(_ build: String) -> String? {
        let canonical = build.lowercased().filter { !$0.isWhitespace }
        guard !canonical.isEmpty, canonical != unknownBuild else { return nil }
        return canonical
    }
}
