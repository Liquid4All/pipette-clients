import Foundation
import Testing

@testable import Pipette

/// Pins the capability flags this client advertises.
///
/// The failure mode these guard against is silent: a misspelled or wrongly
/// versioned flag is still valid canonical input the server accepts, so the
/// client goes on matching zero jobs with no error surfacing anywhere.
struct CapabilitiesTests {
    /// Server-owned namespaces derived from the `device_*` profile. Reporting one
    /// in `capabilities` is a `400` that fails the whole request
    /// (mgmt `httpapi.md` §2.2.1).
    private static let reservedNamespaces = [
        "os:", "os_version:", "device:", "chip:", "form_factor:",
        "ram_bytes:", "gpu:", "gpu_vram_bytes:", "npu:", "npu_vram_bytes:",
    ]

    @Test func advertisesEveryRuntimeTheAppCompilesIn() {
        let flags = Capabilities.flags()
        #expect(flags.contains("runtime:llama_cpp"))
        #expect(flags.contains("runtime:mlx"))
        #expect(flags.contains("runtime:apple_foundation"))
    }

    // Matching compares whole strings, so the general flag must ride along with
    // the versioned one or the client matches only exact-build jobs.
    @Test func advertisesBothTheGeneralAndVersionedLevels() {
        let flags = Capabilities.flags(llamaCpp: .untagged(commit: "f12cc6d0f"), mlxSwiftVersion: "0.31.6")
        #expect(flags.contains("runtime:llama_cpp"))
        #expect(flags.contains("runtime:llama_cpp:f12cc6d0f"))
        #expect(flags.contains("runtime:mlx"))
        #expect(flags.contains("runtime:mlx:0.31.6"))
    }

    // Desktop runtimes are pinned by git tag, so an iOS variant can only be
    // pinned the same way if the device advertises the tag too — alongside the
    // commit, never instead of it.
    @Test func advertisesTheLlamaCppTagAlongsideTheCommit() {
        let flags = Capabilities.flags(llamaCpp: .tagged(tag: "b10216", commit: "876a43211"))
        #expect(flags.contains("runtime:llama_cpp"))
        #expect(flags.contains("runtime:llama_cpp:876a43211"))
        #expect(flags.contains("runtime:llama_cpp:b10216"))
    }

    // A pin that is not an upstream release — a cherry-picked fix — has no tag,
    // and must not invent one.
    @Test func omitsTheTagLevelForAnUntaggedPin() {
        let flags = Capabilities.flags(llamaCpp: .untagged(commit: "876a43211"))
        #expect(flags.contains("runtime:llama_cpp:876a43211"))
        #expect(!flags.contains(where: { $0.hasPrefix("runtime:llama_cpp:b") }))
    }

    // MLX's three packages each get their own flag, so a plan can pin an exact
    // build (all three) or just the component under study (one). Each is short
    // enough to read and write by hand, unlike a single run-together composite.
    @Test func advertisesEachMLXPackageSeparately() {
        let flags = Capabilities.flags(
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision: "f5f18ed9d",
            swiftTransformersVersion: "1.3.3"
        )
        #expect(flags.contains("runtime:mlx"))
        #expect(flags.contains("runtime:mlx:mlx-swift=0.31.6"))
        #expect(flags.contains("runtime:mlx:mlx-swift-lm=f5f18ed9d"))
        #expect(flags.contains("runtime:mlx:swift-transformers=1.3.3"))
    }

    // The bare mlx-swift form is kept alongside the named one: it is what
    // `runtime_capability_flags` derives for an `mlx_ios_pipette` cell in
    // `pipette-ops`, so a plan generated from a runtime spec matches as-is.
    @Test func alsoAdvertisesTheBareMLXSwiftRefForRustDerivedPlans() {
        let flags = Capabilities.flags(mlxSwiftVersion: "0.31.6")
        #expect(flags.contains("runtime:mlx:0.31.6"))
        #expect(flags.contains("runtime:mlx:mlx-swift=0.31.6"))
    }

    // No flag may run two package pins together — that ambiguity is the reason
    // these are separate flags rather than one composite.
    @Test func neverConcatenatesTwoPackagePinsIntoOneFlag() {
        for flag in Capabilities.flags() {
            #expect(
                flag.components(separatedBy: "=").count <= 2,
                "\(flag) packs more than one pin into a single flag"
            )
        }
    }

    // An MLX package whose pin is missing contributes no flag, rather than one
    // ending in a bare `=`.
    @Test func omitsAnMLXPackageFlagWhenItsPinIsBlank() {
        let flags = Capabilities.flags(
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMRevision: "",
            swiftTransformersVersion: ""
        )
        #expect(flags.contains("runtime:mlx:mlx-swift=0.31.6"))
        #expect(!flags.contains(where: { $0.hasPrefix("runtime:mlx:mlx-swift-lm") }))
        #expect(!flags.contains(where: { $0.hasSuffix("=") }))
    }

    // Apple Foundation Models ship with the OS and expose no build id, so there
    // is no versioned level to report.
    @Test func advertisesNoVersionedAppleFoundationFlag() {
        #expect(!Capabilities.flags().contains(where: { $0.hasPrefix("runtime:apple_foundation:") }))
    }

    // A non-canonical flag is a 400 that fails the entire profile update, not
    // just the offending flag.
    @Test func canonicalizesBuildIdsToLowercaseWithoutWhitespace() {
        let flags = Capabilities.flags(llamaCpp: .untagged(commit: "F12CC6D0F"), mlxSwiftVersion: "0.31.6 rc1")
        #expect(flags.contains("runtime:llama_cpp:f12cc6d0f"))
        #expect(flags.contains("runtime:mlx:0.31.6rc1"))
    }

    @Test func everyFlagIsCanonical() {
        for flag in Capabilities.flags() {
            #expect(!flag.isEmpty)
            #expect(flag == flag.lowercased(), "\(flag) must be lowercase")
            #expect(!flag.contains(where: { $0.isWhitespace }), "\(flag) must have no whitespace")
        }
    }

    @Test func avoidsServerOwnedReservedNamespaces() {
        for flag in Capabilities.flags() {
            for reserved in Self.reservedNamespaces {
                #expect(!flag.hasPrefix(reserved), "\(flag) uses reserved namespace \(reserved)")
            }
        }
    }

    // `build-llama.sh` writes the literal `unknown` when the vendored checkout
    // has no git metadata — and resolves no release either, since the recorded
    // sha cannot match an unknown HEAD. Advertising either would publish a
    // versioned flag that matches nothing, and churn the stored set once a real
    // build replaced it.
    @Test func omitsTheVersionedFlagWhenTheBuildIdIsUnknownOrBlank() {
        let unknown = Capabilities.flags(
            llamaCpp: .untagged(commit: "unknown"),
            mlxSwiftVersion: "",
            mlxSwiftLMRevision: "",
            swiftTransformersVersion: ""
        )
        #expect(unknown.contains("runtime:llama_cpp"))
        #expect(unknown.contains("runtime:mlx"))
        #expect(!unknown.contains(where: { $0.hasPrefix("runtime:llama_cpp:") }))
        #expect(!unknown.contains(where: { $0.hasPrefix("runtime:mlx:") }))
    }

    // The server voids queue standing whenever the reported set changes, so an
    // unstable ordering would cost a reindex on every launch.
    @Test func reportsAStableSortedSet() {
        let flags = Capabilities.flags()
        #expect(flags == flags.sorted())
        #expect(flags == Capabilities.flags())
        #expect(Set(flags).count == flags.count)
    }
}
