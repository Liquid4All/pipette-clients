import Foundation
import Testing

@testable import Pipette

/// A misspelled or wrongly versioned flag is still canonical input the server accepts, so
/// the client would silently match zero jobs.
struct CapabilitiesTests {
    /// Server-owned namespaces derived from the `device_*` profile. Reporting one in
    /// `capabilities` is a `400` that fails the whole request (pipette-mgmt's
    /// `docs/httpapi.md` §2.2.1).
    private static let reservedNamespaces = [
        "os:", "os_version:", "device:", "chip:", "form_factor:",
        "ram_bytes:", "gpu:", "gpu_vram_bytes:", "npu:", "npu_vram_bytes:",
    ]

    @Test func everyRuntimeReportsItsGeneralLevel() {
        let flags = installedRuntimeCapabilities()
        #expect(flags.contains("runtime:llama_cpp"))
        #expect(flags.contains("runtime:mlx"))
        #expect(flags.contains("runtime:apple_foundation"))
    }

    // Matching compares whole strings, so without the general flag the client matches only
    // exact-build jobs.
    @Test func advertisesBothTheGeneralAndVersionedLevels() {
        let flags = installedRuntimeCapabilities(llamaCpp: .untagged(commit: "f12cc6d0f"), mlxSwiftVersion: "0.31.6")
        #expect(flags.contains("runtime:llama_cpp"))
        #expect(flags.contains("runtime:llama_cpp:f12cc6d0f"))
        #expect(flags.contains("runtime:mlx"))
        #expect(flags.contains("runtime:mlx:0.31.6"))
    }

    // Desktop runtimes are pinned by git tag, so an iOS variant can only be pinned the
    // same way if the device advertises the tag too.
    @Test func advertisesTheLlamaCppTagAlongsideTheCommit() {
        let flags = installedRuntimeCapabilities(llamaCpp: .tagged(tag: "b10216", commit: "876a43211"))
        #expect(flags.contains("runtime:llama_cpp"))
        #expect(flags.contains("runtime:llama_cpp:876a43211"))
        #expect(flags.contains("runtime:llama_cpp:b10216"))
    }

    // A pin that is not an upstream release (a cherry-picked fix) has no tag.
    @Test func omitsTheTagLevelForAnUntaggedPin() {
        let flags = installedRuntimeCapabilities(llamaCpp: .untagged(commit: "876a43211"))
        #expect(flags.contains("runtime:llama_cpp:876a43211"))
        #expect(!flags.contains(where: { $0.hasPrefix("runtime:llama_cpp:b") }))
    }

    // A plan can pin an exact build (all three) or just the component under study (one),
    // and each flag is short enough to write by hand, unlike a composite.
    @Test func advertisesEachMLXPackageSeparately() {
        let flags = installedRuntimeCapabilities(
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMVersion: "f5f18ed9d",
            swiftTransformersVersion: "1.3.3"
        )
        #expect(flags.contains("runtime:mlx"))
        #expect(flags.contains("runtime:mlx:mlx-swift=0.31.6"))
        #expect(flags.contains("runtime:mlx:mlx-swift-lm=f5f18ed9d"))
        #expect(flags.contains("runtime:mlx:swift-transformers=1.3.3"))
    }

    // The bare form is what `runtime_capability_flags` (pipette-cli's
    // `client/worker/profile.rs`) derives for `mlx_ios_pipette`, so a plan generated from
    // a runtime spec matches as-is.
    @Test func alsoAdvertisesTheBareMLXSwiftRefForRustDerivedPlans() {
        let flags = installedRuntimeCapabilities(mlxSwiftVersion: "0.31.6")
        #expect(flags.contains("runtime:mlx:0.31.6"))
        #expect(flags.contains("runtime:mlx:mlx-swift=0.31.6"))
    }

    @Test func neverConcatenatesTwoPackagePinsIntoOneFlag() {
        for flag in installedRuntimeCapabilities() {
            #expect(
                flag.components(separatedBy: "=").count <= 2,
                "\(flag) packs more than one pin into a single flag"
            )
        }
    }

    @Test func omitsAnMLXPackageFlagWhenItsPinIsBlank() {
        let flags = installedRuntimeCapabilities(
            mlxSwiftVersion: "0.31.6",
            mlxSwiftLMVersion: "",
            swiftTransformersVersion: ""
        )
        #expect(flags.contains("runtime:mlx:mlx-swift=0.31.6"))
        #expect(!flags.contains(where: { $0.hasPrefix("runtime:mlx:mlx-swift-lm") }))
        #expect(!flags.contains(where: { $0.hasSuffix("=") }))
    }

    // Apple Foundation Models ship with the OS and expose no build id.
    @Test func advertisesNoVersionedAppleFoundationFlag() {
        #expect(!installedRuntimeCapabilities().contains(where: { $0.hasPrefix("runtime:apple_foundation:") }))
    }

    // A non-canonical flag is a 400 that fails the entire profile update.
    @Test func canonicalizesBuildIdsToLowercaseWithoutWhitespace() {
        let flags = installedRuntimeCapabilities(llamaCpp: .untagged(commit: "F12CC6D0F"), mlxSwiftVersion: "0.31.6 rc1")
        #expect(flags.contains("runtime:llama_cpp:f12cc6d0f"))
        #expect(flags.contains("runtime:mlx:0.31.6rc1"))
    }

    /// What a profile update reports apart from the hardware flags.
    private static func flags(buildVersion: String) -> [String] {
        installedRuntimeCapabilities() + [clientVersionCapabilityFlag(buildVersion)].compactMap { $0 }
    }

    // A stamped version is passed in so the loop covers `client_version:` on a local build,
    // whose bundle names no release. `CapabilityFlag` is the rule the server applies.
    @Test func everyRuntimeCapabilityFlagIsCanonical() {
        for flag in Self.flags(buildVersion: "2026.08.11-640d96e8") {
            #expect((try? CapabilityFlag(flag)) != nil, "\(flag) is not canonical")
        }
    }

    @Test func avoidsServerOwnedReservedNamespaces() {
        for flag in Self.flags(buildVersion: "2026.08.11-640d96e8") {
            for reserved in Self.reservedNamespaces {
                #expect(!flag.hasPrefix(reserved), "\(flag) uses reserved namespace \(reserved)")
            }
        }
    }

    // `build-llama.sh` writes the literal `unknown` when the vendored checkout has no git
    // metadata. Advertising it would publish a flag that matches nothing, and churn the
    // stored set once a real build replaced it.
    @Test func omitsTheVersionedFlagWhenTheBuildIdIsUnknownOrBlank() {
        let unknown = installedRuntimeCapabilities(
            llamaCpp: .untagged(commit: "unknown"),
            mlxSwiftVersion: "",
            mlxSwiftLMVersion: "",
            swiftTransformersVersion: ""
        )
        #expect(unknown.contains("runtime:llama_cpp"))
        #expect(unknown.contains("runtime:mlx"))
        #expect(!unknown.contains(where: { $0.hasPrefix("runtime:llama_cpp:") }))
        #expect(!unknown.contains(where: { $0.hasPrefix("runtime:mlx:") }))
    }

    // A plan can pin the client that produces the measurement without waiting for a
    // submission.
    @Test func advertisesTheBuildVersionOfAReleasedBuild() {
        #expect(clientVersionCapabilityFlag("2026.08.11-640d96e8")
            == "client_version:2026.08.11-640d96e8\(BuildFlavor.versionSuffix)")
    }

    // A local build and the TestFlight path both name no release, and the flag has no
    // general level to fall back to.
    @Test func omitsTheBuildVersionFlagWhenTheBuildNamesNoRelease() {
        #expect(clientVersionCapabilityFlag(nil) == nil)
    }

    // The build version arrives from a build setting, so nothing guarantees its form.
    @Test func canonicalizesTheBuildVersion() {
        #expect(clientVersionCapabilityFlag("2026.08.11 640D96E8")
            == "client_version:2026.08.11640d96e8\(BuildFlavor.versionSuffix)")
    }

    // Canonicalization runs before the flavor suffix joins, or an internal build would
    // advertise `client_version:-internal`, which names no release.
    @Test func omitsTheBuildVersionFlagWhenTheVersionIsBlank() {
        #expect(clientVersionCapabilityFlag("   ") == nil)
    }

    // The server voids queue standing whenever the reported set changes, so an
    // unstable ordering would cost a reindex on every launch.
    @Test func reportsAStableSortedSet() {
        let flags = installedRuntimeCapabilities()
        #expect(flags == flags.sorted())
        #expect(flags == installedRuntimeCapabilities())
        #expect(Set(flags).count == flags.count)
    }

    // MARK: - Build flags (the crate's `runtime_build_flag`)

    // The catalog offers a build only to a device that reports its build flag, so the worker
    // reports one for llama.cpp main, MLX and ONNX GenAI.
    @Test func reportsOneBuildFlagPerWorkerRuntime() throws {
        let flags = try installedRuntimeBuildFlags()
        #expect(flags.count == 3)
        #expect(Set(flags).count == flags.count)
        #expect(flags == flags.sorted())
        for flag in flags {
            #expect(flag.hasPrefix(runtimeBuildFlagPrefix))
            #expect(flag.dropFirst(runtimeBuildFlagPrefix.count).count == 64)
            #expect((try? CapabilityFlag(flag)) != nil, "\(flag) is not canonical")
        }
    }

    // Apple Foundation ships with the OS, so it has no build to name.
    @Test func appleFoundationHasNoBuildFlag() throws {
        #expect(try installedRuntimeBuildFlags(runtimes: [.appleFoundation(privateApis: .stock)]).isEmpty)
    }

    // The private APIs are part of the build's identity, as in the crate.
    @Test func thePrivateApisChangeTheBuildFlag() throws {
        let source = Runtime.carried(LlamaPins.main)
        guard case let .llamacppIosPipette(repository, flavor, _) = source else {
            Issue.record("a carried llama.cpp pin is an llamacpp_ios_pipette runtime")
            return
        }
        let flags = try installedRuntimeBuildFlags(runtimes: [
            .llamacppIosPipette(source: repository, flavor: flavor, privateApis: .stock),
            .llamacppIosPipette(source: repository, flavor: flavor,
                                privateApis: PrivateApis(thermal: true, memory: true)),
        ])
        #expect(Set(flags).count == 2)
    }

    // MARK: - Hardware flags (`pipette-device/src/capabilities.rs`)

    /// The crate's `hardware_flags_are_canonical_and_sorted`.
    @Test(arguments: [
        ("Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz", false, false, ["cpu_vendor:intel"]),
        ("AMD EPYC 7763 64-Core Processor", false, false, ["cpu_vendor:amd"]),
        ("Apple M4 Pro", false, false, ["cpu_vendor:apple"]),
        ("Snapdragon(R) X Elite - X1E80100", false, false, ["cpu_vendor:qualcomm"]),
        ("Qualcomm Technologies, Inc SM8550", false, false, ["cpu_vendor:qualcomm"]),
        ("QTI", false, false, ["cpu_vendor:qualcomm"]),
        ("Google", false, false, ["cpu_vendor:google"]),
        ("Tensor G3", false, false, ["cpu_vendor:google"]),
        ("Some Unbranded CPU", false, false, []),
        ("Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz", false, true,
         ["cpu_feature:avx512f", "cpu_vendor:intel"]),
        ("Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz", true, false,
         ["accel:gpu", "cpu_vendor:intel"]),
        ("Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz", true, true,
         ["accel:gpu", "cpu_feature:avx512f", "cpu_vendor:intel"]),
    ] as [(String, Bool, Bool, [String])])
    func hardwareFlagsAreCanonicalAndSorted(chipModel: String, hasGpu: Bool, hasAvx512f: Bool,
                                            expected: [String]) {
        #expect(hardwareFlags(chipModel: chipModel, hasGpu: hasGpu, hasAvx512f: hasAvx512f)
            == expected)
    }

    /// Checked against the reserved namespaces the profile's flags are held to as well.
    @Test func everyHardwareFlagIsACanonicalCapability() {
        let chips = [
            "Intel Xeon", "AMD Ryzen", "Apple M4", "Qualcomm Snapdragon", "Samsung Exynos",
            "MediaTek Dimensity", "NVIDIA Tegra", "Google Tensor G", "Ampere Altra",
            "Hisilicon Kirin", "Rockchip RK3588",
        ]
        for chip in chips {
            for (hasGpu, hasAvx512f) in [(false, false), (true, false), (false, true), (true, true)] {
                for flag in hardwareFlags(chipModel: chip, hasGpu: hasGpu, hasAvx512f: hasAvx512f) {
                    #expect(flag == flag.lowercased(), "\(flag) must be lowercase")
                    #expect(!flag.contains(where: { $0.isWhitespace }), "\(flag) has whitespace")
                    for reserved in Self.reservedNamespaces {
                        #expect(!flag.hasPrefix(reserved), "\(flag) uses reserved \(reserved)")
                    }
                }
            }
        }
    }

    @Test func appleSiliconReportsAppleCpuWithoutGpuOrAvx512() {
        #expect(hardwareCapabilities(device: deviceInfoFixture()) == ["cpu_vendor:apple"])
    }

    /// On this platform the vendor is a fact about the device, not the string.
    /// `detectChipModel` falls back to the raw `hw.machine` identifier for a phone newer
    /// than `chipModelMapping`, and the simulator reports the host architecture. Dropping
    /// the flag would stop the device matching apple-gated plans.
    @Test(arguments: ["iPhone19,1", "arm64", "x86_64"])
    func anUnrecognizedChipStillReportsAnAppleCpu(chipModel: String) {
        #expect(cpuVendor(chipModel) == nil, "fixture must name no vendor, or it proves nothing")
        #expect(hardwareCapabilities(device: deviceInfoFixture(chipModel: chipModel))
            == ["cpu_vendor:apple"])
    }

    /// The crate's table stays the rule wherever it applies.
    @Test func aRecognizedChipIsStillReadThroughTheCrateTable() {
        #expect(hardwareCapabilities(device: deviceInfoFixture(chipModel: "Apple M4 Pro"))
            == ["cpu_vendor:apple"])
        #expect(cpuVendor("Apple M4 Pro") == "apple")
    }
}
