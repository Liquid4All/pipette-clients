import Testing
import Foundation
@testable import Pipette

/// Correctness guard for `MLXBuildInfo`, the build-time-generated MLX component
/// versions (`ios/gen-mlx-build-info.sh` reads them from `Package.resolved`).
/// This re-reads the resolved pins and fails if a generated constant doesn't
/// match — catching a generator bug (wrong field/identity) or a committed
/// `MLXBuildInfo.swift` that's stale relative to `Package.resolved`. Add a row
/// to `components` whenever a component is added to the generator.
@Suite struct MLXRuntimeVersionPinTests {
    /// One MLX runtime component: its `Package.resolved` identity, the `state`
    /// key that holds its pin (`version` for semver pins, `revision` for commit
    /// pins), and the generated `MLXBuildInfo` constant that must match it.
    struct Component: Sendable, CustomTestStringConvertible {
        let identity: String
        let stateKey: String
        let expected: String
        /// mlx-swift-lm is committed as a short (9-char) prefix of the resolved
        /// full SHA, so prefix-match it; versions match exactly.
        var abbreviated = false
        var testDescription: String { identity }
    }

    static let components: [Component] = [
        .init(identity: "mlx-swift", stateKey: "version", expected: MLXBuildInfo.mlxSwiftVersion),
        .init(identity: "mlx-swift-lm", stateKey: "revision", expected: MLXBuildInfo.mlxSwiftLMRevision,
              abbreviated: true),
        .init(identity: "swift-transformers", stateKey: "version", expected: MLXBuildInfo.swiftTransformersVersion),
    ]

    /// The project's resolved SwiftPM pins file, located relative to this test's
    /// source (`#filePath`). Only reachable where the source tree is — the Mac
    /// host / simulator / CI — not on a physical device (its filesystem has no
    /// such path). See `generatedConstantMatchesResolvedPin`.
    private static var resolvedURL: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()   // PipetteTests/
            .deletingLastPathComponent()   // ios/Pipette/
            .appending(path: "Pipette.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved")
    }

    private static func resolvedPins() throws -> [[String: Any]] {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: resolvedURL)) as? [String: Any]
        return (json?["pins"] as? [[String: Any]]) ?? []
    }

    /// The `state` dict of the pin whose identity (or location basename) matches.
    /// Exact `identity` / suffix match so `mlx-swift` never picks up `mlx-swift-lm`.
    private static func state(_ pins: [[String: Any]], identity: String) -> [String: Any]? {
        let pin = pins.first {
            ($0["identity"] as? String) == identity
                || ($0["location"] as? String ?? "").lowercased().hasSuffix("/\(identity)")
        }
        return pin?["state"] as? [String: Any]
    }

    /// Each generated constant matches its resolved pin, and its value appears
    /// verbatim in the submitted composite so a result traces back to the exact
    /// component.
    @Test(arguments: components)
    func generatedConstantMatchesResolvedPin(_ c: Component) throws {
        // Host/CI-only consistency guard: it reads the repo's `Package.resolved`
        // through `#filePath`, which isn't present on a physical device. When the
        // source tree is unreachable (on-device runs), skip rather than fail — the
        // check still runs on the Mac/simulator/CI where the repo exists.
        guard FileManager.default.fileExists(atPath: Self.resolvedURL.path) else { return }

        let actual = Self.state(try Self.resolvedPins(), identity: c.identity)?[c.stateKey] as? String
        // Abbreviated components match a prefix of the resolved full SHA; the
        // rest match exactly.
        let resolvedForCompare = c.abbreviated ? actual.map { String($0.prefix(c.expected.count)) } : actual
        #expect(resolvedForCompare == c.expected,
                """
                \(c.identity) is pinned to \(actual ?? "nil") but MLXBuildInfo has \
                \(c.expected). Regenerate via ios/gen-mlx-build-info.sh.
                """)
        #expect(MLXRuntime.submissionRuntimeVersion.contains(c.expected),
                "submissionRuntimeVersion is missing \(c.identity)=\(c.expected)")
    }
}
