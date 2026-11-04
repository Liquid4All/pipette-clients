import Testing
import Foundation
@testable import Pipette

/// Fails when a pin `ios/gen-mlx-build-info.sh` generated into `MLXBuildInfo` disagrees
/// with `Package.resolved`: a generator bug or a stale committed file. Add a row to
/// `components` whenever the generator gains a component.
@Suite struct MLXBuildInfoTests {
    struct Component: Sendable, CustomTestStringConvertible {
        let identity: String
        let build: MLXBuildInfo.Build
        var testDescription: String { identity }
    }

    static let components: [Component] = [
        .init(identity: "mlx-swift", build: MLXBuildInfo.mlxSwift),
        .init(identity: "mlx-swift-lm", build: MLXBuildInfo.mlxSwiftLM),
        .init(identity: "swift-transformers", build: MLXBuildInfo.swiftTransformers),
    ]

    /// Found through `#filePath`, so reachable only where the source tree is (Mac host,
    /// simulator, CI), never on a physical device.
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

    /// Exact identity or `/<identity>` suffix match, so `mlx-swift` never picks up
    /// `mlx-swift-lm`.
    private static func state(_ pins: [[String: Any]], identity: String) -> [String: Any]? {
        let pin = pins.first {
            ($0["identity"] as? String) == identity
                || ($0["location"] as? String ?? "").lowercased().hasSuffix("/\(identity)")
        }
        return pin?["state"] as? [String: Any]
    }

    /// Each generated pin matches its resolved one: these are the values
    /// `Runtime.thisBuild` puts in the MLX runtime descriptor, so a result traces back to
    /// the exact component. The commit is the 9-char prefix of the resolved SHA, and the
    /// tag is the resolved `version`, which SwiftPM records only for a release pin.
    @Test(arguments: components)
    func generatedConstantMatchesResolvedPin(_ c: Component) throws {
        // Skip rather than fail on a device, where the source tree is absent.
        guard FileManager.default.fileExists(atPath: Self.resolvedURL.path) else { return }

        let state = Self.state(try Self.resolvedPins(), identity: c.identity)
        let revision = state?["revision"] as? String
        let version = state?["version"] as? String
        let regenerate = "Regenerate via ios/gen-mlx-build-info.sh."
        #expect(c.build.commit.count == 9 && revision?.hasPrefix(c.build.commit) == true,
                "\(c.identity) is pinned to \(revision ?? "nil") but MLXBuildInfo has \(c.build.commit). \(regenerate)")
        #expect(c.build.tag == version,
                "\(c.identity) resolves to version \(version ?? "nil") but MLXBuildInfo has \(c.build.tag ?? "nil"). \(regenerate)")
    }
}
