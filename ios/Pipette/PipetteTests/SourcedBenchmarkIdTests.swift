import Foundation
import Testing

@testable import Pipette

/// The address that names a catalog half: the crate's `benchmarks/reference.rs`.
struct SourcedBenchmarkIdTests {
    /// A bare id is the distributed form and means the synced catalog; a prefix is an
    /// explicit override.
    @Test(arguments: [
        ("foo", BenchmarkSource.remote),
        ("local/foo", .local),
        ("remote/foo", .remote),
    ])
    func parseAcceptsBareIdsAndExplicitSides(reference: String, source: BenchmarkSource) throws {
        #expect(try SourcedBenchmarkId(parsing: reference)
            == SourcedBenchmarkId(source: source, id: "foo"))
    }

    /// An id the catalog grammar refuses names its character, not the accepted forms.
    @Test(arguments: [
        ("local/remote/foo", nil),
        ("elsewhere/foo", nil),
        ("local/", BenchmarkIdError.empty),
        ("remote/", .empty),
        ("", .empty),
        ("foo bar", .invalidChar(" ")),
        ("local/foo:bar", .invalidChar(":")),
        ("remote/.hidden", .invalidFirst(".")),
        // iOS's own: a flag spelled where an id goes.
        ("-flag", .invalidFirst("-")),
    ] as [(String, BenchmarkIdError?)])
    func aRefusalSaysWhetherTheShapeOrTheIdIsWrong(reference: String,
                                                    reason: BenchmarkIdError?) {
        let expected = reason.map { SourcedBenchmarkIdError.id(reference: reference, reason: $0) }
            ?? .shape(reference)
        #expect(throws: expected) { try SourcedBenchmarkId(parsing: reference) }
    }

    /// The refusal the parser prints: the reference escaped, so an invisible or look-alike
    /// character shows, then the reason. Swift spells the escape's hex in capitals, padded
    /// to four digits, where Rust's `escape_default` writes it lowercase and unpadded.
    @Test(arguments: [
        ("elsewhere/a\nb\u{200B}\u{430}",
         #""elsewhere/a\nb\u{200B}\u{0430}" is not a benchmark reference: expected `<id>`, `local/<id>` or `remote/<id>`"#),
        ("local/a\nb\u{200B}\u{430}",
         #""local/a\nb\u{200B}\u{0430}" is not a benchmark reference: benchmark id contains '\n'; allowed: [A-Za-z0-9][A-Za-z0-9_.-]*"#),
    ])
    func aRefusalEscapesTheReference(reference: String, expected: String) {
        let error = #expect(throws: SourcedBenchmarkIdError.self) {
            try SourcedBenchmarkId(parsing: reference)
        }
        #expect(error?.localizedDescription == expected)
    }

    @Test func displayRoundTripsThroughParse() throws {
        for reference in [SourcedBenchmarkId.local("foo"), .remote("foo")] {
            #expect(try SourcedBenchmarkId(parsing: reference.description) == reference)
        }
    }

    @Test func newMatchesSource() {
        #expect(SourcedBenchmarkId(source: .local, id: "x") == .local("x"))
        #expect(SourcedBenchmarkId(source: .remote, id: "x") == .remote("x"))
    }

    @Test func sourceMatchesVariant() {
        #expect(SourcedBenchmarkId.local("x").source == .local)
        #expect(SourcedBenchmarkId.remote("x").source == .remote)
    }
}
