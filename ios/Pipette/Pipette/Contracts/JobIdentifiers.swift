import Foundation

// Strongly-typed run identifiers. A job id and a cell id are both opaque UUID
// strings, and they are passed *positionally and adjacently* through dozens of
// storage/submission signatures — `cellPayloadURL(jobId:cellId:)`,
// `saveSubmission(_:jobId:cellId:)`, `submitCell(jobId:cellId:)`, … — where two
// bare `String`s in a row make a transposition a silent wrong-directory bug
// instead of a compile error. Distinct newtypes make the swap a type error at
// zero runtime cost (value type, inline layout), the same rationale as the
// client identifiers in `ClientIdentifiers.swift`.
//
// Both are `StringId`s, which supplies the transparent single-value `Codable`
// (so the on-disk manifest/sentinel/submission JSON and wire payloads are
// byte-for-byte unchanged), ordering, `CustomStringConvertible` (so `"\(jobId)"`
// interpolation needs no `.value`), and `ExpressibleByStringLiteral`.

/// A local job identifier (the `jobs/{jobId}/` directory name).
nonisolated struct JobId: StringId, Hashable {
    let value: String
    init(_ value: String) { self.value = value }
}

/// A cell identifier (UUID; doubles as the cell's local result directory name).
nonisolated struct CellId: StringId, Hashable {
    let value: String
    init(_ value: String) { self.value = value }
}
