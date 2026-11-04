import Foundation

// Distinct newtypes because job and cell ids travel adjacently through many storage and
// submission signatures, where two bare `String`s would make a transposition a silent
// wrong-directory bug instead of a compile error. `StringId` keeps the Codable form a
// bare string, so on-disk and wire JSON are unchanged.

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
