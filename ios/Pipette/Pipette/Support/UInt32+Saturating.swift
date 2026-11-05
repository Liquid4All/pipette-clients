import Foundation

/// The crate's `saturating_add` and `saturating_mul`: context sizing chains them over plan
/// values of any size, and wrapping would produce a tiny context.
nonisolated extension UInt32 {
    func addingSaturating(_ other: UInt32) -> UInt32 {
        let (sum, overflow) = addingReportingOverflow(other)
        return overflow ? .max : sum
    }

    func multipliedSaturating(_ other: UInt32) -> UInt32 {
        let (product, overflow) = multipliedReportingOverflow(by: other)
        return overflow ? .max : product
    }
}
