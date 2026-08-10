import Foundation

extension String {
    /// Pluralize this noun for `count`: the bare word for 1, an `s` suffix
    /// otherwise. Shared by the run headers and pocket mode, which build the same
    /// "N model(s) · M benchmark(s)" summaries.
    func pluralized(_ count: Int) -> String {
        count == 1 ? self : "\(self)s"
    }

    /// The normalized form of a search-field string: trimmed and lowercased, for
    /// case-insensitive `contains` filtering. The one place the list views prepare
    /// their query before matching.
    var searchNormalized: String {
        trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }
}
