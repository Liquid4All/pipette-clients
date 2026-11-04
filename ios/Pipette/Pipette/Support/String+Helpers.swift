import Foundation

extension String {
    /// English `s` plural only; fine for the nouns the summaries use.
    func pluralized(_ count: Int) -> String {
        count == 1 ? self : "\(self)s"
    }

    /// A search query prepared for case-insensitive `contains` filtering.
    var searchNormalized: String {
        trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }
}
