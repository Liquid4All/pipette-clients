import Foundation

extension JobCell {
    /// Build one `.pending` cell per benchmark reference for `model`, dropping any that
    /// don't parse to a known `BenchmarkDefinition` (reported via `onSkip`).
    ///
    /// A reference carries its own provenance — `local/<id>`, `remote/<id>`, or a bare id
    /// meaning remote — exactly as the CLI's `--benchmark` takes a `SourcedBenchmarkId`.
    /// That is what puts `benchmarkSource` on the cell, and so what keeps a result from
    /// the generated catalog half out of the submit sweep. Without it every front-end but
    /// the New Job screen produced cells that read as submittable.
    ///
    /// The single resolve→build step shared by every job front-end (the New Job
    /// screen, the headless CLI, and deep links), so a cell is constructed
    /// identically regardless of entry point. AFM needs no special case: its
    /// `DiscoveredModel` (`.appleFoundation`) yields the AFM submission slug for
    /// `hfRepo` and `source == .appleFoundation`, which is exactly the AFM cell shape.
    static func pending(
        benchmarkIds ids: [String], for model: DiscoveredModel,
        onSkip: (String) -> Void = { _ in }
    ) -> [JobCell] {
        ids.compactMap { reference in
            guard let sourced = SourcedBenchmarkId(reference: reference),
                  let def = BenchmarkDefinition(parsingId: sourced.id)
            else {
                onSkip(reference)
                return nil
            }
            return JobCell(
                cellId: CellId(UUID().uuidString), benchmarkId: sourced.id,
                benchmarkType: def.type,
                runStatus: .pending, serverJobId: nil, errorMessage: nil,
                source: model.source, benchmarkSource: sourced.source)
        }
    }
}
