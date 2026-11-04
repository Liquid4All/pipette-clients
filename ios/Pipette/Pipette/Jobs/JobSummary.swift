import Foundation

/// What the job list shows and searches of a job, without its cells.
nonisolated struct JobSummary {
    let jobId: JobId
    let createdAt: String
    let createdDate: Date?
    let status: JobStatus
    let displayTitle: String
    let modelNames: [String]
    let benchmarkIds: [String]
    let totalCells: Int
    let completedCells: Int
    /// Every engine the cells run on: one for a new job, more for a legacy mixed-runtime job.
    let runtimeKinds: [RuntimeKind]

    init(_ manifest: JobManifest) {
        jobId = manifest.jobId
        createdAt = manifest.createdAt
        createdDate = manifest.createdDate
        status = manifest.status
        displayTitle = manifest.displayTitle
        modelNames = manifest.modelNames
        benchmarkIds = manifest.benchmarkIds
        totalCells = manifest.totalCells
        completedCells = manifest.completedCells
        let kinds = Set(manifest.cells.map { RuntimeKind($0.source) })
        runtimeKinds = RuntimeKind.allCases.filter { kinds.contains($0) }
    }
}
