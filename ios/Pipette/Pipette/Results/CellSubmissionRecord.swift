import Foundation

enum CellSubmissionStatus: String, Codable {
    case submitted
    case failed
}

struct CellSubmissionRecord: nonisolated Codable, Equatable {
    let status: CellSubmissionStatus
    let serverJobId: String?
    let submittedAt: String?
    let errors: [String]
    /// The collector's base URL, its only stable identity, so a later sync can re-send
    /// results that went to a different collector. Nil for `.failed` records and for
    /// records written before this field existed.
    let collector: ServerUrl?

    enum CodingKeys: String, CodingKey {
        case status
        case serverJobId = "server_job_id"
        case submittedAt = "submitted_at"
        case errors
        case collector
    }

    static func submitted(
        serverJobId: String,
        collector: ServerUrl? = nil,
        submittedAt: String = JobDateFormat.iso8601.string(from: Date())
    ) -> Self {
        CellSubmissionRecord(
            status: .submitted,
            serverJobId: serverJobId,
            submittedAt: submittedAt,
            errors: [],
            collector: collector
        )
    }

    static func failed(_ errors: [String], submittedAt: String = JobDateFormat.iso8601.string(from: Date())) -> Self {
        CellSubmissionRecord(
            status: .failed,
            serverJobId: nil,
            submittedAt: submittedAt,
            errors: errors,
            collector: nil
        )
    }
}
