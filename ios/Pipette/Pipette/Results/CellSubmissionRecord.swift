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
    /// The collector this result was submitted to, identified by its base URL —
    /// the only stable collector identity we have (there is no server-issued
    /// id). Lets a later sync detect results that belong to a *different*
    /// collector and re-send them. Nil for records written before this field
    /// existed, or for `.failed` records.
    let collector: ServerURL?

    enum CodingKeys: String, CodingKey {
        case status
        case serverJobId = "server_job_id"
        case submittedAt = "submitted_at"
        case errors
        case collector
    }

    static func submitted(
        serverJobId: String,
        collector: ServerURL? = nil,
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

enum ResultSubmissionFeatureGate {
    nonisolated static func canSubmitResults(registration: IdentityRegistration?) -> Bool {
        registration != nil
    }
}
