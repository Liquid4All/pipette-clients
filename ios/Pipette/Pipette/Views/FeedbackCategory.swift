import Foundation

/// Optional feedback category. Kept in sync with the Android `FeedbackDialog` categories and
/// pipette-dashboard's `FEEDBACK_CATEGORIES` so the `category` tag is consistent everywhere.
enum FeedbackCategory: String, CaseIterable, Identifiable {
    case reportBug = "report_bug"
    case reportIncorrectData = "report_incorrect_data"
    case requestModel = "request_model"
    case requestRuntime = "request_runtime"
    case requestHardware = "request_hardware"
    case requestEval = "request_eval"
    case other

    var id: String { rawValue }

    var label: String {
        switch self {
        case .reportBug: return "Report a bug"
        case .reportIncorrectData: return "Report incorrect data"
        case .requestModel: return "Request a model"
        case .requestRuntime: return "Request a runtime"
        case .requestHardware: return "Request hardware"
        case .requestEval: return "Request an evaluation dataset"
        case .other: return "Something else"
        }
    }
}
