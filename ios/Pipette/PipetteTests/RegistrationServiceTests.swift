import Foundation
import Testing

@testable import Pipette

struct RegistrationServiceTests {
    // On the registration path a 401 is the server's pre-auth-key rejection
    // (invalid/expired/used); the user sees a key-specific cause, not the raw
    // "HTTP 401: {body}".
    @Test func humanizes401AsPreauthKeyRejection() {
        let message = humanizedRegistrationError(
            ManagementClientError.httpStatus(statusCode: 401, body: "consumed"),
            preauthContext: true
        )
        #expect(message.contains("pre-auth key"))
        #expect(!message.contains("401"))
    }

    // A 403 means the collector requires a key but none was supplied.
    @Test func humanizes403AsPreauthKeyRequired() {
        let message = humanizedRegistrationError(
            ManagementClientError.httpStatus(statusCode: 403, body: ""),
            preauthContext: true
        )
        #expect(message.contains("requires a pre-auth key"))
        #expect(!message.contains("403"))
    }

    // Without the pre-auth context (e.g. the public benchmark-catalog sync), a
    // 401/403 must not be mislabeled as a key problem.
    @Test func doesNotRemap401OutsidePreauthContext() {
        let message = humanizedRegistrationError(
            ManagementClientError.httpStatus(statusCode: 401, body: "nope")
        )
        #expect(!message.contains("pre-auth key"))
    }

    // Other HTTP failures keep their existing generic surfacing.
    @Test func doesNotRemapUnrelatedHTTPStatus() {
        let message = humanizedRegistrationError(
            ManagementClientError.httpStatus(statusCode: 500, body: "boom"),
            preauthContext: true
        )
        #expect(!message.contains("pre-auth key"))
    }
}
