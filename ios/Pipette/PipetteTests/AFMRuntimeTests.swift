import Testing
@testable import Pipette

/// Unit tests for the pure AFM logic — the chat-message→session mapping used by eval
/// and the cap-reached threshold used by the timing benchmarks. Neither touches the
/// on-device model, so they run anywhere (the model itself is device + Apple
/// Intelligence only, and is exercised via `headlessrun runtime=afm`).
struct AFMRuntimeTests {
    // MARK: - splitMessages

    struct SplitCase {
        let name: String
        let messages: [[String: String]]
        let instructions: String
        let prompt: String
    }

    /// Pins the documented mapping: `system` turns join (blank-line separated) into the
    /// session instructions; a lone user turn is the prompt verbatim; multiple
    /// non-system turns fold into one role-labeled block; a missing role defaults to
    /// `user`; empty input yields empty strings.
    @Test(arguments: [
        SplitCase(name: "single user verbatim",
                  messages: [["role": "user", "content": "What is 2+2?"]],
                  instructions: "", prompt: "What is 2+2?"),
        SplitCase(name: "system + user",
                  messages: [["role": "system", "content": "Be terse."],
                             ["role": "user", "content": "Hi"]],
                  instructions: "Be terse.", prompt: "Hi"),
        SplitCase(name: "two system turns join with blank line",
                  messages: [["role": "system", "content": "A"],
                             ["role": "system", "content": "B"],
                             ["role": "user", "content": "Q"]],
                  instructions: "A\n\nB", prompt: "Q"),
        SplitCase(name: "multi-turn folds role-labeled",
                  messages: [["role": "user", "content": "Q1"],
                             ["role": "assistant", "content": "A1"],
                             ["role": "user", "content": "Q2"]],
                  instructions: "", prompt: "user: Q1\nassistant: A1\nuser: Q2"),
        SplitCase(name: "missing role defaults to user",
                  messages: [["content": "no role"]],
                  instructions: "", prompt: "no role"),
        SplitCase(name: "empty",
                  messages: [],
                  instructions: "", prompt: ""),
    ])
    func splitMessages(_ c: SplitCase) {
        let (instructions, prompt) = AFMRuntime.splitMessages(c.messages)
        #expect(instructions == c.instructions, "instructions [\(c.name)]")
        #expect(prompt == c.prompt, "prompt [\(c.name)]")
    }

    // MARK: - capFloor

    /// The cap-reached threshold is cap − max(2, cap/20): ~5% drift for large caps,
    /// a fixed 2-token floor for small ones.
    @Test(arguments: [
        (cap: 100, floor: 95),   // 5% = 5
        (cap: 200, floor: 190),  // 5% = 10
        (cap: 40, floor: 38),    // 5% = 2 (== the min)
        (cap: 10, floor: 8),     // 5% = 0 → clamped to 2
        (cap: 1, floor: -1),     // degenerate, but must not trap
    ])
    func capFloor(_ c: (cap: Int, floor: Int)) {
        #expect(AFMRuntime.capFloor(c.cap) == c.floor, "capFloor(\(c.cap))")
    }
}
