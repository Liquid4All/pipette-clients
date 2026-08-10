import Foundation
import Testing

@testable import Pipette

/// The split into verbs and options that lets `key=value` and `--key value` reach one
/// declared command tree.
///
/// One table, because every case asks the same question of the same function: given these
/// tokens, what routes and what is a parameter. The cases that matter are the ones where a
/// token is not what it looks like — an `=` inside a *value*, a verb written after a
/// parameter, an empty value — each of which routed wrongly at some point.
struct HeadlessTokensTests {
    struct Split: Sendable, CustomTestStringConvertible {
        let why: String
        let tokens: [String]
        let verbs: [String]
        let options: [String]

        var testDescription: String { why }
    }

    @Test(arguments: [
        Split(why: "a parameter becomes an option",
              tokens: ["models", "pull", "model=X"],
              verbs: ["models", "pull"], options: ["--model=X"]),
        Split(why: "the dashed spelling is already an option",
              tokens: ["models", "pull", "--model=X"],
              verbs: ["models", "pull"], options: ["--model=X"]),
        Split(why: "a dashed key with a space-separated value",
              tokens: ["bench", "--model", "X"],
              verbs: ["bench"], options: ["--model", "X"]),
        Split(why: "split on the first `=` only, so the value keeps its own",
              tokens: ["model=mlx://repo=o/r"],
              verbs: [], options: ["--model=mlx://repo=o/r"]),
        Split(why: "a URI value is not a parameter, however many `=` it carries",
              tokens: ["--model", "gguf-text://repo=o/r&path=m.gguf"],
              verbs: [], options: ["--model", "gguf-text://repo=o/r&path=m.gguf"]),
        Split(why: "a JSON value is not a parameter either",
              tokens: ["--spec", #"{"type":"gguf_text","repo_name"="x"}"#],
              verbs: [], options: ["--spec", #"{"type":"gguf_text","repo_name"="x"}"#]),
        Split(why: "`runtime=llama bench` — a verb written after a parameter is hoisted",
              tokens: ["runtime=llama", "bench", "model=X"],
              verbs: ["bench"], options: ["--runtime=llama", "--model=X"]),
        Split(why: "an empty value stays a value, so `runtime=` is not read as a verb",
              tokens: ["runtime="],
              verbs: [], options: ["--runtime", ""]),
        Split(why: "a value-or-flag parameter with no value",
              tokens: ["benchmarks", "run", "--sync", "--benchmark=b"],
              verbs: ["benchmarks", "run"], options: ["--sync", "--benchmark=b"]),
        Split(why: "no tokens at all is the bare form",
              tokens: [], verbs: [], options: []),
    ])
    func tokensSplitIntoVerbsAndOptions(_ testCase: Split) {
        let split = HeadlessTokens.split(testCase.tokens)

        #expect(split.verbs == testCase.verbs)
        #expect(split.options == testCase.options)
    }
}
