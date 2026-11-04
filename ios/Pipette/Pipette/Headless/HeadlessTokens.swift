import Foundation

/// Splits the argument list into the verbs that route and the options they carry, in the
/// `ArgumentParser` spelling.
///
/// Two things the declared grammar cannot do for itself:
///
///   - **`key=value`.** Both `pipette`'s dashed spelling and this client's bare one must
///     work. A value may contain `=` (`--model gguf-text://repo=o/r&path=m.gguf`), so only a
///     token whose text before the first `=` is a bare identifier is a parameter.
///   - **Parameters before verbs.** `runtime=llama bench` is valid, and a subcommand tree
///     must be routed before it knows its options, so verbs are hoisted first. A leaf's
///     positional (`results show <job>/<cell>`) is hoisted with them, and
///     `HeadlessCommand.parse` takes it back out as the first word past the leaf.
nonisolated enum HeadlessTokens {
    /// One pass rather than two because a bare word is a verb only when it is not the value
    /// of the option before it, and an empty value (`runtime=`) is a bare word too.
    static func split(_ tokens: [String]) -> (verbs: [String], options: [String]) {
        let spelled = tokens.flatMap(asOption)
        var verbs: [String] = []
        var options: [String] = []
        var index = 0
        while index < spelled.count {
            let token = spelled[index]
            index += 1
            guard token.hasPrefix("-") else { verbs.append(token); continue }
            options.append(token)
            // An option written without `=` claims the word that follows, so that word is a
            // value and not a verb — including when it is the empty string.
            if !token.contains("="), index < spelled.count, !spelled[index].hasPrefix("-") {
                options.append(spelled[index])
                index += 1
            }
        }
        return (verbs, options)
    }

    /// A parameter becomes an option; anything else is returned unchanged.
    private static func asOption(_ token: String) -> [String] {
        guard !token.hasPrefix("-"),
              let eq = token.firstIndex(of: "="),
              isIdentifier(token[..<eq])
        else { return [token] }
        // `org=` is present but empty, which verbs refuse by name rather than treat as
        // absent, so the empty string must arrive as a value.
        guard token[token.index(after: eq)...].isEmpty else { return ["--\(token)"] }
        return ["--\(token.dropLast())", ""]
    }

    /// `runtime-flags` is a key; `gguf-text://repo` and `{"type":"x"` are not.
    private static func isIdentifier(_ s: Substring) -> Bool {
        guard let first = s.first, first.isLetter else { return false }
        return s.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" }
    }
}
