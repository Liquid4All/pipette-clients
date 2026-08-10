import Foundation

/// Splits the argument list into the verbs that route and the options they carry, in the
/// `ArgumentParser` spelling.
///
/// Two things the declared grammar cannot do for itself:
///
///   - **This client's `key=value`.** An invocation copied from `pipette` uses dashes, one
///     copied from this client's own docs does not, and both have to keep working. The
///     rewrite cannot be "any token containing `=`", because a *value* may contain one —
///     `--model gguf-text://repo=o/r&path=m.gguf` is a single token whose `=` belongs to
///     the URI. What separates them is the text before the first `=`: a parameter's key is
///     a bare identifier, a URI's is `scheme://host`, a JSON blob's is `{"type":"…"`.
///   - **Parameters before verbs.** `runtime=llama bench` is a real invocation, and a
///     subcommand tree has to be routed before it knows its options, so verbs are hoisted
///     ahead of everything else.
///
/// Splitting on the *first* `=` is what keeps `model=mlx://repo=o/r` intact — key `model`,
/// value `mlx://repo=o/r`.
nonisolated enum HeadlessTokens {
    /// The verbs, and the options in `ArgumentParser` spelling.
    ///
    /// Returned together rather than as two passes because a bare word is only a verb when
    /// it is not the value of the option before it — a distinction the caller would have to
    /// redo to ask "were there any verbs?", and did, wrongly: an empty value is a bare
    /// token too, so `runtime=` used to read as a verb and refuse a valid bare-form run.
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

    /// One token in `ArgumentParser` spelling. A parameter becomes an option; anything else
    /// — a verb, a value, an already-dashed option — is returned unchanged.
    private static func asOption(_ token: String) -> [String] {
        guard !token.hasPrefix("-"),
              let eq = token.firstIndex(of: "="),
              isIdentifier(token[..<eq])
        else { return [token] }
        // `org=` is a real invocation — a present-but-empty parameter, which the verbs
        // refuse by name rather than treat as absent. Split into option and value so the
        // empty string arrives as a value instead of a missing one.
        guard token[token.index(after: eq)...].isEmpty else { return ["--\(token)"] }
        return ["--\(token.dropLast())", ""]
    }

    /// A parameter key: a letter, then letters/digits/dashes. Deliberately narrower than
    /// "no `://`" — `runtime-flags` is a key, `gguf-text://repo` and `{"type":"x"` are not.
    private static func isIdentifier(_ s: Substring) -> Bool {
        guard let first = s.first, first.isLetter else { return false }
        return s.allSatisfy { $0.isLetter || $0.isNumber || $0 == "-" }
    }
}
