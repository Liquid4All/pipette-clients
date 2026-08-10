import ArgumentParser
import Foundation

/// The headless grammar, declared rather than hand-tokenized.
///
/// Each verb is a `ParsableCommand` whose stored properties *are* its accepted parameters,
/// which is the rule the hand-rolled parser had to restate as a per-verb `accept(…)`
/// allowlist. A parameter the verb does not declare is unrecognized by construction.
///
/// Three things this has to preserve, because the existing tests and both front-ends
/// depend on them:
///
///   - **Both spellings.** `model=X` and `--model X` are the same invocation.
///     `HeadlessTokens.normalize` converts the first into the second before parsing.
///   - **Typed refusals.** Every option is declared optional so `ArgumentParser` never
///     raises its own "missing expected argument" prose; required-ness, ranges and value
///     shapes stay in `HeadlessCommand.build`, which still throws `HeadlessUsageError`.
///     Strays are captured by `.allUnrecognized` and named in the refusal.
///   - **Value-or-flag parameters.** `submit=0` and a bare `--sync` are both real, so those
///     are `defaultAsFlag` options rather than `@Flag`s, which cannot carry a value.
///
/// The output is the `(verbs, params)` pair `build` already consumes, so what this
/// replaces is the tokenizer and the per-verb allowlists — `build` keeps the value
/// validation, which a declaration cannot express.
nonisolated protocol HeadlessLeaf: ParsableCommand {
    /// The verb path `build` switches on — also the alias target, which is why it is
    /// declared rather than derived from the type name.
    static var verbPath: [String] { get }
    /// Tokens that matched no declared parameter, in the spelling the caller typed.
    var unrecognized: [String] { get }
    /// The declared parameters that were supplied, keyed as `build` expects.
    var bag: [String: String] { get }
}

/// Drops the `--` and any `=value`, so a refusal names the parameter the way the caller
/// wrote it: `--cooldown=5` is reported as `cooldown`.
nonisolated func headlessStrayKey(_ token: String) -> String {
    let bare = token.hasPrefix("--") ? String(token.dropFirst(2)) : token
    guard let eq = bare.firstIndex(of: "=") else { return bare }
    return String(bare[..<eq])
}

/// Collapses the optional properties a leaf declares into `build`'s string bag.
nonisolated func headlessBag(_ pairs: KeyValuePairs<String, String?>) -> [String: String] {
    pairs.reduce(into: [:]) { bag, pair in
        if let value = pair.value { bag[pair.key] = value }
    }
}

// MARK: - Root

nonisolated enum HeadlessTree {
    struct Root: ParsableCommand {
        static let configuration = CommandConfiguration(
            commandName: "headlessrun",
            subcommands: [
                Auth.self, Diag.self, Models.self, Runtimes.self, Benchmarks.self,
                Results.self, Storage.self, Job.self, Settings.self,
                Jobs.self, Status.self, Worker.self, Sync.self, Version.self,
                Bench.self,
            ])
    }

    // MARK: - auth

    struct Auth: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [Register.self, Whoami.self, Reset.self])

        struct Whoami: HeadlessLeaf {
            static let configuration = CommandConfiguration(commandName: "me")
            static let verbPath = ["auth", "me"]
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { [:] }
        }

        struct Reset: HeadlessLeaf {
            static let verbPath = ["auth", "reset"]
            @Option(defaultAsFlag: "1") var force: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["force": force]) }
        }
    }

    /// Parameter names are the CLI's: `pipette auth register --server-url --organization
    /// --contact-email --preauth-key --client-details --device-name`.
    struct Register: HeadlessLeaf {
        static let verbPath = ["auth", "register"]
        @Option(name: .customLong("server-url")) var serverUrl: String?
        @Option var organization: String?
        @Option(name: .customLong("contact-email")) var contactEmail: String?
        @Option(name: .customLong("preauth-key")) var preauthKey: String?
        @Option(name: .customLong("client-details")) var clientDetails: String?
        @Option(name: .customLong("device-name")) var deviceName: String?
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] {
            headlessBag(["server-url": serverUrl, "organization": organization,
                         "contact-email": contactEmail, "preauth-key": preauthKey,
                         "client-details": clientDetails, "device-name": deviceName])
        }
    }

    // MARK: - diag

    struct Diag: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [Memseq.self, Probe.self])

        struct Probe: HeadlessLeaf {
            static let verbPath = ["diag", "probe"]
            @Option var kind: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["kind": kind]) }
        }
    }

    struct Memseq: HeadlessLeaf {
        static let verbPath = ["diag", "memseq"]
        @Option var models: String?
        @Option var batch: String?
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { headlessBag(["models": models, "batch": batch]) }
    }

    // MARK: - models

    struct Models: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [List.self, Pull.self, Delete.self, RemoveNamed.self],
            defaultSubcommand: List.self)

        struct List: HeadlessLeaf {
            static let verbPath = ["models", "list"]
            @Option var format: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["format": format]) }
        }

        struct Pull: HeadlessLeaf {
            static let verbPath = ["models", "pull"]
            @Option var model: String?
            @Option var spec: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["model": model, "spec": spec]) }
        }

        struct Delete: HeadlessLeaf {
            static let verbPath = ["models", "delete"]
            @Option var model: String?
            @Option var spec: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["model": model, "spec": spec]) }
        }

        struct RemoveNamed: HeadlessLeaf {
            static let configuration = CommandConfiguration(commandName: "rm")
            static let verbPath = ["models", "rm"]
            @Option var name: String?
            @Option var repo: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["name": name, "repo": repo]) }
        }
    }

    // MARK: - runtimes

    struct Runtimes: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [List.self], defaultSubcommand: List.self)

        struct List: HeadlessLeaf {
            static let verbPath = ["runtimes", "list"]
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { [:] }
        }
    }

    // MARK: - benchmarks

    struct Benchmarks: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [List.self, Show.self, InitLocal.self, Run.self],
            defaultSubcommand: List.self)

        struct List: HeadlessLeaf {
            static let verbPath = ["benchmarks", "list"]
            @Option var type: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["type": type]) }
        }

        struct Show: HeadlessLeaf {
            static let verbPath = ["benchmarks", "show"]
            @Option var benchmark: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["benchmark": benchmark]) }
        }

        struct InitLocal: HeadlessLeaf {
            static let configuration = CommandConfiguration(commandName: "init-local")
            static let verbPath = ["benchmarks", "init-local"]
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { [:] }
        }

        struct Run: HeadlessLeaf {
            static let verbPath = ["benchmarks", "run"]
            @Option var benchmark: String?
            @Option var model: String?
            @Option var runtime: String?
            @Option(name: .customLong("runtime-flags")) var runtimeFlags: String?
            @Option(defaultAsFlag: "1") var sync: String?
            @Option(name: .customLong("readiness-max-wait-secs")) var readinessMaxWaitSecs: String?
            @Option(name: .customLong("readiness-skip-thermal"), defaultAsFlag: "1")
            var readinessSkipThermal: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] {
                headlessBag(["benchmark": benchmark, "model": model, "runtime": runtime,
                             "runtime-flags": runtimeFlags, "sync": sync,
                             "readiness-max-wait-secs": readinessMaxWaitSecs,
                             "readiness-skip-thermal": readinessSkipThermal])
            }
        }
    }

    // MARK: - results

    struct Results: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [List.self, Show.self, Delete.self],
            defaultSubcommand: List.self)

        struct List: HeadlessLeaf {
            static let verbPath = ["results", "list"]
            @Option var benchmark: String?
            @Option var type: String?
            @Option var state: String?
            @Option var limit: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] {
                headlessBag(["benchmark": benchmark, "type": type,
                             "state": state, "limit": limit])
            }
        }

        struct Show: HeadlessLeaf {
            static let verbPath = ["results", "show"]
            @Option var result: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["result": result]) }
        }

        struct Delete: HeadlessLeaf {
            static let verbPath = ["results", "delete"]
            @Option var result: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["result": result]) }
        }
    }

    // MARK: - storage

    struct Storage: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [Status.self, Sweep.self])

        struct Status: HeadlessLeaf {
            static let verbPath = ["storage", "status"]
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { [:] }
        }

        struct Sweep: HeadlessLeaf {
            static let configuration = CommandConfiguration(commandName: "gc")
            static let verbPath = ["storage", "gc"]
            @Option(name: .customLong("dry-run"), defaultAsFlag: "1") var dryRun: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["dry-run": dryRun]) }
        }
    }

    // MARK: - job / jobs

    struct Job: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [Remove.self, Run.self, Export.self, Submit.self])

        struct Remove: HeadlessLeaf {
            static let configuration = CommandConfiguration(commandName: "rm")
            static let verbPath = ["job", "rm"]
            @Option var id: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["id": id]) }
        }

        struct Run: HeadlessLeaf {
            static let verbPath = ["job", "run"]
            @Option var id: String?
            @Option var scope: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["id": id, "scope": scope]) }
        }

        struct Export: HeadlessLeaf {
            static let verbPath = ["job", "export"]
            @Option var id: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["id": id]) }
        }

        struct Submit: HeadlessLeaf {
            static let verbPath = ["job", "submit"]
            @Option var id: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["id": id]) }
        }
    }

    struct Jobs: HeadlessLeaf {
        static let verbPath = ["jobs"]
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { [:] }
    }

    // MARK: - settings

    struct Settings: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [Show.self, Set.self, Run.self], defaultSubcommand: Show.self)

        /// `settings` with no leaf — the show form, which has no verb of its own.
        struct Show: HeadlessLeaf {
            static let verbPath = ["settings"]
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { [:] }
        }

        struct Set: HeadlessLeaf {
            static let verbPath = ["settings", "set"]
            @Option var worker: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["worker": worker]) }
        }

        struct Run: HeadlessLeaf {
            static let verbPath = ["settings", "run"]
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { [:] }
        }
    }

    // MARK: - single-word verbs

    struct Status: HeadlessLeaf {
        static let verbPath = ["status"]
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { [:] }
    }

    struct Worker: HeadlessLeaf {
        static let verbPath = ["worker"]
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { [:] }
    }

    struct Version: HeadlessLeaf {
        static let verbPath = ["version"]
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { [:] }
    }

    struct Sync: HeadlessLeaf {
        static let verbPath = ["sync"]
        @Option var job: String?
        @Option var result: String?
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { headlessBag(["job": job, "result": result]) }
    }

    // MARK: - bench, and the bare form

    /// The parameters `bench` and the bare form share. The bare form additionally accepts
    /// nothing of its own — the two differ only in which verb selected them — so one group
    /// serves both and the difference stays in `build`.
    struct BenchOptions: ParsableArguments {
        @Option var spec: String?
        @Option var model: String?
        @Option var match: String?
        @Option var quant: String?
        @Option var runtime: String?
        @Option var batch: String?
        @Option(name: .customLong("runtime-flags")) var runtimeFlags: String?
        @Option var benchmark: String?
        @Option var benchmarks: String?
        @Option var metrics: String?
        @Option var offsets: String?
        @Option(defaultAsFlag: "1") var submit: String?

        var bag: [String: String] {
            headlessBag(["spec": spec, "model": model, "match": match, "quant": quant,
                         "runtime": runtime, "batch": batch, "runtime-flags": runtimeFlags,
                         "benchmark": benchmark, "benchmarks": benchmarks,
                         "metrics": metrics, "offsets": offsets, "submit": submit])
        }
    }

    struct Bench: HeadlessLeaf {
        static let verbPath = ["bench"]
        @OptionGroup var options: BenchOptions
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { options.bag }
    }

    /// No verb at all — the original CLI form, still what the plan runner emits.
    struct Bare: HeadlessLeaf {
        static let verbPath: [String] = []
        @OptionGroup var options: BenchOptions
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { options.bag }
    }
}
