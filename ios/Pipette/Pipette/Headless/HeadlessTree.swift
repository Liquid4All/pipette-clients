import ArgumentParser
import Foundation

// Types are named for the verb they declare (`auth me` is `Auth.Me`) so the grammar's
// spelling stays greppable, and two-letter verbs make two-letter type names.
// swiftlint:disable type_name

/// The headless grammar, declared: a leaf's stored properties are its accepted parameters,
/// so an undeclared one is unrecognized by construction. Value validation stays in
/// `HeadlessCommand.build`, which a declaration cannot express.
///
///   - **Both spellings.** `HeadlessTokens.split` turns `model=X` into `--model X` first.
///   - **Typed refusals.** Every option is optional so `ArgumentParser` never raises its
///     own "missing expected argument" prose; `build` throws `HeadlessUsageError` instead.
///     Strays are captured by `.allUnrecognized` and named in the refusal.
///   - **Value-or-flag parameters.** `submit=0` and a bare `--sync` are both valid, so those
///     are `defaultAsFlag` options; a `@Flag` cannot carry a value.
///   - **Positionals.** A leaf names `positionalKey` rather than declaring an `@Argument`,
///     because filling one drops any dashed token passed on the way (`benchmarks show
///     --frob x` would lose `--frob` rather than refuse it).
nonisolated protocol HeadlessLeaf: ParsableCommand {
    /// Declared rather than derived from the type name because it is also the alias target.
    static var verbPath: [String] { get }
    /// The key the crate's clap takes positionally on this leaf: a bare word after the verb
    /// path fills it as `key=value` would.
    static var positionalKey: String? { get }
    var unrecognized: [String] { get }
    var bag: [String: String] { get }
}

nonisolated extension HeadlessLeaf {
    static var positionalKey: String? { nil }
}

/// Drops the `--` and any `=value`, so a refusal names the parameter the way the caller
/// wrote it: `--cooldown=5` is reported as `cooldown`.
nonisolated func headlessStrayKey(_ token: String) -> String {
    let bare = token.hasPrefix("--") ? String(token.dropFirst(2)) : token
    guard let eq = bare.firstIndex(of: "=") else { return bare }
    return String(bare[..<eq])
}

nonisolated func headlessBag(_ pairs: KeyValuePairs<String, String?>) -> [String: String] {
    pairs.reduce(into: [:]) { bag, pair in
        if let value = pair.value { bag[pair.key] = value }
    }
}

// MARK: - Cli

nonisolated enum HeadlessTree {
    struct Cli: ParsableCommand {
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
            subcommands: [Register.self, Me.self, SetDevice.self, Reset.self,
                          Export.self, Import.self])

        struct Me: HeadlessLeaf {
            static let verbPath = ["auth", "me"]
            /// The CLI's `--local`: answer from disk, with no request.
            @Option(defaultAsFlag: "1") var local: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["local": local]) }
        }

        struct SetDevice: HeadlessLeaf {
            static let configuration = CommandConfiguration(commandName: "set-device")
            static let verbPath = ["auth", "set-device"]
            @Option(name: .customLong("device-name")) var deviceName: String?
            @Option(name: .customLong("device-form-factor")) var deviceFormFactor: String?
            @Option(name: .customLong("client-details")) var clientDetails: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] {
                headlessBag(["device-name": deviceName,
                             "device-form-factor": deviceFormFactor,
                             "client-details": clientDetails])
            }
        }

        struct Reset: HeadlessLeaf {
            static let verbPath = ["auth", "reset"]
            @Option(defaultAsFlag: "1") var force: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["force": force]) }
        }

        struct Export: HeadlessLeaf {
            static let verbPath = ["auth", "export"]
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { [:] }
        }

        struct Import: HeadlessLeaf {
            static let verbPath = ["auth", "import"]
            @Option var bundle: String?
            @Option(defaultAsFlag: "1") var force: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["bundle": bundle, "force": force]) }
        }

        struct Register: HeadlessLeaf {
            static let verbPath = ["auth", "register"]
            @Option(name: .customLong("server-url")) var serverUrl: String?
            @Option var organization: String?
            @Option(name: .customLong("contact-email")) var contactEmail: String?
            @Option(name: .customLong("client-details")) var clientDetails: String?
            @Option(name: .customLong("preauth-key")) var preauthKey: String?
            @Option(name: .customLong("device-name")) var deviceName: String?
            @Option(name: .customLong("device-form-factor")) var deviceFormFactor: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] {
                headlessBag(["server-url": serverUrl, "organization": organization,
                             "contact-email": contactEmail, "client-details": clientDetails,
                             "preauth-key": preauthKey, "device-name": deviceName,
                             "device-form-factor": deviceFormFactor])
            }
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

        struct Memseq: HeadlessLeaf {
            static let verbPath = ["diag", "memseq"]
            @Option var models: String?
            @Option var batch: String?
            @Option var prefill: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] {
                headlessBag(["models": models, "batch": batch, "prefill": prefill])
            }
        }
    }

    // MARK: - models

    struct Models: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [List.self, Pull.self, Delete.self, Rm.self],
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
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["model": model]) }
        }

        struct Delete: HeadlessLeaf {
            static let verbPath = ["models", "delete"]
            @Option var model: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["model": model]) }
        }

        struct Rm: HeadlessLeaf {
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
            @Option var format: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["format": format]) }
        }
    }

    // MARK: - benchmarks

    struct Benchmarks: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [List.self, Show.self, InitLocal.self, Run.self],
            defaultSubcommand: List.self)

        /// The crate's `--benchmark-type`. `results list` keeps `type=`, as the crate's
        /// `--type` there does.
        struct List: HeadlessLeaf {
            static let verbPath = ["benchmarks", "list"]
            @Option(name: .customLong("benchmark-type")) var benchmarkType: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["benchmark-type": benchmarkType]) }
        }

        struct Show: HeadlessLeaf {
            static let verbPath = ["benchmarks", "show"]
            static let positionalKey: String? = "benchmark"
            @Option(name: .customLong("benchmark")) var benchmarkRef: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["benchmark": benchmarkRef]) }
        }

        struct InitLocal: HeadlessLeaf {
            static let configuration = CommandConfiguration(commandName: "init-local")
            static let verbPath = ["benchmarks", "init-local"]
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { [:] }
        }

        struct Run: HeadlessLeaf {
            static let verbPath = ["benchmarks", "run"]
            @Option(name: .customLong("benchmark")) var benchmarkRef: String?
            @Option var model: String?
            @Option var runtime: String?
            @Option(name: .customLong("runtime-flags")) var runtimeFlags: String?
            @Option(defaultAsFlag: "1") var sync: String?
            @Option(name: .customLong("benchmark-flags")) var benchmarkFlags: String?
            @Option(name: .customLong("readiness-max-wait-secs")) var readinessMaxWaitSecs: String?
            @Option(name: .customLong("readiness-skip-thermal"), defaultAsFlag: "1")
            var readinessSkipThermal: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] {
                headlessBag(["benchmark": benchmarkRef, "model": model, "runtime": runtime,
                             "runtime-flags": runtimeFlags, "sync": sync,
                             "benchmark-flags": benchmarkFlags,
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
            @Option(name: .customLong("type")) var benchmarkType: String?
            @Option var state: String?
            @Option var limit: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] {
                headlessBag(["benchmark": benchmark, "type": benchmarkType,
                             "state": state, "limit": limit])
            }
        }

        struct Show: HeadlessLeaf {
            static let verbPath = ["results", "show"]
            static let positionalKey: String? = "result"
            @Option(name: .customLong("result")) var resultId: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["result": resultId]) }
        }

        struct Delete: HeadlessLeaf {
            static let verbPath = ["results", "delete"]
            static let positionalKey: String? = "result"
            @Option(name: .customLong("result")) var resultId: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["result": resultId]) }
        }
    }

    // MARK: - storage

    struct Storage: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [Status.self, Gc.self])

        struct Status: HeadlessLeaf {
            static let verbPath = ["storage", "status"]
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { [:] }
        }

        struct Gc: HeadlessLeaf {
            static let verbPath = ["storage", "gc"]
            @Option(name: .customLong("dry-run"), defaultAsFlag: "1") var dryRun: String?
            @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
            var bag: [String: String] { headlessBag(["dry-run": dryRun]) }
        }
    }

    // MARK: - job / jobs

    struct Job: ParsableCommand {
        static let configuration = CommandConfiguration(
            subcommands: [Rm.self, Run.self, Export.self, Submit.self])

        struct Rm: HeadlessLeaf {
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
        static let configuration = CommandConfiguration(subcommands: [Run.self])

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

    /// The claim loop's knobs, named as `pipette worker` names them.
    struct Worker: HeadlessLeaf {
        static let verbPath = ["worker"]
        @Option(name: .customLong("idle-secs")) var idleSecs: String?
        @Option(name: .customLong("idle-jitter-secs")) var idleJitterSecs: String?
        @Option(name: .customLong("heartbeat-secs")) var heartbeatSecs: String?
        @Option(name: .customLong("max-jobs")) var maxJobs: String?
        @Option(name: .customLong("skip-profile-refresh"), defaultAsFlag: "1")
        var skipProfileRefresh: String?
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] {
            headlessBag(["idle-secs": idleSecs, "idle-jitter-secs": idleJitterSecs,
                         "heartbeat-secs": heartbeatSecs, "max-jobs": maxJobs,
                         "skip-profile-refresh": skipProfileRefresh])
        }
    }

    struct Version: HeadlessLeaf {
        static let verbPath = ["version"]
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { [:] }
    }

    struct Sync: HeadlessLeaf {
        static let verbPath = ["sync"]
        static let positionalKey: String? = "result"
        @Option var job: String?
        @Option var result: String?
        /// The CLI's `--server-url`: where the catalog pull goes when no registration
        /// names one.
        @Option(name: .customLong("server-url")) var serverUrl: String?
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] {
            headlessBag(["job": job, "result": result, "server-url": serverUrl])
        }
    }

    // MARK: - bench, and the bare form

    /// Shared by `bench` and the bare form, which differ only in `build`.
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
        /// The server job `pipette-ios-proxy` claimed for the run.
        @Option var claim: String?
        /// The benchmark flags `benchmarks run` takes, accepted here because
        /// `pipette-ios-proxy` emits the bare form.
        @Option(name: .customLong("benchmark-flags")) var benchmarkFlags: String?
        @Option(name: .customLong("readiness-max-wait-secs")) var readinessMaxWaitSecs: String?
        @Option(name: .customLong("readiness-skip-thermal"), defaultAsFlag: "1")
        var readinessSkipThermal: String?

        var bag: [String: String] {
            headlessBag(["spec": spec, "model": model, "match": match, "quant": quant,
                         "runtime": runtime, "batch": batch, "runtime-flags": runtimeFlags,
                         "benchmark": benchmark, "benchmarks": benchmarks,
                         "metrics": metrics, "offsets": offsets, "submit": submit,
                         "claim": claim, "benchmark-flags": benchmarkFlags,
                         "readiness-max-wait-secs": readinessMaxWaitSecs,
                         "readiness-skip-thermal": readinessSkipThermal])
        }
    }

    struct Bench: HeadlessLeaf {
        static let verbPath = ["bench"]
        @OptionGroup var options: BenchOptions
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { options.bag }
    }

    /// No verb at all: what the plan runner emits.
    struct Bare: HeadlessLeaf {
        static let verbPath: [String] = []
        @OptionGroup var options: BenchOptions
        @Argument(parsing: .allUnrecognized) var unrecognized: [String] = []
        var bag: [String: String] { options.bag }
    }
}

// swiftlint:enable type_name
