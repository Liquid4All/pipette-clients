import Foundation
import Sentry

/// One llama.cpp build the app carries: `Frameworks/<name>.framework`, from
/// `ios/build-llama.sh`. `llamacpp` is `vendor/llama.cpp`, which every build of the app
/// carries; `llamacpp-<variant>` is `vendor/llamacpp-<variant>`, which internal builds carry.
nonisolated struct LlamaPin: Hashable, Sendable {
    let name: String
    /// `<host>/<org>/<repo>` from the submodule's `.gitmodules` URL, so a fork reports itself.
    let repositoryUrl: String
    let build: Build

    /// Any commit is buildable: pinning a cherry-picked fix leaves nothing to tag it with, so
    /// `untagged` is a real state rather than a defensive one.
    nonisolated enum Build: Hashable, Sendable {
        case tagged(tag: String, commit: String)
        case untagged(commit: String)

        /// Always present, and the exact identity: a tag can be moved or deleted upstream, a
        /// commit cannot.
        var commit: String {
            switch self {
            case let .tagged(_, commit), let .untagged(commit): commit
            }
        }

        var tag: String? {
            switch self {
            case let .tagged(tag, _): tag
            case .untagged: nil
            }
        }
    }

    /// What a submission reports as `repository_version`, and what results carry as
    /// `runtime_version`: the tag when the pin has one, so an iPhone names its engine the way
    /// every desktop runtime does (`repository_version = "b10216"`), and the commit
    /// otherwise, which is still exact, just not a name a plan can pin by.
    var submissionVersion: String { build.tag ?? build.commit }

    var source: SourceRepository {
        SourceRepository(repositoryUrl: RepositoryUrl(repositoryUrl),
                         repositoryVersion: NonEmptyString(validated: submissionVersion))
    }
}

/// The builds in `Frameworks/` and the one this process runs. A process loads one build:
/// a loaded image cannot be unloaded, and a second would add its own Metal backend and
/// globals to the next cell's memory.
nonisolated enum LlamaPins {
    /// Main first, then by name. Read from the frameworks' Info.plists, so this loads no code.
    static let carried: [LlamaPin] = {
        guard let directory = Bundle.main.privateFrameworksURL else {
            fatalError("the app bundle has no Frameworks folder")
        }
        let urls: [URL]
        do {
            urls = try FileManager.default.contentsOfDirectory(at: directory,
                                                               includingPropertiesForKeys: nil)
        } catch {
            fatalError("cannot list \(directory.path): \(error)")
        }
        let pins = urls
            .filter { $0.pathExtension == "framework"
                && $0.deletingPathExtension().lastPathComponent.hasPrefix(PIPETTE_LLAMA_MAIN_PIN) }
            .map(pin)
            .sorted { (order($0), $0.name) < (order($1), $1.name) }
        guard pins.first?.name == PIPETTE_LLAMA_MAIN_PIN else {
            fatalError("Frameworks/\(PIPETTE_LLAMA_MAIN_PIN).framework is missing; "
                + "ios/embed-llama.sh embeds it into every build")
        }
        return pins
    }()

    static var main: LlamaPin { carried[0] }

    /// The build this process runs: the one it loaded, else main, which the first llama.cpp
    /// call loads when nothing chose a build before it.
    static var current: LlamaPin {
        guard let loaded = pipette_llama_loaded_pin() else { return main }
        let name = String(cString: loaded)
        guard let pin = carried.first(where: { $0.name == name }) else {
            fatalError("this process loaded \(name), which Frameworks/ does not carry")
        }
        return pin
    }

    /// Makes `pin` this process's build. Called before a cell measures anything, so the
    /// load's own time and pages stay out of it.
    static func load(_ pin: LlamaPin) throws {
        var err = [CChar](repeating: 0, count: 512)
        guard pipette_llama_load(pin.name, &err, err.count) != nil else {
            throw LlamaPinError(pin: pin.name, reason: String(cString: err))
        }
        SentrySDK.configureScope { $0.setTag(value: pin.name, key: "llamacpp") }
    }

    /// For `runtimes list`: a build that does not load fails the listing. Leaves every build
    /// mapped, so only a process that runs no cell calls this.
    static func probeAll() throws -> [LlamaPin] {
        try carried.map { pin in
            var err = [CChar](repeating: 0, count: 512)
            guard pipette_llama_probe(pin.name, &err, err.count) != nil else {
                throw LlamaPinError(pin: pin.name, reason: String(cString: err))
            }
            return pin
        }
    }

    private static func order(_ pin: LlamaPin) -> Int {
        pin.name == PIPETTE_LLAMA_MAIN_PIN ? 0 : 1
    }

    private static func pin(_ framework: URL) -> LlamaPin {
        let name = framework.deletingPathExtension().lastPathComponent
        guard let info = Bundle(url: framework)?.infoDictionary,
              let repositoryUrl = info["PipetteLlamaRepositoryUrl"] as? String,
              let commit = info["PipetteLlamaCommit"] as? String
        else { fatalError("\(name).framework's Info.plist does not name its llama.cpp build") }
        return LlamaPin(name: name, repositoryUrl: repositoryUrl,
                        build: build(tag: info["PipetteLlamaTag"] as? String, commit: commit))
    }

    private static func build(tag: String?, commit: String) -> LlamaPin.Build {
        guard let tag, !tag.isEmpty else { return .untagged(commit: commit) }
        return .tagged(tag: tag, commit: commit)
    }
}

nonisolated struct LlamaPinError: Error, LocalizedError {
    let pin: String
    let reason: String

    var errorDescription: String? { "llama.cpp build \(pin) does not load: \(reason)" }
}
