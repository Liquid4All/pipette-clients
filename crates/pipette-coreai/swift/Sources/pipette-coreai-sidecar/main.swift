// pipette-coreai-sidecar — HTTP server for pipette-coreai benchmarks.
//
// Serves the same JSON contract as the pipette-mlx Python sidecar
// (pipette_mlx_server.py), but over Apple's low-level Core AI runtime. It loads
// an .aimodel bundle directory (metadata.json + *.aimodel/ + tokenizer/) and
// exposes the benchmark endpoints the Rust client drives.
//
// Like Apple's own `llm-benchmark`, this uses the LOW-LEVEL Core AI API
// (LanguageBundle + EngineFactory + InferenceEngine) feeding RAW token IDs
// directly — no chat template. That is what lets pipette drive deterministic
// prompt/decode token counts on decode-only bundles, which the high-level
// LanguageModelSession path cannot do (it requires an embedded chat template).
//
// Endpoints (JSON, POST unless noted):
//   GET  /health                        -> {"ok": true}
//   POST /prepare                       -> {"ok": true}  (50ms settle + KV reset; untimed)
//   POST /prefill_throughput            -> {"prompt_tps": f, "prompt_tokens": N}
//   POST /decode_throughput             -> {"generation_tps": f, "decode_tokens": N}
//   POST /max_memory_usage              -> {"prompt_tokens": N, "completion_tokens": M}
//   POST /end_to_end_latency            -> {"total_ms": f, "prompt_tokens": N, "completion_tokens": M}
//   POST /shutdown                      -> exits

import CoreAILanguageModels
import CoreAIShared
import Foundation
import Network

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

struct Args {
    var modelDir: String
    var port: UInt16
    var seed: UInt64
}

enum SidecarError: Error, CustomStringConvertible {
    case missingArg(String)
    case unknownArg(String)
    case missingModel
    case badRequest(String)
    case invalidArg(String, String)

    var description: String {
        switch self {
        case .missingArg(let a): return "missing value for argument \(a)"
        case .invalidArg(let name, let raw): return "invalid value for --\(name): \(raw)"
        case .unknownArg(let a): return "unknown argument \(a)"
        case .missingModel: return "--model <bundle-dir> is required"
        case .badRequest(let m): return m
        }
    }
}

func parseArgs(_ args: [String]) throws -> Args {
    var modelDir: String?
    var port: UInt16 = 0
    var seed: UInt64 = 0
    var i = 0
    let argv = Array(args.dropFirst())
    func next() throws -> String {
        guard i + 1 < argv.count else { throw SidecarError.missingArg(argv[i]) }
        i += 1
        return argv[i]
    }
    while i < argv.count {
        let a = argv[i]
        switch a {
        case "--model": modelDir = try next()
        case "--port": port = try parseArg(try next(), as: UInt16.self, name: "port")
        case "--seed": seed = try parseArg(try next(), as: UInt64.self, name: "seed")
        default: throw SidecarError.unknownArg(a)
        }
        i += 1
    }
    guard let modelDir else { throw SidecarError.missingModel }
    return Args(modelDir: modelDir, port: port, seed: seed)
}

private func parseArg<T: FixedWidthInteger>(_ raw: String, as _: T.Type, name: String) throws -> T {
    guard let value = T(raw) else {
        throw SidecarError.invalidArg(name, raw)
    }
    return value
}

// ---------------------------------------------------------------------------
// Low-level Core AI engine wrapper
// ---------------------------------------------------------------------------

final class CoreAIEngine: @unchecked Sendable {
    let engine: any InferenceEngine
    let vocabSize: Int
    let name: String
    let seed: UInt64
    let eosTokenIds: Set<Int32>
    /// True when Apple's specialization cache already had this model's
    /// artifacts before `createEngine` (no new cache files appeared).
    let specializationCacheHit: Bool

    init(bundleDir: URL, seed: UInt64) async throws {
        // S=1 zoo decode bundles (catalog hint "pipelined") are static graphs that
        // accept a single token per forward; multi-token prefill chunks are rejected
        // (shape-substitution fatal). COREAI_CHUNK_THRESHOLD=1 makes the runtime
        // chunk the prompt into 1-token steps — the same guard coreai-kit's
        // ModelRuntime sets for its .pipelined engine variant.
        if getenv("COREAI_CHUNK_THRESHOLD") == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
        let bundle = try LanguageBundle(at: bundleDir)
        let modelURL = try bundle.requireModelURL(for: "main")
        let config = ModelConfig(
            name: bundle.name,
            tokenizer: bundle.tokenizer,
            vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength,
            serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main"
        )
        let configData = try JSONEncoder().encode(config)
        let beforeCache = specializationCacheFiles()
        self.engine = try await EngineFactory.createEngine(config: configData, modelURL: modelURL)
        let afterCache = specializationCacheFiles()
        self.specializationCacheHit = !beforeCache.isEmpty && afterCache == beforeCache
        self.vocabSize = bundle.vocabSize
        self.name = bundle.name
        self.seed = seed
        self.eosTokenIds = loadEosTokenIds(from: bundleDir)
    }

    /// Deterministic pseudo-random token prompt of exactly `count` tokens.
    func randomPrompt(count: Int) -> [Int32] {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        var out = [Int32]()
        out.reserveCapacity(count)
        let v = UInt64(vocabSize)
        for _ in 0..<count {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z = z ^ (z >> 31)
            var token = Int32(z % v)
            // Skip EOS so a uniform vocab sample cannot plant a stop token in
            // the prompt (MLX's `_suppress_eos` equivalent for this path).
            if !eosTokenIds.isEmpty && v > 1 {
                var guardCount = 0
                while eosTokenIds.contains(token) && guardCount < 16 {
                    state = state &+ 0x9E37_79B9_7F4A_7C15
                    z = state
                    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                    z = z ^ (z >> 31)
                    token = Int32(z % v)
                    guardCount += 1
                }
                // Retry budget exhausted (pathological vocab): step
                // deterministically to the next non-EOS id so EOS can never
                // land in a prompt, rather than falling through with `token`
                // still an EOS id.
                while eosTokenIds.contains(token) {
                    token = Int32((UInt64(bitPattern: Int64(token)) &+ 1) % v)
                }
            }
            out.append(token)
        }
        return out
    }

    /// Run one generation. Returns per-phase timing derived like Apple's
    /// llm-benchmark: promptTps over the prefill span (to first generated
    /// token), genTps over the decode span (from first token onward).
    ///
    /// `prepare()` must be called before this — the settle sleep and KV reset
    /// are setup, not measurement, and would otherwise land inside an
    /// end-to-end latency reading.
    func prepare() async {
        // A short settle between generations. Without it the Core AI engine can
        // still be draining the previous decode when the next rep starts, which
        // perturbs the very first prompt_seconds measurement. Deliberately best-
        // effort (`try?`): a scheduling miss is not worth failing the run over.
        try? await Task.sleep(for: .milliseconds(50))
        try? await engine.reset()
    }

    func generate(prompt: [Int32], maxTokens: Int) async throws -> GenResult {
        let options = InferenceOptions(maxTokens: maxTokens, includeLogits: false)
        let sampling = SamplingConfiguration(temperature: 0)
        let start = ContinuousClock.now
        let stream = try await engine.generate(
            with: prompt, samplingConfiguration: sampling, inferenceOptions: options
        )

        var promptSeconds: Double = 0
        var genStart = ContinuousClock.now
        var count = 0

        for try await _ in stream {
            if count == 0 {
                let now = ContinuousClock.now
                promptSeconds = seconds(from: start, to: now)
                genStart = now
            }
            count += 1
        }
        let genSeconds = seconds(from: genStart, to: .now)
        let totalSeconds = seconds(from: start, to: .now)
        let promptTps = promptSeconds > 0 ? Double(prompt.count) / promptSeconds : 0
        // Pipette's contract is deterministic: a decode of N tokens must report
        // completion_tokens == N (so a 1-token max-memory cell validates). But
        // the rate over the span [firstToken, end] covers only the N-1
        // inter-token intervals, so the numerator for tps is count - 1 — using
        // `count` would overstate throughput by N/(N-1) (~14% at 8 tokens).
        let decodeCount = count
        let genTps = (genSeconds > 0 && count > 1) ? Double(count - 1) / genSeconds : 0
        return GenResult(
            promptTps: promptTps,
            genTps: genTps,
            promptTokens: prompt.count,
            completionTokens: decodeCount,
            genSeconds: genSeconds,
            totalSeconds: totalSeconds
        )
    }

    private func seconds(from start: ContinuousClock.Instant, to end: ContinuousClock.Instant) -> Double {
        let d = start.duration(to: end)
        return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }
}

struct GenResult {
    let promptTps: Double
    let genTps: Double
    let promptTokens: Int
    let completionTokens: Int
    let genSeconds: Double
    // Whole-generation span (prefill + decode), matching MLX's e2e timing.
    let totalSeconds: Double
}

/// All EOS-like token ids to keep out of sampled prompts. Reads
/// `tokenizer_config.json` then `generation_config.json`; accepts a scalar
/// `eos_token_id` or an array of them (HF configs use both forms).
func loadEosTokenIds(from bundleDir: URL) -> Set<Int32> {
    var ids = Set<Int32>()
    let candidates = [
        bundleDir.appendingPathComponent("tokenizer/tokenizer_config.json"),
        bundleDir.appendingPathComponent("tokenizer/generation_config.json"),
        bundleDir.appendingPathComponent("generation_config.json"),
    ]
    for url in candidates {
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            continue
        }
        switch obj["eos_token_id"] {
        case let n as Int:
            ids.insert(Int32(truncatingIfNeeded: n))
        case let arr as [Any]:
            for item in arr {
                if let n = item as? Int {
                    ids.insert(Int32(truncatingIfNeeded: n))
                }
            }
        default:
            break
        }
    }
    return ids
}

func specializationCacheFiles() -> Set<String> {
    let fm = FileManager.default
    let home = fm.homeDirectoryForCurrentUser
    let roots = [
        home.appendingPathComponent("Library/Caches/com.apple.coreai"),
        home.appendingPathComponent("Library/Caches/AIModelCache"),
    ]
    var files = Set<String>()
    for root in roots {
        guard let enumerator = fm.enumerator(at: root, includingPropertiesForKeys: nil) else {
            continue
        }
        for case let url as URL in enumerator {
            files.insert(url.path)
        }
    }
    return files
}

// ---------------------------------------------------------------------------
// Minimal HTTP server (Network framework)
// ---------------------------------------------------------------------------

private final class ReadBuffer: @unchecked Sendable {
    var data = Data()
}

final class SidecarServer: @unchecked Sendable {
    let port: UInt16
    let engine: CoreAIEngine
    let listener: NWListener
    private let chainLock = NSLock()
    private var postChain: Task<Void, Never> = Task {}

    init(port: UInt16, engine: CoreAIEngine) throws {
        self.port = port
        self.engine = engine
        guard let p = NWEndpoint.Port(rawValue: port) else {
            throw SidecarError.badRequest("invalid port")
        }
        // Restrict the listener to loopback so the benchmark endpoints
        // (including /shutdown) are never reachable from the LAN. The Rust
        // client always connects to 127.0.0.1, and the MLX sidecar binds the
        // same way. requiredInterfaceType must be set on the parameters before
        // the listener is created; NWListener.requiredLocalEndpoint is get-only.
        let params = NWParameters.tcp
        params.requiredInterfaceType = .loopback
        self.listener = try NWListener(using: params, on: p)
    }

    func start() {
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                FileHandle.standardError.write(
                    Data("PIPETTE_COREAI_READY port=\(self.port) model=\(self.engine.name) specialization_cache=\(self.engine.specializationCacheHit ? "hit" : "miss")\n".utf8)
                )
            case .failed(let error):
                FileHandle.standardError.write(
                    Data("PIPETTE_COREAI_ERROR bind failed: \(error)\n".utf8)
                )
                exit(1)
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] conn in
            guard let self else { conn.cancel(); return }
            conn.start(queue: .global())
            self.handle(conn)
        }
        listener.start(queue: .global())
    }

    private func handle(_ conn: NWConnection) {
        let buf = ReadBuffer()
        func read() {
            conn.receive(minimumIncompleteLength: 1, maximumLength: 1_048_576) { data, _, isComplete, error in
                if let data, !data.isEmpty { buf.data.append(data) }
                if self.requestComplete(buf.data) {
                    self.dispatch(buf.data, to: conn)
                    return
                }
                if error != nil || isComplete { conn.cancel(); return }
                read()
            }
        }
        read()
    }

    private func requestComplete(_ data: Data) -> Bool {
        guard let s = String(data: data, encoding: .utf8),
              let headerEnd = s.range(of: "\r\n\r\n") else {
            return false
        }
        let header = s[s.startIndex..<headerEnd.lowerBound]
        // The fixed pipette client sends the whole body in one write, but honor
        // Content-Length anyway so a request whose body arrives in a later TCP
        // segment is still read correctly.
        let length = header
            .components(separatedBy: "\r\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { $0.lowercased().hasPrefix("content-length:") }
            .flatMap { $0.split(separator: ":").last }
            .flatMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard let length else { return true } // no body expected; headers are enough
        return data.count >= headerEnd.upperBound.utf16Offset(in: s) + length
    }

    private func dispatch(_ raw: Data, to conn: NWConnection) {
        guard let s = String(data: raw, encoding: .utf8),
              let headerEnd = s.range(of: "\r\n\r\n"),
              let requestLine = s.components(separatedBy: "\r\n").first?.components(separatedBy: " "),
              requestLine.count >= 2 else {
            respond("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n", to: conn)
            return
        }
        let method = requestLine[0]
        let path = requestLine[1]
        let body = String(s[headerEnd.upperBound...])

        if method == "GET" && path == "/health" {
            respond(json([
                "ok": true,
                "specialization_cache_hit": engine.specializationCacheHit,
            ]), to: conn)
        } else if method == "POST" && path == "/shutdown" {
            respond(json(["ok": true]), to: conn)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { exit(0) }
        } else if method == "POST" {
            // Serialise generations: two concurrent POSTs would interleave
            // engine.reset() / generate() on this @unchecked Sendable class.
            chainLock.lock()
            let previous = postChain
            postChain = Task {
                await previous.value
                let response = await self.post(path, body: body)
                self.respond(response, to: conn)
            }
            chainLock.unlock()
        } else {
            respond(json(["error": "unknown endpoint: \(method) \(path)"], status: 404), to: conn)
        }
    }

    private func post(_ path: String, body: String) async -> String {
        do {
            if path == "/prepare" {
                // Untimed setup. The Rust harness calls this from
                // measurement::run's prepare closure so the 50 ms settle and
                // KV reset cannot land inside an end-to-end wall clock.
                await engine.prepare()
                return json(["ok": true])
            }
            guard let obj = try JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any] else {
                return json(["error": "request body must be a JSON object"], status: 400)
            }
            switch path {
            case "/prefill_throughput":
                guard let n = obj["prompt_tokens"] as? Int, n > 0 else {
                    return json(["error": "'prompt_tokens' must be a positive integer"], status: 400)
                }
                let r = try await engine.generate(prompt: engine.randomPrompt(count: n), maxTokens: 1)
                // Echo the ACTUALS, not the request: the client-side
                // determinism guard compares these against the requested
                // counts and must be able to fire on an early stop.
                return json(["prompt_tps": r.promptTps, "prompt_tokens": r.promptTokens])
            case "/decode_throughput":
                guard let p = obj["prompt_tokens"] as? Int, p > 0,
                      let d = obj["decode_tokens"] as? Int, d > 1 else {
                    return json(["error": "'decode_tokens' must be >= 2 (generation_tps is N-1 inter-token intervals)"], status: 400)
                }
                let r = try await engine.generate(prompt: engine.randomPrompt(count: p), maxTokens: d)
                return json(["generation_tps": r.genTps, "decode_tokens": r.completionTokens])
            case "/max_memory_usage":
                guard let p = obj["prompt_tokens"] as? Int, p > 0,
                      let d = obj["decode_tokens"] as? Int, d > 0 else {
                    return json(["error": "'prompt_tokens'/'decode_tokens' positive ints required"], status: 400)
                }
                let r = try await engine.generate(prompt: engine.randomPrompt(count: p), maxTokens: d)
                return json(["prompt_tokens": r.promptTokens, "completion_tokens": r.completionTokens])
            case "/end_to_end_latency":
                guard let n = obj["prompt_tokens"] as? Int, n > 0,
                      let d = obj["decode_tokens"] as? Int, d > 0 else {
                    return json(["error": "'prompt_tokens'/'decode_tokens' positive ints required"], status: 400)
                }
                let r = try await engine.generate(prompt: engine.randomPrompt(count: n), maxTokens: d)
                return json([
                    "total_ms": r.totalSeconds * 1000.0,
                    "prompt_tokens": r.promptTokens,
                    "completion_tokens": r.completionTokens,
                ])
            default:
                return json(["error": "unknown endpoint: POST \(path)"], status: 404)
            }
        } catch {
            return json(["error": "\(error)"], status: 500)
        }
    }

    private func json(_ obj: [String: Any], status: Int = 200) -> String {
        let body = (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
        let statusText = status == 200 ? "OK" : "Error"
        return "HTTP/1.1 \(status) \(statusText)\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n" + String(data: body, encoding: .utf8)!
    }

    private func respond(_ response: String, to conn: NWConnection) {
        guard let data = response.data(using: .utf8) else { conn.cancel(); return }
        conn.send(content: data, completion: .contentProcessed { _ in conn.cancel() })
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

@main
struct SidecarMain {
    static func main() async {
        do {
            let args = try parseArgs(CommandLine.arguments)
            let dir = URL(fileURLWithPath: args.modelDir, isDirectory: true)
            FileHandle.standardError.write(Data("PIPETTE_COREAI_LOADING \(args.modelDir)\n".utf8))
            let engine = try await CoreAIEngine(bundleDir: dir, seed: args.seed)
            let server = try SidecarServer(port: args.port, engine: engine)
            server.start()
            // Suspend forever so the process stays alive; the NWListener's global
            // queue keeps serving. Never call dispatchMain()/RunLoop.run() from an
            // async @main — dispatchMain() traps there.
            await withCheckedContinuation { (_: CheckedContinuation<Void, Never>) in
                // never resumed
            }
        } catch {
            FileHandle.standardError.write(Data("PIPETTE_COREAI_ERROR \(error)\n".utf8))
            exit(1)
        }
    }
}
